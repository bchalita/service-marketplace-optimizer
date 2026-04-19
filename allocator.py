#!/usr/bin/env python3
"""Stage 1+2 of the clean-slate allocator: candidate pruning + greedy construction.

Given for a target day:
  - a pool of orders (lat/lng, time, duration, service, requested window)
  - monthly per-provider priors (home, certs, windows, capacity, radius)

Produces a feasible schedule without looking at the baseline for that day.
This is the DAY-AHEAD allocator: no baseline leakage.

Pipeline:
  Stage 1 — candidate pruning
    For each order: filter providers by cert + window_for(spec, date) fit
    + radius cap. Rank by zone match → distance → historical load. Keep top-K.
    Guardrail relax order (softest first): zone pref → window fit → radius.
  Stage 2 — greedy insertion
    Sort orders by tightness = 1 / (n_candidates * window_width). Tightest
    first. For each order find the spec+slot minimizing insertion cost.
    Fall back to overflow if no feasible slot exists.

Feasibility uses the locked calibrated travel model:
  speed_public_transport = 10 km/h
  speed_car              = 20 km/h
  handover_slack_min     = 15  (per handover, counted against daily cap)
  max_handover_events_per_day = 3

This file is the production clean-slate track. Stage 3 (local search
refinement) layers on top and warm-starts from this output.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from provider_priors import capacity_for, haversine_km, order_in_window, window_for

DATA_DIR = Path(__file__).resolve().parent / "data"
PREPARED_PATH = DATA_DIR / "prepared_data.json"
PRIORS_PATH = DATA_DIR / "provider_priors.json"

# --- Locked travel model ---
SPEED_PUBLIC_TRANSPORT = 10.0
SPEED_CAR = 20.0
HANDOVER_SLACK_MIN = 15
MAX_HANDOVER_EVENTS_PER_DAY = 3
SAME_ADDR_KM = 0.05
RECENCY_DAYS = 30  # skip specs whose last active date is >N days before target

# --- Allocator tunables ---
TOP_K = 25
INSERTION_IDLE_WEIGHT = 0.1
ZONE_MATCH_RADIUS_KM = 3.0  # within this distance of a historical zone centroid => zone match
OPEN_SPEC_PENALTY_KM = 25.0  # extra "km-equivalent" cost for opening a previously-unused provider.
                             # Swept {0, 2, 5, 8, 12, 16, 20, 25, 35} across 5 validation days.
                             # 25 is the Pareto-best: -2% providers, -2% travel, +2% o/s vs baseline.
                             # Overflow constant ~19/sweep so the penalty
                             # isn't hurting feasibility.


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Order:
    order_id: str
    lat: float
    lng: float
    service: str
    duration: int
    start_min: int  # scheduled start (we treat as hard start)
    end_min: int
    partner_id_baseline: Optional[str] = None  # ONLY for eval, not for allocation

    @property
    def window_width(self) -> int:
        # Current data has fixed start — width = 0. We synthesize a small window
        # later in Stage 3. For now tightness uses a minimum of 30 min as a floor.
        return max(30, self.end_min - self.start_min)


@dataclass
class ScheduledItem:
    order: Order
    start_min: int       # possibly adjusted from order.start_min in future versions
    end_min: int
    used_handover_slack: bool  # True if this transition consumed the slack budget


@dataclass
class SpecialistDay:
    spec_id: str
    prior: dict
    window: list[tuple[int, int]]  # blocks (Approach B)
    capacity: dict
    items: list[ScheduledItem] = field(default_factory=list)
    handover_events_used: int = 0

    @property
    def n_orders(self) -> int:
        return len(self.items)

    @property
    def total_minutes(self) -> int:
        return sum(it.order.duration for it in self.items)

    def sorted_items(self) -> list[ScheduledItem]:
        return sorted(self.items, key=lambda it: it.start_min)


# ---------------------------------------------------------------------------
# Travel model
# ---------------------------------------------------------------------------

def _is_car(prev_svc: str, next_svc: str, car_services: set) -> bool:
    return prev_svc in car_services or next_svc in car_services


def travel_time_min(a: Order, b: Order, car_services: set) -> float:
    dist = haversine_km(a.lat, a.lng, b.lat, b.lng)
    if dist < SAME_ADDR_KM:
        return 0.0
    speed = SPEED_CAR if _is_car(a.service, b.service, car_services) else SPEED_PUBLIC_TRANSPORT
    return (dist / speed) * 60.0


def required_gap_min(travel_time: float) -> float:
    return max(0.0, travel_time - HANDOVER_SLACK_MIN)


def transition_feasible(a: Order, b: Order, car_services: set) -> tuple[bool, bool]:
    """Returns (feasible, used_handover_slack).
    used_handover_slack := True if the transition only fits because of the slack.
    """
    observed_gap = b.start_min - a.end_min
    tt = travel_time_min(a, b, car_services)
    if tt == 0.0:
        return (observed_gap >= 0, False)
    req = required_gap_min(tt)
    if observed_gap < req:
        return (False, False)
    used_slack = observed_gap < tt
    return (True, used_slack)


def can_insert(specday: SpecialistDay, order: Order, car_services: set) -> bool:
    """Can `order` be inserted into `specday` without breaking anything?"""
    # Order must fit in at least one window block
    if not order_in_window(specday.window, order.start_min, order.end_min):
        return False

    # Capacity
    cap = specday.capacity
    if cap:
        if specday.n_orders + 1 > cap["max_orders"]:
            return False
        if specday.total_minutes + order.duration > int(cap["typical_min"] * 1.25):
            # Allow 25% over typical for outlier-but-plausible days
            return False

    # Check transition with surrounding items
    items = specday.sorted_items()
    # Insert position
    pos = 0
    for i, it in enumerate(items):
        if it.start_min > order.start_min:
            break
        pos = i + 1

    prev_it = items[pos - 1] if pos > 0 else None
    next_it = items[pos] if pos < len(items) else None

    extra_slack_used = 0
    if prev_it:
        ok, used = transition_feasible(prev_it.order, order, car_services)
        if not ok:
            return False
        if used:
            extra_slack_used += 1
    if next_it:
        ok, used = transition_feasible(order, next_it.order, car_services)
        if not ok:
            return False
        if used:
            extra_slack_used += 1

    # If inserting between two existing items, we may invalidate the
    # prev->next transition that WAS feasible. That's fine because we're
    # splitting it. But we must not exceed the handover cap:
    if specday.handover_events_used + extra_slack_used > MAX_HANDOVER_EVENTS_PER_DAY:
        return False
    return True


def insertion_cost(
    specday: SpecialistDay,
    order: Order,
    car_services: set,
    open_spec_penalty: float = OPEN_SPEC_PENALTY_KM,
) -> Optional[float]:
    """Cost of inserting `order` into `specday`. None if infeasible.

    Cost model:
        delta_km + INSERTION_IDLE_WEIGHT * delta_idle + (open_spec_penalty if empty)

    The open-spec penalty discourages greedy from spreading work across many
    one-order providers when a comparable existing provider is available. Without it,
    empty providers have cost ~= haversine(home, order) which is bounded below the
    radius cap (~5 km) and usually beats adding a second order to an existing
    provider whose nearest item is more than 5 km away.
    """
    if not can_insert(specday, order, car_services):
        return None

    items = specday.sorted_items()
    pos = 0
    for i, it in enumerate(items):
        if it.start_min > order.start_min:
            break
        pos = i + 1

    prev_it = items[pos - 1] if pos > 0 else None
    next_it = items[pos] if pos < len(items) else None

    # Delta travel: insert order between prev and next
    def dist(a: Order, b: Order) -> float:
        return haversine_km(a.lat, a.lng, b.lat, b.lng)

    if prev_it and next_it:
        old = dist(prev_it.order, next_it.order)
        new = dist(prev_it.order, order) + dist(order, next_it.order)
        delta_km = new - old
    elif prev_it:
        delta_km = dist(prev_it.order, order)
    elif next_it:
        delta_km = dist(order, next_it.order)
    else:
        # First order on this provider-day: anchor at home distance
        home = specday.prior["home"]
        delta_km = haversine_km(home[0], home[1], order.lat, order.lng)

    # Delta idle: the new item splits idle between prev and next
    if prev_it:
        gap_before = order.start_min - prev_it.end_min
    else:
        gap_before = 0
    if next_it:
        gap_after = next_it.start_min - order.end_min
    else:
        gap_after = 0
    # Idle cost is the new gaps (positive idle = waiting time)
    new_idle = max(0, gap_before) + max(0, gap_after)
    # Old idle (what we replaced) — between prev and next if both exist
    if prev_it and next_it:
        old_idle = max(0, next_it.start_min - prev_it.end_min)
    else:
        old_idle = 0
    delta_idle = new_idle - old_idle

    cost = delta_km + INSERTION_IDLE_WEIGHT * delta_idle
    if specday.n_orders == 0:
        cost += open_spec_penalty
    return cost


# ---------------------------------------------------------------------------
# Candidate map
# ---------------------------------------------------------------------------

def zone_match(order: Order, prior: dict) -> bool:
    for z in prior.get("historical_zones", []):
        if z.get("lat") is None or z.get("lng") is None:
            continue
        if haversine_km(z["lat"], z["lng"], order.lat, order.lng) <= ZONE_MATCH_RADIUS_KM:
            return True
    return False


def _candidate_score(order: Order, prior: dict, load_signal: int) -> tuple:
    """(zone_match DESC, distance_from_home ASC, load ASC).
    Lower tuple = better.
    """
    home = prior["home"]
    d = haversine_km(home[0], home[1], order.lat, order.lng)
    zm = 0 if zone_match(order, prior) else 1  # 0 = match (better)
    return (zm, d, load_signal)


def build_candidate_map(
    orders: list[Order],
    priors: dict,
    certifications: dict,
    the_date: date,
    load_signal_by_spec: dict[str, int],
) -> tuple[dict[str, list[str]], dict[str, dict]]:
    """Returns (candidate_map, diagnostic_map).

    candidate_map[order_id] = [spec_id, ...] top-K.
    diagnostic_map[order_id] = {
      'relaxed_zone': bool, 'relaxed_window': bool, 'relaxed_radius': bool,
      'pool_size': int,
    }
    """
    candidate_map: dict[str, list[str]] = {}
    diagnostics: dict[str, dict] = {}

    # Pre-filter: skip providers by recency
    active_priors = {}
    for spec_id, prior in priors.items():
        last_str = prior.get("last_active_date")
        if last_str:
            last_d = date.fromisoformat(last_str)
            if (the_date - last_d).days > RECENCY_DAYS:
                continue
        active_priors[spec_id] = prior

    for order in orders:
        # First pass: full filter
        strict_pool = []
        for spec_id, prior in active_priors.items():
            certs = certifications.get(spec_id, [])
            if order.service not in certs:
                continue
            blocks = window_for(prior, the_date)
            if not order_in_window(blocks, order.start_min, order.end_min):
                continue
            home = prior["home"]
            if haversine_km(home[0], home[1], order.lat, order.lng) > prior["radius_cap_km"]:
                continue
            ls = load_signal_by_spec.get(spec_id, 0)
            strict_pool.append((_candidate_score(order, prior, ls), spec_id))

        relaxed_zone = False
        relaxed_window = False
        relaxed_radius = False
        pool = strict_pool

        # Relax 1: drop zone preference (no-op at filter level; zone is only
        # a ranking signal so strict pool already includes non-zone-match
        # providers). We flag relaxed_zone only if strict_pool was empty.

        # Relax 2: drop window fit
        if not pool:
            relaxed_zone = True  # crossed step 1 trivially
            relaxed_window = True
            fallback = []
            for spec_id, prior in active_priors.items():
                certs = certifications.get(spec_id, [])
                if order.service not in certs:
                    continue
                home = prior["home"]
                if haversine_km(home[0], home[1], order.lat, order.lng) > prior["radius_cap_km"]:
                    continue
                ls = load_signal_by_spec.get(spec_id, 0)
                fallback.append((_candidate_score(order, prior, ls), spec_id))
            pool = fallback

        # Relax 3: drop radius
        if not pool:
            relaxed_radius = True
            fallback = []
            for spec_id, prior in active_priors.items():
                certs = certifications.get(spec_id, [])
                if order.service not in certs:
                    continue
                ls = load_signal_by_spec.get(spec_id, 0)
                fallback.append((_candidate_score(order, prior, ls), spec_id))
            pool = fallback

        pool.sort()
        chosen = [spec_id for _, spec_id in pool[:TOP_K]]
        candidate_map[order.order_id] = chosen
        diagnostics[order.order_id] = {
            "relaxed_zone": relaxed_zone,
            "relaxed_window": relaxed_window,
            "relaxed_radius": relaxed_radius,
            "pool_size": len(pool),
            "top_k_used": len(chosen),
        }

    return candidate_map, diagnostics


# ---------------------------------------------------------------------------
# Greedy allocator
# ---------------------------------------------------------------------------

def allocate_day(
    the_date: date,
    orders: list[Order],
    priors: dict,
    certifications: dict,
    car_services: set,
    open_spec_penalty: float = OPEN_SPEC_PENALTY_KM,
) -> dict:
    """Run Stage 1+2 for a single day. Returns a result dict."""

    # Initial load signal — zero; could be replaced with running count across the day
    load_by_spec: dict[str, int] = defaultdict(int)

    candidate_map, diagnostics = build_candidate_map(
        orders, priors, certifications, the_date, load_by_spec
    )

    specdays: dict[str, SpecialistDay] = {}
    assigned: list[tuple[str, str]] = []  # (order_id, spec_id)
    overflow: list[dict] = []

    def tightness(o: Order) -> float:
        n = max(1, len(candidate_map.get(o.order_id, [])))
        return 1.0 / (n * o.window_width)

    sorted_orders = sorted(orders, key=lambda o: (-tightness(o), o.start_min))

    for order in sorted_orders:
        candidates = candidate_map.get(order.order_id, [])
        if not candidates:
            overflow.append({"order_id": order.order_id, "reason": "no_candidates"})
            continue

        best_spec = None
        best_cost = math.inf
        for spec_id in candidates:
            if spec_id not in specdays:
                prior = priors[spec_id]
                specdays[spec_id] = SpecialistDay(
                    spec_id=spec_id,
                    prior=prior,
                    window=window_for(prior, the_date),
                    capacity=capacity_for(prior, the_date),
                )
            specday = specdays[spec_id]
            cost = insertion_cost(specday, order, car_services, open_spec_penalty)
            if cost is not None and cost < best_cost:
                best_cost = cost
                best_spec = spec_id

        if best_spec is None:
            overflow.append({"order_id": order.order_id, "reason": "no_feasible_slot"})
            continue

        specday = specdays[best_spec]
        # Determine if the insertion used handover slack (for counting)
        items = specday.sorted_items()
        pos = 0
        for i, it in enumerate(items):
            if it.start_min > order.start_min:
                break
            pos = i + 1
        used = 0
        if pos > 0:
            _, u = transition_feasible(items[pos - 1].order, order, car_services)
            if u:
                used += 1
        if pos < len(items):
            _, u = transition_feasible(order, items[pos].order, car_services)
            if u:
                used += 1
        specday.handover_events_used += used
        specday.items.append(
            ScheduledItem(
                order=order,
                start_min=order.start_min,
                end_min=order.end_min,
                used_handover_slack=(used > 0),
            )
        )
        load_by_spec[best_spec] += 1
        assigned.append((order.order_id, best_spec))

    return {
        "date": str(the_date),
        "assigned": assigned,
        "overflow": overflow,
        "specdays": specdays,
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# Metrics for a day's schedule
# ---------------------------------------------------------------------------

def day_travel_km(specdays: dict[str, SpecialistDay]) -> float:
    total = 0.0
    for sd in specdays.values():
        items = sd.sorted_items()
        for i in range(len(items) - 1):
            a = items[i].order
            b = items[i + 1].order
            total += haversine_km(a.lat, a.lng, b.lat, b.lng)
    return total


def baseline_day_travel_km(baseline_by_spec: dict[str, list[Order]]) -> float:
    total = 0.0
    for spec_id, orders in baseline_by_spec.items():
        orders = sorted(orders, key=lambda o: o.start_min)
        for i in range(len(orders) - 1):
            a, b = orders[i], orders[i + 1]
            total += haversine_km(a.lat, a.lng, b.lat, b.lng)
    return total


# ---------------------------------------------------------------------------
# Validation entrypoint
# ---------------------------------------------------------------------------

def load_for_day(
    the_date: date,
    include_unserved: bool = True,
    priors_override: Optional[dict] = None,
) -> tuple[list[Order], dict, dict, set, dict]:
    """Load the full order pool for a target day.

    include_unserved:
      True  -> full pool (finalized_orders + unserved_orders). This is the
              honest DR benchmark: baseline 'dropped' the unserved orders,
              and our pipeline gets to try to place them.
      False -> served pool only. Kept for legacy callers and quick sanity
              checks that don't care about DR.

    priors_override:
      If provided, use this priors dict instead of reading from PRIORS_PATH.
      Used by the look-ahead bias check to inject time-split priors.

    Returns: (orders, priors, certifications, car_services, baseline_by_spec)
      - baseline_by_spec groups the baseline-served orders only. Orders from
        unserved_orders have partner_id_baseline=None and don't appear there.
    """
    prepared = json.loads(PREPARED_PATH.read_text())
    if priors_override is not None:
        priors = priors_override
    else:
        priors = json.loads(PRIORS_PATH.read_text())
    car_services = set(prepared["parameters"]["car_services"])
    certifications = prepared.get("certifications", {})

    the_str = the_date.isoformat()
    orders: list[Order] = []
    baseline_by_spec: dict[str, list[Order]] = defaultdict(list)

    for row in prepared["finalized_orders"]:
        if row["date"] != the_str:
            continue
        o = Order(
            order_id=row["order_id"],
            lat=row["lat"],
            lng=row["lng"],
            service=row["service"],
            duration=row["duration"],
            start_min=row["start_min"],
            end_min=row["end_min"],
            partner_id_baseline=row.get("partner_id"),
        )
        orders.append(o)
        if o.partner_id_baseline:
            baseline_by_spec[o.partner_id_baseline].append(o)

    if include_unserved:
        for row in prepared.get("unserved_orders", []):
            if row["date"] != the_str:
                continue
            o = Order(
                order_id=row["order_id"],
                lat=row["lat"],
                lng=row["lng"],
                service=row["service"],
                duration=row["duration"],
                start_min=row["start_min"],
                end_min=row["end_min"],
                partner_id_baseline=None,  # by definition
            )
            orders.append(o)

    return orders, priors, certifications, car_services, baseline_by_spec


def main() -> None:
    # Example: pick a target date from your prepared_data.json
    target = date(2026, 1, 15)  # replace with a date present in your data
    orders, priors, certifications, car_services, baseline_by_spec = load_for_day(target)
    print(f"== Clean-slate allocator validation: {target} ==")
    print(f"Orders to assign: {len(orders)}")
    print(f"Baseline providers used: {len(baseline_by_spec)}")
    bl_km = baseline_day_travel_km(baseline_by_spec)
    print(f"Baseline travel: {bl_km:.2f} km\n")

    result = allocate_day(target, orders, priors, certifications, car_services)
    specdays = result["specdays"]
    overflow = result["overflow"]
    diagnostics = result["diagnostics"]

    cs_km = day_travel_km(specdays)
    n_assigned = sum(sd.n_orders for sd in specdays.values())
    n_specs_used = sum(1 for sd in specdays.values() if sd.n_orders > 0)

    print(f"Clean-slate assigned: {n_assigned}/{len(orders)}")
    print(f"Clean-slate overflow: {len(overflow)}")
    print(f"Clean-slate providers used: {n_specs_used}")
    print(f"Clean-slate travel: {cs_km:.2f} km")
    if bl_km > 0:
        print(f"Delta vs baseline: {100 * (cs_km - bl_km) / bl_km:+.1f}%")

    # Orders/provider (key utilization metric)
    if n_specs_used:
        print(f"Orders/provider: {n_assigned / n_specs_used:.2f}")
    baseline_n_specs = len(baseline_by_spec)
    print(f"  (baseline orders/provider: {len(orders) / baseline_n_specs:.2f})")

    # Distribution of orders per provider
    loads = [sd.n_orders for sd in specdays.values() if sd.n_orders > 0]
    load_hist: dict[int, int] = defaultdict(int)
    for ln in loads:
        load_hist[ln] += 1
    print(f"  load histogram: ", end="")
    for k in sorted(load_hist):
        print(f"{k}:{load_hist[k]} ", end="")
    print()

    # Baseline load histogram for comparison
    bl_loads = [len(v) for v in baseline_by_spec.values()]
    bl_hist: dict[int, int] = defaultdict(int)
    for ln in bl_loads:
        bl_hist[ln] += 1
    print(f"  baseline load histogram: ", end="")
    for k in sorted(bl_hist):
        print(f"{k}:{bl_hist[k]} ", end="")
    print()

    # Honesty check: travel per order
    print(f"\nHonesty: travel per order")
    print(f"  baseline:    {bl_km / len(orders):.2f} km/order")
    print(f"  clean-slate: {cs_km / max(1, n_assigned):.2f} km/order")

    # Handover slack usage
    total_handovers = sum(sd.handover_events_used for sd in specdays.values())
    specdays_with_handovers = sum(1 for sd in specdays.values() if sd.handover_events_used > 0)
    print(f"\nHandover slack usage: {total_handovers} events across {specdays_with_handovers} provider-days")

    # Candidate pool diagnostics
    relaxed_zone = sum(1 for d in diagnostics.values() if d["relaxed_zone"])
    relaxed_window = sum(1 for d in diagnostics.values() if d["relaxed_window"])
    relaxed_radius = sum(1 for d in diagnostics.values() if d["relaxed_radius"])
    pool_sizes = [d["pool_size"] for d in diagnostics.values()]
    tk_used = [d["top_k_used"] for d in diagnostics.values()]
    from statistics import median
    print()
    print("Candidate map diagnostics:")
    print(f"  Median pool size (before top-K cap): {int(median(pool_sizes))}")
    print(f"  Median top-K used:                   {int(median(tk_used))}")
    print(f"  Orders needing zone relax:           {relaxed_zone}")
    print(f"  Orders needing window relax:         {relaxed_window}")
    print(f"  Orders needing radius relax:         {relaxed_radius}")

    # Overflow reasons
    if overflow:
        reasons: dict[str, int] = defaultdict(int)
        for o in overflow:
            reasons[o["reason"]] += 1
        print("\nOverflow reasons:")
        for r, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {r}: {c}")


if __name__ == "__main__":
    main()
