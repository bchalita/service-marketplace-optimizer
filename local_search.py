#!/usr/bin/env python3
"""Stage 3 (local search): hill-climbing refinement of the clean-slate greedy.

Starts from allocator.allocate_day output and applies relocate
and swap moves under the FULL natural objective:

    + 2_000_000 * unassigned
    +       100 * travel_km     (inter-order, haversine)
    +        10 * idle_min
    +         5 * providers_used
    +         2 * zone_mismatch

Same locked travel model as greedy: 10/20 km/h, handover_slack_min=15,
max_handover_events_per_day=3. Window/cert constraints are HARD; radius is
enforced on the destination side (local search never introduces relaxation).
Local search only moves orders BETWEEN currently-active providers (no opening
new ones; greedy's open-spec penalty already ran).

Why local search and not CP-SAT: the constraint programming approach couldn't
represent travel/idle without successor variables or Circuit constraints, so
its partial objective was misaligned with the natural objective and the
strict-improvement check rejected every solution. Local search optimizes
the full natural objective directly via cheap O(provider_size) deltas.

Strategy:
  1. Build state from greedy (per-provider sorted order list + cached metrics).
  2. Relocate pass: for each order, try moving it to every other active
     provider. First improvement wins. Repeat until no improving move.
  3. Swap pass: for each pair of orders in different providers, try swapping.
     First improvement wins. Repeat until no improving move.
  4. Alternate relocate/swap until both stabilize or time limit hit.
  5. Strict-improvement check vs greedy_obj.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from allocator import (
    MAX_HANDOVER_EVENTS_PER_DAY,
    Order,
    ScheduledItem,
    SpecialistDay,
    allocate_day,
    baseline_day_travel_km,
    day_travel_km,
    haversine_km,
    load_for_day,
    transition_feasible,
    travel_time_min,
    zone_match,
)
from provider_priors import capacity_for, order_in_window, window_for

# --- Locked objective weights ---
W_UNASSIGNED = 2_000_000
W_TRAVEL_KM = 100
W_IDLE_MIN = 10
W_SPECS_USED = 5
W_ZONE_MM = 2

TIME_LIMIT_S = 30


# ---------------------------------------------------------------------------
# Per-provider state + metric helpers
# ---------------------------------------------------------------------------

@dataclass
class SpecMetrics:
    sorted_idxs: list[int]
    km: float
    idle: int
    handover: int
    zone_mm: int
    n: int
    duration: int


@dataclass
class LocalSearchState:
    active_specs: set[str]
    assignment: dict[int, str]
    spec_orders: dict[str, list[int]]
    spec_metrics: dict[str, SpecMetrics]
    # Static per-provider data (cached for fast access)
    priors: dict
    spec_window: dict[str, list[tuple[int, int]]]
    spec_capacity: dict[str, dict]
    spec_certs: dict[str, set]
    spec_radius: dict[str, float]
    spec_home: dict[str, tuple[float, float]]


def _compute_metrics(
    order_idxs: list[int],
    orders: list[Order],
    prior: dict,
    car_services: set,
) -> Optional[SpecMetrics]:
    """Sort + compute metrics for a candidate provider assignment.
    Returns None if pairwise feasibility or handover cap is violated.
    Caller is responsible for cert/window/capacity checks before calling.
    """
    sorted_idxs = sorted(order_idxs, key=lambda i: (orders[i].start_min, i))
    n = len(sorted_idxs)
    km = 0.0
    idle = 0
    handover = 0
    for k in range(n - 1):
        a = orders[sorted_idxs[k]]
        b = orders[sorted_idxs[k + 1]]
        ok, used = transition_feasible(a, b, car_services)
        if not ok:
            return None
        km += haversine_km(a.lat, a.lng, b.lat, b.lng)
        tt = travel_time_min(a, b, car_services)
        gap = b.start_min - a.end_min
        idle += max(0, gap - int(round(tt)))
        if used:
            handover += 1
    if handover > MAX_HANDOVER_EVENTS_PER_DAY:
        return None
    zone_mm = sum(1 for i in sorted_idxs if not zone_match(orders[i], prior))
    duration = sum(orders[i].duration for i in sorted_idxs)
    return SpecMetrics(
        sorted_idxs=sorted_idxs,
        km=km,
        idle=idle,
        handover=handover,
        zone_mm=zone_mm,
        n=n,
        duration=duration,
    )


def _basic_accept(
    new_idxs: list[int],
    delta_duration: int,
    spec_id: str,
    state: LocalSearchState,
    orders: list[Order],
    new_order_idx: Optional[int],
) -> bool:
    """Cert + window + radius (for the newly added order) + capacity check.
    new_order_idx is the order we're ADDING to spec_id (None for pure removal).
    """
    n = len(new_idxs)

    # Capacity (count + minutes)
    cap = state.spec_capacity[spec_id]
    if cap:
        if n > cap["max_orders"]:
            return False
        old_dur = state.spec_metrics[spec_id].duration if spec_id in state.spec_metrics else 0
        new_dur = old_dur + delta_duration
        if new_dur > int(cap["typical_min"] * 1.25):
            return False

    # The newly inserted order must pass cert/window/radius for this provider.
    # Existing orders are already valid (they were validated when assigned).
    if new_order_idx is not None:
        o = orders[new_order_idx]
        if o.service not in state.spec_certs[spec_id]:
            return False
        if not order_in_window(state.spec_window[spec_id], o.start_min, o.end_min):
            return False
        home = state.spec_home[spec_id]
        if haversine_km(home[0], home[1], o.lat, o.lng) > state.spec_radius[spec_id]:
            return False
    return True


# ---------------------------------------------------------------------------
# Build initial state from greedy output
# ---------------------------------------------------------------------------

def _build_state(
    greedy_specdays: dict[str, SpecialistDay],
    orders: list[Order],
    priors: dict,
    certifications: dict,
    car_services: set,
    the_date: date,
) -> LocalSearchState:
    order_idx = {o.order_id: i for i, o in enumerate(orders)}
    assignment: dict[int, str] = {}
    spec_orders: dict[str, list[int]] = {}
    spec_metrics: dict[str, SpecMetrics] = {}
    spec_window: dict[str, list[tuple[int, int]]] = {}
    spec_capacity: dict[str, dict] = {}
    spec_certs: dict[str, set] = {}
    spec_radius: dict[str, float] = {}
    spec_home: dict[str, tuple[float, float]] = {}

    for s, sd in greedy_specdays.items():
        idxs = [order_idx[it.order.order_id] for it in sd.items]
        spec_orders[s] = idxs
        for i in idxs:
            assignment[i] = s
        prior = priors[s]
        spec_window[s] = window_for(prior, the_date)
        spec_capacity[s] = capacity_for(prior, the_date)
        spec_certs[s] = set(certifications.get(s, []))
        spec_radius[s] = prior["radius_cap_km"]
        spec_home[s] = (prior["home"][0], prior["home"][1])
        m = _compute_metrics(idxs, orders, prior, car_services)
        # Greedy output should always be feasible; assert defensively.
        if m is None:
            raise RuntimeError(f"greedy provider {s} failed metric recomputation")
        spec_metrics[s] = m

    return LocalSearchState(
        active_specs=set(spec_orders.keys()),
        assignment=assignment,
        spec_orders=spec_orders,
        spec_metrics=spec_metrics,
        priors=priors,
        spec_window=spec_window,
        spec_capacity=spec_capacity,
        spec_certs=spec_certs,
        spec_radius=spec_radius,
        spec_home=spec_home,
    )


# ---------------------------------------------------------------------------
# Move evaluators
# ---------------------------------------------------------------------------

def _delta_relocate(
    state: LocalSearchState,
    i: int,
    src: str,
    dst: str,
    orders: list[Order],
    car_services: set,
) -> Optional[tuple[float, SpecMetrics, SpecMetrics]]:
    """Compute objective delta for moving order i from src to dst.
    Returns (delta, new_src_metrics, new_dst_metrics) or None if infeasible.
    Sign convention: delta < 0 = improvement.
    """
    o = orders[i]

    # Quick reject: dst must accept on cert/window/radius
    if o.service not in state.spec_certs[dst]:
        return None
    if not order_in_window(state.spec_window[dst], o.start_min, o.end_min):
        return None
    home = state.spec_home[dst]
    if haversine_km(home[0], home[1], o.lat, o.lng) > state.spec_radius[dst]:
        return None

    # Capacity on dst
    new_dst_idxs = state.spec_orders[dst] + [i]
    if not _basic_accept(new_dst_idxs, o.duration, dst, state, orders, i):
        return None

    # Recompute dst (with new order)
    new_dst = _compute_metrics(new_dst_idxs, orders, state.priors[dst], car_services)
    if new_dst is None:
        return None

    # Recompute src (without order i)
    new_src_idxs = [k for k in state.spec_orders[src] if k != i]
    if new_src_idxs:
        new_src = _compute_metrics(new_src_idxs, orders, state.priors[src], car_services)
        if new_src is None:
            return None  # removing an order should never break feasibility, but guard
    else:
        new_src = SpecMetrics(sorted_idxs=[], km=0.0, idle=0, handover=0, zone_mm=0, n=0, duration=0)

    old_src = state.spec_metrics[src]
    old_dst = state.spec_metrics[dst]

    delta_km = (new_src.km + new_dst.km) - (old_src.km + old_dst.km)
    delta_idle = (new_src.idle + new_dst.idle) - (old_src.idle + old_dst.idle)
    delta_zone = (new_src.zone_mm + new_dst.zone_mm) - (old_src.zone_mm + old_dst.zone_mm)
    delta_specs = 0
    if new_src.n == 0:
        delta_specs -= 1  # src dropped

    delta = (
        W_TRAVEL_KM * delta_km
        + W_IDLE_MIN * delta_idle
        + W_SPECS_USED * delta_specs
        + W_ZONE_MM * delta_zone
    )
    return delta, new_src, new_dst


def _delta_swap(
    state: LocalSearchState,
    i: int,
    j: int,
    src: str,
    dst: str,
    orders: list[Order],
    car_services: set,
) -> Optional[tuple[float, SpecMetrics, SpecMetrics]]:
    """Compute objective delta for swapping order i (in src) with order j (in dst)."""
    oi = orders[i]
    oj = orders[j]

    # Both moves must satisfy cert/window/radius on the new provider
    if oi.service not in state.spec_certs[dst]:
        return None
    if oj.service not in state.spec_certs[src]:
        return None
    if not order_in_window(state.spec_window[dst], oi.start_min, oi.end_min):
        return None
    if not order_in_window(state.spec_window[src], oj.start_min, oj.end_min):
        return None
    home_d = state.spec_home[dst]
    if haversine_km(home_d[0], home_d[1], oi.lat, oi.lng) > state.spec_radius[dst]:
        return None
    home_s = state.spec_home[src]
    if haversine_km(home_s[0], home_s[1], oj.lat, oj.lng) > state.spec_radius[src]:
        return None

    new_src_idxs = [k for k in state.spec_orders[src] if k != i] + [j]
    new_dst_idxs = [k for k in state.spec_orders[dst] if k != j] + [i]

    # Capacity (size unchanged but duration may change)
    delta_dur_src = oj.duration - oi.duration
    if not _basic_accept(new_src_idxs, delta_dur_src, src, state, orders, j):
        return None
    delta_dur_dst = oi.duration - oj.duration
    if not _basic_accept(new_dst_idxs, delta_dur_dst, dst, state, orders, i):
        return None

    new_src = _compute_metrics(new_src_idxs, orders, state.priors[src], car_services)
    if new_src is None:
        return None
    new_dst = _compute_metrics(new_dst_idxs, orders, state.priors[dst], car_services)
    if new_dst is None:
        return None

    old_src = state.spec_metrics[src]
    old_dst = state.spec_metrics[dst]

    delta_km = (new_src.km + new_dst.km) - (old_src.km + old_dst.km)
    delta_idle = (new_src.idle + new_dst.idle) - (old_src.idle + old_dst.idle)
    delta_zone = (new_src.zone_mm + new_dst.zone_mm) - (old_src.zone_mm + old_dst.zone_mm)
    # specs_used unchanged (both still non-empty after swap)

    delta = (
        W_TRAVEL_KM * delta_km
        + W_IDLE_MIN * delta_idle
        + W_ZONE_MM * delta_zone
    )
    return delta, new_src, new_dst


# ---------------------------------------------------------------------------
# Apply move (mutates state)
# ---------------------------------------------------------------------------

def _apply_relocate(
    state: LocalSearchState,
    i: int,
    src: str,
    dst: str,
    new_src: SpecMetrics,
    new_dst: SpecMetrics,
) -> None:
    state.assignment[i] = dst
    state.spec_orders[src] = new_src.sorted_idxs
    state.spec_orders[dst] = new_dst.sorted_idxs
    state.spec_metrics[src] = new_src
    state.spec_metrics[dst] = new_dst
    if new_src.n == 0:
        state.active_specs.discard(src)
        del state.spec_orders[src]
        del state.spec_metrics[src]


def _apply_swap(
    state: LocalSearchState,
    i: int,
    j: int,
    src: str,
    dst: str,
    new_src: SpecMetrics,
    new_dst: SpecMetrics,
) -> None:
    state.assignment[i] = dst
    state.assignment[j] = src
    state.spec_orders[src] = new_src.sorted_idxs
    state.spec_orders[dst] = new_dst.sorted_idxs
    state.spec_metrics[src] = new_src
    state.spec_metrics[dst] = new_dst


# ---------------------------------------------------------------------------
# Hill-climb passes
# ---------------------------------------------------------------------------

def _relocate_pass(
    state: LocalSearchState,
    orders: list[Order],
    car_services: set,
    deadline: float,
) -> int:
    """One full relocate pass. Returns the number of moves applied."""
    moves = 0
    n_orders = len(orders)
    # Snapshot the order indexes; iterate in fixed order. Re-evaluate after
    # each accepted move (cheap to keep going).
    for i in range(n_orders):
        if time.time() > deadline:
            return moves
        if i not in state.assignment:
            continue
        src = state.assignment[i]
        # Snapshot active specs to a list (the set may shrink during iteration)
        for dst in list(state.active_specs):
            if dst == src:
                continue
            res = _delta_relocate(state, i, src, dst, orders, car_services)
            if res is None:
                continue
            delta, new_src_m, new_dst_m = res
            if delta < 0:
                _apply_relocate(state, i, src, dst, new_src_m, new_dst_m)
                moves += 1
                # i is now in dst; src may be gone. Stop trying alt destinations
                # for this i and move to the next order.
                break
    return moves


def _swap_pass(
    state: LocalSearchState,
    orders: list[Order],
    car_services: set,
    deadline: float,
) -> int:
    """One full swap pass over distinct (i, j) pairs in different providers."""
    moves = 0
    n_orders = len(orders)
    for i in range(n_orders):
        if time.time() > deadline:
            return moves
        if i not in state.assignment:
            continue
        src = state.assignment[i]
        for j in range(i + 1, n_orders):
            if j not in state.assignment:
                continue
            dst = state.assignment[j]
            if dst == src:
                continue
            res = _delta_swap(state, i, j, src, dst, orders, car_services)
            if res is None:
                continue
            delta, new_src_m, new_dst_m = res
            if delta < 0:
                _apply_swap(state, i, j, src, dst, new_src_m, new_dst_m)
                moves += 1
                # After swapping, i is in dst now; break to outer loop and let
                # the next i picks up the new assignment.
                break
    return moves


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def _score_state(state: LocalSearchState, n_total_orders: int) -> float:
    n_assigned = len(state.assignment)
    n_unassigned = n_total_orders - n_assigned
    travel_km = sum(m.km for m in state.spec_metrics.values())
    idle_min = sum(m.idle for m in state.spec_metrics.values())
    zone_mm = sum(m.zone_mm for m in state.spec_metrics.values())
    n_specs = len(state.active_specs)
    return (
        W_UNASSIGNED * n_unassigned
        + W_TRAVEL_KM * travel_km
        + W_IDLE_MIN * idle_min
        + W_SPECS_USED * n_specs
        + W_ZONE_MM * zone_mm
    )


def _state_to_specdays(
    state: LocalSearchState,
    orders: list[Order],
    priors: dict,
    car_services: set,
    the_date: date,
) -> dict[str, SpecialistDay]:
    out: dict[str, SpecialistDay] = {}
    for s, idxs in state.spec_orders.items():
        prior = priors[s]
        sd = SpecialistDay(
            spec_id=s,
            prior=prior,
            window=window_for(prior, the_date),
            capacity=capacity_for(prior, the_date),
        )
        # Sort by start time and reconstruct ScheduledItems w/ correct slack flag
        sorted_idxs = sorted(idxs, key=lambda i: (orders[i].start_min, i))
        for k, oi in enumerate(sorted_idxs):
            o = orders[oi]
            used_slack = False
            if k > 0:
                a = orders[sorted_idxs[k - 1]]
                _, used_slack = transition_feasible(a, o, car_services)
            sd.items.append(ScheduledItem(order=o, start_min=o.start_min, end_min=o.end_min,
                                           used_handover_slack=used_slack))
        # Recount handovers from scratch
        handovers = 0
        for k in range(len(sorted_idxs) - 1):
            a = orders[sorted_idxs[k]]
            b = orders[sorted_idxs[k + 1]]
            _, used = transition_feasible(a, b, car_services)
            if used:
                handovers += 1
        sd.handover_events_used = handovers
        out[s] = sd
    return out


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def refine_day(
    the_date: date,
    orders: list[Order],
    priors: dict,
    certifications: dict,
    car_services: set,
    time_limit_s: int = TIME_LIMIT_S,
    verbose: bool = True,
) -> dict:
    """Run greedy + local-search refinement. Strict-improvement accept."""
    # 1. Greedy warm start
    greedy = allocate_day(the_date, orders, priors, certifications, car_services)
    greedy_specdays = {s: sd for s, sd in greedy["specdays"].items() if sd.n_orders > 0}
    n_orders = len(orders)
    n_assigned_greedy = sum(sd.n_orders for sd in greedy_specdays.values())

    # 2. Build LS state
    state = _build_state(greedy_specdays, orders, priors, certifications, car_services, the_date)
    greedy_obj = _score_state(state, n_orders)

    if verbose:
        print(f"  greedy: {n_assigned_greedy}/{n_orders} assigned, "
              f"{len(state.active_specs)} providers, "
              f"{sum(m.km for m in state.spec_metrics.values()):.1f} km, "
              f"obj={greedy_obj:.0f}")

    # 3. Hill-climb until both passes find no moves or time runs out
    deadline = time.time() + time_limit_s
    total_relocate = 0
    total_swap = 0
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        rmoves = _relocate_pass(state, orders, car_services, deadline)
        total_relocate += rmoves
        if time.time() > deadline:
            break
        smoves = _swap_pass(state, orders, car_services, deadline)
        total_swap += smoves
        if rmoves == 0 and smoves == 0:
            break  # local optimum
        if verbose:
            obj = _score_state(state, n_orders)
            print(f"  iter {iteration}: +{rmoves} relocate, +{smoves} swap -> obj={obj:.0f}")

    refined_obj = _score_state(state, n_orders)
    refined_specdays = _state_to_specdays(state, orders, priors, car_services, the_date)
    n_assigned_refined = sum(sd.n_orders for sd in refined_specdays.values())

    if verbose:
        print(f"  done: {total_relocate} relocate + {total_swap} swap moves over "
              f"{iteration} iter, wall={time_limit_s - max(0, deadline - time.time()):.1f}s")
        print(f"  refined: {n_assigned_refined}/{n_orders} assigned, "
              f"{len(refined_specdays)} providers, "
              f"{day_travel_km(refined_specdays):.1f} km, obj={refined_obj:.0f}")

    if refined_obj >= greedy_obj:
        if verbose:
            print(f"  no improvement -- keeping greedy")
        return {
            "refined": False,
            "reason": "no_improvement",
            "specdays": greedy_specdays,
            "greedy_obj": greedy_obj,
            "refined_obj": refined_obj,
            "moves": {"relocate": total_relocate, "swap": total_swap},
        }

    return {
        "refined": True,
        "specdays": refined_specdays,
        "greedy_specdays": greedy_specdays,
        "greedy_obj": greedy_obj,
        "refined_obj": refined_obj,
        "moves": {"relocate": total_relocate, "swap": total_swap},
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    # Example: pick a target date from your prepared_data.json
    target = date(2026, 1, 15)  # replace with a date present in your data
    orders, priors, certifications, car_services, baseline_by_spec = load_for_day(target)
    print(f"== Local-search refinement: {target} ==")
    print(f"Orders: {len(orders)}, baseline providers: {len(baseline_by_spec)}, "
          f"baseline travel: {baseline_day_travel_km(baseline_by_spec):.0f} km\n")

    result = refine_day(target, orders, priors, certifications, car_services)
    specdays = result["specdays"]
    n_assigned = sum(sd.n_orders for sd in specdays.values())
    n_specs = sum(1 for sd in specdays.values() if sd.n_orders > 0)
    km = day_travel_km(specdays)

    print()
    print(f"Final: {n_assigned}/{len(orders)} assigned, {n_specs} providers, {km:.1f} km travel")
    if "greedy_specdays" in result:
        g = result["greedy_specdays"]
        gk = day_travel_km(g)
        gn = sum(1 for sd in g.values() if sd.n_orders > 0)
        print(f"  greedy was: {gn} providers, {gk:.1f} km")
        print(f"  delta: {n_specs - gn:+d} providers, {km - gk:+.1f} km")
        print(f"  obj: {result['greedy_obj']:.0f} -> {result['refined_obj']:.0f} "
              f"({result['refined_obj'] - result['greedy_obj']:+.0f})")


if __name__ == "__main__":
    main()
