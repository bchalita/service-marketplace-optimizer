#!/usr/bin/env python3
"""Stage 0 of the clean-slate allocator: per-provider historical priors.

Reads:
    data/prepared_data.json        (certifications, provider_schedules)
    data/behavioral_profiles.json  (centroid_blended, operating_zones)

Writes:
    data/provider_priors.json

For every provider with a non-empty schedule, emits:

    {
      "spec_id": ...,
      "home": [lat, lng],                       # blended centroid anchor
      "certs": [...],
      "historical_zones": [
        {rank, is_primary, lat, lng, fraction}, ...
      ],
      "radius_cap_km": p90 of daily max distance from home,
      "windows": {
        "weekday":  {p20_start, p80_end, n_samples} | null,
        "saturday": ... | null,
        "sunday":   ... | null,
        "flat":     {p20_start, p80_end, n_samples}   (always present)
      },
      "capacity": {
        same segmentation, each {typical_min (p75 daily mins),
                                 max_orders (p90 daily cnt),
                                 n_samples}
      }
    }

Segment fallback rule: a segment emits null when n_samples < 4. Consumers
call window_for / capacity_for, which automatically fall back to flat.

Note on "home": this is the operational anchor (centroid_blended), not the
provider's literal residence. Derived from historical order locations.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from statistics import median
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent / "data"
PREPARED_PATH = DATA_DIR / "prepared_data.json"
BEHAVIORAL_PATH = DATA_DIR / "behavioral_profiles.json"
OUT_PATH = DATA_DIR / "provider_priors.json"

MIN_SEGMENT_SAMPLES = 4
MIN_DOW_SAMPLES = 2         # lower bar for per-dow windows: fewer samples but
                            # effectively stricter (20% of 2 = need 50% hit rate)
MIN_DOW_FRACTION = 0.20     # also require provider works this dow >= 20% of the time
SEGMENTS = ("weekday", "saturday", "sunday")

# Window mode: "single" (original p20/p80), "split" (Approach B), "mask" (Approach A)
WINDOW_MODE = "mask"

# Bimodal detection (Approach B) / hourly mask (Approach A)
BIMODAL_THRESHOLD = 0.20   # hours with <20% of working days active = "dead"
BIMODAL_MIN_GAP_HOURS = 2  # minimum contiguous dead hours to trigger split (B only)

# Fallback dow gate: when falling back from per-dow to segment/flat, require
# that the provider has worked this dow at least this fraction of the time.
# Prevents assigning Thursday orders to a provider who never works Thursdays
# just because her Mon-Wed-Fri pattern says that hour is "available".
# Set to 0.0 to disable (original behavior).
MIN_DOW_FALLBACK_FRACTION = 0.01  # catches only zero-instance (never worked this dow)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def percentile(values: list, pct: float) -> Optional[float]:
    """Linear interpolation percentile. Returns None if empty."""
    if not values:
        return None
    sv = sorted(values)
    if len(sv) == 1:
        return sv[0]
    k = (len(sv) - 1) * pct / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sv[int(k)]
    return sv[f] + (sv[c] - sv[f]) * (k - f)


def segment_of(d: date) -> str:
    wd = d.weekday()
    if wd <= 4:
        return "weekday"
    if wd == 5:
        return "saturday"
    return "sunday"


# ---------------------------------------------------------------------------
# Bimodal window splitting (Approach B)
# ---------------------------------------------------------------------------

def _split_window_blocks(
    p20_start: int,
    p80_end: int,
    hourly_days: list[set[int]],
) -> list[list[int]]:
    """Split a single [p20_start, p80_end] window into blocks if dead zones exist.

    hourly_days: one set of active hours per working day.
    Returns list of [start_min, end_min] blocks.
    """
    n_days = len(hourly_days)
    if n_days < MIN_SEGMENT_SAMPLES:
        return [[p20_start, p80_end]]

    start_h = p20_start // 60
    end_h = (p80_end + 59) // 60  # ceiling to include partial last hour

    # For each hour in range, compute fraction of days with activity
    dead_hours = []
    for h in range(start_h, end_h):
        active_count = sum(1 for day_hours in hourly_days if h in day_hours)
        if active_count / n_days < BIMODAL_THRESHOLD:
            dead_hours.append(h)

    if not dead_hours:
        return [[p20_start, p80_end]]

    # Find contiguous runs of dead hours
    gaps: list[list[int]] = []
    current = [dead_hours[0]]
    for h in dead_hours[1:]:
        if h == current[-1] + 1:
            current.append(h)
        else:
            gaps.append(current)
            current = [h]
    gaps.append(current)

    # Only split on gaps >= minimum length
    significant = [g for g in gaps if len(g) >= BIMODAL_MIN_GAP_HOURS]

    if not significant:
        return [[p20_start, p80_end]]

    # Split window at significant gaps
    significant.sort(key=lambda g: g[0])
    blocks: list[list[int]] = []
    current_start = p20_start
    for gap in significant:
        gap_start_min = gap[0] * 60
        gap_end_min = (gap[-1] + 1) * 60
        if gap_start_min > current_start:
            blocks.append([current_start, gap_start_min])
        current_start = gap_end_min

    if current_start < p80_end:
        blocks.append([current_start, p80_end])

    return blocks if blocks else [[p20_start, p80_end]]


def _hourly_mask_blocks(
    hourly_days: list[set[int]],
    min_samples: int = MIN_SEGMENT_SAMPLES,
) -> list[list[int]] | None:
    """Approach A: build window blocks from hourly availability mask.

    For each hour, check if >= BIMODAL_THRESHOLD fraction of working days
    have activity. Contiguous available hours form blocks.
    Returns list of [start_min, end_min] blocks, or None if insufficient data.
    """
    n_days = len(hourly_days)
    if n_days < min_samples:
        return None

    # Collect all hours that appear in any day
    all_hours: set[int] = set()
    for day_hours in hourly_days:
        all_hours.update(day_hours)

    if not all_hours:
        return None

    # Filter to hours with sufficient activity
    available = []
    for h in range(min(all_hours), max(all_hours) + 1):
        active_count = sum(1 for day_hours in hourly_days if h in day_hours)
        if active_count / n_days >= BIMODAL_THRESHOLD:
            available.append(h)

    if not available:
        return None

    # Form contiguous blocks
    blocks: list[list[int]] = []
    block_start = available[0]
    prev = available[0]
    for h in available[1:]:
        if h != prev + 1:
            blocks.append([block_start * 60, (prev + 1) * 60])
            block_start = h
        prev = h
    blocks.append([block_start * 60, (prev + 1) * 60])

    return blocks


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _window_summary(
    daily: list,
    hourly_days: list[set[int]] | None = None,
    min_samples: int = MIN_SEGMENT_SAMPLES,
) -> Optional[dict]:
    if not daily:
        return None
    starts = [d[0] for d in daily]
    ends = [d[1] for d in daily]
    p20_start = int(round(percentile(starts, 20)))
    p80_end = int(round(percentile(ends, 80)))

    blocks = [[p20_start, p80_end]]
    if hourly_days and len(hourly_days) >= min_samples:
        if WINDOW_MODE == "split":
            blocks = _split_window_blocks(p20_start, p80_end, hourly_days)
        elif WINDOW_MODE == "mask":
            mask_blocks = _hourly_mask_blocks(hourly_days, min_samples=min_samples)
            if mask_blocks:
                blocks = mask_blocks

    return {
        "p20_start": p20_start,
        "p80_end": p80_end,
        "n_samples": len(daily),
        "blocks": blocks,
    }


def _capacity_summary(mins: list, counts: list) -> Optional[dict]:
    if not mins:
        return None
    return {
        "typical_min": int(round(percentile(mins, 50))),
        "max_orders": max(2, int(round(percentile(counts, 75)))),
        "n_samples": len(mins),
    }


# ---------------------------------------------------------------------------
# Prior builder
# ---------------------------------------------------------------------------

def build_priors(cutoff_date: Optional[date] = None) -> dict:
    """Build per-provider priors.

    cutoff_date:
      If provided, only schedule entries with date STRICTLY before cutoff_date
      are used to compute per-provider windows/capacity/radius. Used for the
      look-ahead bias check -- lets us train priors on days 1..N-1 and then
      test on days N..end without the priors knowing anything about the
      test days.
    """
    prepared = json.loads(PREPARED_PATH.read_text())
    behavioral = json.loads(BEHAVIORAL_PATH.read_text())

    certifications = prepared.get("certifications", {})
    schedules = prepared.get("specialist_schedules", {})
    profiles = behavioral.get("profiles", {})

    # Compute total instances of each dow in the training period
    all_training_dates: set[date] = set()
    for sched in schedules.values():
        for date_str, orders in sched.items():
            if not orders:
                continue
            try:
                d = datetime.fromisoformat(date_str).date()
            except ValueError:
                continue
            if cutoff_date is None or d < cutoff_date:
                all_training_dates.add(d)
    total_dow_instances: dict[int, int] = defaultdict(int)
    for d in all_training_dates:
        total_dow_instances[d.weekday()] += 1

    priors = {}
    skipped_no_home = 0
    skipped_no_days = 0

    for spec_id, sched in schedules.items():
        profile = profiles.get(spec_id, {})
        centroid = profile.get("centroid_blended") or profile.get("centroid_all")
        if not centroid:
            skipped_no_home += 1
            continue
        home_lat = centroid["lat"]
        home_lng = centroid["lng"]

        per_seg_windows: dict[str, list] = defaultdict(list)
        per_seg_caps_min: dict[str, list] = defaultdict(list)
        per_seg_caps_cnt: dict[str, list] = defaultdict(list)
        per_seg_hourly: dict[str, list[set[int]]] = defaultdict(list)
        per_dow_windows: dict[int, list] = defaultdict(list)
        per_dow_hourly: dict[int, list[set[int]]] = defaultdict(list)
        flat_windows: list = []
        flat_caps_min: list = []
        flat_caps_cnt: list = []
        flat_hourly: list[set[int]] = []
        daily_max_dists: list = []
        all_active_dates: list[date] = []
        spec_dow_counts: dict[int, int] = defaultdict(int)

        for date_str, orders in sched.items():
            if not orders:
                continue
            try:
                d = datetime.fromisoformat(date_str).date()
            except ValueError:
                continue
            all_active_dates.append(d)
            if cutoff_date is not None and d >= cutoff_date:
                continue  # hold out test days from prior construction
            spec_dow_counts[d.weekday()] += 1
            starts = [o["start_min"] for o in orders if "start_min" in o]
            ends = [o["end_min"] for o in orders if "end_min" in o]
            if not starts or not ends:
                continue
            seg = segment_of(d)
            day_start = min(starts)
            day_end = max(ends)
            per_seg_windows[seg].append((day_start, day_end))
            per_dow_windows[d.weekday()].append((day_start, day_end))
            flat_windows.append((day_start, day_end))

            # Hourly activity for bimodal detection
            active_hours: set[int] = set()
            for o in orders:
                s_h = o.get("start_min", 0) // 60
                e_h = max(s_h, (o.get("end_min", 0) - 1) // 60)
                for h in range(s_h, e_h + 1):
                    active_hours.add(h)
            per_seg_hourly[seg].append(active_hours)
            per_dow_hourly[d.weekday()].append(active_hours)
            flat_hourly.append(active_hours)

            day_minutes = sum((o.get("duration") or 0) for o in orders)
            day_count = len(orders)
            per_seg_caps_min[seg].append(day_minutes)
            per_seg_caps_cnt[seg].append(day_count)
            flat_caps_min.append(day_minutes)
            flat_caps_cnt.append(day_count)

            max_d = 0.0
            for o in orders:
                if "lat" in o and "lng" in o:
                    dkm = haversine_km(home_lat, home_lng, o["lat"], o["lng"])
                    if dkm > max_d:
                        max_d = dkm
            daily_max_dists.append(max_d)

        if not flat_windows:
            skipped_no_days += 1
            continue

        windows = {}
        for seg in SEGMENTS:
            seg_daily = per_seg_windows.get(seg, [])
            if len(seg_daily) >= MIN_SEGMENT_SAMPLES:
                windows[seg] = _window_summary(seg_daily, per_seg_hourly.get(seg))
            else:
                windows[seg] = None
        windows["flat"] = _window_summary(flat_windows, flat_hourly)

        for dow in range(7):
            dow_daily = per_dow_windows.get(dow, [])
            dow_frac = spec_dow_counts.get(dow, 0) / max(1, total_dow_instances.get(dow, 1))
            if len(dow_daily) >= MIN_DOW_SAMPLES and dow_frac >= MIN_DOW_FRACTION:
                windows[f"dow_{dow}"] = _window_summary(
                    dow_daily, per_dow_hourly.get(dow),
                    min_samples=MIN_DOW_SAMPLES,
                )
            else:
                windows[f"dow_{dow}"] = None

        capacity = {}
        for seg in SEGMENTS:
            seg_mins = per_seg_caps_min.get(seg, [])
            seg_cnts = per_seg_caps_cnt.get(seg, [])
            if len(seg_mins) >= MIN_SEGMENT_SAMPLES:
                capacity[seg] = _capacity_summary(seg_mins, seg_cnts)
            else:
                capacity[seg] = None
        capacity["flat"] = _capacity_summary(flat_caps_min, flat_caps_cnt)

        radius_cap = percentile(daily_max_dists, 90) if daily_max_dists else 0.0

        priors[spec_id] = {
            "spec_id": spec_id,
            "home": [round(home_lat, 6), round(home_lng, 6)],
            "last_active_date": str(max(all_active_dates)) if all_active_dates else None,
            "certs": certifications.get(spec_id, []),
            "historical_zones": [
                {
                    "rank": z.get("rank"),
                    "is_primary": z.get("is_primary", False),
                    "lat": z.get("centroid_lat"),
                    "lng": z.get("centroid_lng"),
                    "fraction": z.get("fraction"),
                }
                for z in profile.get("operating_zones", [])
            ],
            "radius_cap_km": round(radius_cap, 2),
            "dow_fractions": {
                str(dow): round(spec_dow_counts.get(dow, 0) / max(1, total_dow_instances.get(dow, 1)), 3)
                for dow in range(7)
            },
            "windows": windows,
            "capacity": capacity,
        }

    return {
        "priors": priors,
        "_skip_no_home": skipped_no_home,
        "_skip_no_days": skipped_no_days,
    }


# ---------------------------------------------------------------------------
# Consumer helpers (imported by Stage 1+2)
# ---------------------------------------------------------------------------

def window_for(prior: dict, d: date) -> list[tuple[int, int]]:
    """Window blocks for a provider on a given date.

    Fallback chain: per-dow -> segment -> flat.
    Per-dow windows use that specific day-of-week's hourly mask (e.g.,
    Thursday-only data for Thursday orders). Falls back to segment when
    fewer than MIN_SEGMENT_SAMPLES instances of that dow exist.
    """
    windows = prior.get("windows") or {}

    # Try per-dow first (e.g., "dow_3" for Thursday)
    dow_key = f"dow_{d.weekday()}"
    w = windows.get(dow_key)
    if w is not None:
        blocks = w.get("blocks")
        if blocks:
            return [(b[0], b[1]) for b in blocks]
        return [(w["p20_start"], w["p80_end"])]

    # Falling back to segment/flat -- check dow activity gate.
    # If provider rarely/never works this dow, don't assign via proxy data.
    if MIN_DOW_FALLBACK_FRACTION > 0:
        dow_frac = prior.get("dow_fractions", {}).get(str(d.weekday()), 0)
        if dow_frac < MIN_DOW_FALLBACK_FRACTION:
            return []  # not available on this dow

    # Fall back to segment -> flat
    seg = segment_of(d)
    w = windows.get(seg)
    if w is None:
        w = windows["flat"]
    blocks = w.get("blocks")
    if blocks:
        return [(b[0], b[1]) for b in blocks]
    return [(w["p20_start"], w["p80_end"])]


def order_in_window(
    blocks: list[tuple[int, int]], start_min: int, end_min: int,
) -> bool:
    """Check if an order [start_min, end_min] fits entirely within at least one block."""
    return any(start_min >= ws and end_min <= we for ws, we in blocks)


def capacity_for(prior: dict, d: date) -> dict:
    """Capacity dict {typical_min, max_orders, n_samples} for a provider on a date."""
    seg = segment_of(d)
    c = (prior.get("capacity") or {}).get(seg)
    if c is None:
        c = prior["capacity"]["flat"]
    return c


# ---------------------------------------------------------------------------
# Sanity prints
# ---------------------------------------------------------------------------

def sanity_prints(priors: dict, skip_no_home: int, skip_no_days: int) -> None:
    n = len(priors)
    print(f"== Provider priors: {n} providers "
          f"(skipped {skip_no_home} no-home, {skip_no_days} no-days) ==\n")

    # Segment availability
    seg_avail = {s: 0 for s in SEGMENTS}
    for p in priors.values():
        for s in SEGMENTS:
            if (p["windows"] or {}).get(s):
                seg_avail[s] += 1
    print("Segment window availability (n >= 4 samples):")
    for s in SEGMENTS:
        c = seg_avail[s]
        print(f"  {s:<9}: {c:>4} / {n} ({c / n:.1%})")
    print()

    # Median window widths
    def widths(seg_key: str) -> list:
        out = []
        for p in priors.values():
            w = p["windows"].get(seg_key)
            if w:
                out.append(w["p80_end"] - w["p20_start"])
        return out

    print("Median window width in minutes (p80_end - p20_start):")
    for s in SEGMENTS + ("flat",):
        ws = widths(s)
        if ws:
            print(f"  {s:<9}: median {int(median(ws))} min   (n={len(ws)})")
        else:
            print(f"  {s:<9}: (no samples)")
    print()

    # Fallback histogram: how many segments per provider fall back to flat
    fb_hist: dict[int, int] = defaultdict(int)
    for p in priors.values():
        n_fb = sum(1 for s in SEGMENTS if p["windows"].get(s) is None)
        fb_hist[n_fb] += 1
    print("Fallback histogram (# of {weekday,saturday,sunday} segments null):")
    for k in sorted(fb_hist):
        c = fb_hist[k]
        print(f"  {k} fallback(s): {c:>4} providers ({c / n:.1%})")
    print()

    # Bimodal detection (Approach B)
    bimodal = 0
    total_blocks = 0
    for p in priors.values():
        flat_w = p["windows"].get("flat")
        if flat_w and flat_w.get("blocks"):
            nb = len(flat_w["blocks"])
            total_blocks += nb
            if nb > 1:
                bimodal += 1
        else:
            total_blocks += 1
    print(f"Bimodal providers (flat window): {bimodal} / {n} ({bimodal / max(1, n):.1%})")
    if bimodal:
        print(f"  avg blocks among bimodal: {total_blocks / max(1, n):.2f} overall, "
              f"{sum(len(p['windows']['flat'].get('blocks', [[]])) for p in priors.values() if len(p['windows']['flat'].get('blocks', [[]])) > 1) / max(1, bimodal):.1f} among bimodal")
    print()

    # Per-dow window availability
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_avail = {d: 0 for d in range(7)}
    for p in priors.values():
        for d in range(7):
            if p["windows"].get(f"dow_{d}"):
                dow_avail[d] += 1
    print(f"Per-dow window availability (n >= {MIN_DOW_SAMPLES} samples):")
    for d in range(7):
        c = dow_avail[d]
        print(f"  {dow_names[d]:<3}: {c:>4} / {n} ({c / n:.1%})")
    print()

    # Capacity + radius medians
    typical = [p["capacity"]["flat"]["typical_min"]
               for p in priors.values() if p["capacity"].get("flat")]
    maxo = [p["capacity"]["flat"]["max_orders"]
            for p in priors.values() if p["capacity"].get("flat")]
    radii = [p["radius_cap_km"] for p in priors.values()]
    if typical:
        print(f"Capacity (flat) -- median typical_min: {int(median(typical))} min, "
              f"median max_orders: {int(median(maxo))}")
    if radii:
        print(f"Radius cap -- median: {median(radii):.2f} km, "
              f"p90: {percentile(radii, 90):.2f} km, "
              f"max: {max(radii):.2f} km")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    result = build_priors()
    priors = result["priors"]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(priors, indent=2, ensure_ascii=False))
    rel = OUT_PATH.relative_to(Path(__file__).resolve().parent.parent)
    print(f"Wrote {len(priors)} priors -> {rel}\n")
    sanity_prints(priors, result["_skip_no_home"], result["_skip_no_days"])


if __name__ == "__main__":
    main()
