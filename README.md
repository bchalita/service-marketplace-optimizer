# Service Marketplace Optimizer

A two-stage optimization pipeline for assigning orders to service providers in on-demand marketplaces (beauty, cleaning, home repair), solving a Vehicle Routing Problem with Time Windows and Skill Matching (VRPTW-SM).

## Problem

On-demand service marketplaces typically let providers self-select which orders to accept. This leads to geographic inefficiency, underutilized providers, and unserved demand. In practice, roughly 10% of orders go unserved because no provider picks them up -- even when a feasible assignment exists.

The core challenge: build provider schedules that respect certification requirements, time windows, travel constraints, and individual capacity limits, while minimizing the number of dropped orders and total travel distance.

## Solution

A three-stage pipeline that constructs and refines daily schedules from scratch, using only behavioral signals derived from historical transactional data.

```
                        Historical Orders
                              |
                              v
               ┌──────────────────────────────┐
               │  Stage 0: Behavioral Priors  │
               │                              │
               │  Per-provider profiles from   │
               │  transactional data:          │
               │  - Availability windows       │
               │    (hourly mask, per-dow)     │
               │  - Capacity caps (p50/p75)    │
               │  - Service radius (p90)       │
               │  - Operating zone centroids   │
               └──────────────┬───────────────┘
                              |
                              v
               ┌──────────────────────────────┐
               │  Stage 1: Candidate Pruning  │
               │                              │
               │  Filter by cert + window +   │
               │  radius. Rank by zone match, │
               │  distance, load. Keep top-K. │
               │                              │
               │  Constraint relaxation chain: │
               │  zone → window → radius      │
               └──────────────┬───────────────┘
                              |
                              v
               ┌──────────────────────────────┐
               │  Stage 2: Greedy Insertion   │
               │                              │
               │  Sort orders by tightness.   │
               │  Insert into best provider   │
               │  slot by delta-km + idle     │
               │  cost. Open-spec penalty to  │
               │  consolidate workloads.      │
               └──────────────┬───────────────┘
                              |
                              v
               ┌──────────────────────────────┐
               │  Stage 3: Local Search       │
               │                              │
               │  Hill-climbing with relocate │
               │  + swap moves. Full natural  │
               │  objective (travel + idle +  │
               │  zone mismatch + provider    │
               │  count). Strict-improvement  │
               │  accept only.                │
               └──────────────────────────────┘
```

## Results

Validated on a real marketplace dataset with temporal train/test split (priors trained on days 1..N-1, tested on days N..end) to prevent look-ahead bias.

| Metric | Baseline | Optimized | Delta |
|---|---|---|---|
| Unserved demand | 10.6% | 2.2% | **-79%** |
| Travel distance | -- | -- | **-19%** |
| Orders per provider | -- | -- | **+21%** |

The optimizer serves nearly 5x more of the previously-dropped orders while simultaneously reducing travel and increasing provider utilization.

## Key Design Decisions

**Local search over CP-SAT.** An earlier iteration used Google OR-Tools CP-SAT as Stage 3. Without successor variables or Circuit constraints, CP-SAT couldn't represent travel and idle time in its objective, so its partial objective was misaligned with the natural objective. The strict-improvement check rejected every CP-SAT solution. Local search optimizes the full objective directly via cheap O(n) deltas and converges in seconds.

**Behavioral priors from transactional data.** Provider availability windows, capacity caps, and service radii are inferred from historical order patterns (hourly masks, percentile aggregations) rather than self-reported schedules. This captures actual behavior -- including bimodal work patterns and day-of-week variation -- without requiring providers to maintain accurate calendars.

**Multi-level constraint relaxation.** Candidate pruning applies constraints in order of softness: zone preference (ranking signal only) -> time window fit -> service radius. This maximizes the candidate pool for hard-to-place orders without sacrificing quality for easy ones.

## Tech Stack

Pure Python, standard library only. No external solvers, no pip dependencies. The travel model uses haversine distance with calibrated speed factors (10 km/h public transit, 20 km/h car) and a handover slack budget.

## File Structure

```
provider_priors.py   # Stage 0: build per-provider behavioral profiles
allocator.py         # Stage 1+2: candidate pruning + greedy construction
local_search.py      # Stage 3: hill-climbing refinement (relocate + swap)
```

Each stage is independently runnable and testable. The pipeline composes as: `provider_priors` -> `allocator` -> `local_search`.

## More

See the [full case study presentation](presentation_en.html) for detailed methodology and visualizations.

Part of the [Op-Era](https://github.com/bchalita/op-era) optimization deployment platform.

## License

MIT
