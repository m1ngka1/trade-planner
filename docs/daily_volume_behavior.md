# Optimizer-derived daily volume behavior

## Model change

For daily trade `q_t`, cumulative executed inventory `h_t`, and remaining order
`r_t`, the planner can now price both states independently:

```text
sum_t inventory_risk_weight * risk(h_t)
    + residual_risk_weight * risk(r_t)
    + trading_cost(q_t)
```

The rebalance reference configuration uses physical rate-times-ADV caps,
accumulated Barra inventory risk, zero pre-event residual-risk pressure, and
quadratic participation impact. This leaves the schedule to the optimizer:

- inventory risk discourages unnecessary early positions;
- country, sector, industry, and other factor risk rewards early hedges;
- hard completion plus remaining capacity makes urgent names start early; and
- quadratic impact spreads flow instead of creating one final block.

The existing earnings-aware configuration is unchanged because avoiding an
earnings event and accumulating an anticipated rebalance are different use
cases.

## Fixed synthetic benchmark

The CLARABEL experiment uses two deterministic ten-day fixtures with a common
event/deadline on day 10:

1. `urgency_ramp`: twelve factor-neutral names. Urgent, medium, and small orders
   require 8.5, 4.5, and 1.0 days of maximum capacity respectively.
2. `factor_balance`: urgent HK/Financials and JP/IT exposures plus smaller
   flexible offsets. The final parent basket remains deliberately imbalanced,
   so the test measures early balance rather than changing the target.

It records 29 candidates: the current adaptive/residual baseline, a physical-
cap residual baseline, and all 27 combinations of:

| Parameter | Values |
|---|---|
| Inventory-risk weight | 0.1, 1, 10 |
| Factor-risk multiplier | 0, 1, 10 |
| Impact bps at 10% ADV | 1, 5, 20 |

Every candidate is solved on both fixtures. Failed candidates remain in the run
ledger rather than being silently discarded.

## Acceptance gates

A retained candidate must complete every order with no material capacity,
direction, or analytic latest-start-floor violation. On the urgency fixture it
must also have:

- daily-volume Spearman correlation at least 0.80;
- final-three-day versus first-three-day mean volume ratio at least 2.0;
- no more than 35% completion by day 5;
- at least seven of nine non-decreasing daily transitions;
- urgent names starting by day 2, medium names on days 5-6, and small names no
  earlier than day 8; and
- at least four days between urgent and small median execution dates.

On the balance fixture, maximum day-2-to-day-4 normalized factor imbalance must
be at most 10% and improve at least 75% over the otherwise identical no-factor
candidate.

## Retained result

The selected reference combination is:

```text
inventory_risk_weight = 1
factor_risk_multiplier = 1
impact_bps_at_10pct_adv = 20
residual_risk_weight = 0
```

Its recorded behavior is:

| Metric | Result |
|---|---:|
| Daily-volume Spearman correlation | 1.000 |
| Non-decreasing transitions | 9 / 9 |
| Final-three / first-three mean volume | 4.297x |
| Completion by day 5 | 27.90% |
| Medium first-trade day | 5 |
| Small first-trade day | 8 |
| Urgent-to-small median gap | 4 days |
| Early factor imbalance | 1.60% |
| Matching no-factor imbalance | 44.20% |
| Factor-balance improvement | 96.39% |

These numbers validate the mechanism on the fixed synthetic fixtures; they are
not a claim that the same risk-aversion and impact calibration is production-
optimal for every desk.

## Reproduce and inspect

```bash
env PYTHONPATH=. \
  PYTHONPYCACHEPREFIX=/private/tmp/trade_planner_pycache \
  MPLCONFIGDIR=/private/tmp/trade_planner_mpl \
  XDG_CACHE_HOME=/private/tmp/trade_planner_cache \
  ./.conda-env/bin/python experiments/daily_volume_behavior.py \
  --solver CLARABEL \
  --output-prefix artifacts/daily_volume_behavior
```

Generated evidence:

- `daily_volume_behavior_runs.csv`: one row per candidate and fixture;
- `daily_volume_behavior_summary.csv`: cross-fixture gates and keep/discard;
- `daily_volume_behavior_profiles.csv`: daily and cumulative gross volume;
- `daily_volume_behavior_schedules.csv`: every date-symbol trade;
- `daily_volume_behavior_exposures.csv`: cumulative factor exposures;
- `daily_volume_behavior_names.csv`: urgency, latest-start, first-trade, and
  median execution diagnostics;
- `daily_volume_behavior_all_profiles.png`: every urgency-fixture profile;
- `daily_volume_behavior_all_profiles_factor_balance.png`: every balance-
  fixture profile; and
- `daily_volume_behavior_selected.png`: baseline comparison, selected urgency
  stack, and factor-balance ablation.
