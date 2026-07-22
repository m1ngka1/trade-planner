# Tail-preserving scenario reduction

## Decision

Keep 96 weighted optimization scenarios as the automatic calibration default.
Keep the full source distribution for coefficient scaling, economic
measurement, and audit. Discard the 64-scenario alternative even though it is
faster and has marginally lower independent CVaR, because it makes the small,
non-urgent group trade on day 1 instead of waiting until day 4.

This is an investment decision rather than a single-metric optimization: a
speedup is useful only if the planner retains the intended urgency, balance,
and volume behavior.

## Method

`reduce_return_scenarios` is deterministic and basket-aware:

1. Normalize scenario probabilities and remove the weighted mean so residual
   scenarios cannot add hidden alpha.
2. Score each scenario by the loss on the supplied full-target dollar basket
   across the horizon. This identifies adverse rebalance-call and market paths
   without referring to a desired daily execution curve.
3. Retain every observation in the worst 10% probability tail when the limit
   permits it.
4. Sort the remaining core by the same loss score and divide it into equal-
   probability strata.
5. Represent each stratum with the observed path nearest its weighted centroid
   in standardized date-by-name return space. Give that path the stratum's full
   probability.
6. Center the reduced distribution again.

The scenario objective therefore adds at most 96 path-loss variables by
default, while the full input scenarios still determine the basket's CVaR
scale, excess-tail coefficient, reported metrics, and risk-profile selection.
Passing `max_optimization_scenarios=None` to `build_rebalance_frontier` or
`calibrate_rebalance_plan` disables reduction for an audit.

## Predefined acceptance gates

The reduced medium-risk plan had to satisfy every gate against the unreduced
256-scenario hybrid frontier:

- at least 2x faster frontier construction;
- expected net P&L within one basis point of parent gross;
- independent mean 95% loss CVaR no more than 0.25% worse;
- urgent flow starts no later and small flow starts no earlier;
- early maximum country/sector/industry imbalance within one percentage point;
  and
- late/early gross volume at least 90% of the control.

Every plan was evaluated on the same five independently seeded samples of
5,000 fat-tail paths. Those 25,000 paths were not used to fit any plan.

## Recorded results

| Trial | Frontier time | Speedup | Expected net P&L | Independent mean 95% loss CVaR | Early factor imbalance | Urgent start | Small start | Late/early volume | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Full 256 | 93.93s | 1.00x | $710,459 | $8,604,480 | 3.30% | Day 1 | Day 4 | 1.350x | Control |
| Reduced 96 | 23.90s | 3.93x | $710,017 | $8,603,830 | 3.22% | Day 1 | Day 4 | 1.347x | Keep |
| Reduced 64 | 18.17s | 5.17x | $709,727 | $8,600,390 | 3.71% | Day 1 | Day 1 | 1.348x | Discard |

The 96-scenario plan gives up only $442 of expected net P&L, improves the
independent mean loss CVaR by $650, and preserves the economically important
mechanics. Its full date-by-name schedule differs from the 256-scenario control
by 1.42% of parent gross in L1 notional, but those changes do not alter the
group start dates or volume shape.

The 64-scenario result is a useful negative trial. Its scalar P&L-risk metrics
look slightly better, yet the smaller core sample changes the optimizer's
relative trade-offs enough to front-load the small group. It is rejected rather
than accepted on runtime or CVaR alone.

The combined feasibility audit over all 34 saved economic, reduction, and
tail-stress schedules found maximum participation-cap excess of 0.0221 share,
maximum wrong-direction solver dust below 0.0004 share, and maximum terminal
residual below 0.000001 share.

## Reproduce and inspect

```bash
env PYTHONPATH=. \
  PYTHONPYCACHEPREFIX=/private/tmp/trade_planner_pycache \
  MPLCONFIGDIR=/private/tmp/trade_planner_mpl \
  XDG_CACHE_HOME=/private/tmp/trade_planner_cache \
  ./.conda-env/bin/python experiments/scenario_reduction.py \
  --solver OSQP \
  --output-prefix artifacts/scenario_reduction
```

Generated evidence:

- `scenario_reduction_trials.csv`: runtime, economics, mechanics, independent
  tail metrics, explicit acceptance decisions, and failure reasons;
- `scenario_reduction_profiles.csv`: daily gross volume and factor imbalance;
- `scenario_reduction_schedules.csv`: every complete date-by-name schedule; and
- `scenario_reduction.png`: volume, factor-balance, and independent-CVaR
  comparison.

## Limitation and next validation

The speed and behavior result is proven only on the fixed synthetic basket.
The recorded values above use evaluation seeds disjoint from the optimization
seed; the experiment asserts that separation before running.
The 96-scenario default must still be validated in point-in-time walk-forward
replays across basket sizes, rebalance types, event distances, and volatility
regimes. A production review should compare both realized tail P&L and schedule
stability; reducing in-sample CVaR alone is not sufficient evidence of better
trading.
