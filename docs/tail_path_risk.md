# Scenario-derived tail path risk

## Decision

Use the conditional tail second moment for automatic **low** risk aversion.
Keep the 96-scenario hybrid CVaR model for **medium** and covariance for
**high**.

This split is evidence-driven:

- the low-profile second-moment plan passed every investment and mechanics gate
  across five independent optimization scenario samples;
- the medium plan passed four of five strict replications but traded 0.57% of
  its small-order group one day earlier in the remaining sample, so it is not
  promoted for medium; and
- neither scenario-derived tail approximation delivered a material high-
  profile improvement, so high remains on the more stable covariance frontier.

Users still select only `high`, `medium`, or `low`. They do not choose the tail
estimator or enter another coefficient.

## Investment hypothesis

Daily covariance prices country, sector, industry, and specific exposure on
each date, but it does not capture a coherent wrong-rebalance-call shock that
persists across dates. Sample CVaR captures that path dependence but adds a
threshold and hinge exposure for every optimization scenario and can react to
a small fitted tail.

Two cheaper quadratic alternatives were tested:

1. **Conditional-mean stress path**: collapse the worst full-basket 10% into
   one adverse cross-date return path and penalize squared P&L exposure to its
   centered tail/non-tail regime.
2. **Conditional tail second moment**: retain every exact-mass worst-10% path
   and penalize the weighted mean of squared path P&L. This preserves dispersion
   around the tail mean and represents multiple adverse directions without a
   CVaR threshold or hinge variables.

Both are generated from scenario returns and the parent basket—not from a
desired daily execution curve.

## Automatic coefficient

For 95% CVaR calibration, the fitting tail is automatically set to twice the
evaluation tail probability:

```text
tail_probability = 2 * (1 - 0.95) = 10%
```

The doubled mass gives the quadratic estimator more observations than a raw 5%
tail. For the full-target dollar path, calculate:

```text
V_cov  = accumulated daily covariance variance
V_tail = E[(scenario path P&L)^2 | full-target loss is in the worst 10%]
f_excess = max(scenario_CVaR / covariance_implied_CVaR - 1, 0)

tail_variance_scale = f_excess * V_cov / V_tail
```

The optimizer then uses one common economically scaled frontier multiplier:

```text
lambda_cov * (daily covariance variance
              + tail_variance_scale * conditional tail second moment)
```

At the full-target reference path, the added quadratic risk therefore equals
the scenario tail excess not already priced by covariance. The user enters no
stress coefficient, tail weight, or multiplier.

## Evidence integrity correction

During this iteration an overlap was found between the original optimization
seed and one evaluation seed. All affected economic, reduction, and tail-path
artifacts were regenerated with disjoint seed ranges, and the experiment now
asserts that the optimization seed is absent from the evaluation set. The
decisions below use only the corrected runs.

## Five-seed robustness results

Each model was fit independently on five 256-scenario samples. Every selected
schedule was evaluated on the same five disjoint 5,000-path samples.

### Conditional-mean stress path

The mean-path approximation was very fast and improved low-profile risk in all
five fits. Medium failed the independent-CVaR ceiling in two fits:

| Profile | Strict passes | Independent CVaR difference versus hybrid | Decision |
|---|---:|---:|---|
| Medium | 3 / 5 | -0.02% to +0.40% | Discard for medium |
| Low | 5 / 5 | -0.47% to -0.04% | Promising, refine |

The failures show that one conditional mean omits economically relevant
within-tail dispersion.

### Conditional tail second moment

| Profile | Strict passes | Runtime speedup | Expected net P&L difference | Volatility difference | Independent CVaR difference | Early factor-balance difference | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Medium | 4 / 5 | 22.4–53.6x | -$1,137 to +$2,557 | -0.08% to +0.18% | -0.03% to +0.23% | -0.74pp to +0.10pp | Discard for medium under strict mechanics gate |
| Low | 5 / 5 | 22.4–53.6x | -$1,868 to -$104 | -0.49% to -0.03% | -0.40% to -0.03% | -0.77pp to -0.04pp | Keep |

All low-profile urgent and small start dates matched hybrid, and every expected-
P&L sacrifice was far inside the one-basis-point parent-gross materiality band.
The retained model therefore simplifies and accelerates the planner while
making the low profile slightly safer.

Across all 34 saved economic, reduction, and tail-stress schedules, the maximum
participation-cap excess is 0.0221 share, wrong-direction solver dust is below
0.0004 share, and terminal residual is below 0.000001 share.

The one medium failure was narrow but real under the predefined rule: hybrid
kept the small group below its 0.5%-of-order start threshold until day 6, while
the second-moment plan executed 0.57% on day 5 and then 2.42% on day 6. Both
plans economically “wait,” but the experiment preserves the strict gate rather
than changing it after seeing the result.

On the primary fit, automatic low changes the retained hybrid-low result from:

| Model | Expected net P&L | P&L volatility | Independent mean 95% loss CVaR | Early factor imbalance | Late/early volume |
|---|---:|---:|---:|---:|---:|
| Hybrid-96 low | $815,935 | $4.509m | $9.979m | 8.20% | 0.521x |
| Tail-second-moment low | $815,458 | $4.503m | $9.967m | 8.07% | 0.524x |

## Reproduce and inspect

```bash
env PYTHONPATH=. \
  PYTHONPYCACHEPREFIX=/private/tmp/trade_planner_pycache \
  MPLCONFIGDIR=/private/tmp/trade_planner_mpl \
  XDG_CACHE_HOME=/private/tmp/trade_planner_cache \
  ./.conda-env/bin/python experiments/stress_path_risk.py

env PYTHONPATH=. \
  PYTHONPYCACHEPREFIX=/private/tmp/trade_planner_pycache \
  MPLCONFIGDIR=/private/tmp/trade_planner_mpl \
  XDG_CACHE_HOME=/private/tmp/trade_planner_cache \
  ./.conda-env/bin/python experiments/stress_path_seed_robustness.py \
  --path-model mean \
  --output-prefix artifacts/stress_path_seed_robustness

env PYTHONPATH=. \
  PYTHONPYCACHEPREFIX=/private/tmp/trade_planner_pycache \
  MPLCONFIGDIR=/private/tmp/trade_planner_mpl \
  XDG_CACHE_HOME=/private/tmp/trade_planner_cache \
  ./.conda-env/bin/python experiments/stress_path_seed_robustness.py \
  --path-model second_moment \
  --output-prefix artifacts/tail_second_moment_seed_robustness
```

Generated evidence:

- `stress_path_risk_*`: single-fit frontiers, profiles, schedules, and chart;
- `stress_path_seed_robustness_*`: five-fit conditional-mean ledger and chart;
- `tail_second_moment_seed_robustness_*`: five-fit second-moment ledger and
  chart; and
- `rebalance_economic_calibration_*`: regenerated primary ledger and chart with
  the retained low-profile policy.

## Limitation

The scenario generator deliberately contains the coherent wrong-call regime
that these models are designed to capture. Multiple disjoint synthetic fits
reduce seed luck but do not prove production profitability. The final gate is
still point-in-time walk-forward replay on actual rebalance predictions,
realized returns, fills, impact, spreads, FX, financing, and borrow data.
