# Investment-driven rebalance calibration

## Decision

Use two validation layers:

1. keep the deterministic shape benchmark as a guardrail for capacity,
   urgency, early factor balance, and the rising-volume mechanism; and
2. select production coefficients from an expected-net-P&L versus accumulated
   P&L-risk frontier built in dollar units.

The second layer is now implemented. A user chooses only `high`, `medium`, or
`low` risk aversion. The desk's forecasts and TCA data set the other economic
inputs.

## Economic objective

For accumulated signed dollar inventory `w_t`, probability-weighted expected
return `mu_t`, daily covariance `Sigma_t`, and trade `q_t`, the calibrated model
minimizes:

```text
sum_t impact(q_t) + spread_and_fees(q_t)
    - sum_t mu_t' w_t
    + lambda * sum_t w_t' Sigma_t w_t
```

This directly separates the investment decisions:

- expected alpha rewards buying predicted additions and selling predicted
  deletions before the event when the forecast justifies the exposure;
- accumulated P&L variance penalizes premature country, sector, industry, and
  specific risk;
- impact discourages blocks and spreads flow across available dates; and
- physical capacity plus hard completion makes urgent names start before future
  liquidity becomes insufficient.

The expected-return forecast must already include confidence and probability of
the rebalance call being correct. Adding another arbitrary alpha multiplier
would hide forecast calibration errors inside the optimizer.

## Automatic coefficient selection

The calibration procedure is:

1. Load date-by-name `impact_bps_at_10pct_adv` and `linear_cost_bps` from TCA and
   use those surfaces directly in the objective. If impact is unavailable, use a
   conservative date-by-name volatility square-root fallback. Target-notional-
   weighted medians remain available only as a controlled scalar-cost ablation.
2. Calculate a basket-specific risk-coefficient scale from the parent basket's
   expected alpha/cost dollars divided by full-horizon P&L variance. This removes
   the dependence on whether a test basket uses one share or production-sized
   orders.
3. Solve a logarithmic grid of risk coefficients with identical constraints,
   forecasts, and cost inputs.
4. Measure expected alpha, impact, fees, net P&L, P&L volatility, 95% loss VaR,
   and probability of positive P&L for every solved schedule.
5. Give each user label a risk budget inside the feasible frontier and select
   the highest expected net P&L that stays inside that budget. Plans within one
   basis point of parent gross are treated as economically tied, and the
   lower-risk plan wins rather than spending risk for solver noise or a trivial
   forecast edge.

| User selection | Feasible P&L-risk range made available | Investment meaning |
|---|---:|---|
| High risk aversion | 15% | Stay close to the minimum achievable holding-P&L risk |
| Medium risk aversion | 50% | Spend risk when forecast alpha materially improves expected net P&L |
| Low risk aversion | 100% | Allow the full frontier and pursue the highest material expected net P&L |

These percentages are portfolio policy, not mathematical necessities. They are
central defaults that should later be validated against the desk's realized
drawdown tolerance; users do not enter raw optimizer coefficients.

## Recorded synthetic experiment

The fixed ten-day economic fixture has twelve additions/deletions across HK,
Japan, and the US, with country, sector, industry, and specific covariance.
Urgent orders need 8.5 days of capacity, medium orders need 4.5 days, and small
orders need one day. Expected rebalance alpha is concentrated near the common
event. All profiles use the same 5,000 return scenarios.

The current recorded outcomes are approximately:

| Profile | Expected net P&L | P&L volatility | Positive-P&L probability | Early max factor imbalance |
|---|---:|---:|---:|---:|
| Prior fixed weight, no alpha in solve | $628k | $3.78m | 56.6% | 5.5% |
| High | $645k | $3.78m | 56.8% | 5.5% |
| Medium | $750k | $3.99m | 57.5% | 3.8% |
| Medium using scalar TCA medians | $762k | $4.07m | 57.4% | 3.5% |
| Low | $817k | $4.52m | 57.2% | 8.5% |
| Medium + moderate forecast event liquidity | $715k | $3.67m | 57.7% | 2.9% |
| Medium + strong forecast event liquidity | $710k | $3.57m | 57.9% | 2.8% |

The result is a genuine trade-off, not a claim that one profile dominates:

- high risk aversion reduces P&L volatility by about 16% versus low while
  giving up about $172k of expected net P&L;
- the high profile begins urgent flow immediately, keeps small-order flow at
  solver dust until day 10, and produces 1.72x as much gross volume late in the
  horizon as early, without fixing a daily schedule;
- removing expected alpha at the medium risk coefficient reduces expected net
  P&L by about $88k;
- removing factor covariance raises early factor imbalance from about 3.8% to
  14.3% and increases full-model P&L volatility;
- replacing date-by-name TCA with scalar basket medians raises expected net P&L
  by only about $11k, below the $48.7k materiality threshold, while adding about
  $75k of P&L volatility. The scalar simplification is therefore discarded by
  the lower-risk tie-break;
- a moderate date-varying event-liquidity forecast improves medium-profile P&L
  volatility by about 8.0%, raises the late/early volume ratio from 1.00x to
  1.81x, and gives up only about $35k of expected net P&L—less than one basis
  point of parent gross—so it is retained;
- a strong event-liquidity forecast raises the late/early ratio to 2.70x, cuts
  P&L volatility by about 10.6%, and gives up about $41k of expected net P&L.
  It is also inside the one-basis-point tie band and is retained as a tested
  alternative, subject to the desk actually forecasting that liquidity surface.

## Goal coverage

| Goal | Current evidence |
|---|---|
| Balance country/sector/industry early | Fixed shape benchmark: 1.6% early imbalance and 96.4% improvement over no-factor. Economic ablation: 3.8% with factor risk versus 14.3% without it. |
| Urgent names early, small orders may wait | Fixed shape benchmark: urgent day 1 and small day 8. Economic high-risk profile: urgent day 1 and small day 10. |
| Gradually increase volume | Fixed shape benchmark: Spearman 1.0 and 4.30x late/early volume. Moderate event-liquidity forecast improves the economic medium profile from 1.00x to 1.81x without a schedule constraint. |
| Optimizer-derived, not hard-coded | Capacity, factor risk, alpha, impact, and forecast ADV are inputs to one convex optimizer; none of the retained plans fixes daily trade amounts. |
| Reduce P&L swing while preserving profit | High versus low demonstrates the profile trade-off; retained moderate event liquidity cuts medium-profile volatility 8.0% for less than 1 bp of expected-P&L sacrifice. |
| Automatic coefficients | Date/name TCA sets cost inputs, basket economics scales the risk grid, and high/medium/low selects a solved plan with a 1 bp materiality rule. |

The post-run feasibility audit across every recorded schedule found a maximum
participation-cap excess of 0.0221 share, maximum wrong-direction solver dust of
0.000018 share, and maximum terminal residual below 0.0000001 share.

The prior unitless fixed risk weight is retained only as a baseline. After
tightening OSQP's feasibility tolerance it solves the realistic-dollar fixture,
but its expected net P&L and positive-P&L probability are much weaker than the
calibrated profiles. The original false `user_limit`/cap-tolerance behavior is
also why the production-sized experiment now uses tighter OSQP feasibility
tolerances and polishing.

## Important limitation

The expected P&L is only as real as the alpha and TCA forecasts. On this
synthetic fixture the positive-P&L probability is only about 57%, so the current
evidence validates calibration mechanics and the risk/profit trade-off—not a
production profitability claim. `CalibratedRebalancePlan.economically_viable`
is false when forecast expected net P&L is non-positive, so downstream workflow
can flag a compulsory cost-minimizing execution instead of calling it a profit
opportunity.

The next high-value research steps are:

1. walk-forward replay on actual rebalance baskets using point-in-time forecast,
   close, VWAP, spread, impact, FX, financing, and borrow data;
2. walk-forward calibration of date-by-name TCA forecasts against realized
   spread, impact, and fill data;
3. forecast-error shrinkage by rebalance type, confidence, and days to event;
4. scenario/CVaR selection for fat tails and event jumps rather than relying
   only on covariance and normal-probability summaries; and
5. risk-profile defaults calibrated to realized PM drawdowns and alpha capture,
   not to the synthetic fixture.

## Reproduce and inspect

```bash
env PYTHONPATH=. \
  PYTHONPYCACHEPREFIX=/private/tmp/trade_planner_pycache \
  MPLCONFIGDIR=/private/tmp/trade_planner_mpl \
  XDG_CACHE_HOME=/private/tmp/trade_planner_cache \
  ./.conda-env/bin/python experiments/rebalance_economic_calibration.py \
  --solver OSQP \
  --output-prefix artifacts/rebalance_economic_calibration
```

Generated evidence:

- `rebalance_economic_calibration_trials.csv`: research ledger with idea,
  metrics, failure reason, and keep/discard decision;
- `rebalance_economic_calibration_frontier.csv`: every tested risk coefficient;
- `rebalance_economic_calibration_profiles.csv`: daily volume and cumulative
  behavior;
- `rebalance_economic_calibration_exposures.csv`: daily factor exposures;
- `rebalance_economic_calibration_scenario_summary.csv`: common-scenario P&L
  distribution statistics;
- `rebalance_economic_calibration_schedules.csv`: complete schedules for every
  solved trial; and
- `rebalance_economic_calibration.png`: profit-risk frontier, daily volume,
  factor-risk ablation, and scenario distributions.
