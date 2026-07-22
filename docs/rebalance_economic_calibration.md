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
return `mu_t`, daily covariance `Sigma_t`, centered residual-return scenario
`epsilon_s,t`, and trade `q_t`, the calibrated hybrid model minimizes:

```text
sum_t impact(q_t) + spread_and_fees(q_t)
    - sum_t mu_t' w_t
    + lambda_cov * sum_t w_t' Sigma_t w_t
    + lambda_tail * CVaR_95(-sum_t epsilon_s,t' w_t)
```

This directly separates the investment decisions:

- expected alpha rewards buying predicted additions and selling predicted
  deletions before the event when the forecast justifies the exposure;
- accumulated P&L variance penalizes premature country, sector, industry, and
  specific risk;
- the scenario term prices asymmetric event-call errors and fat tails that are
  not already explained by covariance;
- impact discourages blocks and spreads flow across available dates; and
- physical capacity plus hard completion makes urgent names start before future
  liquidity becomes insufficient.

Scenario returns are centered before optimization. The expected-return forecast
must already include confidence and probability of the rebalance call being
correct; centering prevents the scenario sample mean from becoming a hidden
second alpha coefficient.

## Automatic coefficient selection

The calibration procedure is:

1. Load date-by-name `impact_bps_at_10pct_adv` and `linear_cost_bps` from TCA and
   use those surfaces directly in the objective. If impact is unavailable, use a
   conservative date-by-name volatility square-root fallback. Target-notional-
   weighted medians remain available only as a controlled scalar-cost ablation.
2. Calculate separate basket-specific covariance and scenario-CVaR scales from
   expected alpha/cost dollars divided by full-horizon variance or CVaR. This
   removes dependence on whether a test basket uses one share or production-
   sized orders.
3. Estimate the scenario overlay as only the excess of full-basket scenario
   CVaR over covariance-implied normal expected shortfall. In this fixture that
   fraction is 5.44%; the user does not enter it.
4. Apply one common internal frontier-strength multiplier to the covariance
   scale and the excess-tail scale, then solve the grid with identical
   constraints, forecasts, and TCA inputs.
5. Measure expected alpha, impact, fees, net P&L, P&L volatility, 95% loss VaR,
   and loss CVaR for every solved schedule.
6. Give each user label a risk budget inside the feasible frontier and select
   the highest expected net P&L that stays inside that budget. Plans within one
   basis point of parent gross are treated as economically tied, and the
   lower-risk plan wins rather than spending risk for solver noise or a trivial
   forecast edge.

| User selection | Feasible P&L-risk range made available | Investment meaning |
|---|---:|---|
| High risk aversion | 5% | Use the stable covariance frontier and stay very close to minimum risk |
| Medium risk aversion | 50% | Spend risk when forecast alpha materially improves expected net P&L |
| Low risk aversion | 100% | Allow the full frontier and pursue the highest material expected net P&L |

These percentages are portfolio policy, not mathematical necessities. They are
central defaults that should later be validated against the desk's realized
drawdown tolerance; users do not enter raw optimizer coefficients. With
scenario data, `medium` and `low` use the hybrid excess-tail frontier. `High`
uses covariance because the experiment found that optimizing the few most
extreme scenarios added estimation risk without robust downside improvement.

## Recorded synthetic experiment

The fixed ten-day economic fixture has twelve additions/deletions across HK,
Japan, and the US, with country, sector, industry, and specific covariance.
Urgent orders need 8.5 days of capacity, medium orders need 4.5 days, and small
orders need one day. Expected rebalance alpha is concentrated near the common
event. The tail model fits on 256 centered fat-tail scenarios with a 10% wrong-
call regime. Every schedule is then evaluated on five separate 5,000-scenario
samples whose seeds are not used by the optimizer.

The current recorded outcomes are approximately:

| Profile | Expected net P&L | P&L volatility | Replicated mean 95% loss CVaR | Late/early volume | Early max factor imbalance |
|---|---:|---:|---:|---:|---:|
| High, covariance policy | $645k | $3.78m | $8.49m | 1.72x | 5.47% |
| Medium, covariance baseline | $750k | $3.99m | $8.99m | 1.00x | 3.83% |
| Medium, automatic hybrid | $710k | $3.86m | $8.68m | 1.35x | 3.30% |
| Medium, pure CVaR | $760k | $4.32m | $9.58m | 0.87x | 21.00% |
| Low, covariance baseline | $817k | $4.52m | $10.10m | 0.51x | 8.46% |
| Low, automatic hybrid | $816k | $4.51m | $10.08m | 0.52x | 8.38% |
| Medium + strong forecast event liquidity | $710k | $3.57m | $8.05m | 2.70x | 2.77% |

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
- pure sample CVaR is discarded: its medium profile increases independent-
  sample loss CVaR by 6.5%, volatility by 8.2%, and early factor imbalance from
  3.8% to 21.0%. It overfits a small set of tail scenarios and loses stable
  factor hedges;
- the automatic hybrid medium profile retains covariance risk and prices only
  the 5.44% excess scenario tail. Against the variance medium profile it cuts
  replicated mean loss CVaR by 3.5% and volatility by 3.2%, improves early
  factor imbalance from 3.83% to 3.30%, delays small orders from day 1 to day 4,
  and raises late/early volume from 1.00x to 1.35x. Expected net P&L falls by
  $40k, inside the $48.7k one-basis-point tie band, so it is retained;
- the automatic hybrid low profile is nearly economically identical to the
  variance low profile and makes small but consistent risk improvements. The
  hybrid high trial is discarded and the automatic high policy stays on the
  covariance frontier;
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
| Balance country/sector/industry early | Fixed shape benchmark: 1.6% early imbalance and 96.4% improvement over no-factor. Automatic hybrid medium: 3.30%, versus 3.83% for variance medium and 14.3% without factor risk. |
| Urgent names early, small orders may wait | Fixed shape benchmark: urgent day 1 and small day 8. Automatic high: urgent day 1 and small day 10. Automatic hybrid medium: urgent day 1 and small day 4. |
| Gradually increase volume | Fixed shape benchmark: Spearman 1.0 and 4.30x late/early volume. Automatic hybrid improves medium from 1.00x to 1.35x; forecast event liquidity raises it to 2.70x without a schedule constraint. |
| Optimizer-derived, not hard-coded | Capacity, factor risk, alpha, impact, scenario tail risk, and forecast ADV are inputs to one convex optimizer; none of the retained plans fixes daily trade amounts. |
| Reduce P&L swing while preserving profit | Automatic hybrid medium cuts volatility 3.2% and replicated loss CVaR 3.5% for less than 1 bp of expected-P&L sacrifice. Strong forecast event liquidity cuts both by about 10.5% when that forecast is available. |
| Automatic coefficients | Date/name TCA sets costs; basket economics scales covariance and CVaR; excess-tail data sets their relative weight; high/medium/low selects a solved plan with a 1 bp materiality rule. |

The post-run feasibility audit across every recorded schedule found a maximum
participation-cap excess of 0.0221 share, maximum wrong-direction solver dust of
0.000026 share, and maximum terminal residual below 0.0000001 share.

The prior unitless fixed risk weight is retained only as a baseline. After
tightening OSQP's feasibility tolerance it solves the realistic-dollar fixture,
but its expected net P&L and positive-P&L probability are much weaker than the
calibrated profiles. The original false `user_limit`/cap-tolerance behavior is
also why the production-sized experiment now uses tighter OSQP feasibility
tolerances and polishing.

## Important limitation

The expected P&L is only as real as the alpha, TCA, covariance, and scenario
forecasts. On this synthetic fixture the first independent sample's positive-
P&L probability is only about 58–59%, so the current evidence validates
calibration mechanics and the risk/profit trade-off—not a production
profitability claim. `CalibratedRebalancePlan.economically_viable` is false when
forecast expected net P&L is non-positive, so downstream workflow can flag a
compulsory cost-minimizing execution instead of calling it a profit opportunity.

The next high-value research steps are:

1. walk-forward replay on actual rebalance baskets using point-in-time forecast,
   close, VWAP, spread, impact, FX, financing, and borrow data;
2. walk-forward calibration of date-by-name TCA forecasts against realized
   spread, impact, and fill data;
3. point-in-time scenario calibration by rebalance type, confidence, days to
   event, wrong-call frequency, and realized tail severity;
4. scenario-reduction and solver work so the hybrid frontier is fast enough for
   an interactive production planner; and
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
- `rebalance_economic_calibration_scenario_summary.csv`: five-seed out-of-sample
  P&L, VaR, and loss-CVaR statistics;
- `rebalance_economic_calibration_schedules.csv`: complete schedules for every
  solved trial; and
- `rebalance_economic_calibration.png`: variance/pure-CVaR/hybrid frontiers,
  daily volume, factor-risk ablation, and replicated loss-CVaR comparison.
