# Point-in-time conditional alpha-decay calibration

## Investment hypothesis

The strongest synthetic execution policies consistently reduced volatility,
tail loss, drawdown, early factor imbalance, and impact, but lost money when
waiting forfeited more holding alpha than the saved execution cost. Another
optimizer penalty cannot resolve that error. The planner instead needs a
point-in-time estimate of the return earned by carrying each signed rebalance
position on each day, together with honest forecast uncertainty.

Model the directional holding return

```text
sign(parent order) * realized daily return
```

as a function of information known at the planning cutoff:

- the raw point-in-time directional return forecast;
- progress through the execution horizon and finite days to the event;
- buy/delete versus sell/add side;
- order size relative to available horizon capacity;
- forecast ADV and forecast uncertainty;
- country, GICS sector, GICS industry, and urgency; and
- optional rebalance type, prediction confidence, and crowding fields when
  present in `orders.csv`.

The calibrated prediction is converted back to signed security return and fed
to the existing inventory-alpha objective. Its predictive standard error is
fed to the existing capacity-slack confidence model. High/Medium/Low therefore
keeps its investment meaning—97.5%, 75%, or 50% optional-alpha confidence—
without asking the user for a regression penalty or alpha coefficient.

## Chronological contract

For event `j`, training may include event `i` only when both conditions hold:

```text
i appears earlier in the bundle
i.realized_available_at <= j.information_cutoff
```

The current event's realized return and every future event remain inaccessible
during fitting and prediction. Development and holdout bundles remain
physically separate. Calibration never opens another bundle or changes cohort
roles.

The first events fall back to their original point-in-time forecast until four
eligible realized events exist. This warm-up is a data-availability rule, not
a favorable-event filter.

## Automatic regularization and uncertainty

Use a linear ridge model with an unpenalized intercept and a deterministic
feature encoder. Candidate penalties are fixed dimensionless multiples of the
training design scale:

```text
0.01, 0.1, 1, 10, 100
```

For each current event, choose the multiplier with the lowest equal-event-
weighted leave-one-event-out RMSE using only its eligible training history.
Refit on all eligible history. Predictive uncertainty comes from the selected
model's held-out residual scale and ridge leverage; no optimizer outcome,
realized P&L gate, or current-event return selects the penalty.

## Protected surfaces

- Do not alter liquidity forecasts, execution costs, optimizer coefficients,
  event ordering, cohort roles, realized scoring, or the 16 promotion gates.
- Do not use the current or a future event to choose features, category levels,
  regularization, coefficients, or uncertainty.
- Do not claim profitability from a controlled synthetic recovery test.
- Do not open a historical holdout until a real development replay passes.

## Controlled mechanics acceptance criteria

A deterministic synthetic population with a known side/sector/progress alpha
pattern is used only to verify estimator mechanics. After the four-event
warm-up, keep the estimator infrastructure only if:

1. calibrated directional-return RMSE is at least 10% lower than the raw
   point-in-time forecast RMSE;
2. directional sign accuracy is not lower than the raw forecast;
3. an 80% predictive interval covers between 65% and 95% of realized returns;
4. perturbing current and future realized returns cannot change the current
   prediction, selected penalty, or training-event list;
5. unseen country/sector/industry labels predict without failure; and
6. the full repository test suite passes.

Passing this test keeps a causal calibration pipeline for real development
data. Only a later untouched real holdout can establish whether calibrated
alpha preserves P&L while the optimizer reduces swing.

## Implementation

`trade_planner/alpha_decay.py` builds date-name observations in signed parent-
order direction. Numeric fields are standardized using training history only;
country, sector, industry, side, urgency, and rebalance-type levels are also
learned from training only, so a new label maps to an all-zero unseen category
instead of changing the fitted schema. Every historical event records its
eligible training IDs, five cross-validation scores, selected penalty,
coefficients, prediction, and predictive uncertainty.

`experiments/historical_replay.py` enables this walk-forward calibration by
default and applies the same calibrated event to the flat-liquidity baseline
and forecast-liquidity challenger. Liquidity, optimizer risk prices, costs,
hard constraints, and realized scoring are unchanged. The calibrated mean
still enters the ordinary inventory-alpha objective, while its uncertainty
enters the existing risk-profile confidence hurdle. There is no new manual
optimizer coefficient and no imposed daily schedule.

## Controlled result

The fixed 14-event population used four warm-up events and scored 480
date-name observations over the following ten events. All predeclared gates
passed:

| Metric | Raw | Calibrated | Result |
|---|---:|---:|---|
| Equal-event directional RMSE | 1.53 bp/day | 0.90 bp/day | 40.8% lower |
| Directional sign accuracy | 74.4% | 85.0% | 10.6 pp higher |
| Nominal 80% interval coverage | — | 94.4% | inside the 65–95% gate |
| Current/future outcome perturbation | — | exactly invariant | pass |
| Unseen country/sector/industry | — | finite prediction | pass |
| Repository verification | — | 102 tests plus 2 subtests | pass |

Decision: **keep the estimator infrastructure for a real development replay**.
This is deliberately not a profit claim: the generating relationship is known
in the controlled population. The next economic decision requires a frozen
real development bundle, followed by one sealed historical holdout only if all
existing profitability, swing, behavior, liquidity, and feasibility gates
pass.

Reproduce the evidence with:

```bash
env PYTHONPATH=. python experiments/alpha_decay_walkforward.py \
  --full-suite-verified \
  --output-prefix artifacts/alpha_decay_mechanics
```

Artifacts include the summary, seven gates, chronological audit, all
predictions and uncertainties, fitted coefficients, and a four-panel PNG at
`artifacts/alpha_decay_mechanics*`.
