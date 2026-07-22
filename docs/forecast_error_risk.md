# Forecast-error path risk experiment

## Predeclared hypothesis

The current rebalance frontier prices daily market covariance but treats the
expected-return forecast as known. In a point-in-time rebalance, estimation
error in that forecast is itself an investment risk: an optimistic call can
pull both additions and deletions forward and create the same premature gross
exposure that the planner is meant to avoid.

Test one research-only candidate. Add the variance of forecast-estimation P&L
to the same accumulated-inventory risk objective used for market P&L. The
candidate has no manually tuned optimizer coefficient:

```text
predictive P&L variance
    = daily market P&L variance
    + independent forecast-error variance
    + persistent basket-direction forecast-error variance
```

The persistent component is estimated from a disjoint synthetic calibration
history. In production, the identical estimator would consume stored
point-in-time forecasts and later outcomes. The high/medium/low risk label
continues to price all dollar-squared risk through the frontier's existing
economic coefficient.

## Protected replay contract

- Do not change event seeds, realized returns, realized costs, scoring, or the
  development/holdout split.
- Compare the candidate with the existing medium-risk variance frontier on the
  same events and TCA surfaces.
- Use only forecast uncertainty and calibration information available before
  each event. Latent returns and realized outcomes remain scoring-only.
- Screen mechanics on development events 1-2. Run all 12 development events
  only if the model solves and preserves hard execution invariants.
- Do not inspect events 13-24 unless every development gate passes.

## Acceptance gates

The candidate must satisfy all existing rolling-replay gates:

- mean realized P&L no worse than one basis point per event;
- event P&L volatility at least 0.05 basis point lower;
- loss CVaR and mean within-event drawdown no more than 0.05 basis point worse;
- urgent names never later and small names never earlier in any paired event;
- mean early factor imbalance within one percentage point of baseline;
- late/early volume ratio at least 90% of baseline and at least 1.0;
- daily-volume rank correlation no more than 0.10 lower and no more than one
  lost nondecreasing transition; and
- participation, direction, and completion hard tolerances unchanged.

## Result

The automatically estimated persistent directional scale was `0.552305`, close
to the synthetic calibration population's `0.55`. The candidate prices the
forecast-error variance with exactly the selected market-risk coefficient, so
there is no additional model coefficient or user input. Mean forecast-error
P&L volatility was about $546k in development and $554k in holdout, versus
roughly $4m of modeled market P&L volatility.

| Split | P&L delta | Volatility delta | Loss-CVaR delta | Mean drawdown delta | Factor delta | Ramp delta | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Development, events 1-12 | +$38.7k | -0.161 bp | -0.214 bp | -0.104 bp | +0.032 pp | +0.012 | Pass to holdout |
| Untouched holdout, events 13-24 | -$33.7k | +0.022 bp | -0.038 bp | -0.037 bp | +0.026 pp | +0.010 | Fail volatility gate |
| Combined, descriptive only | +$5.0k | -0.086 bp | -0.280 bp | -0.071 bp | +0.029 pp | +0.011 | Not a promotion test |

Every capacity, direction, completion, urgent-start, small-start, factor, and
volume-ramp gate passed in both splits. Development passed all 13 gates without
tuning, which permitted the first and only opening of the chronological
holdout. The holdout then missed the required volatility improvement: realized
volatility increased by 0.022 bp instead of falling by at least 0.05 bp.

The economic direction is plausible—tail loss, drawdown, and the ramp improved
slightly in both splits—but the effect is too small relative to event-to-event
noise to justify a production term. The candidate is therefore retained as a
research model and diagnostic decomposition, while the production frontier is
unchanged. The holdout is now spent and must not be reused for tuning.

## Reproduce

```bash
env PYTHONPATH=. python experiments/forecast_error_risk_walkforward.py \
  --solver OSQP --event-start 0 --n-events 12 \
  --risk-aversion medium \
  --output-prefix artifacts/forecast_error_risk_dev

env PYTHONPATH=. python experiments/forecast_error_risk_walkforward.py \
  --solver OSQP --event-start 12 --n-events 12 \
  --risk-aversion medium \
  --output-prefix artifacts/forecast_error_risk_holdout

env PYTHONPATH=. python experiments/forecast_error_risk_walkforward.py \
  --solver OSQP --event-start 0 --n-events 24 \
  --risk-aversion medium \
  --output-prefix artifacts/forecast_error_risk_all
```

Each prefix records the complete schedules, point-in-time trials, paired event
deltas, full frontier, selected coefficients, factor exposures, acceptance
gates, daily P&L, volume profiles, and a six-panel visualization.
