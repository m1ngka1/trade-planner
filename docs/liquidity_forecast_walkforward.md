# Point-in-time event-liquidity forecast experiment

## Predeclared hypothesis

The strongest result in the economic fixture came from a date-varying forecast
of rebalance liquidity, not from a larger risk penalty. If expected ADV and
market impact improve near the event, the optimizer should naturally reserve
more flow for those dates, reduce premature inventory, and pay less impact. No
daily trade amount or volume curve is constrained directly.

Test the current flat-ADV planner against one candidate that receives a
point-in-time lower-quantile ADV forecast estimated from disjoint synthetic
history. Both strategies retain identical orders, alpha, covariance, TCA,
participation rules, hard completion, and medium risk aversion.

## Fresh cohort and protected surfaces

- Use 24 new event, residual-scenario, and liquidity seeds that do not overlap
  the spent forecast-risk events 1-24.
- Fix the liquidity population and high/medium/low mapping before running any
  event. Do not change realized returns, realized costs, scoring, or gates after
  inspecting results.
- Events 25-36 are development; events 37-48 are a sealed holdout.
- Open the holdout only if every development gate passes without tuning.
- Latent liquidity and realized ADV stay outside the optimizer context.

## Synthetic liquidity population

The population median daily ADV multiplier is the previously recorded moderate
event-liquidity curve:

```text
0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.50, 1.80
```

This describes the replay population, not a schedule constraint. Event-common,
date-common, and name-specific log-volume shocks have standard deviations
`0.10`, `0.08`, and `0.10`. A separate 2,000-event calibration history
estimates the date/name log mean and standard deviation.

The user still chooses only a risk label. The planner receives this automatic
liquidity quantile:

| Risk aversion | Forecast ADV quantile | Investment meaning |
|---|---:|---|
| High | 10th percentile | Protect completion and participation against weak liquidity |
| Medium | 25th percentile | Use likely liquidity while retaining a downside buffer |
| Low | 50th percentile | Use median liquidity and accept more forecast risk |

The forecast ADV drives both physical capacity and the dollar impact curve.
Existing economic calibration continues to set covariance, tail-risk, alpha,
and cost coefficients.

## Realized-liquidity scoring

`PointInTimeRebalanceEvent` must keep realized ADV outside `ctx`, just like
realized returns and costs. Realized impact is evaluated using actual ADV, and
the replay records maximum and 95th-percentile actual participation plus any
excess over the forecast-time participation policy. Existing events without
realized ADV retain their current behavior exactly.

## Acceptance gates

The candidate must pass all existing P&L, volatility, loss-CVaR, drawdown,
urgent-start, small-start, factor-balance, ramp, direction, forecast-cap, and
completion gates. It must also:

- reduce realized impact cost;
- not increase mean event-level 95th-percentile realized participation by more
  than 0.5 percentage point; and
- not increase maximum realized participation by more than 1 percentage point
  versus the flat-ADV baseline.

## Result

Pending.
