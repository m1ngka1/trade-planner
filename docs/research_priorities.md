# Trade-planner research priorities

## Current conclusion

The optimizer can produce all four requested execution behaviors without a
hard-coded daily schedule:

- country, sector, and industry exposure is more balanced early;
- urgent names remain on day one while small names wait longer;
- daily volume rises toward the event; and
- liquidity, participation, direction, and completion remain explicit model
  inputs or gates.

Across two untouched synthetic holdouts, the strongest minimax/risk-scaled
candidates consistently reduced volatility, loss-CVaR, drawdown, early factor
imbalance, and impact cost. Neither preserved realized P&L. The model is
therefore a research challenger, not a production default.

## Ranked next work

| Rank | Workstream | Importance / expected potential | Evidence from completed trials | Next decision-quality action |
|---:|---|---|---|---|
| 1 | Real point-in-time rebalance replay | Critical / highest | Synthetic holdouts disagree on P&L even when every risk and behavior metric improves. Money-making claims cannot be proven from generated returns. The leakage-controlled loader and baseline-versus-challenger runner are implemented. | Export the first frozen real development bundle with stored prediction vintages, orders, close/VWAP, realized ADV, spread/impact, FX, financing, borrow, and GICS/country classifications. Run it without changing the predeclared gates; open a sealed holdout only if development passes. |
| 2 | Conditional alpha-decay and timing model | Very high | Holdout P&L losses come from foregone gross holding alpha, especially a few large event misses; forecast-plan P&L has weak event-level correlation with realized P&L. | Estimate expected return paths and uncertainty by rebalance type, add/delete side, days to event, country, sector, industry, liquidity, crowding, and prediction confidence with hierarchical shrinkage. Feed both mean and uncertainty point-in-time to the optimizer. |
| 3 | Numerically scaled execution formulation | Implemented / deployment-critical | The exact quadratic P&L floor now solves on spent events 25–26 under explicit CLARABEL. Both floors clear and cap, direction, completion, and urgency certificates pass. The investment candidate still fails volatility, small-order, and factor gates, so only the numerical formulation is kept. | Use per-name scaling and strict raw-share certificates in the real historical replay. Extend backend-specific certification only when another production solver is actually required; do not use the mechanics pass as profit evidence. |
| 4 | Risk-scaled minimax liquidity challenger | High for swing, unproven for profit | Fresh validation cut volatility 5.49 bp and loss-CVaR 16.08 bp; untouched holdout cut volatility 8.08 bp and loss-CVaR 14.75 bp, but lost 3.24 bp/event. | Preserve as a challenger for real-data replay. Do not promote or tune against spent synthetic holdouts. |
| 5 | Automatic High/Medium/Low calibration from real economics | High after real data exists | Fixed mappings create interpretable behavior, but synthetic holdouts show that coefficient stability is not enough when timing alpha is noisy. | Select mappings by nested chronological validation on realized utility: net P&L first, then volatility/CVaR/drawdown and operational gates. Store calibration date, sample, and uncertainty with every plan. |
| 6 | More synthetic penalty or threshold tuning | Low | Confidence haircuts, uncertainty budgets, recourse, proximal terms, profit floors, plan-selection gates, and baseline-relative regret risk all failed or were numerically unstable. | Defer until real-data error analysis identifies a specific missing risk or constraint. Avoid blind sweeps. |

## Experimental automatic policy map

The latest research policy derives every model strength from the user's desk
risk label. These values are interpretable defaults for replay, not approved
production coefficients:

| Investment control | High | Medium | Low | Economic meaning |
|---|---:|---:|---:|---|
| Feasible P&L-risk frontier fraction | 5% | 50% | 100% | Maximum portion of the solved risk range available to the plan |
| Liquidity forecast quantile | 10% | 25% | 50% | Downside protection against weak realized ADV |
| Event-liquidity shape consumed | 95% | 50% | 0% | How strongly rising event liquidity is allowed to delay flow |
| Optional-alpha confidence | 97.5% | 75% | 50% | Forecast hurdle applied in proportion to unused execution capacity |
| Minimax factor-stress fraction | 95% | 50% | 0% | Correlation-break protection for the worst country/sector/industry exposure |
| Inventory-risk price | automatic per event | automatic per event | automatic per event | Selected from the flat-information investment frontier, then held invariant as liquidity changes |

The mapping deliberately makes high risk aversion more conservative about
liquidity and forecast alpha, while low risk aversion prioritizes expected P&L.
It does not impose start dates or daily trade percentages; the optimizer still
chooses the complete schedule.

## Evidence map

- `artifacts/walkforward_research_ledger.csv`: compact index of all 46 recorded
  screens, development runs, holdouts, crashes, and decisions.
- `artifacts/liquidity_minimax_factor_dev*` and
  `artifacts/liquidity_minimax_factor_holdout*`: full-shape minimax evidence.
- `artifacts/risk_scaled_liquidity_fresh_dev*` and
  `artifacts/risk_scaled_liquidity_fresh_holdout*`: fully fresh risk-scaled
  validation and untouched holdout, including schedules, coefficients, factor
  paths, liquidity audits, gates, and six-panel visualizations.
- `docs/liquidity_forecast_walkforward.md`: chronological hypothesis,
  predeclaration, result, and keep/discard record.
- `docs/historical_replay_bundle.md` and `experiments/historical_replay.py`:
  strict real-data schema, no-leakage validation, explicit cohort authorization,
  source hashing, and the frozen baseline-versus-challenger evaluator.
- `docs/numerical_scaling.md` and `artifacts/numerical_scaling_mechanics*`:
  predeclaration, algebraic invariance evidence, strict raw-share certificates,
  fixed spent-event mechanics result, and six-panel visualization.

## Production decision rule

Do not describe a policy as profitable until a chronologically untouched real
holdout passes every P&L, swing, behavior, liquidity, and hard-execution gate.
Risk improvement alone is not enough: the desk makes money only when saved
impact and reduced downside exceed the holding alpha forfeited by waiting.
