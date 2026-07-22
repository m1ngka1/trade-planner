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
| 1 | Real point-in-time rebalance replay | Critical / highest | Synthetic holdouts disagree on P&L even when every risk and behavior metric improves. The loader, alpha calibration, seven-policy panel, chronological High/Medium/Low selection, selected-vector replay, source hashes, and visuals are now executable end to end. Smoke data is not profit evidence. | Export the first frozen real development bundle with stored prediction vintages, orders, close/VWAP, realized ADV, spread/impact, FX, financing, borrow, and GICS/country classifications. Run `historical_policy_panel.py` without changing the ladder or gates; open a sealed holdout only if development passes. |
| 2 | Conditional alpha-decay and timing model | Implemented mechanics / very high | The leakage-safe ridge calibrator improved controlled RMSE 40.8%, raised sign accuracy 10.6 pp, produced valid uncertainty, and was exactly invariant to current/future realized-return perturbations. This proves mechanics only. | Run the frozen calibrator on the first real development bundle. Preserve the optimizer and 16 economic/behavior gates; open a sealed holdout only if development passes. |
| 3 | Numerically scaled execution formulation | Implemented / deployment-critical | The exact quadratic P&L floor now solves on spent events 25–26 under explicit CLARABEL. Both floors clear and cap, direction, completion, and urgency certificates pass. The investment candidate still fails volatility, small-order, and factor gates, so only the numerical formulation is kept. | Use per-name scaling and strict raw-share certificates in the real historical replay. Extend backend-specific certification only when another production solver is actually required; do not use the mechanics pass as profit evidence. |
| 4 | Risk-scaled minimax liquidity challenger | High for swing, unproven for profit | Fresh validation cut volatility 5.49 bp and loss-CVaR 16.08 bp; untouched holdout cut volatility 8.08 bp and loss-CVaR 14.75 bp, but lost 3.24 bp/event. | Preserve as a challenger for real-data replay. Do not promote or tune against spent synthetic holdouts. |
| 5 | Automatic High/Medium/Low calibration from real economics | Implemented mechanics / high | The chronological selector rejects hard/behavior failures, requires confidence-adjusted positive P&L, applies realized downside budgets, and preserves monotone profile ordering. Its controlled Low policy gained 0.33 bp/event and cut volatility 0.49 bp; this is not real evidence. | Generate the predeclared policy panel on the frozen real development bundle and select profiles without changing the ladder or investment gates. Only a successful real holdout can replace the fixed fallback map. |
| 6 | More synthetic penalty or threshold tuning | Low | Confidence haircuts, uncertainty budgets, recourse, proximal terms, profit floors, plan-selection gates, baseline-relative regret risk, and the systematic-first specific-risk split all failed or were numerically unstable. The latest split improved factor balance but lost another 0.78 bp/event versus its control and worsened CVaR. | Defer until real-data error analysis identifies a specific missing risk or constraint. Avoid blind sweeps. |

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

- `artifacts/walkforward_research_ledger.csv`: compact index of all 50 recorded
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
- `docs/alpha_decay_calibration.md` and `artifacts/alpha_decay_mechanics*`:
  predeclared chronology, automatic ridge/uncertainty method, exact leakage
  perturbation test, recovery metrics, coefficients, and four-panel visual.
- `docs/automatic_risk_profile_calibration.md` and
  `artifacts/automatic_risk_profile_mechanics*`: investment-first selection
  contract, chronology test, failed-policy exclusion, profile coefficients,
  out-of-sample mechanics, and four-panel visual.
- `experiments/historical_policy_panel.py` and
  `artifacts/historical_policy_smoke*`: complete historical candidate-panel
  generation, event hard/behavior audits, chronological fallback selection,
  selected optimizer replay, source hashes, and two reviewable PNGs.
- `docs/systematic_first_risk.md` and
  `artifacts/systematic_first_risk_spent*`: predeclared factor-versus-specific
  risk ablation, automatic profile coefficient, hard sealed-cohort guard,
  full risk decomposition, failed economic gates, and six-panel visual.

## Production decision rule

Do not describe a policy as profitable until a chronologically untouched real
holdout passes every P&L, swing, behavior, liquidity, and hard-execution gate.
Risk improvement alone is not enough: the desk makes money only when saved
impact and reduced downside exceed the holding alpha forfeited by waiting.
