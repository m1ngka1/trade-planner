# Complete experiment summary

This is the review index for every experiment recorded in
[`artifacts/walkforward_research_ledger.csv`](../artifacts/walkforward_research_ledger.csv).
The rows remain in research order so the sequence from hypothesis to validation or
rejection is auditable. See [the detailed handoff](experiment_details.md) for the
model architecture, decisions, code entry points, and takeover checklist.

## How to read the table

The ledger contains **51 experiments**: **4 keep**, **3 keep-for-holdout**, **1 keep-for-fresh-validation**, **34 discard**, **6 inconclusive**, and **3 crash** rows.

- Deltas are candidate minus baseline. Positive P&L is better; negative volatility,
  loss-CVaR, and early-factor deltas are better; positive ramp is a stronger
  late-volume ramp.
- P&L is in thousands of dollars; the other units are shown in the headers.
- A blank metric means the run tested mechanics or failed before comparable economic
  output existed.
- `keep` means reusable mechanics/infrastructure, not proven production alpha.
  Promotion statuses authorized the next frozen cohort only; later ledger rows show
  whether that validation survived.
- No experiment has passed an untouched real historical holdout. The production/default
  selector therefore remains unchanged.

## All experiments

| # | Idea | Risk/model | Policy or variant | Confidence | P&L Δ ($000) | Vol Δ (bp) | Loss-CVaR Δ (bp) | Early factor Δ (pp) | Ramp Δ | Ledger decision | Result / reason | Artifact prefix |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | alpha_confidence | variance | reselect_frontier | 0.60 | 176.7746 | -0.7368 | 1.0903 | 0.2316 | 0.0912 | `discard` | higher loss CVaR and a small-order timing failure | `alpha_confidence_dev_60` |
| 2 | alpha_confidence | variance | reselect_frontier | 0.65 | -794.9053 | 2.3399 | 23.8839 | 0.6562 | 0.1216 | `discard` | lower P&L, higher volatility and CVaR, and a small-order timing failure | `alpha_confidence_dev_65` |
| 3 | alpha_confidence | variance | reselect_frontier | 0.70 | 1204.4593 | 3.3035 | 3.4787 | 0.8543 | 0.1739 | `discard` | higher volatility and CVaR and a small-order timing failure | `alpha_confidence_dev_70` |
| 4 | alpha_confidence | variance | reselect_frontier | 0.75 | 623.8169 | 0.9159 | 4.8488 | 1.0373 | 0.2548 | `discard` | higher volatility and CVaR and early factor balance above tolerance | `alpha_confidence_walkforward_variance` |
| 5 | alpha_confidence | variance | fixed_risk | 0.60 | 210.1581 | -0.9205 | 1.0903 | 0.3392 | 0.1146 | `discard` | higher loss CVaR | `alpha_confidence_fixed_dev_60` |
| 6 | alpha_confidence | variance | fixed_risk | 0.65 | 296.3596 | -1.2773 | 2.1941 | 0.5868 | 0.1738 | `discard` | higher loss CVaR | `alpha_confidence_fixed_dev_65` |
| 7 | alpha_confidence | variance | fixed_risk | 0.70 | 308.3950 | -1.6615 | 3.4787 | 0.8662 | 0.2375 | `discard` | higher loss CVaR | `alpha_confidence_fixed_dev_70` |
| 8 | alpha_confidence | variance | fixed_risk | 0.75 | 260.1385 | -1.9559 | 4.8488 | 1.1567 | 0.3067 | `discard` | higher loss CVaR and early factor balance above tolerance | `alpha_confidence_fixed_dev_75` |
| 9 | alpha_confidence | hybrid_downside | fixed_risk | 0.60 | 102.6549 | -0.8954 | 1.0903 | 0.5185 | 0.1270 | `discard` | higher loss CVaR | `alpha_confidence_hybrid_dev_60` |
| 10 | alpha_confidence | hybrid_downside | fixed_risk | 0.70 | 122.6242 | -1.4991 | 3.4787 | 1.1771 | 0.2703 | `discard` | higher loss CVaR and early factor balance above tolerance | `alpha_confidence_hybrid_dev_70` |
| 11 | reliability_scaled_budget | variance | frontier_budget | — | -170.2530 | 0.2142 | 0.0000 | -0.0453 | 0.0105 | `discard` | did not reduce realized volatility | `uncertainty_budget_dev` |
| 12 | reliability_scaled_budget | hybrid_downside | frontier_budget | — | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `discard` | contracted budget selected the same frontier candidate | `uncertainty_budget_hybrid_dev` |
| 13 | frontier_uncertainty_tie | variance | paired_uncertainty | 0.60 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `inconclusive` | existing one-basis-point materiality threshold dominated forecast uncertainty in every event | `frontier_uncertainty_dev` |
| 14 | frontier_uncertainty_tie | variance | paired_uncertainty | 0.75 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `inconclusive` | existing one-basis-point materiality threshold dominated forecast uncertainty in every event | `frontier_uncertainty_dev` |
| 15 | frontier_uncertainty_tie | variance | paired_uncertainty | 0.90 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `inconclusive` | existing one-basis-point materiality threshold dominated forecast uncertainty in every event | `frontier_uncertainty_dev` |
| 16 | receding_horizon | variance | always_replan | — | -1114.2113 | 0.4636 | -2.5083 | -0.0187 | -0.1489 | `discard` | lower P&L, higher volatility and within-event drawdown, and a materially weaker late-volume ramp | `rolling_horizon_dev` |
| 17 | commitment_aware_recourse | variance | materiality_0.5bp | — | -448.9961 | 0.1360 | -2.1674 | -0.4365 | -0.0394 | `discard` | higher volatility and one small-order timing failure | `commitment_aware_dev_050` |
| 18 | commitment_aware_recourse | variance | materiality_1bp | — | -448.7884 | 0.1608 | -2.2250 | -0.4375 | -0.0316 | `discard` | higher event P&L volatility | `commitment_aware_dev_100` |
| 19 | commitment_aware_recourse | variance | materiality_2bp | — | -242.4933 | -0.0207 | -4.6103 | -0.2021 | 0.0044 | `discard` | volatility reduction was only 0.0207 bp, below the predeclared 0.05 bp materiality floor | `commitment_aware_dev_200` |
| 20 | defensive_recourse | variance | risk_only_2bp | — | -242.4933 | -0.0207 | -4.6103 | -0.2021 | 0.0044 | `discard` | selected the same schedule as the 2 bp materiality policy and remained below the volatility materiality floor | `defensive_rolling_dev_200` |
| 21 | defensive_recourse | variance | risk_only_4bp | — | -579.3219 | 0.4173 | -4.5351 | -0.0220 | 0.0203 | `discard` | higher volatility and higher within-event drawdown | `defensive_rolling_dev_400` |
| 22 | proximal_trade_recourse | variance | automatic_95pct_single | — | -571.1138 | 0.2694 | -5.5809 | -0.5541 | -0.0135 | `discard` | higher volatility and within-event drawdown, plus one small-order timing failure | `proximal_rolling_dev_medium` |
| 23 | proximal_trade_recourse | variance | automatic_95pct_simultaneous | — | -532.3984 | 0.1910 | -5.6890 | -0.3855 | -0.0035 | `discard` | higher volatility and within-event drawdown, plus one small-order timing failure | `proximal_rolling_dev_medium_simultaneous` |
| 24 | forecast_error_path_risk | variance | predictive_variance_development | — | 38.7057 | -0.1607 | -0.2140 | 0.0324 | 0.0124 | `keep_for_holdout` | passed all 13 development gates without tuning | `forecast_error_risk_dev` |
| 25 | forecast_error_path_risk | variance | predictive_variance_holdout | — | -33.6722 | 0.0224 | -0.0377 | 0.0259 | 0.0104 | `discard` | untouched holdout volatility rose 0.0224 bp instead of falling by the required 0.05 bp | `forecast_error_risk_holdout` |
| 26 | event_liquidity_forecast | variance | lower_quantile_adv | 0.25 | -1332.5369 | -9.1825 | -15.9349 | 0.7124 | 1.5952 | `discard` | lower swing and impact, but P&L fell 2.2792 bp per event and one small order started seven days earlier | `liquidity_forecast_dev` |
| 27 | event_liquidity_forecast | variance | median_adv_sensitivity | 0.50 | -2507.7736 | -9.9488 | -13.6479 | -0.0680 | 2.1465 | `discard` | median liquidity strengthened the ramp but worsened the P&L sacrifice and retained a seven-day small-order timing failure | `liquidity_forecast_dev_q50` |
| 28 | event_liquidity_forecast | variance | baseline_locked_risk_price | 0.25 | -580.1960 | -10.0694 | -15.9349 | 0.2048 | 1.5962 | `discard` | passed 15 of 16 gates and preserved P&L within 0.992 bp per event, but three events started a small order one day earlier | `liquidity_forecast_locked_dev` |
| 29 | liquidity_option_value | variance | capacity_slack_confidence_smoke | 0.75 | 567.7313 | 8.0049 | -0.1661 | 0.2861 | 2.4110 | `inconclusive` | two-event mechanics removed the event-26 small-order regression and preserved urgency and hard constraints; full development required | `liquidity_option_value_smoke` |
| 30 | liquidity_option_value | variance | capacity_slack_confidence | 0.75 | -871.7054 | -12.3004 | -25.1377 | 1.3491 | 2.1007 | `discard` | fixed small-order timing and strongly reduced swing, but P&L fell 1.491 bp per event and early factor imbalance exceeded tolerance | `liquidity_option_value_dev` |
| 31 | liquidity_factor_balance | variance | equal_factor_stress_smoke | 0.50 | 332.6860 | -3.0822 | -5.5937 | 1.0375 | 1.4131 | `discard` | events 28-29 passed every other screen but missed the early-factor tolerance by 0.038 percentage point; no stress tuning or full run | `liquidity_factor_stress_smoke` |
| 32 | liquidity_factor_balance | variance | minimax_factor_stress_smoke | 0.50 | 674.2498 | -4.4834 | -10.0899 | -1.0108 | 1.4886 | `inconclusive` | events 28-29 passed every mechanical and economic screen and authorized full development | `liquidity_minimax_factor_smoke` |
| 33 | liquidity_factor_balance | variance | minimax_factor_stress_development | 0.50 | -480.4032 | -15.7405 | -26.9246 | -1.5342 | 1.9221 | `keep_for_holdout` | passed all 16 development gates without tuning | `liquidity_minimax_factor_dev` |
| 34 | liquidity_factor_balance | variance | minimax_factor_stress_holdout | 0.50 | -1040.9155 | -14.0311 | -23.3161 | -1.7153 | 1.6635 | `discard` | untouched holdout lost 1.780 bp per event and event 45 had sub-share OSQP cap and direction audit misses | `liquidity_minimax_factor_holdout` |
| 35 | forecast_profit_floor | variance | clarabel_mechanical_screen | 1bp | — | — | — | — | — | `crash` | CLARABEL declared every forecast-liquidity candidate infeasible before the floor was attached; events 49-72 abandoned without a solver retry | `profit_floor_smoke_failure` |
| 36 | forecast_profit_floor | variance | osqp_net_pnl_floor_screen | 1bp | — | — | — | — | — | `crash` | OSQP returned false infeasibility although the existing event-25 schedule cleared the exact forecast net-P&L floor by $51.1k | `profit_floor_osqp_mechanics_failure` |
| 37 | forecast_profit_floor | variance | osqp_holding_alpha_floor_screen | 1bp | 688.1287 | 7.2252 | -1.9531 | -1.6501 | 2.1148 | `discard` | event 26 violated the alpha floor by $21.5k, event 25 missed sub-share cap and direction tolerances, and volatility increased | `alpha_floor_osqp_mechanics` |
| 38 | optimizer_plan_selection | variance | forecast_profit_and_hard_gate | 1bp | -855.6233 | 0.0630 | 0.0000 | -0.4834 | 0.6481 | `discard` | selected three candidate plans and nine fallbacks, but lost 1.464 bp per event and erased the volatility improvement | `plan_selection_gate_dev` |
| 39 | risk_scaled_liquidity | variance | risk_budget_log_shape | 0.50 | 67.1136 | -10.7058 | -22.6169 | -1.3844 | 0.7656 | `inconclusive` | passed all 16 gates on spent development data; fixed fresh-cohort validation required | `risk_scaled_liquidity_dev` |
| 40 | risk_scaled_liquidity | variance | fresh_validation_events_73_84 | 0.50 | -167.0085 | -5.4939 | -16.0792 | -1.3229 | 0.7001 | `keep_for_holdout` | fresh validation passed all 16 gates with the policy frozen | `risk_scaled_liquidity_fresh_dev` |
| 41 | risk_scaled_liquidity | variance | untouched_holdout_events_85_96 | 0.50 | -1895.6102 | -8.0812 | -14.7540 | -1.5060 | 0.7334 | `discard` | untouched holdout lost 3.242 bp per event and event 93 had a 0.087-share cap excess | `risk_scaled_liquidity_fresh_holdout` |
| 42 | raw_alpha_opportunity | variance | risk_scaled_minimax_raw_alpha_spent | 0.50 | 283.5929 | -8.2624 | -14.8165 | -2.1286 | 0.5191 | `keep_for_fresh_validation` | passed all 16 mechanics and economic gates on spent events; fixed fresh-cohort validation required | `risk_scaled_raw_alpha_spent` |
| 43 | raw_alpha_opportunity | variance | fresh_validation_events_97_108 | 0.50 | -1689.6208 | -2.8105 | 1.8759 | -2.1237 | 0.4853 | `discard` | fresh validation lost 2.890 bp per event and increased loss-CVaR; events 109-120 remain sealed | `raw_alpha_opportunity_fresh_dev` |
| 44 | baseline_relative_regret | variance | cvar95_spent_screen | 0.50 | — | — | — | — | — | `crash` | OSQP reached user_limit before a comparable event set on both the fixed run and one numerical-normalization retry; no fresh cohort allocated | `baseline_regret_spent_failure` |
| 45 | baseline_relative_regret | variance | second_moment_spent_screen | 0.50 | -1481.9005 | -2.9766 | 0.5319 | -1.7142 | 0.4676 | `discard` | quadratic tracking risk solved and improved the raw-alpha candidate but still lost 2.535 bp per event and increased loss-CVaR | `baseline_tracking_risk_spent` |
| 46 | numerical_scaling | variance | per_name_strict_certificate | 1bp_floor | 479.5267 | 6.6578 | -0.2135 | 1.5226 | 1.6455 | `keep` | exact quadratic profit floor solved on spent events 25-26 with explicit CLARABEL; both floors and hard certificates passed, but the investment policy still failed volatility, small-order, and factor gates | `numerical_scaling_mechanics` |
| 47 | conditional_alpha_decay | directional_return | nested_chronological_ridge | automatic_cv | — | — | — | — | — | `keep` | controlled mechanics improved equal-event RMSE 40.8% and sign accuracy 10.6 pp with 94.4% interval coverage; current/future outcomes were exactly inaccessible and unseen GICS labels predicted; real profitability remains untested | `alpha_decay_mechanics` |
| 48 | automatic_risk_profile | realized_downside | chronological_investment_frontier | high_medium_low | — | — | — | — | — | `keep` | controlled mechanics excluded hard/behavior failures, preserved monotone profiles and positive P&L, and improved Low by 0.33 bp/event while cutting volatility 0.49 bp; real development and holdout evidence remain required | `automatic_risk_profile_mechanics` |
| 49 | historical_policy_panel | realized_downside | seven_policy_end_to_end_smoke | medium_fallback | -0.0008 | 0.0000 | 0.3472 | 0.0000 | 0.0000 | `keep` | two-event smoke generated all 14 policy trials, preserved alpha/source audits, matched selected policy IDs to the final optimizer replay, and correctly failed economic promotion gates; plumbing only | `historical_policy_smoke` |
| 50 | systematic_first_risk | variance | medium_specific_fraction_050 | automatic | -2147.3422 | -1.7098 | 3.7899 | -2.7106 | 0.2102 | `discard` | specific-risk scaling improved early factor balance and kept volatility below the flat baseline, but lost another 0.78 bp/event versus the full-specific control, worsened CVaR and ramp strength, and missed the hard cap certificate by 0.261 shares | `systematic_first_risk_spent` |
| 51 | contextual_policy_selection | realized_downside | nested_ridge_event_features | high_medium_low | — | — | — | — | — | `discard` | controlled Medium/Low P&L improved 0.73 bp/event and Low aggressiveness adapted by 0.54 with exact leakage invariance, but volatility rose 0.63 bp for Medium and 1.35 bp for Low; P&L-only contextual selection is not integrated | `contextual_risk_profile_mechanics` |

## Bottom line

The research established that the optimizer can create the desired behavior without a
hard-written schedule: balance factor exposure early, start capacity-constrained names
early, delay small flexible orders, and increase flow toward the event. The strongest
risk-focused challengers repeatedly reduced swing, tail loss, factor imbalance, and
impact, but their synthetic holdouts sacrificed too much holding alpha. The useful kept
outputs are the point-in-time alpha calibrator, automatic High/Medium/Low selector,
historical policy-panel/replay plumbing, and numerical scaling/certification. The latest
P&L-only contextual selector is reproducible but discarded because it increased Medium
and Low volatility.
