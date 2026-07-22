# Trade-planner research handoff

This document explains what was built, what the experiments established, what was
rejected, and how to continue without accidentally promoting synthetic evidence. The
complete 51-row ledger is in [the experiment summary table](experiment_summary_table.md).

## 1. Goal and decision standard

The goal is an **optimizer-derived** multi-day rebalance schedule that jointly delivers:

1. balanced country, sector, industry, and other factor exposure early, while the
   optimizer still has many offsets available;
2. early execution for capacity-constrained or otherwise urgent names, while small
   flexible orders may wait; and
3. increasing volume toward the announcement/rebalance event, with lower realized
   P&L swing.

No start date, daily percentage, or staircase volume curve should be hard-coded. The
optimizer must infer the schedule from point-in-time alpha, liquidity/capacity, impact,
factor and specific risk, forecast uncertainty, and hard completion. A policy is not
successful merely because it lowers volatility: saved cost and downside must exceed the
holding alpha forfeited by waiting.

The production promotion standard is therefore an untouched **real** chronological
holdout that preserves positive realized net P&L and passes every swing, behavior,
liquidity, and hard-execution gate. That evidence does not exist in this repository yet.

## 2. Current decision in one page

### Keep as reusable infrastructure

- The optimizer mechanism for accumulated-inventory risk, factor balance, physical
  capacity, hard completion, and convex impact. It creates all three requested schedule
  behaviors on deterministic fixtures without prescribing daily trades.
- Per-name numerical scaling and independent raw-share cap, direction, and completion
  certificates. These repaired false infeasibility without relaxing economics or limits.
- Leakage-safe conditional alpha-decay calibration using only outcomes available before
  each information cutoff.
- The monotone seven-policy ladder and chronological High/Medium/Low selector, including
  hard/behavior exclusion, confidence-adjusted P&L eligibility, realized-downside
  budgets, and a no-profitable-policy status.
- The strict historical bundle loader, source hashes, candidate policy panel, selected
  optimizer replay, realized scoring, gates, and review visuals.
- The contextual selector implementation as reproducible research infrastructure only.
  It is useful for a future, separately specified risk-aware experiment, but it is not
  called by the historical default.

### Do not promote

- No synthetic challenger is a production investment policy. The best risk-scaled and
  minimax liquidity models reduced volatility, loss-CVaR, drawdown, impact, and early
  factor imbalance, but failed P&L on untouched synthetic holdouts.
- The latest P&L-only contextual selector is discarded. It improved controlled
  Medium/Low P&L by `0.73 bp/event` and adapted Low aggressiveness by `0.54`, but raised
  volatility by `0.63 bp` for Medium and `1.35 bp` for Low.
- The unconditional chronological selector and fixed fallback map remain the default.
  No contextual path has been integrated into `historical_policy_panel.py`.
- Do not reopen spent synthetic holdouts or tune thresholds against them. More penalty
  tuning is lower value than obtaining the frozen real point-in-time bundle.

## 3. Implemented model flow

```text
information cutoff
    -> point-in-time orders, GICS/country factors, alpha, TCA, covariance, liquidity
    -> optional leakage-safe alpha calibration from prior available events
    -> solve the complete seven-policy optimizer panel
    -> cap/direction/completion and behavior audits
    -> chronological High/Medium/Low policy selection from prior available outcomes
    -> re-solve the selected complete policy vector
    -> attach realized returns, ADV, and costs only for out-of-sample scoring
    -> P&L, volatility, loss-CVaR, drawdown, factor, ramp, timing, and hard gates
```

The complete policy is selected as one indivisible vector. The selector never combines
the best individual coefficient from different trials, and the optimizer still chooses
every date-by-name trade.

## 4. Experimental High/Medium/Low policy map

These are interpretable replay defaults, not approved production coefficients:

| Control | High | Medium | Low | Role in the optimizer |
|---|---:|---:|---:|---|
| Feasible P&L-risk frontier fraction | 5% | 50% | 100% | Risk range available to the selected plan |
| Liquidity forecast quantile | P10 | P25 | P50 | Protection against weaker realized ADV |
| Forecast event-liquidity shape used | 95% | 50% | 0% | Permission to delay flow when liquidity should improve |
| Optional-alpha confidence | 97.5% | 75% | 50% | Forecast hurdle on flexible execution capacity |
| Minimax factor-stress fraction | 95% | 50% | 0% | Correlation-break protection for country/sector/industry |
| Inventory-risk price | automatic per event | automatic per event | automatic per event | Selected from the flat-information frontier and held invariant while liquidity changes |

`build_monotone_policy_ladder(...)` interpolates this map into the frozen aggressiveness
grid `0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00`. The unconditional selector learns which
whole policy survived earlier realized economics. The contextual experiment additionally
used five point-in-time event features—capacity pressure, alpha/cost ratio, liquidity
ramp, factor concentration, and forecast-uncertainty ratio—but that P&L-only ranking is
not a retained deployment rule.

## 5. Experiment families and conclusions

The [summary table](experiment_summary_table.md) is the authoritative row-level record.
The foundational shape/economic experiments below predate the compact 51-row ledger;
their dedicated documents and artifacts remain the authority for their exact outputs.

### A. Foundation: schedule behavior and economic units

The deterministic shape screen tested 29 objective combinations on urgency/ramp and
factor-balance fixtures. The retained mechanism used accumulated inventory risk,
factor-risk multiplier `1`, physical ADV caps, impact of `20 bp at 10% ADV`, and no
pre-event residual-pressure term. It produced Spearman volume correlation `1.00`, a
`4.30x` late/early ratio, urgent flow on day 1, small flow on day 8, and `96.4%` better
early factor balance than the no-factor ablation. This validates mechanism, not economic
calibration. See [daily volume behavior](daily_volume_behavior.md).

The dollar economic frontier then separated expected holding alpha, impact/spread,
accumulated covariance risk, and scenario tail risk. High stayed near minimum risk,
Medium used the hybrid excess-tail frontier, and Low used the conditional tail second
moment. Tail-preserving reduction kept 96 scenarios for a `3.93x` speedup while
preserving mechanics; the 64-scenario variant was rejected because small orders moved to
day 1. See [economic calibration](rebalance_economic_calibration.md),
[scenario reduction](scenario_reduction.md), and [tail-path risk](tail_path_risk.md).

### B. Alpha confidence and uncertainty, ledger rows 1–15

Confidence haircuts at several levels often reduced volatility, but increased realized
loss-CVaR, worsened factor balance, or moved small orders earlier. Reliability-scaled
risk budgets either raised volatility or selected the same plan. An uncertainty-aware
frontier tie was inert because the existing one-basis-point materiality rule dominated
it. Result: reject all tested investment rules; keep the simpler materiality tie.

### C. Replanning and recourse, rows 16–23

Always re-planning lowered P&L, raised event volatility, and weakened the ramp.
Commitment-aware recourse at a 2 bp hurdle improved several downside and behavior
measures, but volatility fell only `0.0207 bp`, short of the predeclared `0.05 bp`
materiality floor. Defensive and proximal variants raised volatility or drawdown and
sometimes failed small-order timing. Result: retain their audit plumbing, but keep the
static point-in-time plan as default.

### D. Forecast-error path risk, rows 24–25

The development run passed all 13 gates and authorized one holdout opening. The untouched
holdout preserved most secondary metrics but raised volatility by `0.0224 bp` instead of
reducing it by at least `0.05 bp`. Result: discard, and treat the holdout as spent.

### E. Liquidity, factor stress, and profit preservation, rows 26–45

This family most clearly reproduced the desired schedule shape. Event-liquidity
forecasts, capacity-slack option value, minimax factor stress, and risk-scaled liquidity
repeatedly kept urgent names early, delayed small names, strengthened the ramp, and
reduced volatility, loss-CVaR, drawdown, factor imbalance, and impact. The failure was
economic: untouched holdouts lost holding alpha.

Key checkpoints:

- Minimax liquidity/factor stress passed all 16 development gates, then reduced holdout
  volatility `14.03 bp` and loss-CVaR `23.32 bp` but lost `1.78 bp/event`.
- Risk-scaled liquidity passed a new validation cohort, then reduced holdout volatility
  `8.08 bp` and loss-CVaR `14.75 bp` but lost `3.24 bp/event` and had one `0.087-share`
  cap excess.
- Restoring raw alpha passed on spent data but lost `2.89 bp/event` on fresh validation,
  so its sealed follow-on holdout stayed unopened.
- Exact forecast-profit floors exposed solver/scaling failures; a linear alternative
  violated its own economics/hard tolerances. Complete-plan selection based on forecast
  P&L chose the wrong events. Baseline-relative tracking risk reduced several risk
  measures but still lost `2.54 bp/event` and increased loss-CVaR.

Result: the desired behavior is feasible, but the point-in-time alpha model—not another
synthetic penalty—is the main profitability bottleneck.

### F. Numerical scaling and point-in-time alpha, rows 46–47

Per-name constraint scaling preserved dollar objectives and raw-share public schedules.
On spent events it made the exact quadratic profit floor solve under explicit CLARABEL,
with floor, cap, direction, completion, and urgency certificates passing. The underlying
investment policy still failed volatility, factor, and small-order gates, so only the
formulation/certification was kept. See [numerical scaling](numerical_scaling.md).

The leakage-safe alpha ridge model selected regularization only from earlier available
events. In controlled mechanics it reduced equal-event directional RMSE by `40.8%`,
raised sign accuracy by `10.6 pp`, achieved `94.4%` interval coverage, was invariant to
current/future outcome changes, and handled unseen GICS labels. This is kept for real
development replay, not claimed as profit evidence. See
[alpha-decay calibration](alpha_decay_calibration.md).

### G. Automatic policy selection and historical bridge, rows 48–49

The unconditional chronological selector excluded attractive policies with hard or
behavior failures, preserved `High <= Medium <= Low`, and improved controlled Low P&L by
`0.33 bp/event` while reducing its volatility by `0.49 bp`. The two-event historical
smoke then generated all 14 candidate rows, preserved hashes/alpha audits, re-solved the
selected policy IDs exactly, and correctly failed economic promotion gates. Result: keep
the pipeline; the smoke is not profit evidence.

### H. Systematic-first and contextual selection, rows 50–51

Pricing only 50% of specific variance improved early factor balance, but lost another
`0.78 bp/event` versus the full-specific control, worsened CVaR/ramp, and missed a cap
certificate. It is discarded.

The new contextual ridge selector is exactly chronological: it scales five features from
training events only, selects a ridge multiplier by nested leave-one-event-out error, and
uses the existing operational exclusions, confidence bounds, risk budgets, materiality
tie, and monotone profile order. On 36 scored controlled events it improved Medium/Low
P&L by `0.73 bp/event`; High was unchanged. The volatility gate failed, so the API,
experiment, tests, and artifacts are kept only to make the negative result reproducible.
See [contextual policy calibration](contextual_risk_profile_calibration.md).

## 6. Evidence hierarchy

| Evidence level | What exists | What it authorizes |
|---|---|---|
| Deterministic shape fixtures | Urgency, factor balance, rising-volume mechanism | Keep model mechanics and behavior gates |
| Controlled estimator/selector populations | Alpha calibration, automatic profiles, contextual selector | Keep leakage-safe plumbing when predeclared gates pass |
| Spent synthetic development | Many risk/liquidity/recourse ablations | Reject weak ideas or authorize exactly one frozen next cohort |
| Fresh synthetic validation | Risk-scaled liquidity passed; raw-alpha variant failed | Open a synthetic holdout only for the passing frozen candidate |
| Untouched synthetic holdouts | Forecast-error, minimax factor, and risk-scaled liquidity all ultimately failed promotion | Reject those candidates; never tune against the spent holdouts |
| Historical two-event smoke | Schema, hashes, policy panel, selected replay, artifacts | Plumbing only |
| Frozen real development | Not supplied | Required before any real holdout may be opened |
| Untouched real holdout | Not supplied | Only level that may support a profitability/production claim |

## 7. Code and artifact map

### Main implementation entry points

- `trade_planner/participation.py`: physical capacity and adaptive pre-event caps.
- `trade_planner/risk.py`, `trade_planner/calibration.py`, and
  `trade_planner/rebalance.py`: accumulated-inventory and economic frontier mechanics.
- `trade_planner/alpha_decay.py`: chronological conditional alpha fitting and uncertainty.
- `trade_planner/policy_calibration.py`: policy ladder, unconditional selector, and the
  new opt-in contextual selector.
- `experiments/historical_replay.py`: strict bundle replay and paired realized scoring.
- `experiments/historical_policy_panel.py`: seven-policy panel, chronological selection,
  selected-policy re-solve, and promotion evidence.

### Changes in the final contextual experiment

- `trade_planner/policy_calibration.py` adds
  `DEFAULT_CONTEXTUAL_RIDGE_MULTIPLIERS` and
  `calibrate_contextual_risk_profiles_walk_forward(...)`.
- `trade_planner/__init__.py` exports both public names.
- `experiments/contextual_risk_profile_walkforward.py` generates the fixed 48-event,
  seven-policy controlled comparison and six-panel review image.
- `tests/test_contextual_risk_profile_walkforward.py` covers the experiment and gates.
- `docs/contextual_risk_profile_calibration.md` contains the untouched predeclaration,
  exact result, and discard decision.
- `artifacts/contextual_risk_profile_mechanics*` contains events/features, policies,
  trials, both selectors, evaluations, summaries, gates, comparison, and the PNG.

### Important evidence indexes

- `artifacts/walkforward_research_ledger.csv`: all 51 compact experiment records.
- [Research priorities](research_priorities.md): ranked next work and current policy map.
- [Historical replay bundle](historical_replay_bundle.md): exact six-file real-data
  schema, chronology rules, commands, hashes, and promotion contract.
- [Point-in-time walk-forward record](point_in_time_walkforward.md): detailed sequence of
  the earlier alpha, recourse, downside, liquidity, and holdout experiments.

## 8. Reproduction commands

Run the full repository suite:

```bash
env PYTHONPATH=. \
  PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache \
  MPLCONFIGDIR=/private/tmp/trade_planner_mpl \
  XDG_CACHE_HOME=/private/tmp/trade_planner_cache \
  ./.conda-env/bin/python -m pytest -q
```

Reproduce the latest contextual negative result:

```bash
env PYTHONPATH=. \
  PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache \
  MPLCONFIGDIR=/private/tmp/trade_planner_mpl \
  XDG_CACHE_HOME=/private/tmp/trade_planner_cache \
  ./.conda-env/bin/python experiments/contextual_risk_profile_walkforward.py \
  --full-suite-verified \
  --output-prefix artifacts/contextual_risk_profile_mechanics
```

Run the decision-quality path once a frozen real development bundle exists:

```bash
env PYTHONPATH=. ./.conda-env/bin/python experiments/historical_policy_panel.py \
  --bundle /path/to/frozen/development_bundle \
  --role development \
  --risk-aversion medium \
  --solver CLARABEL \
  --output-prefix artifacts/historical_policy_development
```

Do not run a holdout bundle until the frozen development output passes all existing
profitability, swing, behavior, liquidity, and feasibility gates without changing the
ladder or thresholds.

## 9. Takeover checklist

1. Review the 51 rows in [the summary table](experiment_summary_table.md), especially
   the promotion row followed by its later validation/holdout rejection.
2. Confirm `artifacts/contextual_risk_profile_mechanics_gates.csv` has exactly one failed
   research gate: per-profile volatility within `0.50 bp`; the full-suite gate should
   pass.
3. Confirm no default path calls `calibrate_contextual_risk_profiles_walk_forward(...)`.
4. Review the strict real-bundle schema and choose one return/fill convention consistently
   for alpha, realized P&L, spread/fees, FX, financing, and borrow.
5. Freeze development and holdout directories before the first comparison. Preserve
   prediction vintages, availability timestamps, classifications, and source hashes.
6. Run the existing seven-policy development command without changing the policy ladder,
   warm-up, confidence levels, one-basis-point materiality tie, or 16 promotion gates.
7. If development fails, diagnose the economic error and stop. If it passes, open one
   sealed real holdout exactly once.
8. Treat the contextual P&L-only model as a recorded negative result. If revisited, write
   a new predeclaration for conditional downside/certainty-equivalent selection rather
   than editing this experiment's gates or result.

## 10. Recommended next action

The highest-value next step is not another synthetic coefficient search. Build the first
frozen real development bundle with point-in-time orders/prediction vintages, realized
close or VWAP returns, realized ADV, impact and linear costs, FX/financing/borrow, and
country/GICS classifications. Run the existing historical policy panel exactly as
documented. That test will reveal whether alpha calibration plus optimizer-derived risk,
liquidity, and factor controls preserves actual desk P&L while reducing swing.

## 11. Handoff verification

The final pre-push audit reconciled all 51 Markdown table rows to the CSV ledger,
resolved every local link in both handoff documents, confirmed the contextual selector
is absent from default replay calls, and verified that the contextual artifact has one
expected failed research gate: `profile_volatility_within_050bp`. The repository suite
completed with **117 tests and 2 subtests passing**.
