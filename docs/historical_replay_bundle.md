# Historical rebalance replay bundle

This is the production-facing bridge between the recorded synthetic research
and a decision-quality historical replay. It loads a frozen tabular bundle,
rejects point-in-time leakage, hashes every source file, and compares the
existing flat-ADV optimizer baseline with the frozen risk-scaled/minimax
challenger. A successful smoke run proves only that the pipeline is wired; it
does not prove that the challenger is profitable.

## Cohort isolation

Keep development, holdout, and descriptive backtest data in separate
directories. Every row in `events.csv` must use the same `cohort_role`, and the
runner requires an explicit matching `--role`. The loader checks this manifest
before opening detailed planning or realized files, so a holdout cannot be
opened accidentally by a development command.

All timestamps are parsed as UTC. For each event:

- `orders.csv`, `planning.csv`, `factor_covariance.csv`, and optional
  `scenarios.csv` must have `available_at <= information_cutoff`;
- `as_of` must not be later than the first planner date;
- each realized row must become available after its market date and no later
  than the event's `realized_available_at`; and
- `realized_available_at` must be after the final planner date.

The development and holdout directories should be frozen before any model
comparison. Never move failed holdout events back into development.

## Required files

| File | Grain | Required fields | Investment meaning |
|---|---|---|---|
| `events.csv` | one row per event | `event_id`, `cohort_role`, `as_of`, `information_cutoff`, `realized_available_at` | Defines chronological vintages and data availability. |
| `orders.csv` | event and symbol | `target_shares`, `country`, `sector`, `industry`, `urgency`, `available_at`; optional `rebalance_type`, `prediction_confidence`, `crowding` | Supplies signed parent orders, early-balance dimensions, and point-in-time alpha-conditioning fields. GICS codes or labels can be used for sector and industry. |
| `planning.csv` | event, date, and symbol | `price`, `adv_shares`, `forecast_adv_p10_shares`, `forecast_adv_p25_shares`, `forecast_adv_p50_shares`, `is_open`, `base_participation`, `event_days`, `specific_variance`, `expected_return`, `expected_return_uncertainty`, `impact_bps_at_10pct_adv`, `linear_cost_bps`, `available_at`, plus one or more `factor:<name>` columns | Contains only information known at the planning cutoff. The flat `adv_shares` surface is the baseline; the three forecast quantiles drive the challenger. |
| `factor_covariance.csv` | event, date, factor pair | `factor_left`, `factor_right`, `covariance`, `available_at` | Full symmetric positive-semidefinite covariance matrix for every planner date. Factor names must exactly match the `factor:<name>` columns. |
| `realized.csv` | event, date, and symbol | `realized_return`, `realized_adv_shares`, `realized_impact_bps_at_10pct_adv`, `realized_linear_cost_bps`, `available_at` | Outcomes kept outside the optimizer and used only for scoring. |
| `scenarios.csv` | event, scenario, date, and symbol | `residual_return`, `scenario_weight`, `available_at` | Optional point-in-time residual-return scenarios. At least two complete scenarios are required when the file is present. |

Every event must have a complete date-symbol grid. Liquidity forecasts must be
positive and ordered `P10 <= P25 <= P50`. Prices and realized ADV must be
positive; participation, costs, variance, and forecast uncertainty must be
non-negative. Source SHA-256 values are copied into the planner context and
written with every replay output.

`linear_cost_bps` and its realized counterpart should include every cost that
is approximately linear in shares: spread, fees, taxes, FX, financing, and
borrow. `expected_return` must represent the point-in-time return from owning
the executed inventory over each day. `realized_return` must use the desk's
chosen close/VWAP convention consistently across baseline and challenger.

## Conditional holding-alpha calibration

The default replay uses `--alpha-calibration walk_forward`. For every event it
fits only earlier events satisfying
`realized_available_at <= information_cutoff`; the current and all future
outcomes are inaccessible. The first four eligible-event vintages use the raw
forecast. Later vintages automatically select a dimensionless ridge penalty by
equal-event leave-one-event-out RMSE, then replace `expected_return` and
`expected_return_uncertainty` for both baseline and challenger before either
optimizer solve. This changes economic inputs, never dates or trade fractions.

Use `--alpha-calibration none` only for an explicit raw-forecast comparison.
The output prefix adds `alpha_audit`, `alpha_predictions`, `alpha_summary`, and
`alpha_coefficients` CSVs so every training event, selected penalty, prediction,
uncertainty, and coefficient is reviewable. See
[Point-in-time conditional alpha-decay calibration](alpha_decay_calibration.md)
for the leakage contract and controlled-mechanics evidence.

A chronologically selected policy row can be passed programmatically as one
indivisible vector:

```python
from experiments.historical_replay import run_historical_experiment
from trade_planner import InvestmentPolicyCoefficients

policy = InvestmentPolicyCoefficients.from_mapping(selected_row)
outputs, metadata = run_historical_experiment(
    bundle,
    risk_aversion="medium",
    policy_coefficients=policy,
)
```

The runner audits `policy_id`, aggressiveness, frontier fraction, interpolated
liquidity quantile, shape fraction, alpha confidence, and factor stress for
every event. See
[Automatic investment-policy calibration](automatic_risk_profile_calibration.md)
for the chronological selector and no-profit fallback rule.

## Automatic High/Medium/Low controls

The replay accepts a risk label, not numerical coefficients. It applies the
predeclared mapping below and lets the optimizer determine the schedule.

| Risk aversion | Liquidity quantile | Forecast shape consumed | Optional-alpha confidence | Factor stress |
|---|---:|---:|---:|---:|
| High | P10 | 95% | 97.5% | 95% |
| Medium | P25 | 50% | 75% | 50% |
| Low | P50 | 0% | 50% | 0% |

The challenger first selects the inventory-risk price from the flat-information
frontier and holds that price invariant. Liquidity uncertainty, alpha
confidence, and minimax country/sector/industry stress then change the
optimizer's trade-off; no date or daily percentage is hard-coded.

## Run

```bash
env PYTHONPATH=. python experiments/historical_replay.py \
  --bundle /path/to/frozen/development_bundle \
  --role development \
  --risk-aversion medium \
  --solver CLARABEL \
  --alpha-calibration walk_forward \
  --output-prefix artifacts/historical_replay_development
```

The historical runner enables per-name numerical scaling and strict raw-share
certificates. CLARABEL is the explicit default because the frozen minimax
factor policy is conic; this avoids an implicit OSQP-to-CLARABEL fallback.

The prefix produces event trials, paired deltas, aggregate summaries, all 16
predeclared gates, schedules, daily realized economics, volume profiles,
factor exposures, liquidity diagnostics, automatic coefficients, alpha
calibration audits, frontier points, source hashes, and a six-panel PNG.

To generate the full automatic-coefficient candidate panel and replay the
chronologically selected profile in one command, use:

```bash
env PYTHONPATH=. python experiments/historical_policy_panel.py \
  --bundle /path/to/frozen/development_bundle \
  --role development \
  --risk-aversion medium \
  --solver CLARABEL \
  --output-prefix artifacts/historical_policy_development
```

This is the decision-quality development command. It solves the frozen seven-
policy ladder, keeps candidate scoring separate from chronological selection,
and then re-solves the selected event-policy mapping. Never change the ladder,
warm-up, confidence levels, materiality rule, or operational gates after
viewing development results. Open the holdout directory only if the selected
development replay passes all existing gates.

Development results may authorize opening one sealed holdout only when all
gates pass. A historical policy can be called profitable only after an
untouched real holdout preserves positive net P&L while satisfying volatility,
loss-CVaR, drawdown, factor-balance, urgency, small-order delay, volume-ramp,
liquidity, completion, sign, and participation gates. A `backtest` bundle is
always labeled `descriptive_only` and cannot authorize promotion.
