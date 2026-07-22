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
| `orders.csv` | event and symbol | `target_shares`, `country`, `sector`, `industry`, `urgency`, `available_at` | Supplies signed parent orders and early-balance dimensions. GICS codes or labels can be used for sector and industry. |
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
  --solver OSQP \
  --output-prefix artifacts/historical_replay_development
```

The prefix produces event trials, paired deltas, aggregate summaries, all 16
predeclared gates, schedules, daily realized economics, volume profiles,
factor exposures, liquidity diagnostics, automatic coefficients, frontier
points, source hashes, and a six-panel PNG.

Development results may authorize opening one sealed holdout only when all
gates pass. A historical policy can be called profitable only after an
untouched real holdout preserves positive net P&L while satisfying volatility,
loss-CVaR, drawdown, factor-balance, urgency, small-order delay, volume-ramp,
liquidity, completion, sign, and participation gates. A `backtest` bundle is
always labeled `descriptive_only` and cannot authorize promotion.
