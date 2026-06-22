# Trade Planner

Pluggable daily basket execution planner for signed multi-name orders.

The package is intentionally split by extension point:

- `context.py`: normalized `PlannerContext` data object and date utilities
- `data.py`: provider placeholders, field alignment, and context assembly
- `participation.py`: participation caps and cap multipliers
- `risk.py`: covariance models and residual-risk overlays
- `costs.py`: market-impact and linear-cost objective terms
- `constraints.py`: pluggable cvxpy constraints
- `planner.py`: core solve orchestration
- `config.py`: default model wiring

## Constraint Plugins

Add a new hard constraint by implementing:

```python
class MyConstraint:
    def constraints(self, ctx, state):
        return [...]
```

`state` exposes:

- `state.trades`: `T x N` cvxpy variable
- `state.target`: parent signed target shares
- `state.caps`: participation cap matrix
- `state.cumulative_trades`: cumulative executed expressions by date
- `state.residuals`: unexecuted residual expressions by date
- `state.terminal_residual`: final residual expression

Optional early validation is supported:

```python
class MyConstraint:
    def validate(self, ctx, state):
        ...

    def constraints(self, ctx, state):
        return [...]
```

Then wire it through config:

```python
from trade_planner import (
    DailyGrossNotionalLimit,
    TradePlannerConfig,
    default_earnings_aware_config,
)

base = default_earnings_aware_config()
config = TradePlannerConfig(
    participation_model=base.participation_model,
    risk_model=base.risk_model,
    cost_model=base.cost_model,
    constraints=base.constraints + (
        DailyGrossNotionalLimit(max_dollars=25_000_000),
    ),
)
```

Run the synthetic example with:

```bash
python -m trade_planner.examples
```

## Production Context Flow

The intended production path is that users provide only:

- `start_date`
- `end_date`
- symbols
- signed `target_shares`

Everything else is loaded by a provider adapter:

```python
from trade_planner import TradePlanner, build_context_from_provider, default_earnings_aware_config

ctx = build_context_from_provider(
    orders=user_orders[["target_shares"]],
    start_date="2026-07-01",
    end_date="2026-07-10",
    provider=my_provider,
)

result = TradePlanner(default_earnings_aware_config()).solve(ctx)
```

The provider implements:

- `load_price(symbols, dates)`
- `load_adv_shares(symbols, dates)`
- `load_is_open(symbols, dates)`; defaults to `True`
- `load_base_participation(symbols, dates)`; defaults to the builder's `default_participation`
- `load_event_days(symbols, dates)`, or implement `load_event_dates(symbols, start_date, end_date)` and use the default event-day calculation
- `load_factor_exposure(symbols, dates)`
- `load_factor_covariance(factor_names, dates)`
- `load_specific_variance(symbols, dates)`
- optional `load_event_volatility(symbols, dates)`: returns event jump volatility used by the earnings risk overlay

Minimal provider skeleton:

```python
from trade_planner import PlannerDataProvider

class MyProvider(PlannerDataProvider):
    def load_price(self, symbols, dates):
        ...

    def load_adv_shares(self, symbols, dates):
        ...

    def load_event_days(self, symbols, dates):
        ...

    def load_factor_exposure(self, symbols, dates):
        ...

    def load_factor_covariance(self, factor_names, dates):
        ...

    def load_specific_variance(self, symbols, dates):
        ...
```

When a new context field is needed, add one provider method and one alignment
step in `data.py`; the optimizer code should not need to change.

## Barra-Style Residual Risk

The default risk model is `BarraFactorRiskModel`. For each date it computes:

```text
residual shares -> residual dollars -> factor dollar exposure
```

Then it applies:

```text
factor_exposure.T @ factor_covariance @ factor_exposure
+ specific variance risk
```

This keeps the implementation aligned with a factor-model risk decomposition
instead of applying a full security covariance matrix directly.
