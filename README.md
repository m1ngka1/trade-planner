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

## Announcement Participation Rates

Use `AnnouncementParticipationCurve` when the announcement date is known and
the planning horizon spans both sides of it. The announcement date itself stays
at the cautious pre-event rate; the higher rate begins the following day.

```python
import pandas as pd
from trade_planner import AnnouncementParticipationCurve

dates = pd.date_range("2026-07-01", periods=15, freq="D")
rates = AnnouncementParticipationCurve(
    pre_rate=0.025,
    post_rate=0.15,
).rates(dates, announcement_date=dates[9])

assert rates.iloc[4] == 0.025   # Day 5
assert rates.iloc[9] == 0.025   # Day 10 / announcement
assert rates.iloc[11] == 0.15   # Day 12
```

Set `transition="logistic"` for a smooth post-announcement ramp. Optional
`pre_volatility_sensitivity` and `post_volatility_sensitivity` apply inverse
volatility scaling: higher volatility reduces the rate and falling volatility
raises it. Inputs may be NumPy arrays or pandas Series. To use absolute event
rates inside the planner, add `AnnouncementParticipationModifier` to
`ParticipationCapModel`; modifiers may exceed 1x so a 2.5% base can become 15%.

Run the complete example with:

```bash
python -m examples.announcement_participation
```

## CVXPY Model Diagnostics

Attach domain meaning directly to each CVXPY constraint with
`with_diagnostics(...)`, solve the problem, then inspect it:

```python
report = diagnose_problem(problem)
print(report["text"])
```

The diagnostic iterates over every entry in `problem.constraints`, regardless of
whether it is an equality, inequality, SOC, PSD, exponential-cone, or a future
CVXPY constraint type. Each row contains its original id, type, shape,
variables, parameters, domain metadata, dual value, residual, and slack when
CVXPY exposes those values. `coverage` states exactly how many constraints had
primal metrics, dual values, and attached metadata.

The original object is never mutated and the diagnostic does not create a model
with many artificial slacks. For an infeasible solve it verifies each candidate
on an isolated copy with only that one constraint omitted. A reported
`single_constraint_recovery` therefore means the remaining original objective
and constraints reached an optimum; `witness_violation` records how far the
feasible witness lies outside the omitted constraint. If no individual omission
works, the report says the conflict requires multiple changes.

An unsolved problem is not solved implicitly; opt in to solving the same
original object with `diagnose_problem(problem, solve_if_needed=True)`. Set
`verify_bottlenecks=False` when only a fast evidence snapshot is wanted. The
optional `max_verification_checks` bounds counterfactual solves for very large
models. The older `diagnose_infeasible_problem` name remains as an alias.

`TradePlanner` keeps its automatic failure report fast and attaches the original
problem to `InfeasiblePlanError`, so a PM-facing workflow can request verified
recovery explicitly:

```python
try:
    result = planner.solve(ctx)
except InfeasiblePlanError as error:
    verified = diagnose_problem(error.problem)
    print(verified["text"])
```

CVXPY exposes statuses, dual values, constraint residuals, and solver-specific
`solver_stats`, but infeasible solves generally do not populate primal variable
values. Candidates are prioritized only when the original object provides evidence:
a nonzero residual, an active shadow price, an infeasibility dual stored on the
original constraint, or a solver certificate mapped to its CVXPY constraint id.
If an infeasible solver result exposes only an unmapped raw
certificate, the report inventories every constraint but explicitly leaves the
bottleneck unresolved rather than guessing. MOSEK can produce primal/dual
infeasibility certificates and a presolve report, but CVXPY canonicalization may
prevent a stable one-to-one mapping from low-level rows back to original
constraints; use the native Task API separately when that deeper evidence is
required.

References: [CVXPY statuses and infeasible/unbounded behavior](https://www.cvxpy.org/tutorial/intro/index.html),
[CVXPY constraint residual API](https://www.cvxpy.org/api_reference/cvxpy.constraints.html),
[CVXPY solver statistics](https://www.cvxpy.org/tutorial/solvers/index.html), and
[MOSEK Python API](https://docs.mosek.com/latest/pythonapi/index.html).

## Development Environment

Create the Python 3.12 environment and run the suite:

```bash
conda env create -f environment.yml
conda run -n trade-planner-dev python -m pytest -q
```

ECOS, SCS, CLARABEL, and OSQP are included as license-free fallbacks. MOSEK is
optional because it requires a supported installation and license; install its
Python package in this environment only when that license is available.

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

## Optimization Formulation

For symbols `i = 1..N` and planner dates `t = 1..T`, define:

- `q_i`: signed target shares; positive means buy, negative means sell/short
- `x_{i,t}`: signed shares traded on date `t`
- `p_{i,t}`: security price
- `ADV_{i,t}`: average daily volume
- `rho_{i,t}`: dynamic participation cap
- `m_{i,t}`: market-open flag
- `r_t = q - sum_{\tau=1}^t x_\tau`: residual unexecuted shares after date `t`

The planner solves a convex daily execution problem:

```math
\begin{aligned}
\min_{\{x_t\}_{t=1}^T}\quad
& \sum_{t=1}^T
\lambda_{\mathrm{risk}} R_t(r_t)
+ \sum_{t=1}^T C_t(x_t) \\
\mathrm{s.t.}\quad
& \sum_{t=1}^T x_{i,t} = q_i,\quad \forall i \\
& |x_{i,t}| \le \rho_{i,t} ADV_{i,t} m_{i,t},\quad \forall i,t \\
& \operatorname{sign}(q_i)x_{i,t} \ge 0,\quad \forall i,t .
\end{aligned}
```

The default cost model is:

```math
C_t(x_t)
=
\sum_i \eta_{i,t}x_{i,t}^2
+
\sum_i c_{i,t}|x_{i,t}|,
```

where the quadratic term is market impact and the linear term is spread,
commission, fees, or soft event-window penalties.

The default Barra-style residual risk model is:

```math
w_t = P_t r_t
```

```math
f_t = B_t^\top w_t
```

```math
R_t(r_t)
=
f_t^\top \Sigma^{F}_t f_t
+
\sum_i \sigma^2_{\epsilon,i,t} w_{i,t}^2 .
```

Here:

- `P_t = diag(p_t)` converts residual shares to residual dollars
- `B_t` is the security-by-factor exposure matrix
- `Sigma^F_t` is the factor covariance matrix
- `sigma^2_{\epsilon,i,t}` is specific return variance

With earnings/event risk overlays, the specific variance becomes:

```math
\tilde{\sigma}^2_{\epsilon,i,t}
=
\sigma^2_{\epsilon,i,t}
+
\sigma^2_{\mathrm{event},i}\exp(-d_{i,t}/\tau),
```

where `d_{i,t}` is days to next earnings/event. Participation caps can also be
made event-aware, for example:

```math
\rho_{i,t}
=
\rho_i^{base}
\left[
h_{min}
+
(1-h_{min})
\frac{1}{1+\exp(-k(d_{i,t}-d_0))}
\right].
```

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

Select active Barra factors in the risk model:

```python
from trade_planner import BarraFactorRiskModel, ExponentialEarningsRiskOverlay

risk_model = BarraFactorRiskModel(
    include_factors=["market", "size", "value", "momentum"],
    exclude_factors=["momentum"],
    specific_overlays=[
        ExponentialEarningsRiskOverlay(event_vol_column="event_vol", tau_days=5.0),
    ],
)
```

If `include_factors` is omitted, all factors in `ctx.factor_names` are used.
`exclude_factors` is applied after `include_factors`, preserving the factor
order loaded into context.
