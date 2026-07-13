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

This is a single-solve design. `diagnose_problem()` never disables a constraint,
builds an elastic model, or runs one solve per rule. It reads the status,
constraint metadata, duals/residuals, and structured certificate data already
returned by the original solve. An unsolved problem is not solved implicitly;
`solve_if_needed=True` permits exactly one solve of that same object.

For a continuous MOSEK solve, CVXPY dualizes the canonical model. Therefore an
original primal infeasibility appears as `dual_infeas` inside MOSEK. CVXPY maps
that certificate back to original constraint IDs in
`problem.solver_stats.extra_stats["IIS"]`. The report then:

1. normalizes the arbitrary certificate scale;
2. ranks the rules participating in the conflict;
3. decodes nonzero entries to date, symbol, or factor labels;
4. shows the current business setting and plugin-owned context such as target,
   total horizon capacity, and share shortfall; and
5. prints the action owned by that constraint plugin.

When MOSEK is requested, `TradePlanner` uses `DiagnosticMOSEK`, a thin CVXPY
solver adapter. It snapshots MOSEK problem/solution status, the native bound
certificate activity used by `getslc/getsuc/getslx/getsux`, and the opposite
certificate before CVXPY closes the Task. That opposite certificate maps an
original unbounded problem to a labeled joint variable direction in
`report["improving_direction"]`.

The main PM-facing fields are:

- `decision.what_to_change`: setting, location, current value, and action;
- `bottlenecks`: scale-normalized members of a primal-infeasibility conflict;
- `improving_direction`: labeled components of an unbounded ray;
- `solver_evidence.native_mosek`: canonical MOSEK statuses and solution quality;
- `summary.additional_solves`: always zero for an already-solved object.

An infeasibility certificate identifies a conflict set, not a guaranteed
one-rule repair and not the smallest safe limit change. Its relative weights
also depend on model scaling. The report deliberately says this instead of
promising that its first-ranked member alone will fix the model. Computing a
minimum operational change is a different optimization problem and is not run
automatically.

`TradePlanner` attaches both the report and original problem to
`InfeasiblePlanError`:

```python
try:
    result = planner.solve(ctx)
except InfeasiblePlanError as error:
    print(error.diagnostics["text"])
```

Without a mapped certificate, the report inventories the model and leaves the
cause unresolved rather than guessing. For non-MOSEK solvers, returned
constraint infeasibility duals are shown only as a lower-confidence fallback.
Mixed-integer infeasibility is also left unresolved because integer models do not
provide the continuous dual certificate used here.

References: [CVXPY statuses and infeasible/unbounded behavior](https://www.cvxpy.org/tutorial/intro/index.html),
[CVXPY constraint residual API](https://www.cvxpy.org/api_reference/cvxpy.constraints.html),
[CVXPY's continuous MOSEK dualization](https://www.cvxpy.org/version/1.2/updates/index.html),
[MOSEK infeasibility certificates](https://docs.mosek.com/latest/pythonapi/tutorial-pinfeas-shared.html), and
[MOSEK Task certificate APIs](https://docs.mosek.com/latest/pythonapi/optimizer-task.html).

## Development Environment

Create the Python 3.12 environment and run the suite:

```bash
conda env create -f environment.yml
conda run -n trade-planner-dev python -m pytest -q
```

ECOS, SCS, CLARABEL, and OSQP are included as license-free fallbacks. The MOSEK
Python package is installed for the certificate adapter, but a valid MOSEK
license is still required to solve with it. If MOSEK cannot run, `TradePlanner`
falls back to CLARABEL and marks its infeasibility-dual ranking as lower
confidence.

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
