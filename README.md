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

The default config uses `AdaptiveAnnouncementParticipation`. Users provide the
parent orders, regular participation rates, and each name's next announcement
date through the data provider; they do not provide 2,000 separate
pre-announcement completion percentages.

For name `i`, the model first calculates the unavoidable pre-event shares:

```text
mandatory_pre_i = max(abs(target_i) - regular_post_event_capacity_i, 0)
```

It then:

1. adds a small portfolio-level flexibility allowance (5% by default);
2. aligns long and short pre-event capacity fractions for names whose planning
   horizons cross an announcement, when feasible;
3. gives that optional capacity to names with safer pre-event dates; and
4. water-fills each name toward dates farther from the announcement.

The announcement date gets the lowest weight. The regular participation cap
resumes on the next planner date. This makes the inferred pre-event fraction
different for every name based on target size, ADV, available post-event days,
side balance, and event timing.

```python
from trade_planner import AdaptiveAnnouncementParticipation, ParticipationCapModel

policy = AdaptiveAnnouncementParticipation(
    pre_event_flex=0.05,  # one portfolio policy, not one input per name
    balance_sides=True,
)
model = ParticipationCapModel(modifiers=(policy,))
summary = policy.allocation_summary(ctx)

print(summary[[
    "side",
    "mandatory_pre_fraction",
    "pre_event_cap_fraction",
    "max_pre_participation_rate",
    "max_post_participation_rate",
]])
```

`AnnouncementParticipationCurve` and `AnnouncementParticipationModifier` remain
available when a desk intentionally wants one fixed pre/post ADV-rate policy.
They are no longer the adaptive default.

After solving, inspect gross-notional pacing with start-date prices so terminal
completion is exactly 100% on both sides:

```python
from trade_planner import cumulative_side_completion

completion = cumulative_side_completion(ctx, result.schedule)
print(completion[["cumulative_long_pct", "cumulative_short_pct", "long_short_gap_pp"]])
```

Run the fixed-policy example and the reproducible model comparison with:

```bash
python -m examples.announcement_participation
python experiments/participation_refinement.py
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
confidence. The default config now requests CLARABEL. The adaptive participation
policy itself is NumPy-only, and the planner applies a positive objective-scale
normalization so changing supported CVXPY backends does not change model
trade-offs.

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

where `d_{i,t}` is days to next earnings/event. The adaptive participation
policy starts from the minimum pre-event requirement:

```math
B_i^{mandatory}
=
\max\left(
|q_i| - \sum_{t > event_i}\rho^{base}_{i,t}ADV_{i,t}m_{i,t},
0
\right).
```

The portfolio-level flexibility and side-balance step determines a pre-event
capacity budget at least this large. A bounded water-fill distributes that
budget using a monotone distance weight, with the smallest weight on the
announcement date and the regular cap restored after the event.

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
