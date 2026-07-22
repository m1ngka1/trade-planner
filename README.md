# Trade Planner

Pluggable daily basket execution planner for signed multi-name orders.

The package is intentionally split by extension point:

- `context.py`: normalized `PlannerContext` data object and date utilities
- `data.py`: provider placeholders, field alignment, and context assembly
- `participation.py`: participation caps and cap multipliers
- `risk.py`: covariance models for cumulative inventory or residual positions
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

A validator that needs to normalize the planning order may return a replacement
signed target array. The planner rebuilds `state` immediately so every later
validator, constraint, and objective expression sees those same values.

The built-in `HardCompletionConstraint` uses this preflight step to keep large
baskets operational. If a signed target is larger than its total horizon
capacity, the planner caps that target to the available shares and continues
instead of raising `InfeasiblePlanError`. One aggregated `UserWarning` lists
every affected symbol's original target, signed capped target, and absolute
share shortfall. The input `ctx.orders` remains unchanged; only the target used
for that plan is capped.

To print and assert the exact warning format with an oversized buy and sell:

```bash
python -m examples.hard_completion_warning
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

For a plain-language walkthrough with worked Stock A/Stock B and long/short
examples, see [Adaptive announcement participation model](docs/participation_model.md).

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

## Optimizer-Derived Rebalance Pacing

For anticipatory rebalance orders, use accumulated executed inventory—not the
unexecuted residual—as the pre-event P&L-risk state. Physical participation caps
then provide capacity, hard completion makes urgent names start when future
capacity becomes scarce, and quadratic impact prevents everything from being
left to the final date.

```python
from trade_planner import TradePlanner, default_rebalance_aware_config

planner = TradePlanner(default_rebalance_aware_config())
result = planner.solve(ctx)
```

Country, GICS sector/industry, beta, currency, and other balancing dimensions
are ordinary columns in the existing Barra factor inputs. Their risk on
cumulative executed positions encourages offsetting early trades without
forcing a hard daily schedule.

The fixed CLARABEL experiment records the current baselines, every tested
inventory/factor/impact combination, complete schedules, factor exposures, and
the corresponding daily-volume plots:

```bash
env PYTHONPATH=. python experiments/daily_volume_behavior.py \
  --solver CLARABEL \
  --output-prefix artifacts/daily_volume_behavior
```

See [Optimizer-derived daily volume behavior](docs/daily_volume_behavior.md)
for the benchmark, acceptance gates, selected combination, and artifact map.

## Investment-Driven Rebalance Calibration

The fixed synthetic weights above are useful for testing shape, but production
coefficients should be in economic units. When the provider supplies a
probability-weighted daily rebalance return forecast, the planner can solve one
expected-net-P&L versus accumulated-P&L-risk frontier and let the user choose
only `high`, `medium`, or `low` risk aversion:

```python
from trade_planner import calibrate_rebalance_plan

plan = calibrate_rebalance_plan(ctx, risk_aversion="medium")
schedule = plan.result.schedule

print(plan.metrics)
print(plan.frontier)
print(plan.economically_viable)
```

The calibrated objective is:

```text
expected market impact + spread/fees
    - expected holding alpha
    + selected covariance coefficient * accumulated holding-P&L variance
    + selected tail coefficient * path downside risk
```

There is deliberately no separate alpha coefficient: forecast confidence is
already reflected in probability-weighted `expected_return`. Impact and linear
costs use the date-by-name `impact_bps_at_10pct_adv` and `linear_cost_bps` TCA
surfaces directly, with a volatility-based impact fallback. The inventory-risk
coefficient is selected from solved schedules rather than copied from the
unit-scaled behavior fixture.

When the provider also supplies centered or uncentered residual-return
scenarios through `return_residual_scenarios`, the planner centers them and
prices only the scenario tail in excess of covariance-implied expected
shortfall. Both risk coefficients are scaled from the basket's expected dollars;
the user still chooses only a risk-aversion label. `high` uses the more stable
covariance frontier because the recorded experiment found no robust benefit
from fitting its extreme tail. `medium` uses the hybrid frontier and `low` uses
the quadratic tail second moment when scenario data is present; both fall back
to covariance without scenarios.

The final automatic policy is profile-specific because the recorded evidence
does not support one estimator everywhere: `high` uses covariance, `medium`
uses covariance plus 96-scenario excess-tail CVaR, and `low` uses covariance
plus the worst-tail conditional P&L second moment. The low-profile quadratic
tail model improved independent CVaR, volatility, and early factor balance in
all five optimization-seed replications while solving 22–54 times faster than
hybrid CVaR. Medium remains hybrid because the faster approximation failed one
strict small-order timing gate. Users still select only the risk label.

Scenario frontiers automatically retain every path in the worst 10% full-
basket tail and compress the remaining core into at most 96 weighted
representatives. Economic metrics and selection are still evaluated on the
full input distribution. This preserved the selected medium-risk mechanics and
independent loss CVaR while making the recorded frontier 3.9 times faster;
passing `max_optimization_scenarios=None` requests an unreduced audit.

`high` stays within the lowest 5% of feasible P&L risk, `medium` allows half of the
feasible risk range when expected alpha pays for it, and `low` allows the full
range. Within each budget, expected-P&L differences smaller than one basis
point of parent gross are treated as economically tied and the lower-risk plan
wins. Capacity and completion remain hard constraints for every profile.
`economically_viable` is false when forecast alpha does not cover modeled
impact and fees; the planner never labels a compulsory but negative-edge
execution as profitable.

Run the recorded economic experiment with:

```bash
env PYTHONPATH=. python experiments/rebalance_economic_calibration.py
env PYTHONPATH=. python experiments/scenario_reduction.py
env PYTHONPATH=. python experiments/stress_path_seed_robustness.py \
  --path-model second_moment
```

See [Investment-driven rebalance calibration](docs/rebalance_economic_calibration.md)
for the tested hypotheses, Monte Carlo validation, limitations, and artifact
map, and [Tail-preserving scenario reduction](docs/scenario_reduction.md) for
the 256/96/64 runtime comparison. See
[Scenario-derived tail path risk](docs/tail_path_risk.md) for the five-seed
mean-stress and second-moment decisions. The economic experiment complements
rather than replaces the fixed shape gates in `daily_volume_behavior.py`.

## Point-in-Time Walk-Forward Replay

Use `PointInTimeRebalanceEvent` to keep the optimizer snapshot physically
separate from realized returns and execution costs, then compare strategies
chronologically with `replay_rebalance_events`. The replay validates information
cutoffs and reports realized P&L, costs, volatility, loss CVaR, drawdown, and
completion error in dollars and basis points of parent gross.

```python
from trade_planner import PointInTimeRebalanceEvent, replay_rebalance_events

event = PointInTimeRebalanceEvent(
    event_id="rebalance-001",
    as_of=as_of,
    information_cutoff=as_of,
    ctx=point_in_time_ctx,
    realized_returns=realized_returns,
    realized_impact_bps_at_10pct_adv=realized_impact,
    realized_linear_cost_bps=realized_linear_cost,
    realized_available_at=realized_available_at,
)
replay = replay_rebalance_events([event], strategies)
```

The recorded development experiment rejected automatic alpha-confidence
haircuts and uncertainty-scaled risk budgets because they failed strict realized
downside gates. An uncertainty-aware tie rule was also redundant with the
existing one-basis-point economic-materiality rule. They remain research options
rather than production defaults.
See [Point-in-time rebalance replay](docs/point_in_time_walkforward.md) for the
data contract, every trial, keep/discard logic, and visual artifacts.

## CVXPY Model Diagnostics

For a worked three-name example with a real CLARABEL failure and a transparent
certificate stand-in, see [Diagnostics walkthrough](docs/diagnostics_example.md).

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
- optional `load_expected_return(symbols, dates)`: probability-weighted expected return earned by accumulated inventory after each planner date
- optional `load_expected_return_uncertainty(symbols, dates)`: point-in-time standard error of the expected-return forecast for controlled walk-forward research
- optional `load_impact_bps_at_10pct_adv(symbols, dates)` and `load_linear_cost_bps(symbols, dates)`: date/name TCA surfaces
- optional `load_return_residual_scenarios(symbols, dates)`: scenario-by-date-by-name holding-return residuals for hybrid downside risk
- optional `load_return_scenario_weights(symbols, dates)`: one probability per residual-return scenario

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
- `h_t = sum_{\tau=1}^t x_\tau`: cumulative executed inventory after date `t`
- `r_t = q - sum_{\tau=1}^t x_\tau`: residual unexecuted shares after date `t`

The planner solves a convex daily execution problem:

```math
\begin{aligned}
\min_{\{x_t\}_{t=1}^T}\quad
& \sum_{t=1}^T \left[
\lambda_{\mathrm{inventory}} R_t(h_t)
+ \lambda_{\mathrm{residual}} R_t(r_t)
+ C_t(x_t) \right] \\
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

The Barra-style risk model can score either cumulative inventory or residual
shares. For a generic share position `z_t`:

```math
w_t = P_t z_t
```

```math
f_t = B_t^\top w_t
```

```math
R_t(z_t)
=
f_t^\top \Sigma^{F}_t f_t
+
\sum_i \sigma^2_{\epsilon,i,t} w_{i,t}^2 .
```

Here:

- `P_t = diag(p_t)` converts shares to position dollars
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
