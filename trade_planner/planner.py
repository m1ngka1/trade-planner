"""Core cvxpy trade planner."""

from __future__ import annotations

from dataclasses import dataclass, replace

import cvxpy as cp
import numpy as np
import pandas as pd

from .config import TradePlannerConfig
from .constraints import OptimizationState, VariableDiagnostics, with_variable_diagnostics
from .context import PlannerContext
from .diagnostics import diagnose_problem
from .mosek_diagnostics import diagnostic_mosek_if_requested
from .types import Array, InfeasiblePlanError


@dataclass(frozen=True)
class TradePlannerResult:
    schedule: pd.DataFrame
    diagnostics: dict[str, float | str]


class TradePlanner:
    """Core optimizer. Model behavior is controlled by config plugins."""

    def __init__(self, config: TradePlannerConfig):
        self.config = config

    def solve(self, ctx: PlannerContext) -> TradePlannerResult:
        symbols = ctx.symbols
        dates = ctx.dates
        t_count, n_names = len(dates), len(symbols)
        target = ctx.orders["target_shares"].reindex(symbols).to_numpy(float)
        caps = self.config.participation_model.caps(ctx)

        decision_variable, state = self._new_decision_state(
            target=target,
            caps=caps,
            ctx=ctx,
        )

        # A validator may return a normalized target. Rebuild immediately so
        # later validators and all CVXPY expressions see the same values.
        for plugin in self.config.constraints:
            validate = getattr(plugin, "validate", None)
            if callable(validate):
                adjusted_target = validate(ctx, state)
                if adjusted_target is not None:
                    target = np.asarray(adjusted_target, dtype=float).copy()
                    if target.shape != (n_names,):
                        raise ValueError(
                            f"{type(plugin).__name__}.validate returned target shape {target.shape}; "
                            f"expected {(n_names,)}"
                        )
                    decision_variable, state = self._new_decision_state(
                        target=target,
                        caps=caps,
                        ctx=ctx,
                    )

        constraints = []
        for plugin in self.config.constraints:
            constraints.extend(plugin.constraints(ctx, state))

        objective_ctx = self._objective_context(ctx, state.share_scale)
        objective_terms = self._objective_terms(objective_ctx, state)
        if self.config.terminal_penalty is not None:
            terminal_dollars = cp.multiply(
                objective_ctx.price[-1],
                state.constraint_terminal_residual,
            )
            objective_terms.append(self.config.terminal_penalty * cp.sum_squares(terminal_dollars))

        total_objective: cp.Expression | float = 0.0
        for term in objective_terms:
            total_objective = total_objective + term

        objective_multiplier = self._objective_multiplier(
            total_objective=total_objective,
            decision_variable=decision_variable,
            share_scale=state.share_scale,
            target=target,
            caps=caps,
        )
        problem = cp.Problem(cp.Minimize(objective_multiplier * total_objective), constraints)
        self._solve_problem(problem)

        trade_values = np.asarray(state.trades.value, dtype=float)
        residual_after = target[None, :] - np.cumsum(trade_values, axis=0)
        certificate = self._hard_constraint_certificate(
            trades=trade_values,
            target=target,
            caps=caps,
        )
        diagnostics: dict[str, float | str] = {
            "status": problem.status,
            "objective": float(total_objective.value),
            "solver_name": str(problem.solver_stats.solver_name),
            "solver_iterations": float(problem.solver_stats.num_iters or 0),
            "numerical_scaling": self.config.numerical_scaling,
            "decision_scale_min_shares": float(np.min(state.share_scale)),
            "decision_scale_max_shares": float(np.max(state.share_scale)),
            **certificate,
        }
        if self.config.verify_hard_constraints:
            violations = self._certificate_violations(certificate)
            if violations:
                details = ", ".join(
                    f"{name}={value:.9g} > {limit:.9g}"
                    for name, value, limit in violations
                )
                raise InfeasiblePlanError(
                    "Solver returned an operationally invalid schedule: " + details,
                    diagnostics={
                        "summary": {
                            "message": (
                                "The solver status was acceptable, but the independent "
                                "raw-share certificate breached a hard tolerance."
                            )
                        },
                        "certificate": diagnostics,
                    },
                    problem=problem,
                )
        schedule = self._build_schedule(ctx, trade_values, residual_after, caps)
        return TradePlannerResult(schedule=schedule, diagnostics=diagnostics)

    def _new_decision_state(
        self,
        *,
        target: Array,
        caps: Array,
        ctx: PlannerContext,
    ) -> tuple[cp.Variable, OptimizationState]:
        t_count, n_names = len(ctx.dates), len(ctx.symbols)
        share_scale = self._share_scale(target, caps)
        scaled = self.config.numerical_scaling == "per_name"
        decision_variable = with_variable_diagnostics(
            cp.Variable(
                (t_count, n_names),
                name="trade_parent_units" if scaled else "trade_shares",
            ),
            VariableDiagnostics(
                name="trade parent units" if scaled else "trade shares",
                description=(
                    "Signed parent-order units used by the numerical model."
                    if scaled
                    else "Signed shares scheduled for each date and symbol."
                ),
                units="parent-order units" if scaled else "shares",
                axis_labels={
                    "date": tuple(str(date.date()) for date in ctx.dates),
                    "symbol": tuple(ctx.symbols),
                },
            ),
        )
        trades: cp.Expression = (
            cp.multiply(share_scale[None, :], decision_variable)
            if scaled
            else decision_variable
        )
        trade_units: cp.Expression = decision_variable
        state = self._build_state(
            trades=trades,
            target=target,
            caps=caps,
            ctx=ctx,
            trade_units=trade_units,
            share_scale=share_scale,
        )
        return decision_variable, state

    def _share_scale(self, target: Array, caps: Array) -> Array:
        if self.config.numerical_scaling == "none":
            return np.ones_like(target, dtype=float)
        scale = np.abs(np.asarray(target, dtype=float))
        zero = scale <= 0.0
        if np.any(zero):
            scale[zero] = np.maximum(
                np.max(np.asarray(caps, dtype=float)[:, zero], axis=0),
                1.0,
            )
        return np.maximum(scale, 1e-12)

    @staticmethod
    def _objective_multiplier(
        total_objective: cp.Expression | float,
        decision_variable: cp.Variable,
        share_scale: Array,
        target: Array,
        caps: Array,
    ) -> float:
        """Numerically normalize the objective without changing its optimum."""
        if not isinstance(total_objective, cp.Expression):
            return 1.0
        total_capacity = np.sum(caps, axis=0)
        reference = np.divide(
            caps,
            total_capacity[None, :],
            out=np.zeros_like(caps),
            where=total_capacity[None, :] > 0,
        ) * target[None, :]
        decision_variable.value = reference / share_scale[None, :]
        reference_value = total_objective.value
        if reference_value is None or not np.isfinite(reference_value) or reference_value == 0:
            return 1.0
        # Keep typical objective magnitudes in a range that works well across
        # open-source QP and conic backends. Alpha rewards can make the
        # reference objective negative, so scale by magnitude. This remains a
        # positive scalar only; model trade-offs and the optimum are unchanged.
        return 1_000_000.0 / max(abs(float(reference_value)), 1_000_000.0)

    @staticmethod
    def _build_state(
        trades: cp.Expression,
        target: Array,
        caps: Array,
        ctx: PlannerContext,
        trade_units: cp.Expression | None = None,
        share_scale: Array | None = None,
    ) -> OptimizationState:
        if share_scale is None:
            share_scale = np.ones_like(target, dtype=float)
        if trade_units is None:
            trade_units = trades
        cumulative_trades = []
        residuals = []
        cumulative_trade_units = []
        residual_units = []
        cumulative: cp.Expression | float = 0.0
        cumulative_units: cp.Expression | float = 0.0
        target_units = target / share_scale
        caps_units = caps / share_scale[None, :]
        for date_index in range(len(ctx.dates)):
            cumulative = cumulative + trades[date_index, :]
            residual = target - cumulative
            cumulative_trades.append(cumulative)
            residuals.append(residual)
            cumulative_units = cumulative_units + trade_units[date_index, :]
            residual_unit = target_units - cumulative_units
            cumulative_trade_units.append(cumulative_units)
            residual_units.append(residual_unit)

        terminal_residual = target - cp.sum(trades, axis=0)
        terminal_residual_units = target_units - cp.sum(trade_units, axis=0)
        return OptimizationState(
            trades=trades,
            target=target,
            caps=caps,
            cumulative_trades=tuple(cumulative_trades),
            residuals=tuple(residuals),
            terminal_residual=terminal_residual,
            trade_units=trade_units,
            target_units=target_units,
            caps_units=caps_units,
            cumulative_trade_units=tuple(cumulative_trade_units),
            residual_units=tuple(residual_units),
            terminal_residual_units=terminal_residual_units,
            share_scale=share_scale,
        )

    @staticmethod
    def _hard_constraint_certificate(
        *,
        trades: Array,
        target: Array,
        caps: Array,
    ) -> dict[str, float]:
        cap_excess = np.maximum(np.abs(trades) - caps, 0.0)
        direction = np.sign(target)[None, :]
        wrong_direction = np.where(
            direction == 0.0,
            np.abs(trades),
            np.maximum(-direction * trades, 0.0),
        )
        completion = np.sum(trades, axis=0) - target
        return {
            "max_cap_excess_shares": float(np.max(cap_excess, initial=0.0)),
            "max_wrong_direction_shares": float(
                np.max(wrong_direction, initial=0.0)
            ),
            "max_abs_terminal_residual": float(
                np.max(np.abs(completion), initial=0.0)
            ),
        }

    def _certificate_violations(
        self,
        certificate: dict[str, float],
    ) -> list[tuple[str, float, float]]:
        checks = (
            (
                "max_cap_excess_shares",
                self.config.cap_tolerance_shares,
            ),
            (
                "max_wrong_direction_shares",
                self.config.direction_tolerance_shares,
            ),
            (
                "max_abs_terminal_residual",
                self.config.completion_tolerance_shares,
            ),
        )
        return [
            (name, certificate[name], float(limit))
            for name, limit in checks
            if certificate[name] > float(limit)
        ]

    def _objective_context(
        self,
        ctx: PlannerContext,
        share_scale: Array,
    ) -> PlannerContext:
        """Return an economically equivalent context in solver decision units."""

        if self.config.numerical_scaling == "none":
            return ctx
        orders = ctx.orders.copy()
        orders.loc[ctx.symbols, "target_shares"] = (
            orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
            / share_scale
        )
        return replace(
            ctx,
            orders=orders,
            price=np.asarray(ctx.price, dtype=float) * share_scale[None, :],
            adv_shares=(
                np.asarray(ctx.adv_shares, dtype=float) / share_scale[None, :]
            ),
            metadata={
                **ctx.metadata,
                "numerical_share_scale": share_scale.copy(),
                "numerical_units": "parent_order_units",
            },
        )

    def _objective_terms(
        self,
        ctx: PlannerContext,
        state: OptimizationState,
    ) -> list[cp.Expression]:
        objective_terms: list[cp.Expression] = []
        cumulative_path = state.constraint_cumulative_trades
        residual_path = (
            state.residuals
            if state.residual_units is None
            else state.residual_units
        )
        for date_index, (cumulative, residual) in enumerate(
            zip(cumulative_path, residual_path)
        ):
            trade_t = state.constraint_trades[date_index, :]
            # Inventory risk prices the P&L exposure already accumulated by
            # this date.  It is opt-in so existing residual-risk schedules are
            # unchanged until a caller deliberately enables this behavior.
            if self.config.inventory_risk_weight > 0:
                objective_terms.append(
                    self.config.inventory_risk_weight
                    * self.config.risk_model.objective(cumulative, ctx, date_index)
                )
            if self.config.residual_risk_weight > 0:
                objective_terms.append(
                    self.config.residual_risk_weight
                    * self.config.risk_model.objective(residual, ctx, date_index)
                )
            if self.config.inventory_alpha_model is not None:
                objective_terms.append(
                    self.config.inventory_alpha_model.objective(cumulative, ctx, date_index)
                )
            objective_terms.append(self.config.cost_model.objective(trade_t, ctx, date_index))
        if (
            self.config.inventory_path_risk_weight > 0
            and self.config.inventory_path_risk_model is not None
        ):
            objective_terms.append(
                self.config.inventory_path_risk_weight
                * self.config.inventory_path_risk_model.objective(
                    cumulative_path,
                    ctx,
                )
            )
        return objective_terms

    def _solve_problem(self, problem: cp.Problem) -> None:
        requested_solver = diagnostic_mosek_if_requested(self.config.solver)
        solver_options: dict[str, object] = {}
        if str(self.config.solver).upper() == "OSQP":
            # Production-sized share variables need tighter feasibility
            # tolerances than OSQP's defaults for participation caps to remain
            # operationally hard after numerical scaling.
            solver_options = {
                "eps_abs": 1e-8,
                "eps_rel": 1e-8,
                "max_iter": 500_000,
                "polishing": True,
            }
        elif (
            str(self.config.solver).upper() == "CLARABEL"
            and self.config.numerical_scaling == "per_name"
        ):
            # Do not accept Clarabel's much looser reduced-accuracy fallback
            # for operationally hard share constraints. In parent units,
            # 1e-8 is safely below the desk's raw-share certificate limits.
            solver_options = {
                "max_iter": 1_000,
                "tol_gap_abs": 1e-9,
                "tol_gap_rel": 1e-9,
                "tol_feas": 1e-9,
                "tol_infeas_abs": 1e-9,
                "tol_infeas_rel": 1e-9,
                "reduced_tol_gap_abs": 1e-8,
                "reduced_tol_gap_rel": 1e-8,
                "reduced_tol_feas": 1e-8,
                "reduced_tol_infeas_abs": 1e-9,
                "reduced_tol_infeas_rel": 1e-8,
            }
        try:
            problem.solve(
                solver=requested_solver,
                warm_start=True,
                **solver_options,
            )
        except cp.SolverError:
            problem.solve(solver="CLARABEL", warm_start=True)
        if problem.status not in {"optimal", "optimal_inaccurate"}:
            # Diagnose this exact solved object.  The report consumes attached
            # constraint metadata and existing solver evidence; it does not
            # relax the model or launch a second diagnostic solve.
            diagnostics = diagnose_problem(problem)
            message = diagnostics.get("summary", {}).get("message") or "Optimization did not solve."
            raise InfeasiblePlanError(
                f"Optimization failed with status {problem.status}: {message}",
                diagnostics=diagnostics,
                problem=problem,
            )

    @staticmethod
    def _build_schedule(
        ctx: PlannerContext,
        trades: Array,
        residual_after: Array,
        caps: Array,
    ) -> pd.DataFrame:
        records = []
        for t_idx, date in enumerate(ctx.dates):
            for s_idx, symbol in enumerate(ctx.symbols):
                price = ctx.price[t_idx, s_idx]
                adv = max(ctx.adv_shares[t_idx, s_idx], 1.0)
                trade = trades[t_idx, s_idx]
                records.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "trade_shares": trade,
                        "trade_dollars": trade * price,
                        "abs_pct_adv": abs(trade) / adv,
                        "cap_shares": caps[t_idx, s_idx],
                        "cap_pct_adv": caps[t_idx, s_idx] / adv,
                        "days_to_earnings": ctx.event_days.iloc[t_idx, s_idx],
                        "residual_shares_after": residual_after[t_idx, s_idx],
                        "residual_dollars_after": residual_after[t_idx, s_idx] * price,
                    }
                )
        return pd.DataFrame.from_records(records)
