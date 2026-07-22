"""Core cvxpy trade planner."""

from __future__ import annotations

from dataclasses import dataclass

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

        trades = with_variable_diagnostics(
            cp.Variable((t_count, n_names), name="trade_shares"),
            VariableDiagnostics(
                name="trade shares",
                description="Signed shares scheduled for each date and symbol.",
                units="shares",
                axis_labels={
                    "date": tuple(str(date.date()) for date in ctx.dates),
                    "symbol": tuple(ctx.symbols),
                },
            ),
        )
        state = self._build_state(trades=trades, target=target, caps=caps, ctx=ctx)

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
                    state = self._build_state(trades=trades, target=target, caps=caps, ctx=ctx)

        constraints = []
        for plugin in self.config.constraints:
            constraints.extend(plugin.constraints(ctx, state))

        objective_terms = self._objective_terms(ctx, state)
        if self.config.terminal_penalty is not None:
            terminal_dollars = cp.multiply(ctx.price[-1], state.terminal_residual)
            objective_terms.append(self.config.terminal_penalty * cp.sum_squares(terminal_dollars))

        total_objective: cp.Expression | float = 0.0
        for term in objective_terms:
            total_objective = total_objective + term

        objective_multiplier = self._objective_multiplier(
            total_objective=total_objective,
            trades=trades,
            target=target,
            caps=caps,
        )
        problem = cp.Problem(cp.Minimize(objective_multiplier * total_objective), constraints)
        self._solve_problem(problem)

        trade_values = np.asarray(trades.value, dtype=float)
        residual_after = target[None, :] - np.cumsum(trade_values, axis=0)
        schedule = self._build_schedule(ctx, trade_values, residual_after, caps)

        diagnostics = {
            "status": problem.status,
            "objective": float(total_objective.value),
            "max_abs_terminal_residual": float(np.max(np.abs(residual_after[-1]))),
        }
        return TradePlannerResult(schedule=schedule, diagnostics=diagnostics)

    @staticmethod
    def _objective_multiplier(
        total_objective: cp.Expression | float,
        trades: cp.Variable,
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
        trades.value = reference
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
        trades: cp.Variable,
        target: Array,
        caps: Array,
        ctx: PlannerContext,
    ) -> OptimizationState:
        cumulative_trades = []
        residuals = []
        cumulative: cp.Expression | float = 0.0
        for date_index in range(len(ctx.dates)):
            cumulative = cumulative + trades[date_index, :]
            residual = target - cumulative
            cumulative_trades.append(cumulative)
            residuals.append(residual)

        terminal_residual = target - cp.sum(trades, axis=0)
        return OptimizationState(
            trades=trades,
            target=target,
            caps=caps,
            cumulative_trades=tuple(cumulative_trades),
            residuals=tuple(residuals),
            terminal_residual=terminal_residual,
        )

    def _objective_terms(self, ctx: PlannerContext, state: OptimizationState) -> list[cp.Expression]:
        objective_terms: list[cp.Expression] = []
        for date_index, (cumulative, residual) in enumerate(
            zip(state.cumulative_trades, state.residuals)
        ):
            trade_t = state.trades[date_index, :]
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
                    state.cumulative_trades,
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
