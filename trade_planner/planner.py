"""Core cvxpy trade planner."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from .config import TradePlannerConfig
from .constraints import OptimizationState
from .context import PlannerContext
from .diagnostics import diagnose_infeasible_problem
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

        trades = cp.Variable((t_count, n_names))
        state = self._build_state(trades=trades, target=target, caps=caps, ctx=ctx)

        constraints = []
        for plugin in self.config.constraints:
            validate = getattr(plugin, "validate", None)
            if callable(validate):
                validate(ctx, state)
            constraints.extend(plugin.constraints(ctx, state))

        objective_terms = self._objective_terms(ctx, state)
        if self.config.terminal_penalty is not None:
            terminal_dollars = cp.multiply(ctx.price[-1], state.terminal_residual)
            objective_terms.append(self.config.terminal_penalty * cp.sum_squares(terminal_dollars))

        total_objective: cp.Expression | float = 0.0
        for term in objective_terms:
            total_objective = total_objective + term

        problem = cp.Problem(cp.Minimize(total_objective), constraints)
        self._solve_problem(problem)

        trade_values = np.asarray(trades.value, dtype=float)
        residual_after = target[None, :] - np.cumsum(trade_values, axis=0)
        schedule = self._build_schedule(ctx, trade_values, residual_after, caps)

        diagnostics = {
            "status": problem.status,
            "objective": float(problem.value),
            "max_abs_terminal_residual": float(np.max(np.abs(residual_after[-1]))),
        }
        return TradePlannerResult(schedule=schedule, diagnostics=diagnostics)

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
        for date_index, residual in enumerate(state.residuals):
            trade_t = state.trades[date_index, :]
            objective_terms.append(
                self.config.residual_risk_weight
                * self.config.risk_model.objective(residual, ctx, date_index)
            )
            objective_terms.append(self.config.cost_model.objective(trade_t, ctx, date_index))
        return objective_terms

    def _solve_problem(self, problem: cp.Problem) -> None:
        try:
            problem.solve(solver=self.config.solver, warm_start=True)
        except cp.SolverError:
            problem.solve(solver="CLARABEL", warm_start=True)
        if problem.status not in {"optimal", "optimal_inaccurate"}:
            diagnostics = diagnose_infeasible_problem(problem, run_elastic=False)
            message = diagnostics.get("summary", {}).get("message") or "Optimization did not solve."
            raise InfeasiblePlanError(
                f"Optimization failed with status {problem.status}: {message}",
                diagnostics=diagnostics,
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
