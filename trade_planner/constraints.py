"""Pluggable cvxpy constraints for the trade planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import cvxpy as cp
import numpy as np
import pandas as pd

from .context import PlannerContext
from .types import Array, InfeasiblePlanError


@dataclass(frozen=True)
class OptimizationState:
    """Variables and reusable expressions exposed to constraint plugins."""

    trades: cp.Variable
    target: Array
    caps: Array
    cumulative_trades: tuple[cp.Expression, ...]
    residuals: tuple[cp.Expression, ...]
    terminal_residual: cp.Expression


class ConstraintPlugin(Protocol):
    """Plugin that contributes cvxpy constraints to the planner."""

    def constraints(self, ctx: PlannerContext, state: OptimizationState) -> list[cp.Constraint]:
        ...


@dataclass(frozen=True)
class ParticipationCapacityConstraint:
    """Limit each date-symbol trade by the participation cap model."""

    def constraints(self, ctx: PlannerContext, state: OptimizationState) -> list[cp.Constraint]:
        return [cp.abs(state.trades) <= state.caps]


@dataclass(frozen=True)
class DirectionConstraint:
    """Prevent buy-sell round trips against the parent order direction."""

    def constraints(self, ctx: PlannerContext, state: OptimizationState) -> list[cp.Constraint]:
        direction = np.sign(state.target)
        return [cp.multiply(np.tile(direction, (len(ctx.dates), 1)), state.trades) >= 0]


@dataclass(frozen=True)
class ZeroTargetConstraint:
    """Force no trades for names with a zero parent order."""

    def constraints(self, ctx: PlannerContext, state: OptimizationState) -> list[cp.Constraint]:
        zero_target_idx = np.flatnonzero(state.target == 0)
        if len(zero_target_idx) == 0:
            return []
        return [state.trades[:, zero_target_idx] == 0]


@dataclass(frozen=True)
class HardCompletionConstraint:
    """Require full completion by the final planner date."""

    check_capacity: bool = True

    def validate(self, ctx: PlannerContext, state: OptimizationState) -> None:
        if not self.check_capacity:
            return
        capacity = state.caps.sum(axis=0)
        shortfall = np.abs(state.target) - capacity
        bad = np.flatnonzero(shortfall > 1e-8)
        if not len(bad):
            return
        details = {
            ctx.symbols[i]: {
                "required_abs_shares": float(abs(state.target[i])),
                "available_capacity": float(capacity[i]),
                "shortfall": float(shortfall[i]),
            }
            for i in bad
        }
        raise InfeasiblePlanError(f"Insufficient capacity for hard completion: {details}")

    def constraints(self, ctx: PlannerContext, state: OptimizationState) -> list[cp.Constraint]:
        return [cp.sum(state.trades, axis=0) == state.target]


@dataclass(frozen=True)
class DailyGrossNotionalLimit:
    """Cap total absolute traded notional per date."""

    max_dollars: float | pd.Series | Mapping[pd.Timestamp | str, float]

    def constraints(self, ctx: PlannerContext, state: OptimizationState) -> list[cp.Constraint]:
        limits = _align_daily_limit(self.max_dollars, ctx.dates)
        out = []
        for date_index, limit in enumerate(limits):
            trade_dollars = cp.multiply(ctx.price[date_index], state.trades[date_index, :])
            out.append(cp.sum(cp.abs(trade_dollars)) <= limit)
        return out


@dataclass(frozen=True)
class DailyNetNotionalLimit:
    """Bound signed net traded notional per date."""

    max_abs_dollars: float | pd.Series | Mapping[pd.Timestamp | str, float]

    def constraints(self, ctx: PlannerContext, state: OptimizationState) -> list[cp.Constraint]:
        limits = _align_daily_limit(self.max_abs_dollars, ctx.dates)
        out = []
        for date_index, limit in enumerate(limits):
            net_dollars = cp.sum(cp.multiply(ctx.price[date_index], state.trades[date_index, :]))
            out.extend([net_dollars <= limit, net_dollars >= -limit])
        return out


@dataclass(frozen=True)
class MinCompletionByDate:
    """
    Require at least a fraction of each order to be completed by a milestone date.

    The fraction is applied to signed parent orders, so it works for both buys
    and sells when DirectionConstraint is also present.
    """

    date: pd.Timestamp | str
    fraction: float

    def constraints(self, ctx: PlannerContext, state: OptimizationState) -> list[cp.Constraint]:
        idx = int(np.searchsorted(ctx.dates, pd.Timestamp(self.date).normalize(), side="right") - 1)
        if idx < 0:
            raise ValueError("milestone date is before the planner start date")
        idx = min(idx, len(ctx.dates) - 1)
        direction = np.sign(state.target)
        completed = state.cumulative_trades[idx]
        return [cp.multiply(direction, completed) >= self.fraction * np.abs(state.target)]


@dataclass(frozen=True)
class FactorExposureLimit:
    """Bound cumulative or residual dollar factor exposure each day."""

    exposures: pd.DataFrame
    max_abs_exposure: float
    use_residual: bool = True

    def constraints(self, ctx: PlannerContext, state: OptimizationState) -> list[cp.Constraint]:
        matrix = self.exposures.reindex(ctx.symbols).fillna(0.0).to_numpy(float)
        out = []
        expressions = state.residuals if self.use_residual else state.cumulative_trades
        for date_index, expr in enumerate(expressions):
            dollars = cp.multiply(ctx.price[date_index], expr)
            factor_exposure = matrix.T @ dollars
            out.extend([factor_exposure <= self.max_abs_exposure, factor_exposure >= -self.max_abs_exposure])
        return out


def default_constraints() -> tuple[ConstraintPlugin, ...]:
    """Default execution-mandate constraints used by the planner."""
    return (
        ParticipationCapacityConstraint(),
        DirectionConstraint(),
        ZeroTargetConstraint(),
        HardCompletionConstraint(),
    )


def _align_daily_limit(
    value: float | pd.Series | Mapping[pd.Timestamp | str, float],
    dates: pd.DatetimeIndex,
) -> Array:
    if isinstance(value, (int, float)):
        return np.full(len(dates), float(value), dtype=float)
    if isinstance(value, pd.Series):
        series = value.copy()
    else:
        series = pd.Series(value)
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index)).normalize()
    aligned = series.reindex(dates)
    if aligned.isna().any():
        missing = [str(date.date()) for date in dates[aligned.isna()]]
        raise ValueError(f"missing daily limit values for dates: {missing}")
    return aligned.to_numpy(float)
