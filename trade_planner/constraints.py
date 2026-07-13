"""Pluggable cvxpy constraints for the trade planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol
import weakref

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


@dataclass(frozen=True)
class ConstraintDiagnostics:
    """User-facing diagnostics owned by a high-level planner constraint."""

    name: str
    group: str = ""
    description: str = ""
    potential_cause: str = ""
    suggested_relaxation: str = ""
    units: str = ""
    weight: float = 1.0
    hard: bool = True
    axis_labels: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)
    setting_name: str = ""
    bound_values: Any = field(default=None, repr=False, compare=False)
    element_context: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class VariableDiagnostics:
    """Business labels used to explain an unbounded solver direction."""

    name: str
    description: str = ""
    units: str = ""
    axis_labels: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


_DIAGNOSTICS_ATTR = "_trade_planner_diagnostics"
_DIAGNOSTICS_BY_OBJECT: weakref.WeakKeyDictionary[cp.Constraint, ConstraintDiagnostics] = weakref.WeakKeyDictionary()
_DIAGNOSTICS_BY_ID: dict[int, ConstraintDiagnostics] = {}
_VARIABLE_DIAGNOSTICS_ATTR = "_trade_planner_variable_diagnostics"
_VARIABLE_DIAGNOSTICS_BY_OBJECT: weakref.WeakKeyDictionary[cp.Variable, VariableDiagnostics] = (
    weakref.WeakKeyDictionary()
)
_VARIABLE_DIAGNOSTICS_BY_ID: dict[int, VariableDiagnostics] = {}


def with_diagnostics(constraint: cp.Constraint, diagnostics: ConstraintDiagnostics) -> cp.Constraint:
    """Attach user-facing diagnostics to a CVXPY constraint and return it."""
    try:
        setattr(constraint, _DIAGNOSTICS_ATTR, diagnostics)
    except Exception:
        try:
            _DIAGNOSTICS_BY_OBJECT[constraint] = diagnostics
        except TypeError:
            _DIAGNOSTICS_BY_ID[id(constraint)] = diagnostics
    return constraint


def get_constraint_diagnostics(constraint: cp.Constraint) -> ConstraintDiagnostics | None:
    """Return diagnostics previously attached with `with_diagnostics`."""
    diagnostics = getattr(constraint, _DIAGNOSTICS_ATTR, None)
    if isinstance(diagnostics, ConstraintDiagnostics):
        return diagnostics
    try:
        diagnostics = _DIAGNOSTICS_BY_OBJECT.get(constraint)
    except TypeError:
        diagnostics = None
    if diagnostics is not None:
        return diagnostics
    return _DIAGNOSTICS_BY_ID.get(id(constraint))


def with_variable_diagnostics(variable: cp.Variable, diagnostics: VariableDiagnostics) -> cp.Variable:
    """Attach business labels to a CVXPY variable and return it."""
    try:
        setattr(variable, _VARIABLE_DIAGNOSTICS_ATTR, diagnostics)
    except Exception:
        try:
            _VARIABLE_DIAGNOSTICS_BY_OBJECT[variable] = diagnostics
        except TypeError:
            _VARIABLE_DIAGNOSTICS_BY_ID[id(variable)] = diagnostics
    return variable


def get_variable_diagnostics(variable: cp.Variable) -> VariableDiagnostics | None:
    """Return diagnostics previously attached with `with_variable_diagnostics`."""
    diagnostics = getattr(variable, _VARIABLE_DIAGNOSTICS_ATTR, None)
    if isinstance(diagnostics, VariableDiagnostics):
        return diagnostics
    try:
        diagnostics = _VARIABLE_DIAGNOSTICS_BY_OBJECT.get(variable)
    except TypeError:
        diagnostics = None
    if diagnostics is not None:
        return diagnostics
    return _VARIABLE_DIAGNOSTICS_BY_ID.get(id(variable))


class ConstraintPlugin(Protocol):
    """Plugin that contributes cvxpy constraints to the planner."""

    def constraints(self, ctx: PlannerContext, state: OptimizationState) -> list[cp.Constraint]:
        ...


@dataclass(frozen=True)
class ParticipationCapacityConstraint:
    """Limit each date-symbol trade by the participation cap model."""

    def constraints(self, ctx: PlannerContext, state: OptimizationState) -> list[cp.Constraint]:
        total_capacity = np.sum(state.caps, axis=0)

        def element_context(index: tuple[int, ...]) -> Mapping[str, Any]:
            symbol_index = index[-1]
            required = float(abs(state.target[symbol_index]))
            available = float(total_capacity[symbol_index])
            shortfall = max(required - available, 0.0)
            return {
                "parent_target_abs_shares": required,
                "total_horizon_capacity_shares": available,
                "capacity_shortfall_shares": shortfall,
                "pm_action": (
                    f"Capacity-only adjustment: add at least {shortfall:g} shares of horizon capacity "
                    f"for {ctx.symbols[symbol_index]}, or reduce/leave {shortfall:g} target shares. "
                    "Other reported policy conflicts may still need changes."
                    if shortfall > 0
                    else "Capacity covers the target; adjust another reported policy conflict."
                ),
            }

        return [
            with_diagnostics(
                cp.abs(state.trades) <= state.caps,
                ConstraintDiagnostics(
                    name="participation_capacity",
                    group="capacity",
                    description="Each date-symbol trade must stay within the participation cap model.",
                    potential_cause=(
                        "The requested order size, trading window, market-open mask, or participation "
                        "cap is too tight to satisfy the other requirements."
                    ),
                    suggested_relaxation="Increase participation caps, extend the trading window, or relax full completion.",
                    units="shares",
                    axis_labels={"date": _date_labels(ctx.dates), "symbol": tuple(ctx.symbols)},
                    setting_name="participation cap",
                    bound_values=state.caps,
                    element_context=element_context,
                ),
            )
        ]


@dataclass(frozen=True)
class DirectionConstraint:
    """Prevent buy-sell round trips against the parent order direction."""

    def constraints(self, ctx: PlannerContext, state: OptimizationState) -> list[cp.Constraint]:
        direction = np.sign(state.target)
        return [
            with_diagnostics(
                cp.multiply(np.tile(direction, (len(ctx.dates), 1)), state.trades) >= 0,
                ConstraintDiagnostics(
                    name="order_direction",
                    group="direction",
                    description="Trades must have the same signed direction as the parent order.",
                    potential_cause=(
                        "Other constraints may require temporary buy-sell round trips to reach feasibility."
                    ),
                    suggested_relaxation="Allow limited opposite-direction trades or relax the conflicting exposure/cap constraint.",
                    units="shares",
                    weight=10.0,
                    axis_labels={"date": _date_labels(ctx.dates), "symbol": tuple(ctx.symbols)},
                    setting_name="minimum signed trade",
                    bound_values=0.0,
                ),
            )
        ]


@dataclass(frozen=True)
class ZeroTargetConstraint:
    """Force no trades for names with a zero parent order."""

    def constraints(self, ctx: PlannerContext, state: OptimizationState) -> list[cp.Constraint]:
        zero_target_idx = np.flatnonzero(state.target == 0)
        if len(zero_target_idx) == 0:
            return []
        zero_symbols = tuple(ctx.symbols[i] for i in zero_target_idx)
        return [
            with_diagnostics(
                state.trades[:, zero_target_idx] == 0,
                ConstraintDiagnostics(
                    name="zero_target_no_trade",
                    group="mandate",
                    description="Symbols with zero parent target shares must not trade.",
                    potential_cause="Other constraints may require trading a name that has no parent order.",
                    suggested_relaxation="Permit a small trade for zero-target names or remove the conflicting constraint.",
                    units="shares",
                    weight=10.0,
                    axis_labels={"date": _date_labels(ctx.dates), "symbol": zero_symbols},
                    details={"zero_target_symbols": zero_symbols},
                    setting_name="allowed trade for zero-target names",
                    bound_values=0.0,
                ),
            )
        ]


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
        total_capacity = np.sum(state.caps, axis=0)

        def element_context(index: tuple[int, ...]) -> Mapping[str, Any]:
            symbol_index = index[-1]
            required = float(abs(state.target[symbol_index]))
            available = float(total_capacity[symbol_index])
            shortfall = max(required - available, 0.0)
            return {
                "parent_target_abs_shares": required,
                "total_horizon_capacity_shares": available,
                "capacity_shortfall_shares": shortfall,
                "pm_action": (
                    f"Capacity-only adjustment: add at least {shortfall:g} shares of horizon capacity "
                    f"for {ctx.symbols[symbol_index]}, or reduce/leave {shortfall:g} target shares. "
                    "Other reported policy conflicts may still need changes."
                    if shortfall > 0
                    else "Capacity covers the target; adjust another reported policy conflict."
                ),
            }

        return [
            with_diagnostics(
                cp.sum(state.trades, axis=0) == state.target,
                ConstraintDiagnostics(
                    name="hard_completion",
                    group="completion",
                    description="The final cumulative trade must exactly match each parent order.",
                    potential_cause=(
                        "The order cannot be fully completed under capacity, direction, zero-target, or "
                        "portfolio-level limits."
                    ),
                    suggested_relaxation=(
                        "Allow terminal residual, extend the trading window, reduce the parent order, or relax capacity limits."
                    ),
                    units="shares",
                    axis_labels={"symbol": tuple(ctx.symbols)},
                    setting_name="parent target",
                    bound_values=state.target,
                    element_context=element_context,
                ),
            )
        ]


@dataclass(frozen=True)
class DailyGrossNotionalLimit:
    """Cap total absolute traded notional per date."""

    max_dollars: float | pd.Series | Mapping[pd.Timestamp | str, float]

    def constraints(self, ctx: PlannerContext, state: OptimizationState) -> list[cp.Constraint]:
        limits = _align_daily_limit(self.max_dollars, ctx.dates)
        out = []
        for date_index, limit in enumerate(limits):
            trade_dollars = cp.multiply(ctx.price[date_index], state.trades[date_index, :])
            date_label = str(ctx.dates[date_index].date())
            out.append(
                with_diagnostics(
                    cp.sum(cp.abs(trade_dollars)) <= limit,
                    ConstraintDiagnostics(
                        name=f"daily_gross_notional_limit[{date_label}]",
                        group="notional",
                        description="Total absolute traded notional on a date must stay below the configured limit.",
                        potential_cause="The daily gross notional limit is too low for required completion or milestones.",
                        suggested_relaxation="Increase the gross notional limit for this date or move more volume to other dates.",
                        units="dollars",
                        axis_labels={"date": (date_label,)},
                        details={"limit": float(limit)},
                        setting_name="daily gross notional limit",
                        bound_values=float(limit),
                    ),
                )
            )
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
            date_label = str(ctx.dates[date_index].date())
            out.extend(
                [
                    with_diagnostics(
                        net_dollars <= limit,
                        ConstraintDiagnostics(
                            name=f"daily_net_notional_upper[{date_label}]",
                            group="notional",
                            description="Signed net traded notional on a date must stay below the upper limit.",
                            potential_cause="The date requires more net buying than the configured net notional limit allows.",
                            suggested_relaxation="Increase the positive net notional limit or rebalance buys/sells across dates.",
                            units="dollars",
                            axis_labels={"date": (date_label,)},
                            details={"limit": float(limit)},
                            setting_name="daily net notional upper limit",
                            bound_values=float(limit),
                        ),
                    ),
                    with_diagnostics(
                        net_dollars >= -limit,
                        ConstraintDiagnostics(
                            name=f"daily_net_notional_lower[{date_label}]",
                            group="notional",
                            description="Signed net traded notional on a date must stay above the lower limit.",
                            potential_cause="The date requires more net selling than the configured net notional limit allows.",
                            suggested_relaxation="Increase the negative net notional limit or rebalance buys/sells across dates.",
                            units="dollars",
                            axis_labels={"date": (date_label,)},
                            details={"limit": float(limit)},
                            setting_name="daily net notional lower limit",
                            bound_values=-float(limit),
                        ),
                    ),
                ]
            )
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
        date_label = str(ctx.dates[idx].date())
        return [
            with_diagnostics(
                cp.multiply(direction, completed) >= self.fraction * np.abs(state.target),
                ConstraintDiagnostics(
                    name=f"min_completion_by_date[{date_label}]",
                    group="completion",
                    description="Each order must reach a minimum completion fraction by the milestone date.",
                    potential_cause="The milestone completion fraction is too aggressive for earlier capacity or notional limits.",
                    suggested_relaxation="Lower the milestone fraction, move the milestone later, or increase capacity before the milestone.",
                    units="shares",
                    axis_labels={"symbol": tuple(ctx.symbols)},
                    details={"date": date_label, "fraction": float(self.fraction)},
                    setting_name="minimum completed shares",
                    bound_values=self.fraction * np.abs(state.target),
                ),
            )
        ]


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
        factor_labels = tuple(str(col) for col in self.exposures.columns)
        exposure_basis = "residual" if self.use_residual else "cumulative"
        for date_index, expr in enumerate(expressions):
            dollars = cp.multiply(ctx.price[date_index], expr)
            factor_exposure = matrix.T @ dollars
            date_label = str(ctx.dates[date_index].date())
            out.extend(
                [
                    with_diagnostics(
                        factor_exposure <= self.max_abs_exposure,
                        ConstraintDiagnostics(
                            name=f"{exposure_basis}_factor_exposure_upper[{date_label}]",
                            group="factor_exposure",
                            description=f"{exposure_basis.title()} dollar factor exposure must stay below the upper limit.",
                            potential_cause="The exposure limit is too tight relative to the order mix and available trading capacity.",
                            suggested_relaxation="Increase factor exposure limits, relax completion timing, or adjust the basket composition.",
                            units="dollars",
                            axis_labels={"factor": factor_labels},
                            details={
                                "date": date_label,
                                "max_abs_exposure": float(self.max_abs_exposure),
                                "use_residual": bool(self.use_residual),
                            },
                            setting_name="factor exposure upper limit",
                            bound_values=float(self.max_abs_exposure),
                        ),
                    ),
                    with_diagnostics(
                        factor_exposure >= -self.max_abs_exposure,
                        ConstraintDiagnostics(
                            name=f"{exposure_basis}_factor_exposure_lower[{date_label}]",
                            group="factor_exposure",
                            description=f"{exposure_basis.title()} dollar factor exposure must stay above the lower limit.",
                            potential_cause="The exposure limit is too tight relative to the order mix and available trading capacity.",
                            suggested_relaxation="Increase factor exposure limits, relax completion timing, or adjust the basket composition.",
                            units="dollars",
                            axis_labels={"factor": factor_labels},
                            details={
                                "date": date_label,
                                "max_abs_exposure": float(self.max_abs_exposure),
                                "use_residual": bool(self.use_residual),
                            },
                            setting_name="factor exposure lower limit",
                            bound_values=-float(self.max_abs_exposure),
                        ),
                    ),
                ]
            )
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


def _date_labels(dates: pd.DatetimeIndex) -> tuple[str, ...]:
    return tuple(str(date.date()) for date in dates)
