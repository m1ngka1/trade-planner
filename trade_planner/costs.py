"""Transaction cost and market-impact objective plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import cvxpy as cp
import numpy as np

from .context import PlannerContext
from .utils import safe_numeric


class CostTerm(Protocol):
    """Plugin that returns the cvxpy cost expression for one date's trades."""

    def objective(
        self,
        trade: cp.Expression,
        ctx: PlannerContext,
        date_index: int,
    ) -> cp.Expression:
        ...


@dataclass(frozen=True)
class CompositeCostModel:
    terms: Sequence[CostTerm]

    def objective(self, trade: cp.Expression, ctx: PlannerContext, date_index: int) -> cp.Expression:
        total: cp.Expression | float = 0.0
        for term in self.terms:
            total = total + term.objective(trade, ctx, date_index)
        return total


@dataclass(frozen=True)
class QuadraticParticipationImpact:
    """
    Simple convex impact model.

    The coefficient is calibrated so trading 10% ADV costs roughly
    impact_bps_at_10pct_adv times traded dollars, then represented as a
    quadratic cost in shares.
    """

    impact_bps_at_10pct_adv: float = 5.0

    def objective(self, trade: cp.Expression, ctx: PlannerContext, date_index: int) -> cp.Expression:
        price = ctx.price[date_index]
        adv = safe_numeric(ctx.adv_shares[date_index])
        eta = _impact_coefficient_in_context_units(
            self.impact_bps_at_10pct_adv,
            price,
            adv,
            ctx,
        )
        return cp.sum(cp.multiply(eta, cp.square(trade)))


@dataclass(frozen=True)
class LinearBpsCost:
    """Linear traded-dollar cost for spread, commission, fees, or soft penalties."""

    bps: float = 1.0

    def objective(self, trade: cp.Expression, ctx: PlannerContext, date_index: int) -> cp.Expression:
        price = ctx.price[date_index]
        return (self.bps / 10000.0) * cp.sum(cp.multiply(price, cp.abs(trade)))


@dataclass(frozen=True)
class TCAQuadraticParticipationImpact:
    """Date-by-name quadratic impact calibrated from TCA forecasts."""

    impact_bps_at_10pct_adv: np.ndarray

    def objective(self, trade: cp.Expression, ctx: PlannerContext, date_index: int) -> cp.Expression:
        forecasts = _cost_forecast_row(
            self.impact_bps_at_10pct_adv,
            ctx,
            date_index,
            "impact_bps_at_10pct_adv",
        )
        price = ctx.price[date_index]
        adv = safe_numeric(ctx.adv_shares[date_index])
        eta = _impact_coefficient_in_context_units(
            forecasts,
            price,
            adv,
            ctx,
        )
        return cp.sum(cp.multiply(eta, cp.square(trade)))


@dataclass(frozen=True)
class TCALinearBpsCost:
    """Date-by-name spread, commission, fee, and borrow-like linear cost."""

    linear_cost_bps: np.ndarray

    def objective(self, trade: cp.Expression, ctx: PlannerContext, date_index: int) -> cp.Expression:
        forecasts = _cost_forecast_row(
            self.linear_cost_bps,
            ctx,
            date_index,
            "linear_cost_bps",
        )
        price = ctx.price[date_index]
        return cp.sum(cp.multiply((forecasts / 10_000.0) * price, cp.abs(trade)))


def _cost_forecast_row(
    values: np.ndarray,
    ctx: PlannerContext,
    date_index: int,
    name: str,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    expected_shape = (len(ctx.dates), len(ctx.symbols))
    if matrix.shape != expected_shape:
        raise ValueError(f"{name} shape {matrix.shape} does not match {expected_shape}")
    row = matrix[date_index]
    if not np.all(np.isfinite(row)) or np.any(row < 0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return row


def _impact_coefficient_in_context_units(
    impact_bps_at_10pct_adv: float | np.ndarray,
    price: np.ndarray,
    adv: np.ndarray,
    ctx: PlannerContext,
) -> np.ndarray:
    """Return exact impact coefficients for shares or scaled parent units."""

    raw_scale = ctx.metadata.get("numerical_share_scale")
    if raw_scale is None:
        return (
            np.asarray(impact_bps_at_10pct_adv, dtype=float)
            / 10_000.0
            * price
            / safe_numeric(0.10 * adv)
        )
    share_scale = np.asarray(raw_scale, dtype=float)
    if share_scale.shape != (len(ctx.symbols),) or np.any(share_scale <= 0.0):
        raise ValueError("numerical_share_scale must be one positive value per symbol")
    original_price = np.asarray(price, dtype=float) / share_scale
    original_adv = np.asarray(adv, dtype=float) * share_scale
    share_coefficient = (
        np.asarray(impact_bps_at_10pct_adv, dtype=float)
        / 10_000.0
        * original_price
        / safe_numeric(0.10 * original_adv)
    )
    return share_coefficient * np.square(share_scale)


@dataclass(frozen=True)
class EarningsLinearPenalty:
    """
    Optional soft penalty for trading near earnings.

    This is separate from participation caps. Use it when the desk prefers to
    avoid event windows but still wants the optimizer to trade there if needed.
    """

    max_bps: float = 5.0
    tau_days: float = 5.0

    def objective(self, trade: cp.Expression, ctx: PlannerContext, date_index: int) -> cp.Expression:
        d = ctx.event_days.iloc[date_index].to_numpy(float)
        weight = np.zeros(len(ctx.symbols), dtype=float)
        finite = np.isfinite(d)
        weight[finite] = np.exp(-d[finite] / self.tau_days)
        price = ctx.price[date_index]
        return cp.sum(cp.multiply((self.max_bps / 10000.0) * weight * price, cp.abs(trade)))
