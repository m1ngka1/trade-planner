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
        eta = (self.impact_bps_at_10pct_adv / 10000.0) * price / safe_numeric(0.10 * adv)
        return cp.sum(cp.multiply(eta, cp.square(trade)))


@dataclass(frozen=True)
class LinearBpsCost:
    """Linear traded-dollar cost for spread, commission, fees, or soft penalties."""

    bps: float = 1.0

    def objective(self, trade: cp.Expression, ctx: PlannerContext, date_index: int) -> cp.Expression:
        price = ctx.price[date_index]
        return (self.bps / 10000.0) * cp.sum(cp.multiply(price, cp.abs(trade)))


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
