"""Expected-return objective plugins for accumulated execution inventory."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Protocol

import cvxpy as cp
import numpy as np

from .context import PlannerContext


class InventoryAlphaModel(Protocol):
    """Plugin that prices expected P&L earned by accumulated inventory."""

    def objective(
        self,
        position_shares: cp.Expression,
        ctx: PlannerContext,
        date_index: int,
    ) -> cp.Expression:
        ...


@dataclass(frozen=True)
class ExpectedReturnAlphaModel:
    """Reward inventory for probability-weighted expected close-to-close return.

    ``ctx.expected_return[t, i]`` is the expected return earned after planner
    date ``t`` by one dollar of symbol ``i`` exposure. Positive accumulated
    inventory in a positive-alpha name and negative inventory in a
    negative-alpha name both reduce the minimization objective.

    Forecast confidence belongs in ``expected_return`` itself. Keeping the
    reward in expected-P&L dollars avoids another arbitrary tuning coefficient.
    """

    def objective(
        self,
        position_shares: cp.Expression,
        ctx: PlannerContext,
        date_index: int,
    ) -> cp.Expression:
        if ctx.expected_return is None:
            raise ValueError(
                "ExpectedReturnAlphaModel requires expected_return in PlannerContext"
            )
        expected_return = np.asarray(ctx.expected_return[date_index], dtype=float)
        if not np.all(np.isfinite(expected_return)):
            raise ValueError("expected_return must contain finite values")
        position_dollars = cp.multiply(ctx.price[date_index], position_shares)
        return -cp.sum(cp.multiply(expected_return, position_dollars))


@dataclass(frozen=True)
class ConfidenceAdjustedExpectedReturnAlphaModel:
    """Reward a one-sided lower confidence bound on expected holding alpha.

    ``expected_return_uncertainty`` is the point-in-time standard error of the
    probability-weighted return forecast.  The model subtracts the matching
    one-sided normal quantile from expected P&L, so uncertain alpha must clear a
    higher hurdle before it pulls optional flow forward.  Capacity and hard
    completion still determine when urgent names must start.
    """

    confidence: float = 0.75

    def __post_init__(self) -> None:
        if not 0.5 <= self.confidence < 1.0:
            raise ValueError("confidence must be between 0.5 inclusive and 1.0 exclusive")

    @property
    def uncertainty_multiplier(self) -> float:
        return float(NormalDist().inv_cdf(self.confidence))

    def objective(
        self,
        position_shares: cp.Expression,
        ctx: PlannerContext,
        date_index: int,
    ) -> cp.Expression:
        if ctx.expected_return is None:
            raise ValueError(
                "ConfidenceAdjustedExpectedReturnAlphaModel requires expected_return"
            )
        if ctx.expected_return_uncertainty is None:
            raise ValueError(
                "ConfidenceAdjustedExpectedReturnAlphaModel requires "
                "expected_return_uncertainty"
            )
        expected_return = np.asarray(ctx.expected_return[date_index], dtype=float)
        uncertainty = np.asarray(
            ctx.expected_return_uncertainty[date_index],
            dtype=float,
        )
        if not np.all(np.isfinite(expected_return)):
            raise ValueError("expected_return must contain finite values")
        if not np.all(np.isfinite(uncertainty)) or np.any(uncertainty < 0):
            raise ValueError(
                "expected_return_uncertainty must contain finite non-negative values"
            )
        target_sign = np.sign(
            ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
        )
        # Default direction constraints make target_sign * position non-negative.
        # Writing the lower bound as an equivalent linear return keeps the
        # production problem a QP instead of adding one conic absolute-value
        # epigraph per date and name.
        robust_return = (
            expected_return
            - self.uncertainty_multiplier * uncertainty * target_sign
        )
        position_dollars = cp.multiply(ctx.price[date_index], position_shares)
        return -cp.sum(cp.multiply(robust_return, position_dollars))
