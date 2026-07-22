"""Expected-return objective plugins for accumulated execution inventory."""

from __future__ import annotations

from dataclasses import dataclass
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
