"""Scenario-based downside risk for accumulated execution inventory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import cvxpy as cp
import numpy as np

from .context import PlannerContext


class InventoryPathRiskModel(Protocol):
    """Plugin that prices the full path of accumulated inventory."""

    def objective(
        self,
        cumulative_positions: Sequence[cp.Expression],
        ctx: PlannerContext,
    ) -> cp.Expression:
        ...


@dataclass(frozen=True)
class ScenarioCVaRRiskModel:
    """Penalize expected loss in the worst residual-return scenarios.

    Scenario returns are centered before use because expected rebalance alpha is
    already rewarded separately by :class:`ExpectedReturnAlphaModel`.  This
    prevents a finite-sample scenario mean from becoming a hidden second alpha
    forecast.
    """

    confidence: float = 0.95

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be strictly between zero and one")

    def objective(
        self,
        cumulative_positions: Sequence[cp.Expression],
        ctx: PlannerContext,
    ) -> cp.Expression:
        scenarios, weights = centered_return_scenarios(ctx)
        if len(cumulative_positions) != len(ctx.dates):
            raise ValueError("cumulative position path must match planner dates")

        scenario_pnl: cp.Expression | np.ndarray = np.zeros(len(scenarios), dtype=float)
        for date_index, position_shares in enumerate(cumulative_positions):
            position_dollars = cp.multiply(ctx.price[date_index], position_shares)
            scenario_pnl = scenario_pnl + scenarios[:, date_index, :] @ position_dollars

        loss = -scenario_pnl
        threshold = cp.Variable(name=f"inventory_cvar_{int(100 * self.confidence)}_threshold")
        tail_loss = cp.pos(loss - threshold)
        return threshold + cp.sum(cp.multiply(weights, tail_loss)) / (1.0 - self.confidence)


def centered_return_scenarios(ctx: PlannerContext) -> tuple[np.ndarray, np.ndarray]:
    """Validate and center scenario residual returns with normalized weights."""

    if ctx.return_residual_scenarios is None:
        raise ValueError(
            "ScenarioCVaRRiskModel requires return_residual_scenarios in PlannerContext"
        )
    scenarios = np.asarray(ctx.return_residual_scenarios, dtype=float)
    expected_shape = (len(ctx.dates), len(ctx.symbols))
    if scenarios.ndim != 3 or scenarios.shape[1:] != expected_shape:
        raise ValueError(
            "return_residual_scenarios must have shape "
            f"(scenario, date, symbol) with trailing shape {expected_shape}"
        )
    if len(scenarios) < 2 or not np.all(np.isfinite(scenarios)):
        raise ValueError(
            "return_residual_scenarios must contain at least two finite scenarios"
        )

    if ctx.return_scenario_weights is None:
        weights = np.full(len(scenarios), 1.0 / len(scenarios), dtype=float)
    else:
        weights = np.asarray(ctx.return_scenario_weights, dtype=float)
        if weights.shape != (len(scenarios),):
            raise ValueError(
                "return_scenario_weights must contain one value per return scenario"
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or np.sum(weights) <= 0:
            raise ValueError("return_scenario_weights must be finite, non-negative, and nonzero")
        weights = weights / np.sum(weights)

    mean = np.einsum("s,stn->tn", weights, scenarios)
    return scenarios - mean[None, :, :], weights


def weighted_loss_var_cvar(
    pnl: np.ndarray,
    weights: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return weighted loss VaR and CVaR for a one-dimensional P&L sample."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    pnl = np.asarray(pnl, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if pnl.ndim != 1 or weights.shape != pnl.shape or len(pnl) < 2:
        raise ValueError("pnl and weights must be one-dimensional with matching length")
    if not np.all(np.isfinite(pnl)) or not np.all(np.isfinite(weights)):
        raise ValueError("pnl and weights must be finite")
    if np.any(weights < 0) or np.sum(weights) <= 0:
        raise ValueError("weights must be non-negative and nonzero")

    losses = -pnl
    order = np.argsort(losses)
    losses = losses[order]
    weights = weights[order] / np.sum(weights)
    cumulative = np.cumsum(weights)
    index = min(int(np.searchsorted(cumulative, confidence, side="left")), len(losses) - 1)
    var = float(losses[index])

    # Integrate the worst (1-confidence) probability mass, splitting the VaR
    # observation when its probability straddles the quantile boundary.
    tail_mass = 1.0 - confidence
    mass_above = float(np.sum(weights[index + 1 :]))
    var_mass = max(tail_mass - mass_above, 0.0)
    tail_sum = float(np.dot(losses[index + 1 :], weights[index + 1 :]))
    cvar = (tail_sum + var_mass * var) / tail_mass
    return var, float(cvar)
