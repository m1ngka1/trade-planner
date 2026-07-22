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
    max_scenarios: int | None = None
    tail_probability: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be strictly between zero and one")
        if self.max_scenarios is not None:
            _validate_max_scenarios(self.max_scenarios)
        if not 0.0 < self.tail_probability < 1.0:
            raise ValueError("tail_probability must be strictly between zero and one")

    def objective(
        self,
        cumulative_positions: Sequence[cp.Expression],
        ctx: PlannerContext,
    ) -> cp.Expression:
        if self.max_scenarios is None:
            scenarios, weights = centered_return_scenarios(ctx)
        else:
            scenarios, weights = reduce_return_scenarios(
                ctx,
                max_scenarios=self.max_scenarios,
                tail_probability=self.tail_probability,
            )
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


@dataclass(frozen=True)
class TailStressPathRiskModel:
    """Penalize exposure to the scenario-derived adverse event path.

    The worst full-basket scenarios define one conditional mean return path.
    Treating tail versus non-tail observations as a centered two-state regime
    turns exposure to that path into dollar P&L variance.  This captures
    coherent cross-date call errors without one optimization variable per
    scenario.
    """

    tail_probability: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 < self.tail_probability <= 0.5:
            raise ValueError("tail_probability must be greater than zero and at most 0.5")

    def objective(
        self,
        cumulative_positions: Sequence[cp.Expression],
        ctx: PlannerContext,
    ) -> cp.Expression:
        if len(cumulative_positions) != len(ctx.dates):
            raise ValueError("cumulative position path must match planner dates")
        stress_path, regime_variance = tail_stress_return_path(
            ctx,
            tail_probability=self.tail_probability,
        )
        stress_pnl: cp.Expression | float = 0.0
        for date_index, position_shares in enumerate(cumulative_positions):
            position_dollars = cp.multiply(ctx.price[date_index], position_shares)
            stress_pnl = stress_pnl + stress_path[date_index] @ position_dollars
        return regime_variance * cp.square(stress_pnl)


@dataclass(frozen=True)
class TailSecondMomentPathRiskModel:
    """Penalize the conditional second moment of adverse scenario P&L.

    Unlike :class:`TailStressPathRiskModel`, this retains dispersion around the
    conditional mean and can represent more than one wrong-call or market-tail
    direction.  It is still a quadratic objective and needs no CVaR threshold
    or hinge variables.
    """

    tail_probability: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 < self.tail_probability <= 0.5:
            raise ValueError("tail_probability must be greater than zero and at most 0.5")

    def objective(
        self,
        cumulative_positions: Sequence[cp.Expression],
        ctx: PlannerContext,
    ) -> cp.Expression:
        if len(cumulative_positions) != len(ctx.dates):
            raise ValueError("cumulative position path must match planner dates")
        scenarios, weights = tail_return_scenarios(
            ctx,
            tail_probability=self.tail_probability,
        )
        scenario_pnl: cp.Expression | np.ndarray = np.zeros(len(scenarios), dtype=float)
        for date_index, position_shares in enumerate(cumulative_positions):
            position_dollars = cp.multiply(ctx.price[date_index], position_shares)
            scenario_pnl = scenario_pnl + scenarios[:, date_index, :] @ position_dollars
        return cp.sum(cp.multiply(weights, cp.square(scenario_pnl)))


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


def reduce_return_scenarios(
    ctx: PlannerContext,
    *,
    max_scenarios: int,
    tail_probability: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compress scenario risk while retaining the basket's adverse tail.

    Every observation in the worst ``tail_probability`` of the full-target
    path is retained when the requested limit permits it.  The remaining core
    observations are sorted by the same loss score, divided into equal-mass
    strata, and represented by the scenario nearest each stratum's weighted
    centroid.  The representative receives the stratum's full probability.

    The score depends on the supplied basket and return paths, not on a desired
    execution schedule.  The reduced set is centered again so scenario risk
    cannot become an accidental alpha forecast.
    """

    _validate_max_scenarios(max_scenarios)
    if not 0.0 < tail_probability < 1.0:
        raise ValueError("tail_probability must be strictly between zero and one")

    scenarios, weights = centered_return_scenarios(ctx)
    if len(scenarios) <= max_scenarios:
        return scenarios.copy(), weights.copy()

    target = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    reference_positions = np.asarray(ctx.price, dtype=float) * target[None, :]
    reference_loss = -np.einsum("stn,tn->s", scenarios, reference_positions)
    worst_first = np.argsort(-reference_loss, kind="stable")

    ordered_weights = weights[worst_first]
    probability_starts = np.cumsum(ordered_weights) - ordered_weights
    tail_count = int(np.sum(probability_starts < tail_probability))
    tail_count = min(max(tail_count, 1), max_scenarios - 1)
    tail_indices = worst_first[:tail_count]
    core_indices = worst_first[tail_count:]
    core_slots = max_scenarios - tail_count

    if len(core_indices) <= core_slots:
        representative_indices = np.concatenate([tail_indices, core_indices])
        reduced_weights = weights[representative_indices]
    else:
        flat = scenarios.reshape(len(scenarios), -1)
        feature_scale = np.sqrt(np.einsum("s,sf->f", weights, np.square(flat)))
        positive_scale = feature_scale[feature_scale > 1e-12]
        scale_floor = float(np.median(positive_scale)) * 1e-6 if positive_scale.size else 1.0
        feature_scale = np.maximum(feature_scale, max(scale_floor, 1e-12))

        core_weights = weights[core_indices]
        core_mass = float(np.sum(core_weights))
        core_midpoints = (np.cumsum(core_weights) - 0.5 * core_weights) / core_mass
        stratum_ids = np.minimum((core_midpoints * core_slots).astype(int), core_slots - 1)

        core_representatives: list[int] = []
        core_representative_weights: list[float] = []
        for stratum in range(core_slots):
            members = core_indices[stratum_ids == stratum]
            if not len(members):
                continue
            member_weights = weights[members]
            stratum_mass = float(np.sum(member_weights))
            centroid = np.einsum("s,sf->f", member_weights, flat[members]) / stratum_mass
            standardized_distance = np.sum(
                np.square((flat[members] - centroid[None, :]) / feature_scale[None, :]),
                axis=1,
            )
            representative = int(members[int(np.argmin(standardized_distance))])
            core_representatives.append(representative)
            core_representative_weights.append(stratum_mass)

        representative_indices = np.concatenate(
            [tail_indices, np.asarray(core_representatives, dtype=int)]
        )
        reduced_weights = np.concatenate(
            [weights[tail_indices], np.asarray(core_representative_weights, dtype=float)]
        )

    reduced_weights = reduced_weights / np.sum(reduced_weights)
    reduced = scenarios[representative_indices].copy()
    reduced_mean = np.einsum("s,stn->tn", reduced_weights, reduced)
    return reduced - reduced_mean[None, :, :], reduced_weights


def tail_stress_return_path(
    ctx: PlannerContext,
    *,
    tail_probability: float = 0.10,
) -> tuple[np.ndarray, float]:
    """Return the exact-mass adverse conditional path and regime variance.

    Scenarios are ranked by full-target path loss.  The final observation is
    split when necessary so the conditional mean always represents exactly the
    requested probability mass, including for non-uniform scenario weights.
    """

    if not 0.0 < tail_probability <= 0.5:
        raise ValueError("tail_probability must be greater than zero and at most 0.5")
    tail_scenarios, tail_weights = tail_return_scenarios(
        ctx,
        tail_probability=tail_probability,
    )
    stress_path = np.einsum("s,stn->tn", tail_weights, tail_scenarios)
    regime_variance = tail_probability / (1.0 - tail_probability)
    return stress_path, regime_variance


def tail_return_scenarios(
    ctx: PlannerContext,
    *,
    tail_probability: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return adverse full-target scenarios with exact normalized tail mass."""

    if not 0.0 < tail_probability <= 0.5:
        raise ValueError("tail_probability must be greater than zero and at most 0.5")
    scenarios, weights = centered_return_scenarios(ctx)
    target = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    reference_positions = np.asarray(ctx.price, dtype=float) * target[None, :]
    reference_loss = -np.einsum("stn,tn->s", scenarios, reference_positions)
    worst_first = np.argsort(-reference_loss, kind="stable")

    tail_weights = np.zeros_like(weights)
    remaining = tail_probability
    for scenario_index in worst_first:
        if remaining <= 1e-15:
            break
        allocated = min(float(weights[scenario_index]), remaining)
        tail_weights[scenario_index] = allocated
        remaining -= allocated
    if remaining > 1e-12:
        raise ValueError("scenario probabilities do not cover the requested tail mass")

    selected = tail_weights > 0.0
    return scenarios[selected].copy(), tail_weights[selected] / tail_probability


def _validate_max_scenarios(max_scenarios: int) -> None:
    if isinstance(max_scenarios, bool) or int(max_scenarios) != max_scenarios:
        raise ValueError("max_scenarios must be an integer of at least two")
    if max_scenarios < 2:
        raise ValueError("max_scenarios must be an integer of at least two")


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
