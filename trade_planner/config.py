"""Planner configuration and defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .alpha import InventoryAlphaModel
from .constraints import ConstraintPlugin, default_constraints
from .costs import CompositeCostModel, LinearBpsCost, QuadraticParticipationImpact
from .downside import InventoryPathRiskModel
from .participation import AdaptiveAnnouncementParticipation, ParticipationCapModel
from .risk import BarraFactorRiskModel, ExponentialEarningsRiskOverlay, RiskModel


@dataclass(frozen=True)
class TradePlannerConfig:
    participation_model: ParticipationCapModel
    risk_model: RiskModel
    cost_model: CompositeCostModel
    constraints: tuple[ConstraintPlugin, ...] = default_constraints()
    residual_risk_weight: float = 1.0
    terminal_penalty: float | None = None
    solver: Any = "OSQP"
    inventory_risk_weight: float = 0.0
    inventory_alpha_model: InventoryAlphaModel | None = None
    inventory_path_risk_weight: float = 0.0
    inventory_path_risk_model: InventoryPathRiskModel | None = None

    def __post_init__(self) -> None:
        if self.residual_risk_weight < 0:
            raise ValueError("residual_risk_weight must be non-negative")
        if self.inventory_risk_weight < 0:
            raise ValueError("inventory_risk_weight must be non-negative")
        if self.inventory_path_risk_weight < 0:
            raise ValueError("inventory_path_risk_weight must be non-negative")
        if self.inventory_path_risk_weight > 0 and self.inventory_path_risk_model is None:
            raise ValueError(
                "inventory_path_risk_model is required when inventory_path_risk_weight is positive"
            )


def default_earnings_aware_config() -> TradePlannerConfig:
    """Reasonable default config with earnings-aware participation and risk."""
    return TradePlannerConfig(
        participation_model=ParticipationCapModel(
            modifiers=[
                AdaptiveAnnouncementParticipation()
            ]
        ),
        risk_model=BarraFactorRiskModel(
            specific_overlays=[
                ExponentialEarningsRiskOverlay(
                    event_vol_column="event_vol",
                    tau_days=5.0,
                )
            ],
        ),
        cost_model=CompositeCostModel(
            terms=[
                QuadraticParticipationImpact(impact_bps_at_10pct_adv=5.0),
                LinearBpsCost(bps=1.0),
            ]
        ),
        constraints=default_constraints(),
        residual_risk_weight=1.0,
        terminal_penalty=None,
        solver="CLARABEL",
    )


def default_rebalance_aware_config() -> TradePlannerConfig:
    """Reference configuration for optimizer-derived pre-event accumulation.

    Physical participation caps describe available capacity.  Accumulated
    inventory risk discourages unnecessary early positions, Barra factors make
    early hedging economically useful, and convex impact prevents a final-day
    block.  The weights are a validated synthetic starting point; production
    desks should calibrate them to their own risk and impact units.
    """
    return TradePlannerConfig(
        participation_model=ParticipationCapModel(),
        risk_model=BarraFactorRiskModel(),
        cost_model=CompositeCostModel(
            terms=[
                QuadraticParticipationImpact(impact_bps_at_10pct_adv=20.0),
                LinearBpsCost(bps=1.0),
            ]
        ),
        constraints=default_constraints(),
        residual_risk_weight=0.0,
        terminal_penalty=None,
        solver="CLARABEL",
        inventory_risk_weight=1.0,
    )
