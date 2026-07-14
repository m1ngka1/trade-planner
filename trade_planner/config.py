"""Planner configuration and defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .constraints import ConstraintPlugin, default_constraints
from .costs import CompositeCostModel, LinearBpsCost, QuadraticParticipationImpact
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
