"""Planner configuration and defaults."""

from __future__ import annotations

from dataclasses import dataclass

from .constraints import ConstraintPlugin, default_constraints
from .costs import CompositeCostModel, LinearBpsCost, QuadraticParticipationImpact
from .participation import LogisticEarningsParticipation, ParticipationCapModel
from .risk import ExponentialEarningsRiskOverlay, StaticCovarianceRiskModel


@dataclass(frozen=True)
class TradePlannerConfig:
    participation_model: ParticipationCapModel
    risk_model: StaticCovarianceRiskModel
    cost_model: CompositeCostModel
    constraints: tuple[ConstraintPlugin, ...] = default_constraints()
    residual_risk_weight: float = 1.0
    terminal_penalty: float | None = None
    solver: str = "OSQP"


def default_earnings_aware_config() -> TradePlannerConfig:
    """Reasonable default config with earnings-aware participation and risk."""
    return TradePlannerConfig(
        participation_model=ParticipationCapModel(
            modifiers=[
                LogisticEarningsParticipation(
                    h_min=0.25,
                    midpoint_days=5.0,
                    steepness=1.0,
                )
            ]
        ),
        risk_model=StaticCovarianceRiskModel(
            covariance=None,
            overlays=[
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
        solver="OSQP",
    )
