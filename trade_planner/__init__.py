"""Pluggable daily basket execution planner."""

from .config import TradePlannerConfig, default_earnings_aware_config
from .constraints import (
    ConstraintPlugin,
    DailyGrossNotionalLimit,
    DailyNetNotionalLimit,
    DirectionConstraint,
    FactorExposureLimit,
    HardCompletionConstraint,
    MinCompletionByDate,
    OptimizationState,
    ParticipationCapacityConstraint,
    ZeroTargetConstraint,
    default_constraints,
)
from .context import PlannerContext, days_to_next_event
from .costs import CompositeCostModel, EarningsLinearPenalty, LinearBpsCost, QuadraticParticipationImpact
from .data import (
    FactorRiskData,
    PlannerDataProvider,
    align_date_symbol_field,
    align_factor_covariance,
    align_factor_exposure,
    align_specific_variance,
    assemble_context,
    build_context_from_provider,
    build_market_panel_from_provider,
    build_planner_dates,
    normalize_orders,
)
from .participation import LogisticEarningsParticipation, ParticipationCapModel, PiecewiseEarningsParticipation
from .planner import TradePlanner, TradePlannerResult
from .risk import (
    BarraFactorRiskModel,
    ExponentialEarningsRiskOverlay,
    RiskModel,
    SpecificRiskOverlay,
    StaticCovarianceRiskModel,
)
from .types import InfeasiblePlanError

__all__ = [
    "BarraFactorRiskModel",
    "CompositeCostModel",
    "ConstraintPlugin",
    "DailyGrossNotionalLimit",
    "DailyNetNotionalLimit",
    "DirectionConstraint",
    "EarningsLinearPenalty",
    "ExponentialEarningsRiskOverlay",
    "FactorRiskData",
    "FactorExposureLimit",
    "HardCompletionConstraint",
    "InfeasiblePlanError",
    "LinearBpsCost",
    "LogisticEarningsParticipation",
    "MinCompletionByDate",
    "OptimizationState",
    "ParticipationCapacityConstraint",
    "ParticipationCapModel",
    "PiecewiseEarningsParticipation",
    "PlannerContext",
    "PlannerDataProvider",
    "QuadraticParticipationImpact",
    "RiskModel",
    "SpecificRiskOverlay",
    "StaticCovarianceRiskModel",
    "TradePlanner",
    "TradePlannerConfig",
    "TradePlannerResult",
    "ZeroTargetConstraint",
    "align_date_symbol_field",
    "align_factor_covariance",
    "align_factor_exposure",
    "align_specific_variance",
    "assemble_context",
    "build_market_panel_from_provider",
    "build_planner_dates",
    "build_context_from_provider",
    "days_to_next_event",
    "default_constraints",
    "default_earnings_aware_config",
    "normalize_orders",
]
