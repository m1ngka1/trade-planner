"""Point-in-time replay of flat versus forecast event liquidity.

The candidate receives a date/name ADV forecast learned from disjoint history.
Capacity and impact then cause the optimizer to choose the schedule; no daily
trade amount or target volume curve is constrained directly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import NormalDist

import cvxpy as cp
import numpy as np
import pandas as pd

from trade_planner import (
    BarraFactorRiskModel,
    DEFAULT_RISK_PREFERENCES,
    PlannerContext,
    PointInTimeRebalanceEvent,
    RebalanceRiskMeasure,
    RiskAversion,
    TradePlanner,
    build_rebalance_frontier,
    evaluate_rebalance_schedule,
    evaluate_realized_rebalance_schedule,
)
from trade_planner.downside import centered_return_scenarios

from experiments.alpha_confidence_walkforward import (
    LAMBDA_MULTIPLIERS,
    _build_event_with_truth,
    _calibrated_alpha_uncertainty,
)
from experiments.rebalance_economic_calibration import (
    EVENT_LIQUIDITY_CURVES,
    _behavior_metrics,
    economic_fixture,
)
from experiments.rolling_horizon_walkforward import (
    CAP_TOLERANCE_SHARES,
    COMPLETION_TOLERANCE_SHARES,
    DIRECTION_TOLERANCE_SHARES,
    _decision,
    _schedule_audit,
    _summary,
)


LIQUIDITY_EVENT_SEEDS = tuple(20271001 + offset for offset in range(24))
LIQUIDITY_SCENARIO_SEEDS = tuple(20271101 + offset for offset in range(24))
REALIZED_LIQUIDITY_SEEDS = tuple(20271201 + offset for offset in range(24))
LIQUIDITY_CALIBRATION_SEED = 20270901
N_LIQUIDITY_CALIBRATION_EVENTS = 2_000
EVENT_LOG_LIQUIDITY_STD = 0.10
DATE_LOG_LIQUIDITY_STD = 0.08
NAME_LOG_LIQUIDITY_STD = 0.10
LIQUIDITY_QUANTILES = {
    RiskAversion.HIGH: 0.10,
    RiskAversion.MEDIUM: 0.25,
    RiskAversion.LOW: 0.50,
}
FRESH_EVENT_INDEX_OFFSET = 24
MEAN_P95_PARTICIPATION_TOLERANCE = 0.005
MAX_PARTICIPATION_TOLERANCE = 0.01
FORECAST_PROFIT_MATERIALITY_BPS = 1.0

_all_seeds = (
    LIQUIDITY_EVENT_SEEDS
    + LIQUIDITY_SCENARIO_SEEDS
    + REALIZED_LIQUIDITY_SEEDS
    + (LIQUIDITY_CALIBRATION_SEED,)
)
assert len(set(_all_seeds)) == len(_all_seeds)


def alpha_confidence_for_risk_profile(risk_aversion: RiskAversion | str) -> float:
    """Map the desk risk budget to a one-sided optional-inventory confidence."""

    preference = DEFAULT_RISK_PREFERENCES[RiskAversion.parse(risk_aversion)]
    return 1.0 - 0.5 * preference.risk_frontier_fraction


def capacity_slack_fraction(ctx: PlannerContext) -> np.ndarray:
    """Return unused horizon-capacity fractions for each parent order."""

    target = np.abs(
        ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    )
    total_capacity = np.sum(
        ctx.base_participation
        * np.asarray(ctx.adv_shares, dtype=float)
        * np.asarray(ctx.is_open, dtype=float),
        axis=0,
    )
    utilization = np.divide(
        target,
        total_capacity,
        out=np.ones_like(target),
        where=total_capacity > 0.0,
    )
    return np.clip(1.0 - utilization, 0.0, 1.0)


@dataclass(frozen=True)
class CapacitySlackConfidenceAlphaModel:
    """Charge forecast uncertainty only where execution has capacity slack."""

    risk_aversion: RiskAversion
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.50 <= self.confidence < 1.0:
            raise ValueError(
                "confidence must be between 0.50 inclusive and 1.0 exclusive"
            )

    def objective(
        self,
        position_shares: cp.Expression,
        ctx: PlannerContext,
        date_index: int,
    ) -> cp.Expression:
        if ctx.expected_return is None:
            raise ValueError(
                "CapacitySlackConfidenceAlphaModel requires expected_return"
            )
        if ctx.expected_return_uncertainty is None:
            raise ValueError(
                "CapacitySlackConfidenceAlphaModel requires "
                "expected_return_uncertainty"
            )
        expected_return = np.asarray(ctx.expected_return[date_index], dtype=float)
        uncertainty = np.asarray(
            ctx.expected_return_uncertainty[date_index],
            dtype=float,
        )
        if not np.all(np.isfinite(expected_return)):
            raise ValueError("expected_return must contain finite values")
        if not np.all(np.isfinite(uncertainty)) or np.any(uncertainty < 0.0):
            raise ValueError(
                "expected_return_uncertainty must contain finite non-negative values"
            )
        target_sign = np.sign(
            ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
        )
        confidence = (
            alpha_confidence_for_risk_profile(self.risk_aversion)
            if self.confidence is None
            else float(self.confidence)
        )
        hurdle = (
            NormalDist().inv_cdf(confidence)
            * uncertainty
            * target_sign
            * capacity_slack_fraction(ctx)
        )
        robust_return = expected_return - hurdle
        position_dollars = cp.multiply(ctx.price[date_index], position_shares)
        return -cp.sum(cp.multiply(robust_return, position_dollars))


def factor_stress_fraction_for_risk_profile(
    risk_aversion: RiskAversion | str,
) -> float:
    """Use unused risk budget as an automatic correlation-break stress level."""

    preference = DEFAULT_RISK_PREFERENCES[RiskAversion.parse(risk_aversion)]
    return 1.0 - preference.risk_frontier_fraction


def liquidity_shape_fraction_for_risk_profile(
    risk_aversion: RiskAversion | str,
) -> float:
    """Map risk appetite to the fraction of event-liquidity shape consumed."""

    preference = DEFAULT_RISK_PREFERENCES[RiskAversion.parse(risk_aversion)]
    return 1.0 - preference.risk_frontier_fraction


def regret_weight_for_risk_profile(
    risk_aversion: RiskAversion | str,
) -> float:
    """Price one dollar of relative tail regret from the desk risk budget."""

    preference = DEFAULT_RISK_PREFERENCES[RiskAversion.parse(risk_aversion)]
    return 1.0 - preference.risk_frontier_fraction


def risk_scaled_liquidity_forecast(
    flat_adv: np.ndarray,
    forecast_adv: np.ndarray,
    risk_aversion: RiskAversion | str,
    *,
    shape_fraction: float | None = None,
) -> np.ndarray:
    """Shrink a positive ADV forecast toward flat ADV in log space."""

    flat = np.asarray(flat_adv, dtype=float)
    forecast = np.asarray(forecast_adv, dtype=float)
    if flat.shape != forecast.shape:
        raise ValueError("flat and forecast ADV must align")
    if np.any(flat <= 0.0) or np.any(forecast <= 0.0):
        raise ValueError("flat and forecast ADV must be positive")
    fraction = (
        liquidity_shape_fraction_for_risk_profile(risk_aversion)
        if shape_fraction is None
        else float(shape_fraction)
    )
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("shape_fraction must be between zero and one")
    return flat * np.exp(fraction * np.log(forecast / flat))


@dataclass(frozen=True)
class EqualFactorStressRiskModel:
    """Add equal country/sector/industry stress in dollar-variance units."""

    risk_aversion: RiskAversion
    specific_overlays: tuple[object, ...] = ()

    @staticmethod
    def factor_indices(ctx: PlannerContext) -> list[int]:
        """Retain the Barra evaluator interface for base economic reporting."""

        return BarraFactorRiskModel().factor_indices(ctx)

    def objective(
        self,
        position_shares: cp.Expression,
        ctx: PlannerContext,
        date_index: int,
    ) -> cp.Expression:
        base_risk = BarraFactorRiskModel().objective(
            position_shares,
            ctx,
            date_index,
        )
        if ctx.factor_exposure is None or ctx.factor_covariance is None:
            raise ValueError(
                "EqualFactorStressRiskModel requires factor exposures and covariance"
            )
        factor_covariance = np.asarray(
            ctx.factor_covariance[date_index],
            dtype=float,
        )
        stress_variance = (
            factor_stress_fraction_for_risk_profile(self.risk_aversion)
            * float(np.median(np.diag(factor_covariance)))
        )
        position_dollars = cp.multiply(ctx.price[date_index], position_shares)
        factor_dollars = (
            np.asarray(ctx.factor_exposure[date_index], dtype=float).T
            @ position_dollars
        )
        return base_risk + stress_variance * cp.sum_squares(factor_dollars)


@dataclass(frozen=True)
class MinimaxFactorStressRiskModel:
    """Price the largest categorical factor exposure as concentration risk."""

    risk_aversion: RiskAversion
    specific_overlays: tuple[object, ...] = ()
    stress_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.stress_fraction is not None and not 0.0 <= self.stress_fraction <= 1.0:
            raise ValueError("stress_fraction must be between zero and one")

    @staticmethod
    def factor_indices(ctx: PlannerContext) -> list[int]:
        """Retain the Barra evaluator interface for base economic reporting."""

        return BarraFactorRiskModel().factor_indices(ctx)

    def objective(
        self,
        position_shares: cp.Expression,
        ctx: PlannerContext,
        date_index: int,
    ) -> cp.Expression:
        base_risk = BarraFactorRiskModel().objective(
            position_shares,
            ctx,
            date_index,
        )
        if ctx.factor_exposure is None or ctx.factor_covariance is None:
            raise ValueError(
                "MinimaxFactorStressRiskModel requires factor exposures and covariance"
            )
        factor_covariance = np.asarray(
            ctx.factor_covariance[date_index],
            dtype=float,
        )
        fraction = (
            factor_stress_fraction_for_risk_profile(self.risk_aversion)
            if self.stress_fraction is None
            else float(self.stress_fraction)
        )
        stress_variance = fraction * float(np.median(np.diag(factor_covariance)))
        exposure = np.asarray(ctx.factor_exposure[date_index], dtype=float)
        position_dollars = cp.multiply(ctx.price[date_index], position_shares)
        factor_dollars = exposure.T @ position_dollars
        worst_factor_dollars = cp.max(cp.abs(factor_dollars))
        return (
            base_risk
            + stress_variance
            * exposure.shape[1]
            * cp.square(worst_factor_dollars)
        )


@dataclass(frozen=True)
class BaselineRelativeCVaRRiskModel:
    """Penalize forecast tail loss relative to a solved baseline schedule."""

    baseline_cumulative_shares: np.ndarray
    confidence: float = 0.95

    def __post_init__(self) -> None:
        baseline = np.asarray(self.baseline_cumulative_shares, dtype=float)
        if baseline.ndim != 2 or not np.all(np.isfinite(baseline)):
            raise ValueError(
                "baseline_cumulative_shares must be a finite date-by-name matrix"
            )
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be strictly between zero and one")

    def objective(
        self,
        cumulative_positions: tuple[cp.Expression, ...],
        ctx: PlannerContext,
    ) -> cp.Expression:
        expected_shape = (len(ctx.dates), len(ctx.symbols))
        baseline = np.asarray(self.baseline_cumulative_shares, dtype=float)
        if baseline.shape != expected_shape:
            raise ValueError(
                "baseline_cumulative_shares must align with planner dates and symbols"
            )
        if len(cumulative_positions) != len(ctx.dates):
            raise ValueError("cumulative position path must match planner dates")
        if ctx.expected_return is None:
            raise ValueError(
                "BaselineRelativeCVaRRiskModel requires expected_return"
            )
        expected_return = np.asarray(ctx.expected_return, dtype=float)
        if expected_return.shape != expected_shape or not np.all(
            np.isfinite(expected_return)
        ):
            raise ValueError("expected_return must be finite and align with the planner")
        residual_scenarios, weights = centered_return_scenarios(ctx)
        scenario_returns = residual_scenarios + expected_return[None, :, :]
        relative_pnl: cp.Expression | np.ndarray = np.zeros(
            len(scenario_returns),
            dtype=float,
        )
        for date_index, candidate_shares in enumerate(cumulative_positions):
            relative_dollars = cp.multiply(
                ctx.price[date_index],
                candidate_shares - baseline[date_index],
            )
            relative_pnl = (
                relative_pnl
                + scenario_returns[:, date_index, :] @ relative_dollars
            )
        loss = -relative_pnl
        threshold = cp.Variable(
            name=f"baseline_regret_cvar_{int(100 * self.confidence)}_threshold"
        )
        # The planner evaluates the complete objective at a feasible reference
        # schedule before solving.  Seed this auxiliary variable so the CVaR
        # expression participates in that purely numerical normalization.
        threshold.value = 0.0
        tail_loss = cp.pos(loss - threshold)
        return threshold + cp.sum(cp.multiply(weights, tail_loss)) / (
            1.0 - self.confidence
        )


@dataclass(frozen=True)
class BaselineRelativeSecondMomentRiskModel:
    """Price residual-return tracking risk versus a solved baseline plan."""

    baseline_cumulative_shares: np.ndarray

    def __post_init__(self) -> None:
        baseline = np.asarray(self.baseline_cumulative_shares, dtype=float)
        if baseline.ndim != 2 or not np.all(np.isfinite(baseline)):
            raise ValueError(
                "baseline_cumulative_shares must be a finite date-by-name matrix"
            )

    def objective(
        self,
        cumulative_positions: tuple[cp.Expression, ...],
        ctx: PlannerContext,
    ) -> cp.Expression:
        expected_shape = (len(ctx.dates), len(ctx.symbols))
        baseline = np.asarray(self.baseline_cumulative_shares, dtype=float)
        if baseline.shape != expected_shape:
            raise ValueError(
                "baseline_cumulative_shares must align with planner dates and symbols"
            )
        if len(cumulative_positions) != len(ctx.dates):
            raise ValueError("cumulative position path must match planner dates")
        residual_scenarios, weights = centered_return_scenarios(ctx)
        relative_pnl: cp.Expression | np.ndarray = np.zeros(
            len(residual_scenarios),
            dtype=float,
        )
        for date_index, candidate_shares in enumerate(cumulative_positions):
            relative_dollars = cp.multiply(
                ctx.price[date_index],
                candidate_shares - baseline[date_index],
            )
            relative_pnl = (
                relative_pnl
                + residual_scenarios[:, date_index, :] @ relative_dollars
            )
        return cp.sum(cp.multiply(weights, cp.square(relative_pnl)))


def cumulative_schedule_shares(
    ctx: PlannerContext,
    schedule: pd.DataFrame,
) -> np.ndarray:
    """Align a solved schedule and return its cumulative share inventory."""

    required = {"date", "symbol", "trade_shares"}
    missing = required.difference(schedule.columns)
    if missing:
        raise ValueError(f"schedule is missing columns: {sorted(missing)}")
    trades = (
        schedule.pivot_table(
            index="date",
            columns="symbol",
            values="trade_shares",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=ctx.dates, columns=ctx.symbols, fill_value=0.0)
        .to_numpy(float)
    )
    if not np.all(np.isfinite(trades)):
        raise ValueError("schedule trade_shares must be finite")
    return np.cumsum(trades, axis=0)


@dataclass(frozen=True)
class ExpectedNetPnlFloorConstraint:
    """Keep raw forecast net P&L inside the desk materiality band."""

    floor_dollars: float
    normalization_dollars: float
    impact_bps_at_10pct_adv: np.ndarray
    linear_cost_bps: np.ndarray

    def constraints(self, ctx: PlannerContext, state: object) -> list[cp.Constraint]:
        if ctx.expected_return is None:
            raise ValueError(
                "ExpectedNetPnlFloorConstraint requires expected_return"
            )
        if self.normalization_dollars <= 0.0:
            raise ValueError("profit-floor normalization must be positive")
        expected_alpha_normalized: cp.Expression | float = 0.0
        impact_cost_normalized: cp.Expression | float = 0.0
        linear_cost_normalized: cp.Expression | float = 0.0
        share_scale = np.asarray(
            getattr(state, "share_scale", np.ones(len(ctx.symbols))),
            dtype=float,
        )
        cumulative_path = getattr(
            state,
            "constraint_cumulative_trades",
            state.cumulative_trades,
        )
        trade_path = getattr(state, "constraint_trades", state.trades)
        for date_index, cumulative in enumerate(cumulative_path):
            price_units = (
                np.asarray(ctx.price[date_index], dtype=float) * share_scale
            )
            position_dollars = cp.multiply(price_units, cumulative)
            expected_alpha_normalized = expected_alpha_normalized + cp.sum(
                cp.multiply(
                    ctx.expected_return[date_index]
                    / self.normalization_dollars,
                    position_dollars,
                )
            )
            trade = trade_path[date_index, :]
            adv_units = np.maximum(
                np.asarray(ctx.adv_shares[date_index], dtype=float)
                / share_scale,
                1e-12,
            )
            eta = (
                self.impact_bps_at_10pct_adv[date_index]
                / 10_000.0
                * price_units
                / (0.10 * adv_units)
            )
            impact_cost_normalized = impact_cost_normalized + cp.sum(
                cp.multiply(
                    eta / self.normalization_dollars,
                    cp.square(trade),
                )
            )
            linear_cost_normalized = linear_cost_normalized + cp.sum(
                cp.multiply(
                    self.linear_cost_bps[date_index]
                    / 10_000.0
                    * price_units,
                    cp.abs(trade) / self.normalization_dollars,
                )
            )
        forecast_net_pnl_normalized = (
            expected_alpha_normalized
            - impact_cost_normalized
            - linear_cost_normalized
        )
        return [
            forecast_net_pnl_normalized
            >= self.floor_dollars / self.normalization_dollars
        ]


@dataclass(frozen=True)
class ExpectedHoldingAlphaFloorConstraint:
    """Keep linear forecast holding alpha inside the materiality band."""

    floor_dollars: float
    normalization_dollars: float

    def constraints(self, ctx: PlannerContext, state: object) -> list[cp.Constraint]:
        if ctx.expected_return is None:
            raise ValueError(
                "ExpectedHoldingAlphaFloorConstraint requires expected_return"
            )
        if self.normalization_dollars <= 0.0:
            raise ValueError("holding-alpha-floor normalization must be positive")
        expected_alpha_normalized: cp.Expression | float = 0.0
        share_scale = np.asarray(
            getattr(state, "share_scale", np.ones(len(ctx.symbols))),
            dtype=float,
        )
        cumulative_path = getattr(
            state,
            "constraint_cumulative_trades",
            state.cumulative_trades,
        )
        for date_index, cumulative in enumerate(cumulative_path):
            price_units = (
                np.asarray(ctx.price[date_index], dtype=float) * share_scale
            )
            position_dollars = cp.multiply(price_units, cumulative)
            expected_alpha_normalized = expected_alpha_normalized + cp.sum(
                cp.multiply(
                    ctx.expected_return[date_index]
                    / self.normalization_dollars,
                    position_dollars,
                )
            )
        return [
            expected_alpha_normalized
            >= self.floor_dollars / self.normalization_dollars
        ]


def run_experiment(
    solver: str = "OSQP",
    n_events: int = 12,
    event_start: int = 0,
    risk_aversion: str = "medium",
    liquidity_quantile: float | None = None,
    coefficient_policy: str = "reselect",
    alpha_policy: str = "raw",
    factor_policy: str = "barra",
    profit_policy: str = "none",
    plan_selection_policy: str = "always_candidate",
    liquidity_shape_policy: str = "full",
    regret_policy: str = "none",
    numerical_scaling: str = "none",
    verify_hard_constraints: bool = False,
    event_seeds: tuple[int, ...] = LIQUIDITY_EVENT_SEEDS,
    scenario_seeds: tuple[int, ...] = LIQUIDITY_SCENARIO_SEEDS,
    realized_liquidity_seeds: tuple[int, ...] = REALIZED_LIQUIDITY_SEEDS,
    event_index_offset: int = FRESH_EVENT_INDEX_OFFSET,
    development_event_count: int = 12,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Compare flat ADV with an automatically buffered event-liquidity curve."""

    if (
        n_events < 2
        or event_start < 0
        or event_start + n_events > len(event_seeds)
    ):
        raise ValueError("event_start and n_events must select at least two fresh events")
    if not (
        len(event_seeds)
        == len(scenario_seeds)
        == len(realized_liquidity_seeds)
    ):
        raise ValueError("event, scenario, and liquidity seed cohorts must align")
    if not 0 < development_event_count < len(event_seeds):
        raise ValueError("development_event_count must split the seed cohort")
    if numerical_scaling not in {"none", "per_name"}:
        raise ValueError("numerical_scaling must be 'none' or 'per_name'")
    parsed_aversion = RiskAversion.parse(risk_aversion)
    resolved_quantile = (
        LIQUIDITY_QUANTILES[parsed_aversion]
        if liquidity_quantile is None
        else float(liquidity_quantile)
    )
    if not 0.0 < resolved_quantile < 1.0:
        raise ValueError("liquidity_quantile must be strictly between zero and one")
    if coefficient_policy not in {"reselect", "baseline_locked"}:
        raise ValueError(
            "coefficient_policy must be 'reselect' or 'baseline_locked'"
        )
    if alpha_policy not in {"raw", "capacity_slack_confidence"}:
        raise ValueError(
            "alpha_policy must be 'raw' or 'capacity_slack_confidence'"
        )
    if factor_policy not in {
        "barra",
        "equal_factor_stress",
        "minimax_factor_stress",
    }:
        raise ValueError(
            "factor_policy must be 'barra', 'equal_factor_stress', or "
            "'minimax_factor_stress'"
        )
    if profit_policy not in {
        "none",
        "baseline_materiality_floor",
        "baseline_alpha_materiality_floor",
    }:
        raise ValueError(
            "profit_policy must be 'none', 'baseline_materiality_floor', or "
            "'baseline_alpha_materiality_floor'"
        )
    if plan_selection_policy not in {
        "always_candidate",
        "forecast_profit_and_hard_gate",
    }:
        raise ValueError(
            "plan_selection_policy must be 'always_candidate' or "
            "'forecast_profit_and_hard_gate'"
        )
    if liquidity_shape_policy not in {"full", "risk_scaled"}:
        raise ValueError(
            "liquidity_shape_policy must be 'full' or 'risk_scaled'"
        )
    if regret_policy not in {
        "none",
        "baseline_relative_cvar",
        "baseline_relative_second_moment",
    }:
        raise ValueError(
            "regret_policy must be 'none', 'baseline_relative_cvar', or "
            "'baseline_relative_second_moment'"
        )
    base_ctx, classifications = economic_fixture()
    alpha_uncertainty = _calibrated_alpha_uncertainty(base_ctx)
    calibration = calibrate_liquidity_distribution(base_ctx)
    full_forecast_adv = forecast_adv_for_risk_profile(
        base_ctx,
        calibration,
        parsed_aversion,
        quantile=resolved_quantile,
    )
    forecast_adv = (
        risk_scaled_liquidity_forecast(
            base_ctx.adv_shares,
            full_forecast_adv,
            parsed_aversion,
        )
        if liquidity_shape_policy == "risk_scaled"
        else full_forecast_adv
    )
    trial_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    schedule_rows: list[pd.DataFrame] = []
    daily_rows: list[pd.DataFrame] = []
    profile_rows: list[pd.DataFrame] = []
    exposure_rows: list[pd.DataFrame] = []
    forecast_rows: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, object]] = []
    frontier_rows: list[pd.DataFrame] = []

    selected = zip(
        event_seeds[event_start : event_start + n_events],
        scenario_seeds[event_start : event_start + n_events],
        realized_liquidity_seeds[event_start : event_start + n_events],
    )
    for cohort_index, (event_seed, scenario_seed, liquidity_seed) in enumerate(
        selected,
        start=event_start,
    ):
        event_index = event_index_offset + cohort_index
        event, forecast_rmse_bps, _ = _build_event_with_truth(
            base_ctx,
            alpha_uncertainty,
            event_index,
            event_seed,
            scenario_seed,
        )
        realized_multiplier = simulate_liquidity_multipliers(
            base_ctx,
            n_events=1,
            seed=liquidity_seed,
        )[0]
        realized_adv = base_ctx.adv_shares * realized_multiplier
        event = replace(event, realized_adv_shares=realized_adv)
        liquidity_ctx = replace(
            event.ctx,
            adv_shares=forecast_adv.copy(),
            metadata={
                **event.ctx.metadata,
                "liquidity_forecast_quantile": resolved_quantile,
                "liquidity_calibration_seed": LIQUIDITY_CALIBRATION_SEED,
            },
        )
        contexts = {
            "static_open_loop": event.ctx,
            "forecast_liquidity": liquidity_ctx,
        }
        event_rows: dict[str, dict[str, object]] = {}
        baseline_plan = None
        for strategy, planning_ctx in contexts.items():
            applied_alpha_policy = (
                alpha_policy
                if strategy == "forecast_liquidity"
                else "raw"
            )
            applied_factor_policy = (
                factor_policy
                if strategy == "forecast_liquidity"
                else "barra"
            )
            applied_profit_policy = (
                profit_policy
                if strategy == "forecast_liquidity"
                else "none"
            )
            applied_liquidity_shape_policy = (
                liquidity_shape_policy
                if strategy == "forecast_liquidity"
                else "flat"
            )
            applied_regret_policy = (
                regret_policy
                if strategy == "forecast_liquidity"
                else "none"
            )
            applied_regret_fraction = (
                regret_weight_for_risk_profile(parsed_aversion)
                if applied_regret_policy != "none"
                else 0.0
            )
            applied_regret_weight = 0.0
            applied_liquidity_shape_fraction = (
                (
                    liquidity_shape_fraction_for_risk_profile(parsed_aversion)
                    if liquidity_shape_policy == "risk_scaled"
                    else 1.0
                )
                if strategy == "forecast_liquidity"
                else 0.0
            )
            inventory_alpha_model = (
                CapacitySlackConfidenceAlphaModel(parsed_aversion)
                if applied_alpha_policy == "capacity_slack_confidence"
                else None
            )
            frontier = build_rebalance_frontier(
                planning_ctx,
                solver=solver,
                lambda_multipliers=LAMBDA_MULTIPLIERS,
                risk_measure=RebalanceRiskMeasure.VARIANCE,
                inventory_alpha_model=inventory_alpha_model,
                numerical_scaling=numerical_scaling,
                verify_hard_constraints=verify_hard_constraints,
            )
            frontier_plan = frontier.select(parsed_aversion)
            applied_config = frontier_plan.config
            applied_result = frontier_plan.result
            applied_metrics = frontier_plan.metrics
            applied_risk_budget = frontier_plan.risk_budget_dollars
            forecast_profit_floor_dollars = np.nan
            forecast_alpha_floor_dollars = np.nan
            requires_resolve = False
            if strategy == "static_open_loop":
                baseline_plan = frontier_plan
            else:
                if coefficient_policy == "baseline_locked":
                    if baseline_plan is None:
                        raise RuntimeError(
                            "baseline plan must be solved before candidate"
                        )
                    applied_config = replace(
                        applied_config,
                        inventory_risk_weight=(
                            baseline_plan.config.inventory_risk_weight
                        ),
                    )
                    applied_risk_budget = baseline_plan.risk_budget_dollars
                    requires_resolve = True
                if applied_factor_policy in {
                    "equal_factor_stress",
                    "minimax_factor_stress",
                }:
                    stress_model = (
                        EqualFactorStressRiskModel(parsed_aversion)
                        if applied_factor_policy == "equal_factor_stress"
                        else MinimaxFactorStressRiskModel(parsed_aversion)
                    )
                    applied_config = replace(
                        applied_config,
                        risk_model=stress_model,
                    )
                    requires_resolve = True
                if applied_regret_policy in {
                    "baseline_relative_cvar",
                    "baseline_relative_second_moment",
                }:
                    if baseline_plan is None:
                        raise RuntimeError(
                            "baseline plan must be solved before applying regret risk"
                        )
                    baseline_inventory_risk_weight = (
                        baseline_plan.config.inventory_risk_weight
                    )
                    if applied_regret_policy == "baseline_relative_cvar":
                        applied_regret_weight = applied_regret_fraction
                        regret_model = BaselineRelativeCVaRRiskModel(
                            baseline_cumulative_shares=cumulative_schedule_shares(
                                event.ctx,
                                baseline_plan.result.schedule,
                            )
                        )
                    else:
                        applied_regret_weight = (
                            applied_regret_fraction
                            * baseline_inventory_risk_weight
                        )
                        regret_model = BaselineRelativeSecondMomentRiskModel(
                            baseline_cumulative_shares=cumulative_schedule_shares(
                                event.ctx,
                                baseline_plan.result.schedule,
                            )
                        )
                    applied_config = replace(
                        applied_config,
                        inventory_path_risk_weight=applied_regret_weight,
                        inventory_path_risk_model=regret_model,
                    )
                    requires_resolve = applied_regret_weight > 0.0
                else:
                    applied_regret_weight = 0.0
                if applied_profit_policy == "baseline_materiality_floor":
                    if baseline_plan is None:
                        raise RuntimeError(
                            "baseline plan must be solved before applying profit floor"
                        )
                    parent_target = (
                        planning_ctx.orders["target_shares"]
                        .reindex(planning_ctx.symbols)
                        .to_numpy(float)
                    )
                    parent_gross = float(
                        np.sum(np.abs(parent_target * planning_ctx.price[0]))
                    )
                    materiality_dollars = (
                        FORECAST_PROFIT_MATERIALITY_BPS
                        / 10_000.0
                        * parent_gross
                    )
                    forecast_profit_floor_dollars = (
                        baseline_plan.metrics.expected_net_pnl_dollars
                        - materiality_dollars
                    )
                    applied_config = replace(
                        applied_config,
                        constraints=applied_config.constraints
                        + (
                            ExpectedNetPnlFloorConstraint(
                                floor_dollars=forecast_profit_floor_dollars,
                                normalization_dollars=materiality_dollars,
                                impact_bps_at_10pct_adv=(
                                    frontier.impact_bps_matrix
                                ),
                                linear_cost_bps=frontier.linear_cost_bps_matrix,
                            ),
                        ),
                    )
                    requires_resolve = True
                elif (
                    applied_profit_policy
                    == "baseline_alpha_materiality_floor"
                ):
                    if baseline_plan is None:
                        raise RuntimeError(
                            "baseline plan must be solved before applying alpha floor"
                        )
                    parent_target = (
                        planning_ctx.orders["target_shares"]
                        .reindex(planning_ctx.symbols)
                        .to_numpy(float)
                    )
                    parent_gross = float(
                        np.sum(np.abs(parent_target * planning_ctx.price[0]))
                    )
                    materiality_dollars = (
                        FORECAST_PROFIT_MATERIALITY_BPS
                        / 10_000.0
                        * parent_gross
                    )
                    forecast_alpha_floor_dollars = (
                        baseline_plan.metrics.expected_alpha_dollars
                        - materiality_dollars
                    )
                    applied_config = replace(
                        applied_config,
                        constraints=applied_config.constraints
                        + (
                            ExpectedHoldingAlphaFloorConstraint(
                                floor_dollars=forecast_alpha_floor_dollars,
                                normalization_dollars=materiality_dollars,
                            ),
                        ),
                    )
                    requires_resolve = True
            if requires_resolve:
                applied_result = TradePlanner(applied_config).solve(planning_ctx)
                applied_metrics = evaluate_rebalance_schedule(
                    planning_ctx,
                    applied_result.schedule,
                    risk_model=applied_config.risk_model,
                    impact_bps_at_10pct_adv=frontier.impact_bps_matrix,
                    linear_cost_bps=frontier.linear_cost_bps_matrix,
                )
            selected_ctx = planning_ctx
            selected_plan_source = strategy
            selection_reason = "Policy selects this optimizer plan directly."
            selection_profit_floor_dollars = np.nan
            selection_candidate_forecast_net_pnl_dollars = np.nan
            selection_candidate_profit_slack_dollars = np.nan
            selection_candidate_hard_passed = True
            if (
                strategy == "forecast_liquidity"
                and plan_selection_policy
                == "forecast_profit_and_hard_gate"
            ):
                if baseline_plan is None:
                    raise RuntimeError(
                        "baseline plan must be solved before plan selection"
                    )
                parent_target = (
                    planning_ctx.orders["target_shares"]
                    .reindex(planning_ctx.symbols)
                    .to_numpy(float)
                )
                parent_gross = float(
                    np.sum(np.abs(parent_target * planning_ctx.price[0]))
                )
                selection_profit_floor_dollars = (
                    baseline_plan.metrics.expected_net_pnl_dollars
                    - FORECAST_PROFIT_MATERIALITY_BPS
                    / 10_000.0
                    * parent_gross
                )
                selection_candidate_forecast_net_pnl_dollars = (
                    applied_metrics.expected_net_pnl_dollars
                )
                selection_candidate_profit_slack_dollars = (
                    selection_candidate_forecast_net_pnl_dollars
                    - selection_profit_floor_dollars
                )
                candidate_audit = _schedule_audit(
                    planning_ctx,
                    applied_result.schedule,
                )
                selection_candidate_hard_passed = bool(
                    candidate_audit["max_cap_excess_shares"]
                    <= CAP_TOLERANCE_SHARES
                    and candidate_audit["max_wrong_direction_shares"]
                    <= DIRECTION_TOLERANCE_SHARES
                    and candidate_audit[
                        "terminal_completion_error_shares_audit"
                    ]
                    <= COMPLETION_TOLERANCE_SHARES
                )
                if (
                    selection_candidate_profit_slack_dollars < 0.0
                    or not selection_candidate_hard_passed
                ):
                    applied_config = baseline_plan.config
                    applied_result = baseline_plan.result
                    applied_metrics = baseline_plan.metrics
                    applied_risk_budget = baseline_plan.risk_budget_dollars
                    selected_ctx = event.ctx
                    selected_plan_source = "static_open_loop_fallback"
                    failed_checks = []
                    if selection_candidate_profit_slack_dollars < 0.0:
                        failed_checks.append("forecast profit below materiality floor")
                    if not selection_candidate_hard_passed:
                        failed_checks.append("candidate hard audit failed")
                    selection_reason = "Fallback: " + " and ".join(failed_checks) + "."
                else:
                    selected_plan_source = "forecast_liquidity"
                    selection_reason = (
                        "Forecast-liquidity plan passed profit materiality and "
                        "hard audit."
                    )
            schedule = applied_result.schedule
            realized, daily = evaluate_realized_rebalance_schedule(event, schedule)
            behavior, profile, exposures = _behavior_metrics(
                selected_ctx,
                classifications,
                schedule,
            )
            audit = _schedule_audit(selected_ctx, schedule)
            row = {
                "event_id": event.event_id,
                "as_of": pd.Timestamp(event.as_of),
                "strategy": strategy,
                "plan_selection_policy": plan_selection_policy,
                "selected_plan_source": selected_plan_source,
                "selection_reason": selection_reason,
                "selection_profit_floor_dollars": (
                    selection_profit_floor_dollars
                ),
                "selection_candidate_forecast_net_pnl_dollars": (
                    selection_candidate_forecast_net_pnl_dollars
                ),
                "selection_candidate_profit_slack_dollars": (
                    selection_candidate_profit_slack_dollars
                ),
                "selection_candidate_hard_passed": (
                    selection_candidate_hard_passed
                ),
                "coefficient_policy": coefficient_policy,
                "alpha_policy": applied_alpha_policy,
                "factor_policy": applied_factor_policy,
                "profit_policy": applied_profit_policy,
                "liquidity_shape_policy": applied_liquidity_shape_policy,
                "regret_policy": applied_regret_policy,
                "regret_fraction": applied_regret_fraction,
                "regret_weight": applied_regret_weight,
                "liquidity_shape_fraction": (
                    applied_liquidity_shape_fraction
                ),
                "forecast_profit_floor_dollars": (
                    forecast_profit_floor_dollars
                ),
                "forecast_profit_floor_slack_dollars": (
                    applied_metrics.expected_net_pnl_dollars
                    - forecast_profit_floor_dollars
                    if np.isfinite(forecast_profit_floor_dollars)
                    else np.nan
                ),
                "forecast_alpha_floor_dollars": forecast_alpha_floor_dollars,
                "forecast_alpha_floor_slack_dollars": (
                    applied_metrics.expected_alpha_dollars
                    - forecast_alpha_floor_dollars
                    if np.isfinite(forecast_alpha_floor_dollars)
                    else np.nan
                ),
                "factor_stress_fraction": (
                    factor_stress_fraction_for_risk_profile(parsed_aversion)
                    if applied_factor_policy != "barra"
                    else 0.0
                ),
                "alpha_confidence": (
                    alpha_confidence_for_risk_profile(parsed_aversion)
                    if applied_alpha_policy == "capacity_slack_confidence"
                    else 0.50
                ),
                "mean_capacity_slack_fraction": float(
                    np.mean(capacity_slack_fraction(selected_ctx))
                ),
                "solver": applied_config.solver,
                "risk_aversion": parsed_aversion.value,
                "liquidity_forecast_quantile": (
                    0.50
                    if strategy == "static_open_loop"
                    else resolved_quantile
                ),
                "forecast_rmse_bps": forecast_rmse_bps,
                "selected_inventory_risk_weight": (
                    applied_config.inventory_risk_weight
                ),
                "forecast_expected_net_pnl_dollars": (
                    applied_metrics.expected_net_pnl_dollars
                ),
                **realized.as_dict(),
                **behavior,
                **audit,
            }
            event_rows[strategy] = row
            trial_rows.append(row)
            schedule_rows.append(
                schedule.assign(
                    event_id=event.event_id,
                    strategy=strategy,
                    selected_plan_source=selected_plan_source,
                )
            )
            daily_rows.append(
                daily.assign(
                    event_id=event.event_id,
                    strategy=strategy,
                    selected_plan_source=selected_plan_source,
                )
            )
            profile_rows.append(
                profile.assign(
                    event_id=event.event_id,
                    strategy=strategy,
                    selected_plan_source=selected_plan_source,
                    day_index=np.arange(1, len(profile) + 1),
                )
            )
            exposure_rows.append(
                exposures.assign(
                    event_id=event.event_id,
                    strategy=strategy,
                    selected_plan_source=selected_plan_source,
                )
            )
            coefficient_rows.append(
                {
                    "event_id": event.event_id,
                    "strategy": strategy,
                    "plan_selection_policy": plan_selection_policy,
                    "selected_plan_source": selected_plan_source,
                    "selection_reason": selection_reason,
                    "coefficient_policy": coefficient_policy,
                    "alpha_policy": applied_alpha_policy,
                    "factor_policy": applied_factor_policy,
                    "profit_policy": applied_profit_policy,
                    "liquidity_shape_policy": (
                        applied_liquidity_shape_policy
                    ),
                    "regret_policy": applied_regret_policy,
                    "regret_fraction": applied_regret_fraction,
                    "regret_weight": applied_regret_weight,
                    "liquidity_shape_fraction": (
                        applied_liquidity_shape_fraction
                    ),
                    "forecast_profit_floor_dollars": (
                        forecast_profit_floor_dollars
                    ),
                    "forecast_profit_floor_slack_dollars": (
                        applied_metrics.expected_net_pnl_dollars
                        - forecast_profit_floor_dollars
                        if np.isfinite(forecast_profit_floor_dollars)
                        else np.nan
                    ),
                    "forecast_alpha_floor_dollars": (
                        forecast_alpha_floor_dollars
                    ),
                    "forecast_alpha_floor_slack_dollars": (
                        applied_metrics.expected_alpha_dollars
                        - forecast_alpha_floor_dollars
                        if np.isfinite(forecast_alpha_floor_dollars)
                        else np.nan
                    ),
                    "risk_aversion": parsed_aversion.value,
                    "liquidity_forecast_quantile": (
                        0.50
                        if strategy == "static_open_loop"
                        else resolved_quantile
                    ),
                    "frontier_selected_inventory_risk_weight": (
                        frontier_plan.config.inventory_risk_weight
                    ),
                    "inventory_risk_weight": (
                        applied_config.inventory_risk_weight
                    ),
                    "risk_budget_dollars": applied_risk_budget,
                }
            )
            frontier_rows.append(
                frontier.frontier.assign(
                    event_id=event.event_id,
                    strategy=strategy,
                    plan_selection_policy=plan_selection_policy,
                    selected_plan_source=selected_plan_source,
                    coefficient_policy=coefficient_policy,
                    alpha_policy=applied_alpha_policy,
                    factor_policy=applied_factor_policy,
                    profit_policy=applied_profit_policy,
                    liquidity_shape_policy=(
                        applied_liquidity_shape_policy
                    ),
                    regret_policy=applied_regret_policy,
                    regret_fraction=applied_regret_fraction,
                    regret_weight=applied_regret_weight,
                    liquidity_shape_fraction=(
                        applied_liquidity_shape_fraction
                    ),
                    applied_inventory_risk_weight=(
                        applied_config.inventory_risk_weight
                    ),
                )
            )

        forecast_rows.append(
            _liquidity_audit_frame(
                event,
                base_ctx.adv_shares,
                forecast_adv,
                realized_adv,
                calibration,
            )
        )
        baseline = event_rows["static_open_loop"]
        candidate = event_rows["forecast_liquidity"]
        paired_rows.append(
            {
                "event_id": event.event_id,
                "as_of": pd.Timestamp(event.as_of),
                "realized_net_pnl_delta_dollars": (
                    candidate["net_pnl_dollars"] - baseline["net_pnl_dollars"]
                ),
                "realized_net_pnl_delta_bps": (
                    candidate["net_pnl_bps"] - baseline["net_pnl_bps"]
                ),
                "realized_impact_cost_delta_dollars": (
                    candidate["impact_cost_dollars"]
                    - baseline["impact_cost_dollars"]
                ),
                "within_event_drawdown_delta_bps": (
                    candidate["within_event_max_drawdown_bps"]
                    - baseline["within_event_max_drawdown_bps"]
                ),
                "early_factor_imbalance_delta_pp": (
                    candidate["early_factor_imbalance_pct"]
                    - baseline["early_factor_imbalance_pct"]
                ),
                "late_early_ratio_delta": (
                    candidate["late_early_gross_ratio"]
                    - baseline["late_early_gross_ratio"]
                ),
                "daily_gross_spearman_delta": (
                    candidate["daily_gross_spearman"]
                    - baseline["daily_gross_spearman"]
                ),
                "nondecreasing_transitions_delta": (
                    candidate["nondecreasing_transitions"]
                    - baseline["nondecreasing_transitions"]
                ),
                "urgent_start_delta_days": (
                    candidate["urgent_first_trade_day"]
                    - baseline["urgent_first_trade_day"]
                ),
                "small_start_delta_days": (
                    candidate["small_first_trade_day"]
                    - baseline["small_first_trade_day"]
                ),
                "p95_realized_participation_delta": (
                    candidate["p95_realized_participation_rate"]
                    - baseline["p95_realized_participation_rate"]
                ),
                "max_realized_participation_delta": (
                    candidate["max_realized_participation_rate"]
                    - baseline["max_realized_participation_rate"]
                ),
            }
        )

    trials = pd.DataFrame(trial_rows)
    paired = pd.DataFrame(paired_rows)
    summary = _liquidity_summary(trials)
    gate_decision, gate_reason, gates = _liquidity_decision(summary, paired)
    if event_start >= development_event_count:
        decision = (
            "holdout_pass"
            if gate_decision == "keep_for_holdout"
            else "holdout_fail"
        )
        reason = (
            "Untouched liquidity holdout passed every predeclared gate."
            if decision == "holdout_pass"
            else "Untouched liquidity holdout failed: "
            + gate_reason.removeprefix("Failed: ")
        )
    elif (
        event_start == 0
        and event_start + n_events > development_event_count
    ):
        decision = "descriptive_only"
        reason = (
            "Combined development and holdout report; the separate untouched "
            "holdout decision controls production promotion."
        )
    else:
        decision = gate_decision
        reason = gate_reason
    summary["decision"] = np.where(
        summary["strategy"] == "forecast_liquidity",
        decision,
        "baseline",
    )
    summary["decision_reason"] = np.where(
        summary["strategy"] == "forecast_liquidity",
        reason,
        "Current flat-ADV medium-risk policy.",
    )
    outputs = {
        "trials": trials,
        "paired": paired,
        "summary": summary,
        "gates": gates,
        "schedules": pd.concat(schedule_rows, ignore_index=True),
        "daily": pd.concat(daily_rows, ignore_index=True),
        "profiles": pd.concat(profile_rows, ignore_index=True),
        "exposures": pd.concat(exposure_rows, ignore_index=True),
        "liquidity": pd.concat(forecast_rows, ignore_index=True),
        "coefficients": pd.DataFrame(coefficient_rows),
        "frontiers": pd.concat(frontier_rows, ignore_index=True),
    }
    metadata: dict[str, object] = {
        "decision": decision,
        "decision_reason": reason,
        "solver": solver,
        "n_events": n_events,
        "event_start": event_start,
        "risk_aversion": parsed_aversion.value,
        "liquidity_forecast_quantile": resolved_quantile,
        "coefficient_policy": coefficient_policy,
        "alpha_policy": alpha_policy,
        "factor_policy": factor_policy,
        "profit_policy": profit_policy,
        "plan_selection_policy": plan_selection_policy,
        "liquidity_shape_policy": liquidity_shape_policy,
        "regret_policy": regret_policy,
        "numerical_scaling": numerical_scaling,
        "verify_hard_constraints": verify_hard_constraints,
        "regret_fraction": (
            regret_weight_for_risk_profile(parsed_aversion)
            if regret_policy != "none"
            else 0.0
        ),
        "regret_weight": (
            regret_weight_for_risk_profile(parsed_aversion)
            if regret_policy == "baseline_relative_cvar"
            else np.nan
            if regret_policy == "baseline_relative_second_moment"
            else 0.0
        ),
        "liquidity_shape_fraction": (
            liquidity_shape_fraction_for_risk_profile(parsed_aversion)
            if liquidity_shape_policy == "risk_scaled"
            else 1.0
        ),
        "forecast_profit_materiality_bps": (
            FORECAST_PROFIT_MATERIALITY_BPS
            if (
                profit_policy != "none"
                or plan_selection_policy
                == "forecast_profit_and_hard_gate"
            )
            else np.nan
        ),
        "factor_stress_fraction": (
            factor_stress_fraction_for_risk_profile(parsed_aversion)
            if factor_policy != "barra"
            else 0.0
        ),
        "alpha_confidence": (
            alpha_confidence_for_risk_profile(parsed_aversion)
            if alpha_policy == "capacity_slack_confidence"
            else 0.50
        ),
        "event_index_offset": event_index_offset,
        "development_event_count": development_event_count,
        "holdout_untouched": (
            event_start + n_events <= development_event_count
        ),
    }
    return outputs, metadata


def simulate_liquidity_multipliers(
    ctx: PlannerContext,
    *,
    n_events: int,
    seed: int,
) -> np.ndarray:
    """Generate realized ADV multipliers for the fixed replay population."""

    if n_events <= 0:
        raise ValueError("n_events must be positive")
    population_curve = np.asarray(
        EVENT_LIQUIDITY_CURVES["medium_event_liquidity"],
        dtype=float,
    )
    if population_curve.shape != (len(ctx.dates),):
        raise ValueError("event-liquidity population must align with planner dates")
    rng = np.random.default_rng(seed)
    event_shock = rng.normal(
        0.0,
        EVENT_LOG_LIQUIDITY_STD,
        size=(n_events, 1, 1),
    )
    date_shock = rng.normal(
        0.0,
        DATE_LOG_LIQUIDITY_STD,
        size=(n_events, len(ctx.dates), 1),
    )
    name_shock = rng.normal(
        0.0,
        NAME_LOG_LIQUIDITY_STD,
        size=(n_events, len(ctx.dates), len(ctx.symbols)),
    )
    return np.exp(
        np.log(population_curve)[None, :, None]
        + event_shock
        + date_shock
        + name_shock
    )


def calibrate_liquidity_distribution(ctx: PlannerContext) -> dict[str, np.ndarray]:
    """Estimate date/name log-liquidity moments from disjoint history."""

    samples = simulate_liquidity_multipliers(
        ctx,
        n_events=N_LIQUIDITY_CALIBRATION_EVENTS,
        seed=LIQUIDITY_CALIBRATION_SEED,
    )
    log_samples = np.log(samples)
    return {
        "log_mean": np.mean(log_samples, axis=0),
        "log_std": np.std(log_samples, axis=0, ddof=1),
    }


def forecast_adv_for_risk_profile(
    ctx: PlannerContext,
    calibration: dict[str, np.ndarray],
    risk_aversion: RiskAversion,
    *,
    quantile: float | None = None,
) -> np.ndarray:
    """Convert a desk risk label into a lower-quantile ADV forecast."""

    log_mean = np.asarray(calibration["log_mean"], dtype=float)
    log_std = np.asarray(calibration["log_std"], dtype=float)
    expected_shape = (len(ctx.dates), len(ctx.symbols))
    if log_mean.shape != expected_shape or log_std.shape != expected_shape:
        raise ValueError("liquidity calibration must align with planner context")
    resolved_quantile = (
        LIQUIDITY_QUANTILES[RiskAversion.parse(risk_aversion)]
        if quantile is None
        else float(quantile)
    )
    if not 0.0 < resolved_quantile < 1.0:
        raise ValueError("quantile must be strictly between zero and one")
    z_score = NormalDist().inv_cdf(resolved_quantile)
    multiplier = np.exp(log_mean + z_score * log_std)
    base_adv = np.asarray(ctx.adv_shares, dtype=float)
    return base_adv * multiplier


def _liquidity_summary(trials: pd.DataFrame) -> pd.DataFrame:
    summary = _summary(trials)
    liquidity_rows = []
    for strategy, group in trials.groupby("strategy", sort=False):
        liquidity_rows.append(
            {
                "strategy": strategy,
                "total_realized_impact_cost_dollars": float(
                    group["impact_cost_dollars"].sum()
                ),
                "mean_p95_realized_participation_rate": float(
                    group["p95_realized_participation_rate"].mean()
                ),
                "max_realized_participation_rate": float(
                    group["max_realized_participation_rate"].max()
                ),
                "max_realized_participation_excess_shares": float(
                    group["max_realized_participation_excess_shares"].max()
                ),
            }
        )
    return summary.merge(pd.DataFrame(liquidity_rows), on="strategy", how="left")


def _liquidity_decision(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
) -> tuple[str, str, pd.DataFrame]:
    base_decision, base_reason, base_gates = _decision(
        summary,
        paired,
        "forecast_liquidity",
    )
    by_strategy = summary.set_index("strategy")
    baseline = by_strategy.loc["static_open_loop"]
    candidate = by_strategy.loc["forecast_liquidity"]
    liquidity_gates = pd.DataFrame(
        [
            (
                "lower_realized_impact_cost",
                candidate["total_realized_impact_cost_dollars"]
                < baseline["total_realized_impact_cost_dollars"],
                baseline["total_realized_impact_cost_dollars"],
                candidate["total_realized_impact_cost_dollars"],
                "candidate total realized impact cost < baseline",
            ),
            (
                "p95_realized_participation_preserved",
                candidate["mean_p95_realized_participation_rate"]
                <= baseline["mean_p95_realized_participation_rate"]
                + MEAN_P95_PARTICIPATION_TOLERANCE,
                baseline["mean_p95_realized_participation_rate"]
                + MEAN_P95_PARTICIPATION_TOLERANCE,
                candidate["mean_p95_realized_participation_rate"],
                "candidate mean event p95 <= baseline + 0.5 percentage point",
            ),
            (
                "max_realized_participation_preserved",
                candidate["max_realized_participation_rate"]
                <= baseline["max_realized_participation_rate"]
                + MAX_PARTICIPATION_TOLERANCE,
                baseline["max_realized_participation_rate"]
                + MAX_PARTICIPATION_TOLERANCE,
                candidate["max_realized_participation_rate"],
                "candidate maximum <= baseline + 1 percentage point",
            ),
        ],
        columns=("gate", "passed", "baseline_or_limit", "candidate", "criterion"),
    )
    gates = pd.concat([base_gates, liquidity_gates], ignore_index=True)
    failed = gates.loc[~gates["passed"], "gate"].tolist()
    if failed:
        return "discard", "Failed: " + ", ".join(failed) + ".", gates
    if base_decision != "keep_for_holdout":
        return base_decision, base_reason, gates
    return (
        "keep_for_holdout",
        "All development economics, behavior, liquidity, and completion gates passed.",
        gates,
    )


def _liquidity_audit_frame(
    event: PointInTimeRebalanceEvent,
    flat_adv: np.ndarray,
    forecast_adv: np.ndarray,
    realized_adv: np.ndarray,
    calibration: dict[str, np.ndarray],
) -> pd.DataFrame:
    records = []
    for date_index, date in enumerate(event.ctx.dates):
        for symbol_index, symbol in enumerate(event.ctx.symbols):
            records.append(
                {
                    "event_id": event.event_id,
                    "date": pd.Timestamp(date),
                    "day_index": date_index + 1,
                    "symbol": symbol,
                    "flat_adv_shares": float(flat_adv[date_index, symbol_index]),
                    "forecast_adv_shares": float(
                        forecast_adv[date_index, symbol_index]
                    ),
                    "realized_adv_shares": float(
                        realized_adv[date_index, symbol_index]
                    ),
                    "forecast_adv_multiplier": float(
                        forecast_adv[date_index, symbol_index]
                        / flat_adv[date_index, symbol_index]
                    ),
                    "realized_adv_multiplier": float(
                        realized_adv[date_index, symbol_index]
                        / flat_adv[date_index, symbol_index]
                    ),
                    "calibrated_log_std": float(
                        calibration["log_std"][date_index, symbol_index]
                    ),
                }
            )
    return pd.DataFrame(records)


def plot_results(outputs: dict[str, pd.DataFrame], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials = outputs["trials"]
    paired = outputs["paired"].sort_values("as_of")
    summary = outputs["summary"].set_index("strategy")
    profiles = outputs["profiles"]
    liquidity = outputs["liquidity"]
    colors = {"static_open_loop": "#8A929A", "forecast_liquidity": "#2F6B9A"}
    labels = {
        "static_open_loop": "Flat ADV",
        "forecast_liquidity": "Forecast liquidity",
    }
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 9.0))

    axis = axes[0, 0]
    for strategy, group in trials.groupby("strategy", sort=False):
        ordered = group.sort_values("as_of")
        axis.plot(
            np.arange(1, len(ordered) + 1),
            ordered["net_pnl_bps"].cumsum(),
            marker="o",
            linewidth=2,
            color=colors[strategy],
            label=labels[strategy],
        )
    axis.set_title("Cumulative realized net P&L")
    axis.set_ylabel("Cumulative bps of parent gross")
    axis.legend(frameon=False)

    axis = axes[0, 1]
    axis.bar(
        np.arange(1, len(paired) + 1),
        paired["realized_net_pnl_delta_bps"],
        color=np.where(
            paired["realized_net_pnl_delta_bps"] >= 0.0,
            "#70A288",
            "#B04A4A",
        ),
    )
    axis.axhline(0.0, color="#59636E", linewidth=0.9)
    axis.set_title("Paired event P&L difference")
    axis.set_ylabel("Forecast liquidity minus flat ADV (bps)")

    axis = axes[0, 2]
    measures = ["pnl_vol_bps", "loss_cvar_95_bps", "mean_within_event_drawdown_bps"]
    x = np.arange(len(measures))
    width = 0.34
    for offset, strategy in zip((-width / 2, width / 2), colors):
        axis.bar(
            x + offset,
            [summary.loc[strategy, measure] for measure in measures],
            width,
            color=colors[strategy],
            label=labels[strategy],
        )
    axis.set_xticks(x, ["P&L vol", "Loss CVaR", "Within-event DD"])
    axis.set_title("Realized P&L swing and downside")
    axis.set_ylabel("bps")

    axis = axes[1, 0]
    mean_profiles = (
        profiles.groupby(["strategy", "day_index"], as_index=False)["daily_gross_pct"]
        .mean()
    )
    for strategy, group in mean_profiles.groupby("strategy", sort=False):
        axis.plot(
            group["day_index"],
            group["daily_gross_pct"],
            marker="o",
            linewidth=2,
            color=colors[strategy],
            label=labels[strategy],
        )
    axis.set_title("Mean optimizer-derived daily volume")
    axis.set_xlabel("Planner day")
    axis.set_ylabel("Parent gross traded (%)")

    axis = axes[1, 1]
    mean_liquidity = liquidity.groupby("day_index").agg(
        forecast=("forecast_adv_multiplier", "mean"),
        realized=("realized_adv_multiplier", "mean"),
    )
    axis.axhline(
        1.0,
        color=colors["static_open_loop"],
        linewidth=2,
        label="Flat ADV",
    )
    axis.plot(
        mean_liquidity.index,
        mean_liquidity["forecast"],
        marker="o",
        linewidth=2,
        color=colors["forecast_liquidity"],
        label="Forecast quantile",
    )
    axis.plot(
        mean_liquidity.index,
        mean_liquidity["realized"],
        linestyle="--",
        linewidth=1.7,
        color="#70A288",
        label="Realized ADV",
    )
    axis.set_title("Point-in-time forecast versus realized liquidity")
    axis.set_xlabel("Planner day")
    axis.set_ylabel("ADV multiplier")
    axis.legend(frameon=False)

    axis = axes[1, 2]
    x = np.arange(2)
    impact = [
        summary.loc[strategy, "total_realized_impact_cost_dollars"] / 1_000.0
        for strategy in colors
    ]
    axis.bar(x, impact, color=[colors[strategy] for strategy in colors])
    axis.set_xticks(x, [labels[strategy] for strategy in colors], rotation=8)
    axis.set_ylabel("Total realized impact cost ($000)")
    participation_axis = axis.twinx()
    participation_axis.plot(
        x,
        [
            100.0 * summary.loc[strategy, "mean_p95_realized_participation_rate"]
            for strategy in colors
        ],
        marker="o",
        linewidth=2,
        color="#D97732",
    )
    participation_axis.set_ylabel("Mean event p95 actual participation (%)")
    participation_axis.spines["top"].set_visible(False)
    axis.set_title("Realized cost and liquidity usage")

    for axis in axes.ravel():
        axis.grid(axis="y", color="#E1E5E8", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Point-in-time replay: event-liquidity forecast",
        x=0.04,
        y=0.995,
        ha="left",
        fontsize=16,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.995, 0.95), h_pad=2.6, w_pad=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="OSQP")
    parser.add_argument("--n-events", type=int, default=12)
    parser.add_argument("--event-start", type=int, default=0)
    parser.add_argument(
        "--risk-aversion",
        choices=("high", "medium", "low"),
        default="medium",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("artifacts/liquidity_forecast_dev"),
    )
    parser.add_argument(
        "--liquidity-quantile",
        type=float,
        default=None,
        help="Research override; production derives this from risk aversion.",
    )
    parser.add_argument(
        "--coefficient-policy",
        choices=("reselect", "baseline_locked"),
        default="reselect",
        help=(
            "Research ablation: keep the automatically selected baseline risk "
            "price fixed when applying the liquidity forecast."
        ),
    )
    parser.add_argument(
        "--alpha-policy",
        choices=("raw", "capacity_slack_confidence"),
        default="raw",
        help=(
            "Research combination: price forecast uncertainty in proportion "
            "to each order's unused capacity."
        ),
    )
    parser.add_argument(
        "--factor-policy",
        choices=("barra", "equal_factor_stress", "minimax_factor_stress"),
        default="barra",
        help=(
            "Research combination: add an automatically scaled equal-factor "
            "correlation-break stress."
        ),
    )
    parser.add_argument(
        "--profit-policy",
        choices=(
            "none",
            "baseline_materiality_floor",
            "baseline_alpha_materiality_floor",
        ),
        default="none",
        help=(
            "Research combination: preserve raw forecast net P&L within the "
            "existing desk materiality band."
        ),
    )
    parser.add_argument(
        "--plan-selection-policy",
        choices=("always_candidate", "forecast_profit_and_hard_gate"),
        default="always_candidate",
        help=(
            "Select the forecast-liquidity optimizer plan only when forecast "
            "profit materiality and hard audits pass."
        ),
    )
    parser.add_argument(
        "--liquidity-shape-policy",
        choices=("full", "risk_scaled"),
        default="full",
        help=(
            "Shrink the event-liquidity shape toward flat ADV using the "
            "automatic risk-profile fraction."
        ),
    )
    parser.add_argument(
        "--regret-policy",
        choices=(
            "none",
            "baseline_relative_cvar",
            "baseline_relative_second_moment",
        ),
        default="none",
        help=(
            "Research combination: price scenario CVaR of holding-P&L regret "
            "relative to the flat-ADV optimizer plan."
        ),
    )
    args = parser.parse_args()
    outputs, metadata = run_experiment(
        solver=args.solver,
        n_events=args.n_events,
        event_start=args.event_start,
        risk_aversion=args.risk_aversion,
        liquidity_quantile=args.liquidity_quantile,
        coefficient_policy=args.coefficient_policy,
        alpha_policy=args.alpha_policy,
        factor_policy=args.factor_policy,
        profit_policy=args.profit_policy,
        plan_selection_policy=args.plan_selection_policy,
        liquidity_shape_policy=args.liquidity_shape_policy,
        regret_policy=args.regret_policy,
    )
    prefix: Path = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(prefix.with_name(prefix.name + f"_{name}.csv"), index=False)
    chart = prefix.with_name(prefix.name + ".png")
    plot_results(outputs, chart)
    print(outputs["summary"].round(4).to_string(index=False))
    print("\nAcceptance gates:")
    print(outputs["gates"].to_string(index=False))
    print(f"\ndecision: {metadata['decision']}")
    print(f"reason: {metadata['decision_reason']}")
    print(
        "liquidity forecast quantile: "
        f"{100.0 * metadata['liquidity_forecast_quantile']:.1f}%"
    )
    print(f"coefficient policy: {metadata['coefficient_policy']}")
    print(f"alpha policy: {metadata['alpha_policy']}")
    print(f"factor policy: {metadata['factor_policy']}")
    print(f"profit policy: {metadata['profit_policy']}")
    print(f"plan selection policy: {metadata['plan_selection_policy']}")
    print(f"liquidity shape policy: {metadata['liquidity_shape_policy']}")
    print(f"regret policy: {metadata['regret_policy']}")
    print(f"holdout untouched: {metadata['holdout_untouched']}")
    print(f"artifacts: {prefix.parent / (prefix.name + '*')}")


if __name__ == "__main__":
    main()
