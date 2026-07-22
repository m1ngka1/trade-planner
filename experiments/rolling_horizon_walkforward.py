"""Point-in-time test of open-loop versus daily receding-horizon execution.

The production planner currently chooses a full-horizon schedule from one
snapshot.  This research experiment asks whether executing only today's slice,
then rebuilding the remaining-order frontier from the next causal forecast
vintage, reduces realized P&L swings without hard-coding a daily curve.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import NormalDist
from typing import Iterable

import cvxpy as cp
import numpy as np
import pandas as pd

from trade_planner import (
    CompositeCostModel,
    InfeasiblePlanError,
    PlannerContext,
    PointInTimeRebalanceEvent,
    RebalanceRiskMeasure,
    RiskAversion,
    TradePlanner,
    build_rebalance_frontier,
    evaluate_realized_rebalance_schedule,
    evaluate_rebalance_schedule,
    infer_execution_cost_matrices,
    weighted_loss_var_cvar,
)

from experiments.alpha_confidence_walkforward import (
    EVENT_SEEDS,
    LAMBDA_MULTIPLIERS,
    SCENARIO_SEEDS,
    _build_event_with_truth,
    _calibrated_alpha_uncertainty,
)
from experiments.rebalance_economic_calibration import (
    _behavior_metrics,
    economic_fixture,
)


UPDATE_SEEDS = tuple(20261201 + offset for offset in range(len(EVENT_SEEDS)))
DEFAULT_FORECAST_ERROR_RETENTION = 0.65
DEFAULT_FORECAST_INNOVATION_SCALE = 0.20
CAP_TOLERANCE_SHARES = 0.05
DIRECTION_TOLERANCE_SHARES = 0.001
COMPLETION_TOLERANCE_SHARES = 0.001
MINIMUM_REVISION_MATERIALITY_BPS = 1.0
REVISION_CONFIDENCE_LEVELS = {
    RiskAversion.HIGH: 0.99,
    RiskAversion.MEDIUM: 0.95,
    RiskAversion.LOW: 0.80,
}
assert set(UPDATE_SEEDS).isdisjoint(EVENT_SEEDS + SCENARIO_SEEDS)


@dataclass(frozen=True)
class ScheduleRevisionCost:
    """Economic shadow cost for moving dollars away from the active plan.

    The factor of one half avoids double-counting a dollar shifted from one
    date to another: the old date contributes one dollar of absolute change
    and the new date contributes another.
    """

    reference_trades: np.ndarray
    revision_cost_bps: float

    def __post_init__(self) -> None:
        reference = np.asarray(self.reference_trades, dtype=float)
        if reference.ndim != 2 or not np.all(np.isfinite(reference)):
            raise ValueError("reference_trades must be a finite date-by-symbol matrix")
        if not np.isfinite(self.revision_cost_bps) or self.revision_cost_bps < 0.0:
            raise ValueError("revision_cost_bps must be finite and non-negative")

    def objective(
        self,
        trade: cp.Expression,
        ctx: PlannerContext,
        date_index: int,
    ) -> cp.Expression:
        expected_shape = (len(ctx.dates), len(ctx.symbols))
        reference = np.asarray(self.reference_trades, dtype=float)
        if reference.shape != expected_shape:
            raise ValueError(
                f"reference_trades shape {reference.shape} does not match {expected_shape}"
            )
        changed_dollars = cp.multiply(
            ctx.price[date_index],
            trade - reference[date_index],
        )
        return (
            0.5
            * self.revision_cost_bps
            / 10_000.0
            * cp.norm1(changed_dollars)
        )


@dataclass(frozen=True)
class InventoryPathRevisionRiskModel:
    """Penalize dollar-days away from the previously selected inventory path."""

    reference_cumulative_trades: np.ndarray
    revision_cost_bps: float
    reference_scale_dollars: float

    def __post_init__(self) -> None:
        reference = np.asarray(self.reference_cumulative_trades, dtype=float)
        if reference.ndim != 2 or not np.all(np.isfinite(reference)):
            raise ValueError(
                "reference_cumulative_trades must be a finite date-by-symbol matrix"
            )
        if not np.isfinite(self.revision_cost_bps) or self.revision_cost_bps < 0.0:
            raise ValueError("revision_cost_bps must be finite and non-negative")
        if (
            not np.isfinite(self.reference_scale_dollars)
            or self.reference_scale_dollars <= 0.0
        ):
            raise ValueError("reference_scale_dollars must be finite and positive")

    def objective(
        self,
        cumulative_trades: tuple[cp.Expression, ...],
        ctx: PlannerContext,
    ) -> cp.Expression:
        expected_shape = (len(ctx.dates), len(ctx.symbols))
        reference = np.asarray(self.reference_cumulative_trades, dtype=float)
        if reference.shape != expected_shape:
            raise ValueError(
                "reference_cumulative_trades shape "
                f"{reference.shape} does not match {expected_shape}"
            )
        total: cp.Expression | float = 0.0
        for date_index, cumulative in enumerate(cumulative_trades):
            changed_inventory_dollars = cp.multiply(
                ctx.price[date_index],
                cumulative - reference[date_index],
            )
            total = total + (
                self.revision_cost_bps
                / 10_000.0
                / self.reference_scale_dollars
                * cp.sum_squares(changed_inventory_dollars)
            )
        return total


def run_experiment(
    solver: str = "OSQP",
    daily_solver: str | None = None,
    risk_measure: str = "variance",
    n_events: int = 12,
    event_start: int = 0,
    forecast_error_retention: float = DEFAULT_FORECAST_ERROR_RETENTION,
    forecast_innovation_scale: float = DEFAULT_FORECAST_INNOVATION_SCALE,
    replan_policy: str = "materiality",
    replan_threshold_bps: float = 1.0,
    risk_aversion: str = "medium",
    proximal_basis: str = "trade",
    lambda_multipliers: Iterable[float] = LAMBDA_MULTIPLIERS,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Compare one initial schedule with causal daily re-optimization."""

    if n_events < 2 or event_start < 0 or event_start + n_events > len(EVENT_SEEDS):
        raise ValueError("event_start and n_events must select at least two available events")
    if not 0.0 < forecast_error_retention < 1.0:
        raise ValueError("forecast_error_retention must be strictly between zero and one")
    if not 0.0 <= forecast_innovation_scale <= 1.0:
        raise ValueError("forecast_innovation_scale must be between zero and one")
    if replan_policy not in {"always", "materiality", "defensive", "proximal"}:
        raise ValueError(
            "replan_policy must be always, materiality, defensive, or proximal"
        )
    if not np.isfinite(replan_threshold_bps) or replan_threshold_bps < 0.0:
        raise ValueError("replan_threshold_bps must be finite and non-negative")
    if proximal_basis not in {"trade", "inventory"}:
        raise ValueError("proximal_basis must be trade or inventory")
    measure = RebalanceRiskMeasure.parse(risk_measure)
    parsed_aversion = RiskAversion.parse(risk_aversion)
    if measure not in {
        RebalanceRiskMeasure.VARIANCE,
        RebalanceRiskMeasure.HYBRID_DOWNSIDE,
    }:
        raise ValueError("risk_measure must be variance or hybrid_downside")
    multipliers = tuple(float(value) for value in lambda_multipliers)
    if not multipliers:
        raise ValueError("lambda_multipliers must not be empty")

    resolved_daily_solver = daily_solver or solver
    candidate_strategy = {
        "always": "rolling_reoptimization",
        "materiality": "commitment_aware_rolling",
        "defensive": "defensive_rolling",
        "proximal": f"proximal_{proximal_basis}_rolling",
    }[replan_policy]
    base_ctx, classifications = economic_fixture()
    alpha_uncertainty = _calibrated_alpha_uncertainty(base_ctx)
    trial_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    schedule_rows: list[pd.DataFrame] = []
    daily_rows: list[pd.DataFrame] = []
    profile_rows: list[pd.DataFrame] = []
    exposure_rows: list[pd.DataFrame] = []
    coefficient_rows: list[pd.DataFrame] = []
    forecast_rows: list[pd.DataFrame] = []

    selected = zip(
        EVENT_SEEDS[event_start : event_start + n_events],
        SCENARIO_SEEDS[event_start : event_start + n_events],
        UPDATE_SEEDS[event_start : event_start + n_events],
    )
    for event_index, (event_seed, scenario_seed, update_seed) in enumerate(
        selected,
        start=event_start,
    ):
        event, initial_rmse_bps, latent_expected_return = _build_event_with_truth(
            base_ctx,
            alpha_uncertainty,
            event_index,
            event_seed,
            scenario_seed,
        )
        initial_frontier = build_rebalance_frontier(
            event.ctx,
            solver=solver,
            lambda_multipliers=multipliers,
            risk_measure=measure,
        )
        initial_plan = initial_frontier.select(parsed_aversion)
        static_schedule = initial_plan.result.schedule
        rolling_schedule, rolling_coefficients, forecast_audit = _rolling_schedule(
            event=event,
            latent_expected_return=latent_expected_return,
            initial_uncertainty=alpha_uncertainty,
            initial_frontier=initial_frontier,
            initial_plan=initial_plan,
            update_seed=update_seed,
            solver=resolved_daily_solver,
            risk_measure=measure,
            forecast_error_retention=forecast_error_retention,
            forecast_innovation_scale=forecast_innovation_scale,
            replan_policy=replan_policy,
            replan_threshold_bps=replan_threshold_bps,
            risk_aversion=parsed_aversion,
            proximal_basis=proximal_basis,
            lambda_multipliers=multipliers,
        )
        static_coefficients = _coefficient_row(
            event=event,
            decision_day=0,
            ctx=event.ctx,
            frontier=initial_frontier,
            plan=initial_plan,
            strategy="static_open_loop",
            forecast_rmse_bps=initial_rmse_bps,
            forecast_uncertainty_bps=float(10_000.0 * np.mean(alpha_uncertainty)),
        )
        coefficient_rows.extend([static_coefficients, rolling_coefficients])
        forecast_rows.append(forecast_audit)

        strategy_schedules = {
            "static_open_loop": static_schedule,
            candidate_strategy: rolling_schedule,
        }
        event_rows: dict[str, dict[str, object]] = {}
        for strategy, schedule in strategy_schedules.items():
            realized, daily = evaluate_realized_rebalance_schedule(event, schedule)
            behavior, profile, exposures = _behavior_metrics(
                event.ctx,
                classifications,
                schedule,
            )
            audit = _schedule_audit(event.ctx, schedule)
            row = {
                "event_id": event.event_id,
                "as_of": pd.Timestamp(event.as_of),
                "strategy": strategy,
                "solver": solver,
                "daily_solver": resolved_daily_solver,
                "risk_measure": measure.value,
                "forecast_error_retention": forecast_error_retention,
                "forecast_innovation_scale": forecast_innovation_scale,
                "replan_policy": replan_policy,
                "replan_threshold_bps": replan_threshold_bps,
                "risk_aversion": parsed_aversion.value,
                "proximal_basis": proximal_basis,
                "initial_forecast_rmse_bps": initial_rmse_bps,
                **realized.as_dict(),
                **behavior,
                **audit,
            }
            event_rows[strategy] = row
            trial_rows.append(row)
            schedule_rows.append(
                schedule.assign(event_id=event.event_id, strategy=strategy)
            )
            daily_rows.append(
                daily.assign(event_id=event.event_id, strategy=strategy)
            )
            profile_rows.append(
                profile.assign(
                    event_id=event.event_id,
                    strategy=strategy,
                    day_index=np.arange(1, len(profile) + 1),
                )
            )
            exposure_rows.append(
                exposures.assign(event_id=event.event_id, strategy=strategy)
            )

        baseline = event_rows["static_open_loop"]
        candidate = event_rows[candidate_strategy]
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
            }
        )

    trials = pd.DataFrame(trial_rows)
    paired = pd.DataFrame(paired_rows)
    summary = _summary(trials)
    decision, reason, gates = _decision(summary, paired, candidate_strategy)
    summary["decision"] = np.where(
        summary["strategy"] == candidate_strategy,
        decision,
        "baseline",
    )
    summary["decision_reason"] = np.where(
        summary["strategy"] == candidate_strategy,
        reason,
        "Current one-snapshot, full-horizon medium-risk policy.",
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
        "coefficients": pd.concat(coefficient_rows, ignore_index=True),
        "forecasts": pd.concat(forecast_rows, ignore_index=True),
    }
    metadata: dict[str, object] = {
        "decision": decision,
        "decision_reason": reason,
        "risk_measure": measure.value,
        "initial_solver": solver,
        "daily_solver": resolved_daily_solver,
        "n_events": n_events,
        "event_start": event_start,
        "forecast_error_retention": forecast_error_retention,
        "forecast_innovation_scale": forecast_innovation_scale,
        "replan_policy": replan_policy,
        "replan_threshold_bps": replan_threshold_bps,
        "risk_aversion": parsed_aversion.value,
        "proximal_basis": proximal_basis,
        "candidate_strategy": candidate_strategy,
        "holdout_untouched": event_start == 0 and event_start + n_events <= 12,
    }
    return outputs, metadata


def _automatic_revision_cost_bps(
    ctx: PlannerContext,
    remaining_target: np.ndarray,
    risk_aversion: RiskAversion,
) -> float:
    """Turn forecast standard error into an investment-scale revision hurdle."""

    uncertainty = ctx.expected_return_uncertainty
    if uncertainty is None:
        return MINIMUM_REVISION_MATERIALITY_BPS
    matrix = np.asarray(uncertainty, dtype=float)
    expected_shape = (len(ctx.dates), len(ctx.symbols))
    if matrix.shape != expected_shape or not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"expected_return_uncertainty must be finite with shape {expected_shape}"
        )
    target = np.asarray(remaining_target, dtype=float)
    if target.shape != (len(ctx.symbols),):
        raise ValueError("remaining_target must contain one value per symbol")
    name_uncertainty = np.sqrt(np.mean(np.square(matrix), axis=0))
    notional_weights = np.abs(target * ctx.price[0])
    if np.sum(notional_weights) <= 1e-12:
        aggregate_uncertainty_bps = 0.0
    else:
        aggregate_uncertainty_bps = float(
            10_000.0 * np.average(name_uncertainty, weights=notional_weights)
        )
    confidence = REVISION_CONFIDENCE_LEVELS[risk_aversion]
    effective_revision_trials = max(len(ctx.dates), 1)
    simultaneous_tail_probability = (
        1.0 - confidence
    ) / effective_revision_trials
    simultaneous_quantile = NormalDist().inv_cdf(
        1.0 - simultaneous_tail_probability
    )
    return max(
        MINIMUM_REVISION_MATERIALITY_BPS,
        np.sqrt(2.0) * simultaneous_quantile
        * aggregate_uncertainty_bps,
    )


def _schedule_trade_matrix(
    ctx: PlannerContext,
    schedule: pd.DataFrame,
) -> np.ndarray:
    return (
        schedule.assign(date=pd.to_datetime(schedule["date"]).dt.normalize())
        .pivot_table(
            index="date",
            columns="symbol",
            values="trade_shares",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=ctx.dates, columns=ctx.symbols, fill_value=0.0)
        .to_numpy(float)
    )


def _solve_proximal_plan(
    *,
    ctx: PlannerContext,
    plan,
    reference_schedule: pd.DataFrame,
    revision_cost_bps: float,
    proximal_basis: str,
):
    reference_trades = _schedule_trade_matrix(ctx, reference_schedule)
    if proximal_basis == "trade":
        revision_cost = ScheduleRevisionCost(
            reference_trades=reference_trades,
            revision_cost_bps=revision_cost_bps,
        )
        config = replace(
            plan.config,
            cost_model=CompositeCostModel(
                terms=tuple(plan.config.cost_model.terms) + (revision_cost,)
            ),
        )
    elif proximal_basis == "inventory":
        if plan.config.inventory_path_risk_weight > 0.0:
            raise ValueError(
                "inventory proximal research requires a frontier without another path model"
            )
        config = replace(
            plan.config,
            inventory_path_risk_weight=1.0,
            inventory_path_risk_model=InventoryPathRevisionRiskModel(
                reference_cumulative_trades=np.cumsum(reference_trades, axis=0),
                revision_cost_bps=revision_cost_bps,
                reference_scale_dollars=max(
                    float(
                        np.sum(
                            np.abs(
                                ctx.orders["target_shares"]
                                .reindex(ctx.symbols)
                                .to_numpy(float)
                                * ctx.price[0]
                            )
                        )
                    )
                    / len(ctx.dates),
                    1.0,
                ),
            ),
        )
    else:
        raise ValueError("proximal_basis must be trade or inventory")
    try:
        result = TradePlanner(config).solve(ctx)
    except InfeasiblePlanError:
        alternate_solver = (
            "OSQP" if str(config.solver).upper() == "CLARABEL" else "CLARABEL"
        )
        config = replace(config, solver=alternate_solver)
        result = TradePlanner(config).solve(ctx)
    impact_matrix, linear_matrix = infer_execution_cost_matrices(ctx)
    metrics = evaluate_rebalance_schedule(
        ctx,
        result.schedule,
        impact_bps_at_10pct_adv=impact_matrix,
        linear_cost_bps=linear_matrix,
    )
    return replace(plan, config=config, result=result, metrics=metrics)


def _rolling_schedule(
    *,
    event: PointInTimeRebalanceEvent,
    latent_expected_return: np.ndarray,
    initial_uncertainty: np.ndarray,
    initial_frontier,
    initial_plan,
    update_seed: int,
    solver: str,
    risk_measure: RebalanceRiskMeasure,
    forecast_error_retention: float,
    forecast_innovation_scale: float,
    replan_policy: str,
    replan_threshold_bps: float,
    risk_aversion: RiskAversion,
    proximal_basis: str,
    lambda_multipliers: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute one day from each newly solved remaining-order frontier."""

    ctx = event.ctx
    target = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    remaining = target.copy()
    forecasts, uncertainties = _forecast_vintages(
        initial_forecast=np.asarray(ctx.expected_return, dtype=float),
        latent_expected_return=latent_expected_return,
        initial_uncertainty=initial_uncertainty,
        retention=forecast_error_retention,
        innovation_scale=forecast_innovation_scale,
        seed=update_seed,
    )
    schedules: list[pd.DataFrame] = []
    coefficients: list[pd.DataFrame] = []
    forecast_rows: list[dict[str, object]] = []
    active_schedule = initial_plan.result.schedule.copy()
    active_plan = initial_plan
    candidate_strategy = {
        "always": "rolling_reoptimization",
        "materiality": "commitment_aware_rolling",
        "defensive": "defensive_rolling",
        "proximal": f"proximal_{proximal_basis}_rolling",
    }[replan_policy]

    for day_index, date in enumerate(ctx.dates):
        snapshot = _slice_context(
            ctx,
            start=day_index,
            remaining_target=remaining,
            expected_return=forecasts[day_index],
            expected_return_uncertainty=uncertainties[day_index],
        )
        if day_index == 0:
            frontier = initial_frontier
            plan = initial_plan
        else:
            frontier = build_rebalance_frontier(
                snapshot,
                solver=solver,
                lambda_multipliers=lambda_multipliers,
                risk_measure=risk_measure,
            )
            try:
                plan = frontier.select(risk_aversion)
            except RuntimeError:
                fallback_solver = (
                    "OSQP" if str(solver).upper() == "CLARABEL" else "CLARABEL"
                )
                fallback_frontier = build_rebalance_frontier(
                    snapshot,
                    solver=fallback_solver,
                    lambda_multipliers=lambda_multipliers,
                    risk_measure=risk_measure,
                )
                try:
                    fallback_plan = fallback_frontier.select(risk_aversion)
                except RuntimeError as fallback_error:
                    failures = frontier.frontier[
                        ["candidate", "status", "failure_reason"]
                    ].to_dict("records")
                    fallback_failures = fallback_frontier.frontier[
                        ["candidate", "status", "failure_reason"]
                    ].to_dict("records")
                    horizon_capacity = np.sum(
                        snapshot.base_participation
                        * snapshot.adv_shares
                        * snapshot.is_open.astype(float),
                        axis=0,
                    )
                    remaining_capacity = pd.DataFrame(
                        {
                            "symbol": snapshot.symbols,
                            "remaining_target": snapshot.orders["target_shares"]
                            .reindex(snapshot.symbols)
                            .to_numpy(float),
                            "horizon_capacity": horizon_capacity,
                        }
                    )
                    remaining_capacity["shortfall"] = (
                        np.abs(remaining_capacity["remaining_target"])
                        - remaining_capacity["horizon_capacity"]
                    )
                    raise RuntimeError(
                        f"{event.event_id} decision day {day_index + 1} has no "
                        f"solved frontier candidate; remaining capacity: "
                        f"{remaining_capacity.to_dict('records')}; "
                        f"{solver} failures: {failures}; "
                        f"{fallback_solver} failures: {fallback_failures}"
                    ) from fallback_error
                frontier = fallback_frontier
                plan = fallback_plan
        carried_schedule = active_schedule.loc[
            pd.to_datetime(active_schedule["date"]).dt.normalize()
            >= pd.Timestamp(date).normalize()
        ].copy()
        replan_accepted = True
        expected_pnl_gain_bps = np.nan
        forecast_vol_reduction_bps = np.nan
        forecast_cvar_reduction_bps = np.nan
        revision_cost_bps = np.nan
        revision_notional_dollars = 0.0
        if replan_policy == "proximal" and day_index > 0:
            impact_matrix, linear_matrix = infer_execution_cost_matrices(snapshot)
            carried_metrics = evaluate_rebalance_schedule(
                snapshot,
                carried_schedule,
                impact_bps_at_10pct_adv=impact_matrix,
                linear_cost_bps=linear_matrix,
            )
            remaining_gross = float(
                np.sum(np.abs(remaining * snapshot.price[0]))
            )
            scale = 10_000.0 / max(remaining_gross, 1e-12)
            revision_cost_bps = _automatic_revision_cost_bps(
                snapshot,
                remaining,
                risk_aversion,
            )
            proximal_plan = _solve_proximal_plan(
                ctx=snapshot,
                plan=plan,
                reference_schedule=carried_schedule,
                revision_cost_bps=revision_cost_bps,
                proximal_basis=proximal_basis,
            )
            execution_schedule = proximal_plan.result.schedule
            active_schedule = execution_schedule.copy()
            active_plan = proximal_plan
            expected_pnl_gain_bps = scale * (
                proximal_plan.metrics.expected_net_pnl_dollars
                - carried_metrics.expected_net_pnl_dollars
            )
            forecast_vol_reduction_bps = scale * (
                carried_metrics.pnl_vol_dollars
                - proximal_plan.metrics.pnl_vol_dollars
            )
            forecast_cvar_reduction_bps = scale * (
                carried_metrics.loss_cvar_95_dollars
                - proximal_plan.metrics.loss_cvar_95_dollars
            )
            reference_trades = _schedule_trade_matrix(snapshot, carried_schedule)
            proximal_trades = _schedule_trade_matrix(snapshot, execution_schedule)
            revision_notional_dollars = float(
                0.5
                * np.sum(
                    np.abs(proximal_trades - reference_trades)
                    * snapshot.price
                )
            )
        elif replan_policy in {"materiality", "defensive"} and day_index > 0:
            impact_matrix, linear_matrix = infer_execution_cost_matrices(snapshot)
            carried_metrics = evaluate_rebalance_schedule(
                snapshot,
                carried_schedule,
                impact_bps_at_10pct_adv=impact_matrix,
                linear_cost_bps=linear_matrix,
            )
            remaining_gross = float(
                np.sum(np.abs(remaining * snapshot.price[0]))
            )
            scale = 10_000.0 / max(remaining_gross, 1e-12)
            expected_pnl_gain_bps = scale * (
                plan.metrics.expected_net_pnl_dollars
                - carried_metrics.expected_net_pnl_dollars
            )
            forecast_vol_reduction_bps = scale * (
                carried_metrics.pnl_vol_dollars - plan.metrics.pnl_vol_dollars
            )
            forecast_cvar_reduction_bps = scale * (
                carried_metrics.loss_cvar_95_dollars
                - plan.metrics.loss_cvar_95_dollars
            )
            replan_accepted = _material_replan_decision(
                expected_pnl_gain_bps=expected_pnl_gain_bps,
                forecast_vol_reduction_bps=forecast_vol_reduction_bps,
                forecast_cvar_reduction_bps=forecast_cvar_reduction_bps,
                threshold_bps=replan_threshold_bps,
                allow_profit_case=replan_policy == "materiality",
            )
        if replan_policy != "proximal" or day_index == 0:
            if replan_accepted:
                execution_schedule = plan.result.schedule
                active_schedule = execution_schedule.copy()
                active_plan = plan
            else:
                execution_schedule = carried_schedule
        today = execution_schedule.loc[
            pd.to_datetime(execution_schedule["date"]).dt.normalize()
            == pd.Timestamp(date).normalize()
        ].copy()
        if len(today) != len(ctx.symbols):
            raise RuntimeError("daily re-optimization did not return one row per symbol")
        raw_today_trades = (
            today.set_index("symbol")["trade_shares"]
            .reindex(ctx.symbols)
            .to_numpy(float)
        )
        today_caps = (
            today.set_index("symbol")["cap_shares"]
            .reindex(ctx.symbols)
            .to_numpy(float)
        )
        future_rows = execution_schedule.loc[
            pd.to_datetime(execution_schedule["date"]).dt.normalize()
            > pd.Timestamp(date).normalize()
        ].copy()
        if future_rows.empty:
            future_executable_capacity = np.zeros(len(ctx.symbols), dtype=float)
        else:
            future_rows["executable_cap_shares"] = np.floor(
                (future_rows["cap_shares"].to_numpy(float) + 1e-9)
            )
            future_executable_capacity = (
                future_rows.groupby("symbol")["executable_cap_shares"]
                .sum()
                .reindex(ctx.symbols, fill_value=0.0)
                .to_numpy(float)
            )
        today_trades = _executable_lot_trades(
            raw_today_trades,
            remaining,
            today_caps,
            future_executable_capacity,
        )
        trade_by_symbol = pd.Series(today_trades, index=ctx.symbols)
        residual_by_symbol = pd.Series(remaining - today_trades, index=ctx.symbols)
        today["trade_shares"] = today["symbol"].map(trade_by_symbol).to_numpy(float)
        today["trade_dollars"] = (
            today["trade_shares"].to_numpy(float)
            * today["symbol"].map(
                pd.Series(snapshot.price[0], index=ctx.symbols)
            ).to_numpy(float)
        )
        today["abs_pct_adv"] = np.abs(today["trade_shares"].to_numpy(float)) / today[
            "symbol"
        ].map(pd.Series(snapshot.adv_shares[0], index=ctx.symbols)).to_numpy(float)
        today["residual_shares_after"] = today["symbol"].map(
            residual_by_symbol
        ).to_numpy(float)
        today["residual_dollars_after"] = (
            today["residual_shares_after"].to_numpy(float)
            * today["symbol"].map(
                pd.Series(snapshot.price[0], index=ctx.symbols)
            ).to_numpy(float)
        )
        schedules.append(today)
        forecast_rmse_bps = float(
            10_000.0
            * np.sqrt(
                np.mean(
                    np.square(
                        forecasts[day_index][day_index:]
                        - latent_expected_return[day_index:]
                    )
                )
            )
        )
        forecast_uncertainty_bps = float(
            10_000.0 * np.mean(uncertainties[day_index][day_index:])
        )
        coefficient = _coefficient_row(
                event=event,
                decision_day=day_index,
                ctx=snapshot,
                frontier=frontier,
                plan=plan,
                strategy=candidate_strategy,
                forecast_rmse_bps=forecast_rmse_bps,
                forecast_uncertainty_bps=forecast_uncertainty_bps,
        )
        coefficient["replan_policy"] = replan_policy
        coefficient["replan_threshold_bps"] = replan_threshold_bps
        coefficient["replan_accepted"] = replan_accepted
        coefficient["expected_pnl_gain_bps"] = expected_pnl_gain_bps
        coefficient["forecast_vol_reduction_bps"] = forecast_vol_reduction_bps
        coefficient["forecast_cvar_reduction_bps"] = forecast_cvar_reduction_bps
        coefficient["revision_cost_bps"] = revision_cost_bps
        coefficient["revision_notional_dollars"] = revision_notional_dollars
        coefficient["active_solver"] = active_plan.config.solver
        coefficient["active_inventory_risk_weight"] = (
            active_plan.config.inventory_risk_weight
        )
        coefficient["active_path_risk_weight"] = (
            active_plan.config.inventory_path_risk_weight
        )
        coefficients.append(coefficient)
        forecast_rows.append(
            {
                "event_id": event.event_id,
                "decision_day": day_index + 1,
                "decision_date": pd.Timestamp(date),
                "information_cutoff": pd.Timestamp(date) + pd.Timedelta(hours=8),
                "remaining_dates": len(ctx.dates) - day_index,
                "remaining_parent_gross_dollars": float(
                    np.sum(np.abs(remaining * ctx.price[day_index]))
                ),
                "forecast_rmse_bps": forecast_rmse_bps,
                "forecast_uncertainty_bps": forecast_uncertainty_bps,
                "replan_accepted": replan_accepted,
                "expected_pnl_gain_bps": expected_pnl_gain_bps,
                "forecast_vol_reduction_bps": forecast_vol_reduction_bps,
                "forecast_cvar_reduction_bps": forecast_cvar_reduction_bps,
                "revision_cost_bps": revision_cost_bps,
                "revision_notional_dollars": revision_notional_dollars,
            }
        )
        # Parent orders and simulated fills are whole shares. Snap subtraction
        # back to that lattice so binary floating remnants do not poison the
        # next equality-constrained solve.
        remaining = np.rint(remaining - today_trades)

    schedule = pd.concat(schedules, ignore_index=True).sort_values(
        ["date", "symbol"]
    )
    return (
        schedule.reset_index(drop=True),
        pd.concat(coefficients, ignore_index=True),
        pd.DataFrame(forecast_rows),
    )


def _material_replan_decision(
    *,
    expected_pnl_gain_bps: float,
    forecast_vol_reduction_bps: float,
    forecast_cvar_reduction_bps: float,
    threshold_bps: float,
    allow_profit_case: bool = True,
) -> bool:
    """Accept a new plan only for a material, bounded economic improvement."""

    values = np.array(
        [
            expected_pnl_gain_bps,
            forecast_vol_reduction_bps,
            forecast_cvar_reduction_bps,
            threshold_bps,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)) or threshold_bps < 0.0:
        raise ValueError("replan economics must be finite and threshold must be non-negative")
    profit_case = (
        expected_pnl_gain_bps >= threshold_bps
        and forecast_vol_reduction_bps >= -threshold_bps
        and forecast_cvar_reduction_bps >= -threshold_bps
    )
    defensive_case = (
        (
            forecast_vol_reduction_bps >= threshold_bps
            or forecast_cvar_reduction_bps >= threshold_bps
        )
        and expected_pnl_gain_bps >= -threshold_bps
    )
    return bool((allow_profit_case and profit_case) or defensive_case)


def _executable_lot_trades(
    raw_trades: np.ndarray,
    remaining_target: np.ndarray,
    caps: np.ndarray,
    future_capacity: np.ndarray,
    *,
    lot_size: float = 1.0,
) -> np.ndarray:
    """Convert a daily solve into feasible executable lots without pacing it."""

    raw = np.asarray(raw_trades, dtype=float)
    remaining = np.asarray(remaining_target, dtype=float)
    daily_caps = np.asarray(caps, dtype=float)
    future = np.asarray(future_capacity, dtype=float)
    if (
        raw.shape != remaining.shape
        or raw.shape != daily_caps.shape
        or raw.shape != future.shape
    ):
        raise ValueError(
            "raw trades, remaining target, caps, and future capacity must align"
        )
    if not np.isfinite(lot_size) or lot_size <= 0.0:
        raise ValueError("lot_size must be finite and positive")
    rounded_abs = np.rint(np.abs(raw) / lot_size) * lot_size
    executable_cap = np.floor((daily_caps + 1e-9) / lot_size) * lot_size
    executable_remaining = np.rint(np.abs(remaining) / lot_size) * lot_size
    required_today = (
        np.ceil(
            np.maximum(executable_remaining - future, 0.0) / lot_size
            - 1e-12
        )
        * lot_size
    )
    executable_abs = np.minimum(
        np.maximum(rounded_abs, required_today),
        np.minimum(executable_cap, executable_remaining),
    )
    return np.sign(remaining) * executable_abs


def _forecast_vintages(
    *,
    initial_forecast: np.ndarray,
    latent_expected_return: np.ndarray,
    initial_uncertainty: np.ndarray,
    retention: float,
    innovation_scale: float,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Simulate noisy causal forecast revisions whose error shrinks over time.

    Latent alpha is used only by this data-generating process and hindsight
    scoring.  Each optimizer receives a noisy vintage, never latent or realized
    returns.  In production, these vintages must be replaced by stored forecasts.
    """

    initial = np.asarray(initial_forecast, dtype=float)
    latent = np.asarray(latent_expected_return, dtype=float)
    uncertainty = np.asarray(initial_uncertainty, dtype=float)
    if initial.shape != latent.shape or initial.shape != uncertainty.shape:
        raise ValueError("initial forecast, latent return, and uncertainty must align")
    rng = np.random.default_rng(seed)
    initial_error = initial - latent
    forecasts: list[np.ndarray] = []
    uncertainties: list[np.ndarray] = []
    for day_index in range(initial.shape[0]):
        decay = retention**day_index
        if day_index == 0:
            forecast = initial.copy()
        else:
            noise = (
                rng.standard_t(5.0, size=initial.shape)
                / np.sqrt(5.0 / 3.0)
            )
            forecast = (
                latent
                + decay * initial_error
                + innovation_scale * (1.0 - decay) * uncertainty * noise
            )
        posterior_scale = np.sqrt(
            decay**2 + np.square(innovation_scale * (1.0 - decay))
        )
        forecasts.append(forecast)
        uncertainties.append(uncertainty * posterior_scale)
    return forecasts, uncertainties


def _slice_context(
    ctx: PlannerContext,
    *,
    start: int,
    remaining_target: np.ndarray,
    expected_return: np.ndarray,
    expected_return_uncertainty: np.ndarray,
) -> PlannerContext:
    """Create a remaining-horizon snapshot without changing the source event."""

    if not 0 <= start < len(ctx.dates):
        raise ValueError("start must select a remaining planner date")
    target = np.asarray(remaining_target, dtype=float)
    if target.shape != (len(ctx.symbols),):
        raise ValueError("remaining_target must contain one value per symbol")
    dates = ctx.dates[start:]
    index = pd.MultiIndex.from_product(
        [dates, ctx.symbols],
        names=ctx.panel.index.names,
    )
    orders = ctx.orders.copy()
    orders.loc[ctx.symbols, "target_shares"] = target
    scenarios = ctx.return_residual_scenarios
    if scenarios is not None:
        scenarios = np.asarray(scenarios, dtype=float)[:, start:, :].copy()
    return replace(
        ctx,
        dates=dates,
        orders=orders,
        panel=ctx.panel.reindex(index).copy(),
        price=np.asarray(ctx.price, dtype=float)[start:].copy(),
        adv_shares=np.asarray(ctx.adv_shares, dtype=float)[start:].copy(),
        is_open=np.asarray(ctx.is_open)[start:].copy(),
        base_participation=np.asarray(ctx.base_participation, dtype=float)[start:].copy(),
        event_days=ctx.event_days.reindex(index=dates, columns=ctx.symbols).copy(),
        factor_exposure=_slice_optional(ctx.factor_exposure, start),
        factor_covariance=_slice_optional(ctx.factor_covariance, start),
        specific_variance=_slice_optional(ctx.specific_variance, start),
        expected_return=np.asarray(expected_return, dtype=float)[start:].copy(),
        expected_return_uncertainty=np.asarray(
            expected_return_uncertainty,
            dtype=float,
        )[start:].copy(),
        impact_bps_at_10pct_adv=_slice_optional(ctx.impact_bps_at_10pct_adv, start),
        linear_cost_bps=_slice_optional(ctx.linear_cost_bps, start),
        return_residual_scenarios=scenarios,
        metadata={
            **ctx.metadata,
            "information_cutoff": str(pd.Timestamp(dates[0]) + pd.Timedelta(hours=8)),
            "rolling_horizon_day": start + 1,
        },
    )


def _slice_optional(value, start: int):
    if value is None:
        return None
    return np.asarray(value)[start:].copy()


def _coefficient_row(
    *,
    event: PointInTimeRebalanceEvent,
    decision_day: int,
    ctx: PlannerContext,
    frontier,
    plan,
    strategy: str,
    forecast_rmse_bps: float,
    forecast_uncertainty_bps: float,
) -> pd.DataFrame:
    candidate = next(
        name for name, result in frontier.results.items() if result is plan.result
    )
    frontier_row = frontier.frontier.loc[
        frontier.frontier["candidate"].astype(str) == str(candidate)
    ].iloc[0]
    target = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    row = {
        "event_id": event.event_id,
        "strategy": strategy,
        "decision_day": decision_day + 1,
        "decision_date": pd.Timestamp(ctx.dates[0]),
        "candidate": str(candidate),
        "solver": plan.config.solver,
        "risk_measure": plan.risk_measure.value,
        "inventory_risk_weight": plan.config.inventory_risk_weight,
        "path_risk_weight": plan.config.inventory_path_risk_weight,
        "risk_budget_dollars": plan.risk_budget_dollars,
        "forecast_expected_net_pnl_dollars": plan.metrics.expected_net_pnl_dollars,
        "forecast_pnl_vol_dollars": plan.metrics.pnl_vol_dollars,
        "forecast_loss_cvar_95_dollars": plan.metrics.loss_cvar_95_dollars,
        "remaining_parent_gross_dollars": float(
            np.sum(np.abs(target * ctx.price[0]))
        ),
        "forecast_rmse_bps": forecast_rmse_bps,
        "forecast_uncertainty_bps": forecast_uncertainty_bps,
    }
    for column in ("lambda_multiplier", "cvar_lambda_multiplier"):
        if column in frontier_row:
            row[column] = float(frontier_row[column])
    return pd.DataFrame([row])


def _schedule_audit(ctx: PlannerContext, schedule: pd.DataFrame) -> dict[str, float]:
    target = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    sign = pd.Series(np.sign(target), index=ctx.symbols)
    trade = schedule["trade_shares"].to_numpy(float)
    signed = schedule["symbol"].map(sign).to_numpy(float)
    cap_excess = np.maximum(
        np.abs(trade) - schedule["cap_shares"].to_numpy(float),
        0.0,
    )
    wrong_direction = np.maximum(-trade * signed, 0.0)
    executed = (
        schedule.groupby("symbol")["trade_shares"]
        .sum()
        .reindex(ctx.symbols, fill_value=0.0)
        .to_numpy(float)
    )
    return {
        "max_cap_excess_shares": float(np.max(cap_excess)),
        "max_wrong_direction_shares": float(np.max(wrong_direction)),
        "terminal_completion_error_shares_audit": float(
            np.max(np.abs(executed - target))
        ),
    }


def _summary(trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy, group in trials.groupby("strategy", sort=False):
        ordered = group.sort_values("as_of")
        pnl_bps = ordered["net_pnl_bps"].to_numpy(float)
        weights = np.full(len(pnl_bps), 1.0 / len(pnl_bps))
        loss_var, loss_cvar = weighted_loss_var_cvar(pnl_bps, weights)
        path = np.concatenate(([0.0], np.cumsum(pnl_bps)))
        rows.append(
            {
                "strategy": strategy,
                "event_count": len(group),
                "total_net_pnl_dollars": float(group["net_pnl_dollars"].sum()),
                "mean_net_pnl_bps": float(np.mean(pnl_bps)),
                "pnl_vol_bps": float(np.std(pnl_bps, ddof=1)),
                "loss_var_95_bps": loss_var,
                "loss_cvar_95_bps": loss_cvar,
                "probability_profitable": float(np.mean(pnl_bps > 0.0)),
                "worst_event_pnl_bps": float(np.min(pnl_bps)),
                "event_sequence_max_drawdown_bps": float(
                    np.max(np.maximum.accumulate(path) - path)
                ),
                "mean_within_event_drawdown_bps": float(
                    group["within_event_max_drawdown_bps"].mean()
                ),
                "mean_early_factor_imbalance_pct": float(
                    group["early_factor_imbalance_pct"].mean()
                ),
                "mean_late_early_gross_ratio": float(
                    group["late_early_gross_ratio"].mean()
                ),
                "mean_daily_gross_spearman": float(
                    group["daily_gross_spearman"].mean()
                ),
                "mean_nondecreasing_transitions": float(
                    group["nondecreasing_transitions"].mean()
                ),
                "mean_urgent_first_trade_day": float(
                    group["urgent_first_trade_day"].mean()
                ),
                "mean_small_first_trade_day": float(
                    group["small_first_trade_day"].mean()
                ),
                "max_cap_excess_shares": float(group["max_cap_excess_shares"].max()),
                "max_wrong_direction_shares": float(
                    group["max_wrong_direction_shares"].max()
                ),
                "max_terminal_completion_error_shares": float(
                    group["terminal_completion_error_shares_audit"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def _decision(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    candidate_strategy: str,
) -> tuple[str, str, pd.DataFrame]:
    by_strategy = summary.set_index("strategy")
    baseline = by_strategy.loc["static_open_loop"]
    candidate = by_strategy.loc[candidate_strategy]
    definitions = [
        (
            "mean_pnl_within_1bp_per_event",
            candidate["mean_net_pnl_bps"] >= baseline["mean_net_pnl_bps"] - 1.0,
            baseline["mean_net_pnl_bps"],
            candidate["mean_net_pnl_bps"],
            "candidate >= baseline - 1 bp/event",
        ),
        (
            "lower_event_pnl_volatility",
            candidate["pnl_vol_bps"] <= baseline["pnl_vol_bps"] - 0.05,
            baseline["pnl_vol_bps"],
            candidate["pnl_vol_bps"],
            "candidate <= baseline - 0.05 bp",
        ),
        (
            "no_higher_loss_cvar",
            candidate["loss_cvar_95_bps"] <= baseline["loss_cvar_95_bps"] + 0.05,
            baseline["loss_cvar_95_bps"],
            candidate["loss_cvar_95_bps"],
            "candidate <= baseline + 0.05 bp",
        ),
        (
            "no_higher_within_event_drawdown",
            candidate["mean_within_event_drawdown_bps"]
            <= baseline["mean_within_event_drawdown_bps"] + 0.05,
            baseline["mean_within_event_drawdown_bps"],
            candidate["mean_within_event_drawdown_bps"],
            "candidate <= baseline + 0.05 bp",
        ),
        (
            "urgent_never_later",
            bool((paired["urgent_start_delta_days"] <= 0.0).all()),
            0.0,
            float(paired["urgent_start_delta_days"].max()),
            "maximum paired start delta <= 0 days",
        ),
        (
            "small_never_earlier",
            bool((paired["small_start_delta_days"] >= 0.0).all()),
            0.0,
            float(paired["small_start_delta_days"].min()),
            "minimum paired start delta >= 0 days",
        ),
        (
            "early_factor_within_1pp",
            candidate["mean_early_factor_imbalance_pct"]
            <= baseline["mean_early_factor_imbalance_pct"] + 1.0,
            baseline["mean_early_factor_imbalance_pct"],
            candidate["mean_early_factor_imbalance_pct"],
            "candidate <= baseline + 1 percentage point",
        ),
        (
            "late_early_ramp_preserves_90pct",
            candidate["mean_late_early_gross_ratio"]
            >= max(1.0, 0.90 * baseline["mean_late_early_gross_ratio"]),
            baseline["mean_late_early_gross_ratio"],
            candidate["mean_late_early_gross_ratio"],
            "candidate >= max(1.0, 90% of baseline)",
        ),
        (
            "rank_ramp_preserved",
            candidate["mean_daily_gross_spearman"]
            >= baseline["mean_daily_gross_spearman"] - 0.10,
            baseline["mean_daily_gross_spearman"],
            candidate["mean_daily_gross_spearman"],
            "candidate >= baseline - 0.10",
        ),
        (
            "nondecreasing_steps_preserved",
            candidate["mean_nondecreasing_transitions"]
            >= baseline["mean_nondecreasing_transitions"] - 1.0,
            baseline["mean_nondecreasing_transitions"],
            candidate["mean_nondecreasing_transitions"],
            "candidate >= baseline - 1 transition",
        ),
        (
            "participation_caps_hard",
            candidate["max_cap_excess_shares"] <= CAP_TOLERANCE_SHARES,
            CAP_TOLERANCE_SHARES,
            candidate["max_cap_excess_shares"],
            f"maximum cap excess <= {CAP_TOLERANCE_SHARES} shares",
        ),
        (
            "direction_hard",
            candidate["max_wrong_direction_shares"] <= DIRECTION_TOLERANCE_SHARES,
            DIRECTION_TOLERANCE_SHARES,
            candidate["max_wrong_direction_shares"],
            f"maximum wrong-way trade <= {DIRECTION_TOLERANCE_SHARES} shares",
        ),
        (
            "completion_hard",
            candidate["max_terminal_completion_error_shares"]
            <= COMPLETION_TOLERANCE_SHARES,
            COMPLETION_TOLERANCE_SHARES,
            candidate["max_terminal_completion_error_shares"],
            f"maximum terminal error <= {COMPLETION_TOLERANCE_SHARES} shares",
        ),
    ]
    gates = pd.DataFrame(
        definitions,
        columns=("gate", "passed", "baseline_or_limit", "candidate", "criterion"),
    )
    failed = gates.loc[~gates["passed"], "gate"].tolist()
    if failed:
        return "discard", "Failed: " + ", ".join(failed) + ".", gates
    return (
        "keep_for_holdout",
        "All development economics, behavior, liquidity, and completion gates passed.",
        gates,
    )


def plot_results(outputs: dict[str, pd.DataFrame], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials = outputs["trials"]
    paired = outputs["paired"].sort_values("as_of")
    summary = outputs["summary"].set_index("strategy")
    profiles = outputs["profiles"]
    coefficients = outputs["coefficients"]
    candidate_strategy = next(
        strategy
        for strategy in trials["strategy"].drop_duplicates()
        if strategy != "static_open_loop"
    )
    colors = {
        "static_open_loop": "#8A929A",
        candidate_strategy: "#2F6B9A",
    }
    labels = {
        "static_open_loop": "Static open-loop",
        candidate_strategy: (
            "Rolling re-optimization"
            if candidate_strategy == "rolling_reoptimization"
            else "Commitment-aware rolling"
            if candidate_strategy == "commitment_aware_rolling"
            else "Defensive rolling"
            if candidate_strategy == "defensive_rolling"
            else "Proximal rolling"
        ),
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
    axis.set_ylabel("Rolling minus static (bps)")

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
    mechanics = [
        "early_factor_imbalance_delta_pp",
        "late_early_ratio_delta",
        "urgent_start_delta_days",
        "small_start_delta_days",
    ]
    values = [float(paired[column].mean()) for column in mechanics]
    axis.barh(
        np.arange(len(mechanics)),
        values,
        color=["#7C5C9E", "#D97732", "#2F6B9A", "#70A288"],
    )
    axis.axvline(0.0, color="#59636E", linewidth=0.9)
    axis.set_yticks(
        np.arange(len(mechanics)),
        ["Early factor (pp)", "Late/early ratio", "Urgent start (days)", "Small start (days)"],
    )
    axis.set_title("Mean execution-mechanics difference")

    axis = axes[1, 2]
    rolling = coefficients.loc[
        coefficients["strategy"] == candidate_strategy
    ]
    coefficient_column = (
        "active_inventory_risk_weight"
        if "active_inventory_risk_weight" in rolling
        else "inventory_risk_weight"
    )
    coefficient_profile = rolling.groupby("decision_day")[coefficient_column].median()
    if candidate_strategy.startswith("proximal_"):
        revision_profile = rolling.groupby("decision_day")["revision_cost_bps"].median()
        axis.plot(
            revision_profile.index,
            revision_profile.to_numpy(float),
            marker="o",
            linewidth=2,
            color="#D97732",
        )
        axis.set_ylabel("Median revision shadow cost (bps)", color="#D97732")
        risk_axis = axis.twinx()
        risk_axis.plot(
            coefficient_profile.index,
            coefficient_profile.to_numpy(float),
            marker="s",
            linewidth=1.7,
            color=colors[candidate_strategy],
        )
        risk_axis.set_ylabel("Median inventory-risk coefficient", color=colors[candidate_strategy])
        risk_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        risk_axis.spines["top"].set_visible(False)
        axis.set_title("Automatically calibrated coefficients")
    else:
        axis.plot(
            coefficient_profile.index,
            coefficient_profile.to_numpy(float),
            marker="o",
            linewidth=2,
            color=colors[candidate_strategy],
        )
        axis.set_title("Automatically recalibrated risk coefficient")
        axis.set_ylabel("Median inventory-risk coefficient")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axis.set_xlabel("Decision day")

    for axis in axes.ravel():
        axis.grid(axis="y", color="#E1E5E8", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Point-in-time replay: daily receding-horizon execution",
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
    parser.add_argument("--daily-solver", default=None)
    parser.add_argument(
        "--risk-measure",
        choices=("variance", "hybrid_downside"),
        default="variance",
    )
    parser.add_argument("--n-events", type=int, default=12)
    parser.add_argument("--event-start", type=int, default=0)
    parser.add_argument(
        "--forecast-error-retention",
        type=float,
        default=DEFAULT_FORECAST_ERROR_RETENTION,
    )
    parser.add_argument(
        "--forecast-innovation-scale",
        type=float,
        default=DEFAULT_FORECAST_INNOVATION_SCALE,
    )
    parser.add_argument(
        "--replan-policy",
        choices=("always", "materiality", "defensive", "proximal"),
        default="materiality",
    )
    parser.add_argument(
        "--replan-threshold-bps",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--risk-aversion",
        choices=("high", "medium", "low"),
        default="medium",
    )
    parser.add_argument(
        "--proximal-basis",
        choices=("trade", "inventory"),
        default="trade",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("artifacts/rolling_horizon_dev"),
    )
    args = parser.parse_args()
    outputs, metadata = run_experiment(
        solver=args.solver,
        daily_solver=args.daily_solver,
        risk_measure=args.risk_measure,
        n_events=args.n_events,
        event_start=args.event_start,
        forecast_error_retention=args.forecast_error_retention,
        forecast_innovation_scale=args.forecast_innovation_scale,
        replan_policy=args.replan_policy,
        replan_threshold_bps=args.replan_threshold_bps,
        risk_aversion=args.risk_aversion,
        proximal_basis=args.proximal_basis,
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
    print(f"holdout untouched: {metadata['holdout_untouched']}")
    print(f"artifacts: {prefix.parent / (prefix.name + '*')}")


if __name__ == "__main__":
    main()
