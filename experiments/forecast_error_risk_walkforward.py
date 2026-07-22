"""Point-in-time replay of market risk versus total predictive P&L risk.

The candidate keeps the existing open-loop optimizer and adds forecast-error
P&L variance to market covariance.  Both terms are dollars squared, so the
same automatically calibrated inventory-risk coefficient prices them.  The
user still chooses only high, medium, or low risk aversion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

from trade_planner import (
    InfeasiblePlanError,
    PlannerContext,
    RebalanceFrontier,
    RebalanceRiskMeasure,
    RiskAversion,
    TradePlanner,
    build_rebalance_frontier,
    evaluate_realized_rebalance_schedule,
    evaluate_rebalance_schedule,
    infer_execution_cost_matrices,
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
from experiments.rolling_horizon_walkforward import (
    _decision,
    _schedule_audit,
    _summary,
)


FORECAST_ERROR_CALIBRATION_SEED = 20260917
SYNTHETIC_HISTORY_PERSISTENT_SCALE = 0.55
N_FORECAST_ERROR_CALIBRATION_EVENTS = 4_000
assert FORECAST_ERROR_CALIBRATION_SEED not in EVENT_SEEDS + SCENARIO_SEEDS


@dataclass(frozen=True)
class ForecastErrorPathRiskModel:
    """Predictive P&L variance from independent and persistent forecast errors.

    ``expected_return_uncertainty`` is the point-in-time standard error of each
    date/name forecast.  Independent errors contribute the sum of squared
    uncertainty-dollar exposures.  A persistent error in the parent-order
    direction contributes one squared horizon exposure.  The latter captures
    an event-call strength error that makes additions and deletions look
    jointly more attractive than they truly are.
    """

    persistent_directional_scale: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.persistent_directional_scale)
            or self.persistent_directional_scale < 0.0
        ):
            raise ValueError(
                "persistent_directional_scale must be finite and non-negative"
            )

    def objective(
        self,
        cumulative_trades: tuple[cp.Expression, ...],
        ctx: PlannerContext,
    ) -> cp.Expression:
        uncertainty, target_sign = _forecast_error_inputs(ctx)
        if len(cumulative_trades) != len(ctx.dates):
            raise ValueError("cumulative_trades must contain one expression per date")
        independent: cp.Expression | float = 0.0
        persistent_exposure: cp.Expression | float = 0.0
        for date_index, cumulative in enumerate(cumulative_trades):
            position_dollars = cp.multiply(ctx.price[date_index], cumulative)
            uncertainty_dollars = cp.multiply(
                uncertainty[date_index],
                position_dollars,
            )
            independent = independent + cp.sum_squares(uncertainty_dollars)
            persistent_exposure = persistent_exposure + cp.sum(
                cp.multiply(
                    uncertainty[date_index] * target_sign,
                    position_dollars,
                )
            )
        return independent + np.square(self.persistent_directional_scale) * cp.square(
            persistent_exposure
        )

    def variance(self, ctx: PlannerContext, schedule: pd.DataFrame) -> float:
        """Evaluate the same forecast-error variance on a solved schedule."""

        uncertainty, target_sign = _forecast_error_inputs(ctx)
        trades = (
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
        cumulative_dollars = np.cumsum(trades, axis=0) * ctx.price
        uncertainty_dollars = uncertainty * cumulative_dollars
        independent = float(np.sum(np.square(uncertainty_dollars)))
        persistent_exposure = float(
            np.sum(uncertainty_dollars * target_sign[None, :])
        )
        return independent + float(
            np.square(self.persistent_directional_scale * persistent_exposure)
        )


def estimate_persistent_directional_scale(
    standardized_forecast_errors: np.ndarray,
    target_sign: np.ndarray,
) -> float:
    """Estimate common directional error after removing independent noise.

    Inputs are historical forecast errors divided by their point-in-time
    standard errors.  Averaging signed errors across dates and names isolates
    the shared event-call component.  The known sampling variance of the
    independent standardized errors is removed before taking the square root.
    """

    errors = np.asarray(standardized_forecast_errors, dtype=float)
    signs = np.asarray(target_sign, dtype=float)
    if errors.ndim != 3 or errors.shape[2] != len(signs):
        raise ValueError(
            "standardized_forecast_errors must be sample-by-date-by-symbol"
        )
    if not np.all(np.isfinite(errors)) or not np.all(np.isfinite(signs)):
        raise ValueError("forecast errors and target signs must be finite")
    active = np.abs(signs) > 0.0
    if not np.any(active):
        return 0.0
    directional_scores = np.mean(
        errors[:, :, active] * np.sign(signs[active])[None, None, :],
        axis=(1, 2),
    )
    observations_per_score = errors.shape[1] * int(np.sum(active))
    independent_noise_variance = 1.0 / observations_per_score
    persistent_variance = max(
        float(np.var(directional_scores, ddof=1)) - independent_noise_variance,
        0.0,
    )
    return float(np.sqrt(persistent_variance))


def calibrated_persistent_directional_scale(ctx: PlannerContext) -> float:
    """Estimate the synthetic population coefficient from disjoint history."""

    rng = np.random.default_rng(FORECAST_ERROR_CALIBRATION_SEED)
    target_sign = np.sign(
        ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    )
    independent = rng.standard_t(
        5.0,
        size=(
            N_FORECAST_ERROR_CALIBRATION_EVENTS,
            len(ctx.dates),
            len(ctx.symbols),
        ),
    ) / np.sqrt(5.0 / 3.0)
    persistent = rng.normal(
        0.0,
        SYNTHETIC_HISTORY_PERSISTENT_SCALE,
        size=(N_FORECAST_ERROR_CALIBRATION_EVENTS, 1, 1),
    )
    standardized_errors = (
        independent + persistent * target_sign[None, None, :]
    )
    return estimate_persistent_directional_scale(
        standardized_errors,
        target_sign,
    )


def run_experiment(
    solver: str = "OSQP",
    n_events: int = 12,
    event_start: int = 0,
    risk_aversion: str = "medium",
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Compare market-only and total predictive-risk open-loop schedules."""

    if n_events < 2 or event_start < 0 or event_start + n_events > len(EVENT_SEEDS):
        raise ValueError("event_start and n_events must select at least two available events")
    parsed_aversion = RiskAversion.parse(risk_aversion)
    base_ctx, classifications = economic_fixture()
    uncertainty = _calibrated_alpha_uncertainty(base_ctx)
    persistent_scale = calibrated_persistent_directional_scale(base_ctx)
    forecast_risk_model = ForecastErrorPathRiskModel(persistent_scale)
    trial_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    schedule_rows: list[pd.DataFrame] = []
    daily_rows: list[pd.DataFrame] = []
    profile_rows: list[pd.DataFrame] = []
    exposure_rows: list[pd.DataFrame] = []
    frontier_rows: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, object]] = []

    selected = zip(
        EVENT_SEEDS[event_start : event_start + n_events],
        SCENARIO_SEEDS[event_start : event_start + n_events],
    )
    for event_index, (event_seed, scenario_seed) in enumerate(
        selected,
        start=event_start,
    ):
        event, forecast_rmse_bps, _ = _build_event_with_truth(
            base_ctx,
            uncertainty,
            event_index,
            event_seed,
            scenario_seed,
        )
        baseline_frontier = build_rebalance_frontier(
            event.ctx,
            solver=solver,
            lambda_multipliers=LAMBDA_MULTIPLIERS,
            risk_measure=RebalanceRiskMeasure.VARIANCE,
        )
        predictive_frontier = _predictive_risk_frontier(
            event.ctx,
            baseline_frontier,
            forecast_risk_model,
        )
        plans = {
            "static_open_loop": baseline_frontier.select(parsed_aversion),
            "forecast_error_risk": predictive_frontier.select(parsed_aversion),
        }
        event_rows: dict[str, dict[str, object]] = {}
        for strategy, plan in plans.items():
            schedule = plan.result.schedule
            realized, daily = evaluate_realized_rebalance_schedule(event, schedule)
            behavior, profile, exposures = _behavior_metrics(
                event.ctx,
                classifications,
                schedule,
            )
            audit = _schedule_audit(event.ctx, schedule)
            forecast_error_variance = forecast_risk_model.variance(
                event.ctx,
                schedule,
            )
            predictive_vol = float(
                np.sqrt(np.square(plan.metrics.pnl_vol_dollars) + forecast_error_variance)
            )
            row = {
                "event_id": event.event_id,
                "as_of": pd.Timestamp(event.as_of),
                "strategy": strategy,
                "solver": plan.config.solver,
                "risk_aversion": parsed_aversion.value,
                "forecast_rmse_bps": forecast_rmse_bps,
                "calibrated_persistent_directional_scale": persistent_scale,
                "selected_inventory_risk_weight": plan.config.inventory_risk_weight,
                "selected_forecast_path_risk_weight": (
                    plan.config.inventory_path_risk_weight
                ),
                "forecast_market_pnl_vol_dollars": plan.metrics.pnl_vol_dollars,
                "forecast_error_pnl_vol_dollars": float(
                    np.sqrt(forecast_error_variance)
                ),
                "forecast_predictive_pnl_vol_dollars": predictive_vol,
                "forecast_expected_net_pnl_dollars": (
                    plan.metrics.expected_net_pnl_dollars
                ),
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
            coefficient_rows.append(
                {
                    "event_id": event.event_id,
                    "strategy": strategy,
                    "inventory_risk_weight": plan.config.inventory_risk_weight,
                    "forecast_path_risk_weight": (
                        plan.config.inventory_path_risk_weight
                    ),
                    "forecast_error_pnl_vol_dollars": float(
                        np.sqrt(forecast_error_variance)
                    ),
                    "forecast_predictive_pnl_vol_dollars": predictive_vol,
                    "calibrated_persistent_directional_scale": persistent_scale,
                }
            )

        frontier_rows.extend(
            [
                baseline_frontier.frontier.assign(
                    event_id=event.event_id,
                    strategy="static_open_loop",
                ),
                predictive_frontier.frontier.assign(
                    event_id=event.event_id,
                    strategy="forecast_error_risk",
                ),
            ]
        )
        baseline = event_rows["static_open_loop"]
        candidate = event_rows["forecast_error_risk"]
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
    gate_decision, gate_reason, gates = _decision(
        summary,
        paired,
        "forecast_error_risk",
    )
    if event_start >= 12:
        decision = (
            "holdout_pass"
            if gate_decision == "keep_for_holdout"
            else "holdout_fail"
        )
        reason = (
            "Untouched holdout passed every predeclared gate."
            if decision == "holdout_pass"
            else "Untouched holdout failed: "
            + gate_reason.removeprefix("Failed: ")
        )
    elif event_start == 0 and event_start + n_events > 12:
        decision = "descriptive_only"
        reason = (
            "Combined development and holdout report; the separate untouched "
            "holdout decision controls production promotion."
        )
    else:
        decision = gate_decision
        reason = gate_reason
    summary["decision"] = np.where(
        summary["strategy"] == "forecast_error_risk",
        decision,
        "baseline",
    )
    summary["decision_reason"] = np.where(
        summary["strategy"] == "forecast_error_risk",
        reason,
        "Current market-covariance medium-risk policy.",
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
        "frontiers": pd.concat(frontier_rows, ignore_index=True),
        "coefficients": pd.DataFrame(coefficient_rows),
    }
    metadata: dict[str, object] = {
        "decision": decision,
        "decision_reason": reason,
        "solver": solver,
        "n_events": n_events,
        "event_start": event_start,
        "risk_aversion": parsed_aversion.value,
        "calibrated_persistent_directional_scale": persistent_scale,
        "holdout_untouched": event_start == 0 and event_start + n_events <= 12,
    }
    return outputs, metadata


def _predictive_risk_frontier(
    ctx: PlannerContext,
    market_frontier: RebalanceFrontier,
    forecast_risk_model: ForecastErrorPathRiskModel,
) -> RebalanceFrontier:
    """Re-solve one market frontier with equally priced forecast-error risk."""

    impact_matrix, linear_matrix = infer_execution_cost_matrices(ctx)
    rows: list[dict[str, object]] = []
    configs = {}
    results = {}
    for _, market_row in market_frontier.frontier.iterrows():
        market_candidate = str(market_row["candidate"])
        market_config = market_frontier.configs[market_candidate]
        forecast_weight = market_config.inventory_risk_weight
        candidate = market_candidate + "__forecast_error_path"
        config = replace(
            market_config,
            inventory_path_risk_weight=forecast_weight,
            inventory_path_risk_model=(
                forecast_risk_model if forecast_weight > 0.0 else None
            ),
        )
        configs[candidate] = config
        row = market_row.to_dict()
        row["candidate"] = candidate
        row["inventory_path_risk_weight"] = forecast_weight
        try:
            result = TradePlanner(config).solve(ctx)
        except InfeasiblePlanError as error:
            alternate_solver = (
                "CLARABEL"
                if str(config.solver).upper() == "OSQP"
                else "OSQP"
            )
            config = replace(config, solver=alternate_solver)
            configs[candidate] = config
            try:
                result = TradePlanner(config).solve(ctx)
            except Exception as fallback_error:
                row.update(
                    {
                        "status": type(fallback_error).__name__,
                        "failure_reason": (
                            f"{type(error).__name__}: {error}; "
                            f"{type(fallback_error).__name__}: {fallback_error}"
                        ),
                    }
                )
                rows.append(row)
                continue
        except Exception as error:
            row.update(
                {
                    "status": type(error).__name__,
                    "failure_reason": str(error),
                }
            )
            rows.append(row)
            continue
        metrics = evaluate_rebalance_schedule(
            ctx,
            result.schedule,
            impact_bps_at_10pct_adv=impact_matrix,
            linear_cost_bps=linear_matrix,
        )
        forecast_error_variance = forecast_risk_model.variance(
            ctx,
            result.schedule,
        )
        results[candidate] = result
        row.update(
            {
                "status": str(result.diagnostics["status"]),
                **metrics.as_dict(),
                "forecast_error_pnl_vol_dollars": float(
                    np.sqrt(forecast_error_variance)
                ),
                "predictive_pnl_vol_dollars": float(
                    np.sqrt(
                        np.square(metrics.pnl_vol_dollars)
                        + forecast_error_variance
                    )
                ),
            }
        )
        rows.append(row)
    return replace(
        market_frontier,
        frontier=pd.DataFrame(rows),
        configs=configs,
        results=results,
        risk_metric_column="predictive_pnl_vol_dollars",
    )


def _forecast_error_inputs(ctx: PlannerContext) -> tuple[np.ndarray, np.ndarray]:
    uncertainty = ctx.expected_return_uncertainty
    if uncertainty is None:
        raise ValueError(
            "ForecastErrorPathRiskModel requires expected_return_uncertainty"
        )
    matrix = np.asarray(uncertainty, dtype=float)
    expected_shape = (len(ctx.dates), len(ctx.symbols))
    if matrix.shape != expected_shape or not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"expected_return_uncertainty must be finite with shape {expected_shape}"
        )
    if np.any(matrix < 0.0):
        raise ValueError("expected_return_uncertainty must be non-negative")
    target_sign = np.sign(
        ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    )
    return matrix, target_sign


def plot_results(outputs: dict[str, pd.DataFrame], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials = outputs["trials"]
    paired = outputs["paired"].sort_values("as_of")
    summary = outputs["summary"].set_index("strategy")
    profiles = outputs["profiles"]
    coefficients = outputs["coefficients"]
    colors = {
        "static_open_loop": "#8A929A",
        "forecast_error_risk": "#2F6B9A",
    }
    labels = {
        "static_open_loop": "Market covariance",
        "forecast_error_risk": "Total predictive risk",
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
    axis.set_ylabel("Predictive risk minus market risk (bps)")

    axis = axes[0, 2]
    measures = [
        "pnl_vol_bps",
        "loss_cvar_95_bps",
        "mean_within_event_drawdown_bps",
    ]
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
        profiles.groupby(["strategy", "day_index"], as_index=False)[
            "daily_gross_pct"
        ].mean()
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
        [
            "Early factor (pp)",
            "Late/early ratio",
            "Urgent start (days)",
            "Small start (days)",
        ],
    )
    axis.set_title("Mean execution-mechanics difference")

    axis = axes[1, 2]
    coefficient_summary = coefficients.groupby("strategy").agg(
        market_coefficient=("inventory_risk_weight", "median"),
        forecast_coefficient=("forecast_path_risk_weight", "median"),
        forecast_error_vol=("forecast_error_pnl_vol_dollars", "mean"),
    )
    x = np.arange(len(coefficient_summary))
    axis.bar(
        x,
        coefficient_summary["forecast_error_vol"] / 1_000.0,
        color=[colors[strategy] for strategy in coefficient_summary.index],
    )
    axis.set_xticks(
        x,
        [labels[strategy] for strategy in coefficient_summary.index],
        rotation=8,
    )
    axis.set_ylabel("Mean forecast-error P&L vol ($000)")
    axis.set_title("Automatically estimated predictive-risk scale")

    for axis in axes.ravel():
        axis.grid(axis="y", color="#E1E5E8", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Point-in-time replay: forecast-error path risk",
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
        default=Path("artifacts/forecast_error_risk_dev"),
    )
    args = parser.parse_args()
    outputs, metadata = run_experiment(
        solver=args.solver,
        n_events=args.n_events,
        event_start=args.event_start,
        risk_aversion=args.risk_aversion,
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
        "calibrated persistent directional scale: "
        f"{metadata['calibrated_persistent_directional_scale']:.6f}"
    )
    print(f"holdout untouched: {metadata['holdout_untouched']}")
    print(f"artifacts: {prefix.parent / (prefix.name + '*')}")


if __name__ == "__main__":
    main()
