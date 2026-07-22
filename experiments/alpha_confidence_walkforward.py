"""Point-in-time replay of raw versus confidence-adjusted rebalance alpha."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from trade_planner import (
    ConfidenceAdjustedExpectedReturnAlphaModel,
    ExpectedReturnAlphaModel,
    PlannerContext,
    PointInTimeRebalanceEvent,
    RebalanceRiskMeasure,
    RiskAversion,
    TradePlanner,
    build_rebalance_frontier,
    days_to_next_event,
    evaluate_realized_rebalance_schedule,
    evaluate_rebalance_schedule,
    infer_execution_cost_matrices,
    weighted_loss_var_cvar,
)

from experiments.rebalance_economic_calibration import (
    _behavior_metrics,
    _stress_residual_returns,
    economic_fixture,
)


ALPHA_CALIBRATION_SEED = 20260901
EVENT_SEEDS = tuple(20261001 + offset for offset in range(24))
SCENARIO_SEEDS = tuple(20261101 + offset for offset in range(24))
LAMBDA_MULTIPLIERS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)
DEFAULT_MEDIUM_ALPHA_CONFIDENCE = 0.75
N_OPTIMIZATION_SCENARIOS = 64
assert set(EVENT_SEEDS).isdisjoint(SCENARIO_SEEDS)
assert ALPHA_CALIBRATION_SEED not in EVENT_SEEDS + SCENARIO_SEEDS


def run_experiment(
    solver: str = "OSQP",
    risk_measure: str = "variance",
    n_events: int = 12,
    event_start: int = 0,
    alpha_confidence: float = DEFAULT_MEDIUM_ALPHA_CONFIDENCE,
    selection_policy: str = "reselect_frontier",
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    if n_events < 2 or event_start < 0 or event_start + n_events > len(EVENT_SEEDS):
        raise ValueError("event_start and n_events must select at least two available events")
    if not 0.5 <= alpha_confidence < 1.0:
        raise ValueError("alpha_confidence must be between 0.5 inclusive and 1.0 exclusive")
    if selection_policy not in {"reselect_frontier", "fixed_risk"}:
        raise ValueError("selection_policy must be reselect_frontier or fixed_risk")
    measure = RebalanceRiskMeasure.parse(risk_measure)
    if measure not in {
        RebalanceRiskMeasure.VARIANCE,
        RebalanceRiskMeasure.HYBRID_DOWNSIDE,
    }:
        raise ValueError("risk_measure must be variance or hybrid_downside")

    base_ctx, classifications = economic_fixture()
    uncertainty = _calibrated_alpha_uncertainty(base_ctx)
    strategies = {
        "raw_expected_alpha": ExpectedReturnAlphaModel(),
        "confidence_adjusted_alpha": ConfidenceAdjustedExpectedReturnAlphaModel(
            confidence=alpha_confidence
        ),
    }
    trial_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    schedule_rows: list[pd.DataFrame] = []
    daily_rows: list[pd.DataFrame] = []

    for event_index, (event_seed, scenario_seed) in enumerate(
        zip(
            EVENT_SEEDS[event_start : event_start + n_events],
            SCENARIO_SEEDS[event_start : event_start + n_events],
        ),
        start=event_start,
    ):
        event, forecast_rmse_bps = _build_event(
            base_ctx,
            uncertainty,
            event_index,
            event_seed,
            scenario_seed,
        )
        event_strategy_rows: dict[str, dict[str, object]] = {}
        raw_plan = None
        for strategy, alpha_model in strategies.items():
            if strategy == "confidence_adjusted_alpha" and selection_policy == "fixed_risk":
                assert raw_plan is not None
                config = replace(
                    raw_plan.config,
                    inventory_alpha_model=alpha_model,
                )
                result = TradePlanner(config).solve(event.ctx)
                schedule = result.schedule
                impact_matrix, linear_matrix = infer_execution_cost_matrices(event.ctx)
                forecast_metrics = evaluate_rebalance_schedule(
                    event.ctx,
                    schedule,
                    impact_bps_at_10pct_adv=impact_matrix,
                    linear_cost_bps=linear_matrix,
                )
            else:
                frontier = build_rebalance_frontier(
                    event.ctx,
                    solver=solver,
                    lambda_multipliers=LAMBDA_MULTIPLIERS,
                    risk_measure=measure,
                    inventory_alpha_model=alpha_model,
                )
                plan = frontier.select(RiskAversion.MEDIUM)
                if strategy == "raw_expected_alpha":
                    raw_plan = plan
                config = plan.config
                schedule = plan.result.schedule
                forecast_metrics = plan.metrics
            realized, daily = evaluate_realized_rebalance_schedule(event, schedule)
            behavior, _, _ = _behavior_metrics(
                event.ctx,
                classifications,
                schedule,
            )
            row = {
                "event_id": event.event_id,
                "as_of": pd.Timestamp(event.as_of),
                "strategy": strategy,
                "risk_measure": measure.value,
                "alpha_confidence": (
                    alpha_model.confidence
                    if isinstance(alpha_model, ConfidenceAdjustedExpectedReturnAlphaModel)
                    else 0.50
                ),
                "forecast_rmse_bps": forecast_rmse_bps,
                "selection_policy": selection_policy,
                "selected_inventory_risk_weight": config.inventory_risk_weight,
                "selected_path_risk_weight": config.inventory_path_risk_weight,
                "forecast_expected_net_pnl_dollars": forecast_metrics.expected_net_pnl_dollars,
                **realized.as_dict(),
                **behavior,
            }
            event_strategy_rows[strategy] = row
            trial_rows.append(row)
            schedule_rows.append(
                schedule.assign(event_id=event.event_id, strategy=strategy)
            )
            daily_rows.append(
                daily.assign(event_id=event.event_id, strategy=strategy)
            )

        raw = event_strategy_rows["raw_expected_alpha"]
        adjusted = event_strategy_rows["confidence_adjusted_alpha"]
        paired_rows.append(
            {
                "event_id": event.event_id,
                "as_of": pd.Timestamp(event.as_of),
                "realized_net_pnl_delta_dollars": (
                    adjusted["net_pnl_dollars"] - raw["net_pnl_dollars"]
                ),
                "realized_net_pnl_delta_bps": (
                    adjusted["net_pnl_bps"] - raw["net_pnl_bps"]
                ),
                "within_event_drawdown_delta_bps": (
                    adjusted["within_event_max_drawdown_bps"]
                    - raw["within_event_max_drawdown_bps"]
                ),
                "early_factor_imbalance_delta_pp": (
                    adjusted["early_factor_imbalance_pct"]
                    - raw["early_factor_imbalance_pct"]
                ),
                "late_early_ratio_delta": (
                    adjusted["late_early_gross_ratio"]
                    - raw["late_early_gross_ratio"]
                ),
                "urgent_start_delta_days": (
                    adjusted["urgent_first_trade_day"]
                    - raw["urgent_first_trade_day"]
                ),
                "small_start_delta_days": (
                    adjusted["small_first_trade_day"]
                    - raw["small_first_trade_day"]
                ),
            }
        )

    trials = pd.DataFrame(trial_rows)
    paired = pd.DataFrame(paired_rows)
    summary = _summary(trials)
    decision, reason = _decision(summary, paired)
    summary["decision"] = np.where(
        summary["strategy"] == "confidence_adjusted_alpha",
        decision,
        "baseline",
    )
    summary["decision_reason"] = np.where(
        summary["strategy"] == "confidence_adjusted_alpha",
        reason,
        "Current raw expected-alpha policy.",
    )
    outputs = {
        "trials": trials,
        "paired": paired,
        "summary": summary,
        "schedules": pd.concat(schedule_rows, ignore_index=True),
        "daily": pd.concat(daily_rows, ignore_index=True),
    }
    metadata = {
        "decision": decision,
        "decision_reason": reason,
        "risk_measure": measure.value,
        "n_events": n_events,
        "alpha_confidence": alpha_confidence,
        "event_start": event_start,
        "selection_policy": selection_policy,
        "uncertainty_mean_bps": float(10_000.0 * np.mean(uncertainty)),
    }
    return outputs, metadata


def _calibrated_alpha_uncertainty(ctx: PlannerContext) -> np.ndarray:
    """Estimate forecast standard error from a disjoint pre-replay sample."""

    rng = np.random.default_rng(ALPHA_CALIBRATION_SEED)
    base_scale = 0.35 / 10_000.0 + 0.80 * np.abs(ctx.expected_return)
    degrees_of_freedom = 5.0
    standardized = rng.standard_t(
        degrees_of_freedom,
        size=(2_000, len(ctx.dates), len(ctx.symbols)),
    ) / np.sqrt(degrees_of_freedom / (degrees_of_freedom - 2.0))
    errors = standardized * base_scale[None, :, :]
    return np.std(errors, axis=0, ddof=1)


def _build_event(
    base_ctx: PlannerContext,
    uncertainty: np.ndarray,
    event_index: int,
    event_seed: int,
    scenario_seed: int,
) -> tuple[PointInTimeRebalanceEvent, float]:
    rng = np.random.default_rng(event_seed)
    dates = pd.bdate_range("2027-01-04", periods=len(base_ctx.dates))
    dates = dates + pd.offsets.BDay(event_index * (len(base_ctx.dates) + 3))
    target_sign = np.sign(
        base_ctx.orders["target_shares"].reindex(base_ctx.symbols).to_numpy(float)
    )
    true_strength = float(np.clip(rng.normal(0.65, 0.35), -0.25, 1.25))
    true_expected_return = base_ctx.expected_return * true_strength
    forecast_error = (
        rng.standard_t(5.0, size=true_expected_return.shape)
        / np.sqrt(5.0 / 3.0)
        * uncertainty
    )
    # Include coherent estimation error along the rebalance direction. This is
    # the optimizer's-curse case: raw alpha pulls flow forward when a noisy
    # forecast looks strongest, even though the error is not realized alpha.
    coherent_error = (
        rng.normal(0.0, 0.55)
        * uncertainty
        * target_sign[None, :]
    )
    forecast_return = true_expected_return + forecast_error + coherent_error
    event_dates = {symbol: dates[-1] for symbol in base_ctx.symbols}
    point_in_time_ctx = replace(
        base_ctx,
        dates=dates,
        panel=pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [dates, base_ctx.symbols],
                names=["date", "symbol"],
            )
        ),
        event_days=days_to_next_event(dates, base_ctx.symbols, event_dates),
        expected_return=forecast_return,
        expected_return_uncertainty=uncertainty,
        return_residual_scenarios=_stress_residual_returns(
            base_ctx,
            N_OPTIMIZATION_SCENARIOS,
            scenario_seed,
        ),
        metadata={
            **base_ctx.metadata,
            "information_cutoff": str((dates[0] - pd.offsets.BDay(1)).date()),
        },
    )
    realized_residual = _stress_residual_returns(
        base_ctx,
        256,
        event_seed + 50_000,
    )[0]
    realized_returns = true_expected_return + realized_residual
    realized_impact = base_ctx.impact_bps_at_10pct_adv * rng.lognormal(
        mean=-0.5 * 0.12**2,
        sigma=0.12,
        size=base_ctx.impact_bps_at_10pct_adv.shape,
    )
    realized_linear = base_ctx.linear_cost_bps * rng.lognormal(
        mean=-0.5 * 0.08**2,
        sigma=0.08,
        size=base_ctx.linear_cost_bps.shape,
    )
    as_of = dates[0] - pd.offsets.BDay(1) + pd.Timedelta(hours=16)
    event = PointInTimeRebalanceEvent(
        event_id=f"event_{event_index + 1:02d}",
        as_of=as_of,
        information_cutoff=as_of,
        ctx=point_in_time_ctx,
        realized_returns=realized_returns,
        realized_impact_bps_at_10pct_adv=realized_impact,
        realized_linear_cost_bps=realized_linear,
        realized_available_at=dates[-1] + pd.Timedelta(hours=18),
    )
    forecast_rmse_bps = float(
        10_000.0 * np.sqrt(np.mean(np.square(forecast_return - true_expected_return)))
    )
    return event, forecast_rmse_bps


def _summary(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, group in trials.groupby("strategy", sort=False):
        pnl_bps = group.sort_values("as_of")["net_pnl_bps"].to_numpy(float)
        weights = np.full(len(pnl_bps), 1.0 / len(pnl_bps))
        loss_var, loss_cvar = weighted_loss_var_cvar(pnl_bps, weights)
        path = np.concatenate(([0.0], np.cumsum(pnl_bps)))
        max_drawdown = float(np.max(np.maximum.accumulate(path) - path))
        rows.append(
            {
                "strategy": strategy,
                "event_count": len(group),
                "total_net_pnl_dollars": float(group["net_pnl_dollars"].sum()),
                "mean_net_pnl_bps": float(np.mean(pnl_bps)),
                "pnl_vol_bps": float(np.std(pnl_bps, ddof=1)),
                "loss_var_95_bps": loss_var,
                "loss_cvar_95_bps": loss_cvar,
                "probability_profitable": float(np.mean(pnl_bps > 0)),
                "worst_event_pnl_bps": float(np.min(pnl_bps)),
                "event_sequence_max_drawdown_bps": max_drawdown,
                "mean_within_event_drawdown_bps": float(
                    group["within_event_max_drawdown_bps"].mean()
                ),
                "mean_early_factor_imbalance_pct": float(
                    group["early_factor_imbalance_pct"].mean()
                ),
                "mean_late_early_gross_ratio": float(
                    group["late_early_gross_ratio"].mean()
                ),
                "mean_urgent_first_trade_day": float(
                    group["urgent_first_trade_day"].mean()
                ),
                "mean_small_first_trade_day": float(
                    group["small_first_trade_day"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _decision(summary: pd.DataFrame, paired: pd.DataFrame) -> tuple[str, str]:
    by_strategy = summary.set_index("strategy")
    raw = by_strategy.loc["raw_expected_alpha"]
    adjusted = by_strategy.loc["confidence_adjusted_alpha"]
    gates = {
        "higher_total_realized_pnl": (
            adjusted["total_net_pnl_dollars"] >= raw["total_net_pnl_dollars"]
        ),
        "no_higher_event_pnl_volatility": (
            adjusted["pnl_vol_bps"] <= raw["pnl_vol_bps"]
        ),
        "no_higher_loss_cvar": (
            adjusted["loss_cvar_95_bps"] <= raw["loss_cvar_95_bps"]
        ),
        "urgent_never_later": bool((paired["urgent_start_delta_days"] <= 0).all()),
        "small_never_earlier": bool((paired["small_start_delta_days"] >= 0).all()),
        "factor_within_1pp_on_average": (
            adjusted["mean_early_factor_imbalance_pct"]
            <= raw["mean_early_factor_imbalance_pct"] + 1.0
        ),
        "ramp_preserves_90pct": (
            adjusted["mean_late_early_gross_ratio"]
            >= 0.90 * raw["mean_late_early_gross_ratio"]
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        return "discard", "Failed: " + ", ".join(failed) + "."
    return "keep", "All realized-economics and execution-mechanics gates passed."


def plot_results(outputs: dict[str, pd.DataFrame], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials = outputs["trials"]
    paired = outputs["paired"].sort_values("as_of")
    summary = outputs["summary"].set_index("strategy")
    colors = {
        "raw_expected_alpha": "#8A929A",
        "confidence_adjusted_alpha": "#2F6B9A",
    }
    labels = {
        "raw_expected_alpha": "Raw expected alpha",
        "confidence_adjusted_alpha": "Confidence-adjusted alpha",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))

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
    bar_colors = np.where(
        paired["realized_net_pnl_delta_bps"] >= 0,
        "#70A288",
        "#B04A4A",
    )
    axis.bar(
        np.arange(1, len(paired) + 1),
        paired["realized_net_pnl_delta_bps"],
        color=bar_colors,
    )
    axis.axhline(0.0, color="#59636E", linewidth=0.9)
    axis.set_title("Paired event P&L difference")
    axis.set_ylabel("Adjusted minus raw (bps)")

    axis = axes[1, 0]
    measures = ["pnl_vol_bps", "loss_cvar_95_bps", "event_sequence_max_drawdown_bps"]
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
    axis.set_xticks(x, ["P&L vol", "Loss CVaR 95", "Max drawdown"])
    axis.set_title("Realized downside across events")
    axis.set_ylabel("bps")

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

    for axis in axes.ravel():
        axis.grid(axis="y", color="#E1E5E8", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Point-in-time replay: confidence-adjusted alpha",
        x=0.05,
        y=0.995,
        ha="left",
        fontsize=16,
    )
    fig.tight_layout(rect=(0.025, 0.02, 0.995, 0.95), h_pad=2.6, w_pad=2.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="OSQP")
    parser.add_argument(
        "--risk-measure",
        choices=("variance", "hybrid_downside"),
        default="variance",
    )
    parser.add_argument("--n-events", type=int, default=12)
    parser.add_argument("--event-start", type=int, default=0)
    parser.add_argument(
        "--alpha-confidence",
        type=float,
        default=DEFAULT_MEDIUM_ALPHA_CONFIDENCE,
    )
    parser.add_argument(
        "--selection-policy",
        choices=("reselect_frontier", "fixed_risk"),
        default="reselect_frontier",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("artifacts/alpha_confidence_walkforward"),
    )
    args = parser.parse_args()
    outputs, metadata = run_experiment(
        solver=args.solver,
        risk_measure=args.risk_measure,
        n_events=args.n_events,
        event_start=args.event_start,
        alpha_confidence=args.alpha_confidence,
        selection_policy=args.selection_policy,
    )
    prefix: Path = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(prefix.with_name(prefix.name + f"_{name}.csv"), index=False)
    chart = prefix.with_name(prefix.name + ".png")
    plot_results(outputs, chart)
    print(outputs["summary"].round(4).to_string(index=False))
    print(f"\ndecision: {metadata['decision']}")
    print(f"reason: {metadata['decision_reason']}")
    print(f"mean forecast uncertainty: {metadata['uncertainty_mean_bps']:.4f} bps")
    print(f"artifacts: {prefix.parent / (prefix.name + '*')}")


if __name__ == "__main__":
    main()
