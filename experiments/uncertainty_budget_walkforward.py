"""Test signal-reliability scaling of the automatic P&L-risk budget."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trade_planner import (
    DEFAULT_RISK_PREFERENCES,
    RebalanceRiskMeasure,
    RiskAversion,
    RiskPreference,
    build_rebalance_frontier,
    evaluate_realized_rebalance_schedule,
    weighted_loss_var_cvar,
)

from experiments.alpha_confidence_walkforward import (
    EVENT_SEEDS,
    LAMBDA_MULTIPLIERS,
    SCENARIO_SEEDS,
    _build_event,
    _calibrated_alpha_uncertainty,
)
from experiments.rebalance_economic_calibration import (
    _behavior_metrics,
    economic_fixture,
)


def run_experiment(
    solver: str = "OSQP",
    risk_measure: str = "variance",
    n_events: int = 12,
    event_start: int = 0,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    if n_events < 2 or event_start < 0 or event_start + n_events > len(EVENT_SEEDS):
        raise ValueError("event_start and n_events must select at least two available events")
    measure = RebalanceRiskMeasure.parse(risk_measure)
    if measure not in {
        RebalanceRiskMeasure.VARIANCE,
        RebalanceRiskMeasure.HYBRID_DOWNSIDE,
    }:
        raise ValueError("risk_measure must be variance or hybrid_downside")

    base_ctx, classifications = economic_fixture()
    uncertainty = _calibrated_alpha_uncertainty(base_ctx)
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
        frontier = build_rebalance_frontier(
            event.ctx,
            solver=solver,
            lambda_multipliers=LAMBDA_MULTIPLIERS,
            risk_measure=measure,
        )
        reliability = _forecast_reliability(event.ctx)
        base_fraction = DEFAULT_RISK_PREFERENCES[RiskAversion.MEDIUM].risk_frontier_fraction
        effective_fraction = base_fraction * (0.5 + 0.5 * reliability)
        preferences = {
            **DEFAULT_RISK_PREFERENCES,
            RiskAversion.MEDIUM: RiskPreference(
                risk_frontier_fraction=effective_fraction,
                description="Scale the medium risk budget by point-in-time forecast reliability.",
            ),
        }
        plans = {
            "fixed_medium_budget": frontier.select(RiskAversion.MEDIUM),
            "reliability_scaled_budget": frontier.select(
                RiskAversion.MEDIUM,
                preferences=preferences,
            ),
        }
        event_rows: dict[str, dict[str, object]] = {}
        for strategy, plan in plans.items():
            schedule = plan.result.schedule
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
                "forecast_rmse_bps": forecast_rmse_bps,
                "forecast_reliability": reliability,
                "risk_frontier_fraction": (
                    base_fraction
                    if strategy == "fixed_medium_budget"
                    else effective_fraction
                ),
                "risk_budget_dollars": plan.risk_budget_dollars,
                "selected_inventory_risk_weight": plan.config.inventory_risk_weight,
                "selected_path_risk_weight": plan.config.inventory_path_risk_weight,
                "forecast_expected_net_pnl_dollars": plan.metrics.expected_net_pnl_dollars,
                **realized.as_dict(),
                **behavior,
            }
            event_rows[strategy] = row
            trial_rows.append(row)
            schedule_rows.append(schedule.assign(event_id=event.event_id, strategy=strategy))
            daily_rows.append(daily.assign(event_id=event.event_id, strategy=strategy))

        baseline = event_rows["fixed_medium_budget"]
        candidate = event_rows["reliability_scaled_budget"]
        paired_rows.append(
            {
                "event_id": event.event_id,
                "as_of": pd.Timestamp(event.as_of),
                "forecast_reliability": reliability,
                "risk_frontier_fraction": effective_fraction,
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
    decision, reason = _decision(summary, paired)
    summary["decision"] = np.where(
        summary["strategy"] == "reliability_scaled_budget",
        decision,
        "baseline",
    )
    summary["decision_reason"] = np.where(
        summary["strategy"] == "reliability_scaled_budget",
        reason,
        "Current fixed 50% medium risk budget.",
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
        "event_start": event_start,
        "mean_reliability": float(paired["forecast_reliability"].mean()),
        "mean_effective_risk_fraction": float(paired["risk_frontier_fraction"].mean()),
    }
    return outputs, metadata


def _forecast_reliability(ctx) -> float:
    if ctx.expected_return_uncertainty is None:
        return 1.0
    target = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    target_dollars = np.abs(ctx.price * target[None, :])
    directional_alpha = (
        ctx.expected_return
        * np.sign(target)[None, :]
    )
    signal_dollars = float(
        np.sum(target_dollars * np.maximum(directional_alpha, 0.0))
    )
    uncertainty_dollars = float(
        np.sum(target_dollars * ctx.expected_return_uncertainty)
    )
    return signal_dollars / max(signal_dollars + uncertainty_dollars, 1e-12)


def _summary(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, group in trials.groupby("strategy", sort=False):
        pnl_bps = group.sort_values("as_of")["net_pnl_bps"].to_numpy(float)
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
                "probability_profitable": float(np.mean(pnl_bps > 0)),
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
                "mean_urgent_first_trade_day": float(
                    group["urgent_first_trade_day"].mean()
                ),
                "mean_small_first_trade_day": float(
                    group["small_first_trade_day"].mean()
                ),
                "mean_risk_frontier_fraction": float(
                    group["risk_frontier_fraction"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _decision(summary: pd.DataFrame, paired: pd.DataFrame) -> tuple[str, str]:
    by_strategy = summary.set_index("strategy")
    baseline = by_strategy.loc["fixed_medium_budget"]
    candidate = by_strategy.loc["reliability_scaled_budget"]
    gates = {
        # One basis point per event is the same economic materiality rule used
        # by the single-event frontier selector.
        "pnl_within_1bp_per_event": (
            candidate["mean_net_pnl_bps"] >= baseline["mean_net_pnl_bps"] - 1.0
        ),
        "lower_event_pnl_volatility": (
            candidate["pnl_vol_bps"] < baseline["pnl_vol_bps"]
        ),
        "no_higher_loss_cvar": (
            candidate["loss_cvar_95_bps"] <= baseline["loss_cvar_95_bps"]
        ),
        "urgent_never_later": bool((paired["urgent_start_delta_days"] <= 0).all()),
        "small_never_earlier": bool((paired["small_start_delta_days"] >= 0).all()),
        "factor_no_worse": (
            candidate["mean_early_factor_imbalance_pct"]
            <= baseline["mean_early_factor_imbalance_pct"]
        ),
        "ramp_preserves_90pct": (
            candidate["mean_late_early_gross_ratio"]
            >= 0.90 * baseline["mean_late_early_gross_ratio"]
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
        "fixed_medium_budget": "#8A929A",
        "reliability_scaled_budget": "#2F6B9A",
    }
    labels = {
        "fixed_medium_budget": "Fixed 50% medium budget",
        "reliability_scaled_budget": "Reliability-scaled budget",
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
    axis.scatter(
        paired["forecast_reliability"],
        paired["realized_net_pnl_delta_bps"],
        c=paired["risk_frontier_fraction"],
        cmap="Blues",
        s=65,
        edgecolor="white",
        linewidth=0.6,
    )
    axis.axhline(0.0, color="#59636E", linewidth=0.9)
    axis.set_title("Reliability versus paired P&L effect")
    axis.set_xlabel("Forecast reliability")
    axis.set_ylabel("Scaled minus fixed (bps)")

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
        "Point-in-time replay: reliability-scaled risk budget",
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
        "--output-prefix",
        type=Path,
        default=Path("artifacts/uncertainty_budget_walkforward"),
    )
    args = parser.parse_args()
    outputs, metadata = run_experiment(
        solver=args.solver,
        risk_measure=args.risk_measure,
        n_events=args.n_events,
        event_start=args.event_start,
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
    print(f"mean reliability: {metadata['mean_reliability']:.4f}")
    print(f"mean effective risk fraction: {metadata['mean_effective_risk_fraction']:.4f}")
    print(f"artifacts: {prefix.parent / (prefix.name + '*')}")


if __name__ == "__main__":
    main()
