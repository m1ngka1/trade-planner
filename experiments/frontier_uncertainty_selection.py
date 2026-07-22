"""Test forecast-uncertainty-aware selection on an unchanged solved frontier."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from trade_planner import (
    DEFAULT_RISK_PREFERENCES,
    RebalanceFrontier,
    RebalanceRiskMeasure,
    RiskAversion,
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


DEFAULT_CONFIDENCES = (0.60, 0.75, 0.90)


def run_experiment(
    solver: str = "OSQP",
    risk_measure: str = "variance",
    n_events: int = 12,
    event_start: int = 0,
    confidences: tuple[float, ...] = DEFAULT_CONFIDENCES,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    if n_events < 2 or event_start < 0 or event_start + n_events > len(EVENT_SEEDS):
        raise ValueError("event_start and n_events must select at least two available events")
    parsed_confidences = tuple(sorted({float(value) for value in confidences}))
    if not parsed_confidences or any(
        not 0.5 < value < 1.0 for value in parsed_confidences
    ):
        raise ValueError("confidences must be strictly between 0.5 and 1.0")
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
        baseline_plan = frontier.select(RiskAversion.MEDIUM)
        baseline_candidate = _candidate_for_result(
            frontier,
            baseline_plan.result,
        )
        selections: dict[str, dict[str, object]] = {
            "fixed_1bp_tie": {
                "candidate": baseline_candidate,
                "confidence": 0.50,
                "paired_alpha_uncertainty_dollars": 0.0,
                "pnl_gap_dollars": 0.0,
                "tie_threshold_dollars": _parent_gross(event.ctx) / 10_000.0,
            }
        }
        for confidence in parsed_confidences:
            strategy = _strategy_name(confidence)
            selections[strategy] = _select_uncertainty_aware(
                frontier,
                confidence=confidence,
            )

        event_rows: dict[str, dict[str, object]] = {}
        for strategy, selection in selections.items():
            candidate = str(selection["candidate"])
            schedule = frontier.results[candidate].schedule
            frontier_row = frontier.frontier.loc[
                frontier.frontier["candidate"] == candidate
            ].iloc[0]
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
                "confidence": selection["confidence"],
                "forecast_rmse_bps": forecast_rmse_bps,
                "selected_candidate": candidate,
                "selected_inventory_risk_weight": frontier_row[
                    "inventory_risk_weight"
                ],
                "selected_path_risk_weight": frontier_row[
                    "inventory_path_risk_weight"
                ],
                "forecast_expected_net_pnl_dollars": frontier_row[
                    "expected_net_pnl_dollars"
                ],
                "forecast_risk_dollars": frontier_row[frontier.risk_metric_column],
                "paired_alpha_uncertainty_dollars": selection[
                    "paired_alpha_uncertainty_dollars"
                ],
                "pnl_gap_dollars": selection["pnl_gap_dollars"],
                "tie_threshold_dollars": selection["tie_threshold_dollars"],
                "changed_from_baseline": float(candidate != baseline_candidate),
                **realized.as_dict(),
                **behavior,
            }
            event_rows[strategy] = row
            trial_rows.append(row)
            schedule_rows.append(
                schedule.assign(event_id=event.event_id, strategy=strategy)
            )
            daily_rows.append(
                daily.assign(event_id=event.event_id, strategy=strategy)
            )

        baseline = event_rows["fixed_1bp_tie"]
        for confidence in parsed_confidences:
            strategy = _strategy_name(confidence)
            candidate = event_rows[strategy]
            paired_rows.append(
                {
                    "event_id": event.event_id,
                    "as_of": pd.Timestamp(event.as_of),
                    "strategy": strategy,
                    "confidence": confidence,
                    "changed_from_baseline": candidate["changed_from_baseline"],
                    "paired_alpha_uncertainty_dollars": candidate[
                        "paired_alpha_uncertainty_dollars"
                    ],
                    "pnl_gap_dollars": candidate["pnl_gap_dollars"],
                    "tie_threshold_dollars": candidate["tie_threshold_dollars"],
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
    decisions = _decisions(summary, paired)
    summary["decision"] = summary["strategy"].map(
        {strategy: decision for strategy, (decision, _) in decisions.items()}
    )
    summary["decision_reason"] = summary["strategy"].map(
        {strategy: reason for strategy, (_, reason) in decisions.items()}
    )
    outputs = {
        "trials": trials,
        "paired": paired,
        "summary": summary,
        "schedules": pd.concat(schedule_rows, ignore_index=True),
        "daily": pd.concat(daily_rows, ignore_index=True),
    }
    kept = summary.loc[summary["decision"] == "keep", "strategy"].tolist()
    metadata = {
        "risk_measure": measure.value,
        "n_events": n_events,
        "event_start": event_start,
        "kept_strategies": kept,
        "best_kept_strategy": _best_kept_strategy(summary),
    }
    return outputs, metadata


def _select_uncertainty_aware(
    frontier: RebalanceFrontier,
    *,
    confidence: float,
    minimum_edge_bps: float = 1.0,
) -> dict[str, object]:
    valid = frontier.frontier[
        frontier.frontier["status"].isin(("optimal", "optimal_inaccurate"))
    ].copy()
    preference = DEFAULT_RISK_PREFERENCES[RiskAversion.MEDIUM]
    min_risk = float(valid[frontier.risk_metric_column].min())
    max_risk = float(valid[frontier.risk_metric_column].max())
    budget = min_risk + preference.risk_frontier_fraction * (max_risk - min_risk)
    eligible = valid[
        valid[frontier.risk_metric_column]
        <= budget + max(1e-8, abs(budget) * 1e-8)
    ].copy()
    if eligible.empty:
        eligible = valid.nsmallest(1, frontier.risk_metric_column)
    best_row = eligible.sort_values(
        ["expected_net_pnl_dollars", frontier.risk_metric_column],
        ascending=[False, True],
    ).iloc[0]
    best_candidate = str(best_row["candidate"])
    best_schedule = frontier.results[best_candidate].schedule
    best_pnl = float(best_row["expected_net_pnl_dollars"])
    materiality = minimum_edge_bps / 10_000.0 * _parent_gross(frontier.ctx)
    quantile = float(NormalDist().inv_cdf(confidence))

    rows = []
    for row in eligible.itertuples(index=False):
        candidate = str(row.candidate)
        uncertainty = _paired_alpha_uncertainty(
            frontier.ctx,
            best_schedule,
            frontier.results[candidate].schedule,
        )
        threshold = max(materiality, quantile * uncertainty)
        gap = best_pnl - float(row.expected_net_pnl_dollars)
        rows.append(
            {
                "candidate": candidate,
                "paired_alpha_uncertainty_dollars": uncertainty,
                "pnl_gap_dollars": gap,
                "tie_threshold_dollars": threshold,
                "is_tied": gap <= threshold + 1e-8,
            }
        )
    diagnostics = pd.DataFrame(rows).set_index("candidate")
    tied = eligible[
        eligible["candidate"].isin(
            diagnostics.index[diagnostics["is_tied"]].tolist()
        )
    ]
    selected = tied.sort_values(
        [
            frontier.risk_metric_column,
            "pnl_vol_dollars",
            "expected_net_pnl_dollars",
            "impact_cost_dollars",
        ],
        ascending=[True, True, False, True],
    ).iloc[0]
    candidate = str(selected["candidate"])
    return {
        "candidate": candidate,
        "confidence": confidence,
        **diagnostics.loc[candidate].drop(labels="is_tied").to_dict(),
    }


def _paired_alpha_uncertainty(ctx, left: pd.DataFrame, right: pd.DataFrame) -> float:
    if ctx.expected_return_uncertainty is None:
        return 0.0
    left_positions = _position_dollars(ctx, left)
    right_positions = _position_dollars(ctx, right)
    difference = (left_positions - right_positions) * ctx.expected_return_uncertainty
    # Standard errors are treated as independent here. The actual production
    # replay should replace this diagonal approximation when an error covariance
    # estimate is available.
    return float(np.sqrt(np.sum(np.square(difference))))


def _position_dollars(ctx, schedule: pd.DataFrame) -> np.ndarray:
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
    return np.cumsum(trades, axis=0) * ctx.price


def _parent_gross(ctx) -> float:
    target = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    return float(np.sum(np.abs(target * ctx.price[0])))


def _candidate_for_result(frontier: RebalanceFrontier, result: object) -> str:
    for candidate, candidate_result in frontier.results.items():
        if candidate_result is result:
            return str(candidate)
    raise KeyError("selected result is missing from frontier")


def _strategy_name(confidence: float) -> str:
    return f"uncertainty_tie_{int(round(100 * confidence)):02d}"


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
                "schedule_change_rate": float(group["changed_from_baseline"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _decisions(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
) -> dict[str, tuple[str, str]]:
    by_strategy = summary.set_index("strategy")
    baseline = by_strategy.loc["fixed_1bp_tie"]
    decisions = {
        "fixed_1bp_tie": ("baseline", "Current fixed one-basis-point tie rule.")
    }
    for strategy, candidate in by_strategy.drop(index="fixed_1bp_tie").iterrows():
        differences = paired[paired["strategy"] == strategy]
        if float(candidate["schedule_change_rate"]) == 0.0:
            decisions[strategy] = (
                "inconclusive",
                "The uncertainty threshold selected the baseline schedule in every event.",
            )
            continue
        gates = {
            "pnl_within_1bp_per_event": (
                candidate["mean_net_pnl_bps"] >= baseline["mean_net_pnl_bps"] - 1.0
            ),
            "lower_event_pnl_volatility": (
                candidate["pnl_vol_bps"] < baseline["pnl_vol_bps"]
            ),
            "no_higher_loss_cvar": (
                candidate["loss_cvar_95_bps"] <= baseline["loss_cvar_95_bps"]
            ),
            "urgent_never_later": bool(
                (differences["urgent_start_delta_days"] <= 0).all()
            ),
            "small_never_earlier": bool(
                (differences["small_start_delta_days"] >= 0).all()
            ),
            "factor_within_1pp": (
                candidate["mean_early_factor_imbalance_pct"]
                <= baseline["mean_early_factor_imbalance_pct"] + 1.0
            ),
            "ramp_preserves_90pct": (
                candidate["mean_late_early_gross_ratio"]
                >= 0.90 * baseline["mean_late_early_gross_ratio"]
            ),
        }
        failed = [name for name, passed in gates.items() if not passed]
        decisions[strategy] = (
            ("discard", "Failed: " + ", ".join(failed) + ".")
            if failed
            else ("keep", "All realized-economics and execution-mechanics gates passed.")
        )
    return decisions


def _best_kept_strategy(summary: pd.DataFrame) -> str | None:
    kept = summary[summary["decision"] == "keep"]
    if kept.empty:
        return None
    return str(
        kept.sort_values(
            ["loss_cvar_95_bps", "pnl_vol_bps", "mean_net_pnl_bps"],
            ascending=[True, True, False],
        ).iloc[0]["strategy"]
    )


def plot_results(outputs: dict[str, pd.DataFrame], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials = outputs["trials"]
    paired = outputs["paired"]
    summary = outputs["summary"].set_index("strategy")
    strategies = list(summary.index)
    palette = ["#8A929A", "#70A288", "#D97732", "#7C5C9E", "#2F6B9A"]
    colors = {strategy: palette[index % len(palette)] for index, strategy in enumerate(strategies)}
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))

    axis = axes[0, 0]
    for strategy, group in trials.groupby("strategy", sort=False):
        ordered = group.sort_values("as_of")
        axis.plot(
            np.arange(1, len(ordered) + 1),
            ordered["net_pnl_bps"].cumsum(),
            marker="o",
            linewidth=1.8,
            color=colors[strategy],
            label=strategy.replace("_", " ").title(),
        )
    axis.set_title("Cumulative realized net P&L")
    axis.set_ylabel("Cumulative bps of parent gross")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[0, 1]
    for strategy, group in paired.groupby("strategy", sort=False):
        ordered = group.sort_values("as_of")
        axis.plot(
            np.arange(1, len(ordered) + 1),
            ordered["realized_net_pnl_delta_bps"],
            marker="o",
            linewidth=1.6,
            color=colors[strategy],
            label=strategy.replace("uncertainty_tie_", "") + "%",
        )
    axis.axhline(0.0, color="#59636E", linewidth=0.9)
    axis.set_title("Paired event P&L difference")
    axis.set_ylabel("Candidate minus baseline (bps)")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 0]
    measures = ["pnl_vol_bps", "loss_cvar_95_bps", "event_sequence_max_drawdown_bps"]
    x = np.arange(len(measures))
    width = 0.8 / len(strategies)
    for index, strategy in enumerate(strategies):
        axis.bar(
            x - 0.4 + width / 2 + index * width,
            [summary.loc[strategy, measure] for measure in measures],
            width,
            color=colors[strategy],
            label=strategy.replace("_", " ").title(),
        )
    axis.set_xticks(x, ["P&L vol", "Loss CVaR 95", "Max drawdown"])
    axis.set_title("Realized downside across events")
    axis.set_ylabel("bps")

    axis = axes[1, 1]
    candidates = [strategy for strategy in strategies if strategy != "fixed_1bp_tie"]
    change_rate = [100.0 * summary.loc[strategy, "schedule_change_rate"] for strategy in candidates]
    pnl = [summary.loc[strategy, "mean_net_pnl_bps"] for strategy in candidates]
    bars = axis.bar(
        np.arange(len(candidates)),
        change_rate,
        color=[colors[strategy] for strategy in candidates],
    )
    for bar, value in zip(bars, pnl):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.0,
            f"{value:+.2f} bp P&L",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xticks(
        np.arange(len(candidates)),
        [strategy.replace("uncertainty_tie_", "") + "%" for strategy in candidates],
    )
    axis.set_title("Schedule changes and mean realized P&L")
    axis.set_ylabel("Events changed (%)")
    axis.set_ylim(0.0, 105.0)

    for axis in axes.ravel():
        axis.grid(axis="y", color="#E1E5E8", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Point-in-time replay: uncertainty-aware frontier selection",
        x=0.05,
        y=0.995,
        ha="left",
        fontsize=16,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.08,
        top=0.92,
        hspace=0.35,
        wspace=0.28,
    )
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
        "--confidences",
        nargs="+",
        type=float,
        default=list(DEFAULT_CONFIDENCES),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("artifacts/frontier_uncertainty_selection"),
    )
    args = parser.parse_args()
    outputs, metadata = run_experiment(
        solver=args.solver,
        risk_measure=args.risk_measure,
        n_events=args.n_events,
        event_start=args.event_start,
        confidences=tuple(args.confidences),
    )
    prefix: Path = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(prefix.with_name(prefix.name + f"_{name}.csv"), index=False)
    chart = prefix.with_name(prefix.name + ".png")
    plot_results(outputs, chart)
    print(outputs["summary"].round(4).to_string(index=False))
    print(f"\nbest_kept_strategy: {metadata['best_kept_strategy']}")
    print(f"artifacts: {prefix.parent / (prefix.name + '*')}")


if __name__ == "__main__":
    main()
