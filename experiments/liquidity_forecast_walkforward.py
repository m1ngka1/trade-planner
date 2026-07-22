"""Point-in-time replay of flat versus forecast event liquidity.

The candidate receives a date/name ADV forecast learned from disjoint history.
Capacity and impact then cause the optimizer to choose the schedule; no daily
trade amount or target volume curve is constrained directly.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from trade_planner import (
    PlannerContext,
    PointInTimeRebalanceEvent,
    RebalanceRiskMeasure,
    RiskAversion,
    build_rebalance_frontier,
    evaluate_realized_rebalance_schedule,
)

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

_all_seeds = (
    LIQUIDITY_EVENT_SEEDS
    + LIQUIDITY_SCENARIO_SEEDS
    + REALIZED_LIQUIDITY_SEEDS
    + (LIQUIDITY_CALIBRATION_SEED,)
)
assert len(set(_all_seeds)) == len(_all_seeds)


def run_experiment(
    solver: str = "OSQP",
    n_events: int = 12,
    event_start: int = 0,
    risk_aversion: str = "medium",
    liquidity_quantile: float | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Compare flat ADV with an automatically buffered event-liquidity curve."""

    if (
        n_events < 2
        or event_start < 0
        or event_start + n_events > len(LIQUIDITY_EVENT_SEEDS)
    ):
        raise ValueError("event_start and n_events must select at least two fresh events")
    parsed_aversion = RiskAversion.parse(risk_aversion)
    resolved_quantile = (
        LIQUIDITY_QUANTILES[parsed_aversion]
        if liquidity_quantile is None
        else float(liquidity_quantile)
    )
    if not 0.0 < resolved_quantile < 1.0:
        raise ValueError("liquidity_quantile must be strictly between zero and one")
    base_ctx, classifications = economic_fixture()
    alpha_uncertainty = _calibrated_alpha_uncertainty(base_ctx)
    calibration = calibrate_liquidity_distribution(base_ctx)
    forecast_adv = forecast_adv_for_risk_profile(
        base_ctx,
        calibration,
        parsed_aversion,
        quantile=resolved_quantile,
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
        LIQUIDITY_EVENT_SEEDS[event_start : event_start + n_events],
        LIQUIDITY_SCENARIO_SEEDS[event_start : event_start + n_events],
        REALIZED_LIQUIDITY_SEEDS[event_start : event_start + n_events],
    )
    for cohort_index, (event_seed, scenario_seed, liquidity_seed) in enumerate(
        selected,
        start=event_start,
    ):
        event_index = FRESH_EVENT_INDEX_OFFSET + cohort_index
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
        for strategy, planning_ctx in contexts.items():
            frontier = build_rebalance_frontier(
                planning_ctx,
                solver=solver,
                lambda_multipliers=LAMBDA_MULTIPLIERS,
                risk_measure=RebalanceRiskMeasure.VARIANCE,
            )
            plan = frontier.select(parsed_aversion)
            schedule = plan.result.schedule
            realized, daily = evaluate_realized_rebalance_schedule(event, schedule)
            behavior, profile, exposures = _behavior_metrics(
                planning_ctx,
                classifications,
                schedule,
            )
            audit = _schedule_audit(planning_ctx, schedule)
            row = {
                "event_id": event.event_id,
                "as_of": pd.Timestamp(event.as_of),
                "strategy": strategy,
                "solver": plan.config.solver,
                "risk_aversion": parsed_aversion.value,
                "liquidity_forecast_quantile": (
                    0.50
                    if strategy == "static_open_loop"
                    else resolved_quantile
                ),
                "forecast_rmse_bps": forecast_rmse_bps,
                "selected_inventory_risk_weight": (
                    plan.config.inventory_risk_weight
                ),
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
                    "risk_aversion": parsed_aversion.value,
                    "liquidity_forecast_quantile": (
                        0.50
                        if strategy == "static_open_loop"
                        else resolved_quantile
                    ),
                    "inventory_risk_weight": plan.config.inventory_risk_weight,
                    "risk_budget_dollars": plan.risk_budget_dollars,
                }
            )
            frontier_rows.append(
                frontier.frontier.assign(
                    event_id=event.event_id,
                    strategy=strategy,
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
    if event_start >= 12:
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
        "holdout_untouched": event_start == 0 and event_start + n_events <= 12,
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
    args = parser.parse_args()
    outputs, metadata = run_experiment(
        solver=args.solver,
        n_events=args.n_events,
        event_start=args.event_start,
        risk_aversion=args.risk_aversion,
        liquidity_quantile=args.liquidity_quantile,
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
    print(f"holdout untouched: {metadata['holdout_untouched']}")
    print(f"artifacts: {prefix.parent / (prefix.name + '*')}")


if __name__ == "__main__":
    main()
