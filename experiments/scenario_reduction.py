"""Benchmark tail-preserving scenario reduction for hybrid rebalance risk.

The optimizer is fit with either all 256 scenarios or a deterministic weighted
subset.  Every resulting schedule is evaluated on the same five independent
5,000-path samples, so faster fitting cannot improve its score by changing the
test distribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from trade_planner import (
    RebalanceRiskMeasure,
    build_rebalance_frontier,
    evaluate_rebalance_schedule,
    infer_execution_cost_matrices,
    weighted_loss_var_cvar,
)

from experiments.rebalance_economic_calibration import (
    EVALUATION_SEEDS,
    N_SCENARIOS,
    _behavior_metrics,
    _scenario_pnl,
    _stress_residual_returns,
    _trade_matrix,
    economic_fixture,
)


SCENARIO_LIMITS: tuple[int | None, ...] = (None, 96, 64)


def run_experiment(
    solver: str = "OSQP",
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    ctx, classifications = economic_fixture()
    impact, linear = infer_execution_cost_matrices(ctx)
    evaluation_returns = [
        _stress_residual_returns(ctx, N_SCENARIOS, seed)
        + ctx.expected_return[None, :, :]
        for seed in EVALUATION_SEEDS
    ]
    evaluation_weights = np.full(N_SCENARIOS, 1.0 / N_SCENARIOS)
    parent_target = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    parent_gross = float(np.sum(np.abs(parent_target * ctx.price[0])))

    rows: list[dict[str, object]] = []
    profiles: list[pd.DataFrame] = []
    schedules: list[pd.DataFrame] = []
    for limit in SCENARIO_LIMITS:
        trial = "full_256" if limit is None else f"reduced_{limit}"
        started = perf_counter()
        frontier = build_rebalance_frontier(
            ctx,
            solver=solver,
            risk_measure=RebalanceRiskMeasure.HYBRID_DOWNSIDE,
            max_optimization_scenarios=limit,
        )
        elapsed = perf_counter() - started
        plan = frontier.select("medium")
        economics = evaluate_rebalance_schedule(
            ctx,
            plan.result.schedule,
            impact_bps_at_10pct_adv=impact,
            linear_cost_bps=linear,
        )
        behavior, profile, _ = _behavior_metrics(
            ctx,
            classifications,
            plan.result.schedule,
        )
        replicated_cvars = []
        replicated_profitable = []
        for returns in evaluation_returns:
            pnl = _scenario_pnl(ctx, plan.result.schedule, economics, returns)
            _, cvar = weighted_loss_var_cvar(pnl, evaluation_weights)
            replicated_cvars.append(cvar)
            replicated_profitable.append(float(np.mean(pnl > 0.0)))
        rows.append(
            {
                "trial": trial,
                "scenario_limit": "all" if limit is None else limit,
                "optimization_scenario_count": frontier.optimization_scenario_count,
                "runtime_seconds": elapsed,
                **economics.as_dict(),
                **behavior,
                "scenario_loss_cvar_95_mean_dollars": float(np.mean(replicated_cvars)),
                "scenario_loss_cvar_95_std_dollars": float(
                    np.std(replicated_cvars, ddof=1)
                ),
                "scenario_loss_cvar_95_worst_dollars": float(np.max(replicated_cvars)),
                "scenario_probability_profitable_mean": float(
                    np.mean(replicated_profitable)
                ),
            }
        )
        profiles.append(profile.assign(trial=trial))
        schedules.append(plan.result.schedule.assign(trial=trial))

    trials = pd.DataFrame(rows)
    baseline = trials.loc[trials["trial"] == "full_256"].iloc[0]
    baseline_schedule = schedules[0]
    baseline_trades = _trade_matrix(ctx, baseline_schedule)
    decisions = []
    decision_reasons = []
    speedups = []
    schedule_differences = []
    for row, schedule in zip(trials.itertuples(index=False), schedules):
        speedup = float(baseline["runtime_seconds"] / row.runtime_seconds)
        speedups.append(speedup)
        trades = _trade_matrix(ctx, schedule)
        schedule_difference = float(
            np.sum(np.abs(trades - baseline_trades) * ctx.price)
            / parent_gross
        )
        schedule_differences.append(schedule_difference)
        if row.trial == "full_256":
            decisions.append("baseline")
            decision_reasons.append("Unreduced control.")
            continue
        gates = {
            "runtime_2x": speedup >= 2.0,
            "pnl_within_1bp": (
                row.expected_net_pnl_dollars
                >= baseline["expected_net_pnl_dollars"] - parent_gross / 10_000.0
            ),
            "oos_cvar_within_0.25pct": (
                row.scenario_loss_cvar_95_mean_dollars
                <= 1.0025 * baseline["scenario_loss_cvar_95_mean_dollars"]
            ),
            "urgent_not_later": (
                row.urgent_first_trade_day <= baseline["urgent_first_trade_day"]
            ),
            "small_not_earlier": (
                row.small_first_trade_day >= baseline["small_first_trade_day"]
            ),
            "factor_within_1pp": (
                row.early_factor_imbalance_pct
                <= baseline["early_factor_imbalance_pct"] + 1.0
            ),
            "ramp_within_10pct": (
                row.late_early_gross_ratio
                >= 0.90 * baseline["late_early_gross_ratio"]
            ),
        }
        failed = [name for name, passed in gates.items() if not passed]
        decisions.append("keep" if not failed else "discard")
        decision_reasons.append(
            "All acceptance gates passed."
            if not failed
            else "Failed: " + ", ".join(failed) + "."
        )
    trials["runtime_speedup"] = speedups
    trials["schedule_l1_difference_parent_pct"] = 100.0 * np.asarray(
        schedule_differences
    )
    trials["decision"] = decisions
    trials["decision_reason"] = decision_reasons

    outputs = {
        "trials": trials,
        "profiles": pd.concat(profiles, ignore_index=True),
        "schedules": pd.concat(schedules, ignore_index=True),
    }
    metadata: dict[str, object] = {
        "parent_gross": parent_gross,
        "n_evaluation_scenarios": N_SCENARIOS,
        "n_evaluation_replications": len(EVALUATION_SEEDS),
    }
    return outputs, metadata


def plot_results(outputs: dict[str, pd.DataFrame], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials = outputs["trials"].set_index("trial")
    profiles = outputs["profiles"]
    colors = {"full_256": "#8A929A", "reduced_96": "#2F6B9A", "reduced_64": "#D97732"}
    labels = {"full_256": "Full 256", "reduced_96": "Reduced 96", "reduced_64": "Reduced 64"}
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))

    axis = axes[0]
    for trial in labels:
        curve = profiles[profiles["trial"] == trial].sort_values("date")
        axis.plot(
            np.arange(1, len(curve) + 1),
            curve["daily_gross_pct"],
            marker="o",
            linewidth=2,
            color=colors[trial],
            label=labels[trial],
        )
    axis.set_title("Optimizer-derived daily volume")
    axis.set_xlabel("Planner day")
    axis.set_ylabel("Daily gross (% of parent)")
    axis.legend(frameon=False)

    axis = axes[1]
    for trial in labels:
        curve = profiles[profiles["trial"] == trial].sort_values("date")
        axis.plot(
            np.arange(1, len(curve) + 1),
            curve["max_factor_imbalance_pct"],
            marker="o",
            linewidth=2,
            color=colors[trial],
            label=labels[trial],
        )
    axis.set_title("Country/sector/industry balance")
    axis.set_xlabel("Planner day")
    axis.set_ylabel("Maximum factor imbalance (%)")

    axis = axes[2]
    order = list(labels)
    x = np.arange(len(order))
    cvar = trials.loc[order, "scenario_loss_cvar_95_mean_dollars"].to_numpy(float) / 1e6
    cvar_error = trials.loc[order, "scenario_loss_cvar_95_std_dollars"].to_numpy(float) / 1e6
    bars = axis.bar(
        x,
        cvar,
        yerr=cvar_error,
        capsize=3,
        color=[colors[trial] for trial in order],
        alpha=0.75,
    )
    for bar, trial in zip(bars, order):
        row = trials.loc[trial]
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.11,
            f"{row['runtime_speedup']:.1f}x · net ${row['expected_net_pnl_dollars']/1e3:.0f}k",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xticks(x, [labels[trial] for trial in order])
    axis.set_title("Independent 95% loss CVaR")
    axis.set_ylabel("Mean loss CVaR ($m, lower is better)")

    for axis in axes:
        axis.grid(axis="y", color="#E1E5E8", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Tail-preserving scenario reduction",
        x=0.035,
        y=0.995,
        ha="left",
        fontsize=16,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.995, 0.94), w_pad=2.4)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="OSQP")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("artifacts/scenario_reduction"),
    )
    args = parser.parse_args()
    outputs, _ = run_experiment(args.solver)
    prefix: Path = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(prefix.with_name(prefix.name + f"_{name}.csv"), index=False)
    chart = prefix.with_name(prefix.name + ".png")
    plot_results(outputs, chart)
    print(outputs["trials"].round(4).to_string(index=False))
    print(f"artifacts: {prefix.parent / (prefix.name + '*')}")


if __name__ == "__main__":
    main()
