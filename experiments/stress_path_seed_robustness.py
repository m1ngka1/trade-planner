"""Replicate tail-stress versus hybrid CVaR across optimization seeds."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from trade_planner import (
    RebalanceRiskMeasure,
    RiskAversion,
    build_rebalance_frontier,
    evaluate_rebalance_schedule,
    infer_execution_cost_matrices,
    infer_execution_costs,
    weighted_loss_var_cvar,
)

from experiments.rebalance_economic_calibration import (
    EVALUATION_SEEDS,
    N_OPTIMIZATION_SCENARIOS,
    N_SCENARIOS,
    _behavior_metrics,
    _scenario_pnl,
    _stress_residual_returns,
    economic_fixture,
)
from experiments.stress_path_risk import (
    _build_stress_frontier,
    _select_candidate,
)


OPTIMIZATION_SEEDS = (20260724, 20260731, 20260807, 20260814, 20260821)
PROFILES = ("medium", "low")
assert set(OPTIMIZATION_SEEDS).isdisjoint(EVALUATION_SEEDS)


def run_experiment(
    solver: str = "OSQP",
    path_model: str = "mean",
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    if path_model not in {"mean", "second_moment"}:
        raise ValueError("path_model must be 'mean' or 'second_moment'")
    stress_model_name = f"stress_{path_model}"
    base_ctx, classifications = economic_fixture()
    evaluation_returns = [
        _stress_residual_returns(base_ctx, N_SCENARIOS, seed)
        + base_ctx.expected_return[None, :, :]
        for seed in EVALUATION_SEEDS
    ]
    evaluation_weights = np.full(N_SCENARIOS, 1.0 / N_SCENARIOS)
    target = base_ctx.orders["target_shares"].reindex(base_ctx.symbols).to_numpy(float)
    parent_gross = float(np.sum(np.abs(target * base_ctx.price[0])))

    trial_rows: list[dict[str, object]] = []
    for optimization_seed in OPTIMIZATION_SEEDS:
        ctx = replace(
            base_ctx,
            return_residual_scenarios=_stress_residual_returns(
                base_ctx,
                N_OPTIMIZATION_SCENARIOS,
                optimization_seed,
            ),
        )
        impact_matrix, linear_matrix = infer_execution_cost_matrices(ctx)
        impact_bps, linear_bps = infer_execution_costs(ctx)

        started = perf_counter()
        hybrid_frontier = build_rebalance_frontier(
            ctx,
            solver=solver,
            risk_measure=RebalanceRiskMeasure.HYBRID_DOWNSIDE,
        )
        hybrid_runtime = perf_counter() - started
        stress_frontier, stress_results, _, stress_metadata = _build_stress_frontier(
            ctx,
            solver=solver,
            impact_matrix=impact_matrix,
            linear_matrix=linear_matrix,
            impact_bps=impact_bps,
            linear_bps=linear_bps,
            path_model=path_model,
        )

        for profile in PROFILES:
            hybrid_schedule = hybrid_frontier.select(profile).result.schedule
            stress_candidate = _select_candidate(
                stress_frontier,
                RiskAversion.parse(profile),
                parent_gross,
            )
            stress_schedule = stress_results[stress_candidate].schedule
            for model, schedule, runtime in (
                ("hybrid_96", hybrid_schedule, hybrid_runtime),
                (
                    stress_model_name,
                    stress_schedule,
                    stress_metadata["frontier_runtime_seconds"],
                ),
            ):
                economics = evaluate_rebalance_schedule(
                    ctx,
                    schedule,
                    impact_bps_at_10pct_adv=impact_matrix,
                    linear_cost_bps=linear_matrix,
                )
                behavior, _, _ = _behavior_metrics(ctx, classifications, schedule)
                replicated_cvars = []
                for returns in evaluation_returns:
                    pnl = _scenario_pnl(ctx, schedule, economics, returns)
                    _, cvar = weighted_loss_var_cvar(pnl, evaluation_weights)
                    replicated_cvars.append(cvar)
                trial_rows.append(
                    {
                        "optimization_seed": optimization_seed,
                        "model": model,
                        "profile": profile,
                        "frontier_runtime_seconds": runtime,
                        "stress_variance_scale": (
                            stress_metadata["stress_variance_scale"]
                            if model == stress_model_name
                            else np.nan
                        ),
                        **economics.as_dict(),
                        **behavior,
                        "scenario_loss_cvar_95_mean_dollars": float(
                            np.mean(replicated_cvars)
                        ),
                        "scenario_loss_cvar_95_std_dollars": float(
                            np.std(replicated_cvars, ddof=1)
                        ),
                        "scenario_loss_cvar_95_worst_dollars": float(
                            np.max(replicated_cvars)
                        ),
                    }
                )

    trials = pd.DataFrame(trial_rows)
    paired_rows: list[dict[str, object]] = []
    for (seed, profile), group in trials.groupby(["optimization_seed", "profile"]):
        by_model = group.set_index("model")
        hybrid = by_model.loc["hybrid_96"]
        stress = by_model.loc[stress_model_name]
        gates = {
            "runtime_10x": (
                hybrid["frontier_runtime_seconds"]
                / stress["frontier_runtime_seconds"]
                >= 10.0
            ),
            "pnl_within_1bp": (
                stress["expected_net_pnl_dollars"]
                >= hybrid["expected_net_pnl_dollars"] - parent_gross / 10_000.0
            ),
            "cvar_within_0.25pct": (
                stress["scenario_loss_cvar_95_mean_dollars"]
                <= 1.0025 * hybrid["scenario_loss_cvar_95_mean_dollars"]
            ),
            "volatility_within_0.25pct": (
                stress["pnl_vol_dollars"] <= 1.0025 * hybrid["pnl_vol_dollars"]
            ),
            "urgent_not_later": (
                stress["urgent_first_trade_day"] <= hybrid["urgent_first_trade_day"]
            ),
            "small_not_earlier": (
                stress["small_first_trade_day"] >= hybrid["small_first_trade_day"]
            ),
            "factor_within_1pp": (
                stress["early_factor_imbalance_pct"]
                <= hybrid["early_factor_imbalance_pct"] + 1.0
            ),
            "ramp_within_10pct": (
                stress["late_early_gross_ratio"]
                >= 0.90 * hybrid["late_early_gross_ratio"]
            ),
        }
        failed = [name for name, passed in gates.items() if not passed]
        paired_rows.append(
            {
                "optimization_seed": seed,
                "profile": profile,
                "path_model": path_model,
                "runtime_speedup": (
                    hybrid["frontier_runtime_seconds"]
                    / stress["frontier_runtime_seconds"]
                ),
                "expected_net_pnl_delta_dollars": (
                    stress["expected_net_pnl_dollars"]
                    - hybrid["expected_net_pnl_dollars"]
                ),
                "pnl_vol_delta_pct": 100.0
                * (stress["pnl_vol_dollars"] / hybrid["pnl_vol_dollars"] - 1.0),
                "scenario_loss_cvar_95_mean_delta_pct": 100.0
                * (
                    stress["scenario_loss_cvar_95_mean_dollars"]
                    / hybrid["scenario_loss_cvar_95_mean_dollars"]
                    - 1.0
                ),
                "early_factor_imbalance_delta_pp": (
                    stress["early_factor_imbalance_pct"]
                    - hybrid["early_factor_imbalance_pct"]
                ),
                "late_early_ratio_delta": (
                    stress["late_early_gross_ratio"]
                    - hybrid["late_early_gross_ratio"]
                ),
                "urgent_start_delta_days": (
                    stress["urgent_first_trade_day"]
                    - hybrid["urgent_first_trade_day"]
                ),
                "small_start_delta_days": (
                    stress["small_first_trade_day"]
                    - hybrid["small_first_trade_day"]
                ),
                "decision": "keep" if not failed else "discard",
                "decision_reason": (
                    "All acceptance gates passed."
                    if not failed
                    else "Failed: " + ", ".join(failed) + "."
                ),
            }
        )
    paired = pd.DataFrame(paired_rows)
    metadata = {
        "parent_gross": parent_gross,
        "all_pairs_pass": float((paired["decision"] == "keep").all()),
        "path_model": path_model,
    }
    return {"trials": trials, "paired": paired}, metadata


def plot_results(
    outputs: dict[str, pd.DataFrame],
    output: Path,
    *,
    path_model: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paired = outputs["paired"]
    colors = {"medium": "#7C5C9E", "low": "#70A288"}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8))
    x = np.arange(len(OPTIMIZATION_SEEDS))

    axis = axes[0, 0]
    for profile in PROFILES:
        rows = paired[paired["profile"] == profile].sort_values("optimization_seed")
        axis.plot(
            x,
            rows["scenario_loss_cvar_95_mean_delta_pct"],
            marker="o",
            linewidth=2,
            color=colors[profile],
            label=profile.title(),
        )
    axis.axhline(0.25, color="#B04A4A", linestyle="--", linewidth=1.2, label="Acceptance ceiling")
    axis.axhline(0.0, color="#8A929A", linewidth=0.8)
    axis.set_title("Stress minus hybrid independent CVaR")
    axis.set_ylabel("Difference (%)")
    axis.legend(frameon=False)

    axis = axes[0, 1]
    for profile in PROFILES:
        rows = paired[paired["profile"] == profile].sort_values("optimization_seed")
        axis.plot(
            x,
            rows["expected_net_pnl_delta_dollars"] / 1e3,
            marker="o",
            linewidth=2,
            color=colors[profile],
            label=profile.title(),
        )
    axis.axhline(0.0, color="#8A929A", linewidth=0.8)
    axis.set_title("Expected net P&L difference")
    axis.set_ylabel("Stress minus hybrid ($000)")

    axis = axes[1, 0]
    for profile in PROFILES:
        rows = paired[paired["profile"] == profile].sort_values("optimization_seed")
        axis.plot(
            x,
            rows["early_factor_imbalance_delta_pp"],
            marker="o",
            linewidth=2,
            color=colors[profile],
            label=profile.title(),
        )
    axis.axhline(1.0, color="#B04A4A", linestyle="--", linewidth=1.2, label="Acceptance ceiling")
    axis.axhline(0.0, color="#8A929A", linewidth=0.8)
    axis.set_title("Early factor-imbalance difference")
    axis.set_ylabel("Stress minus hybrid (percentage points)")

    axis = axes[1, 1]
    for profile in PROFILES:
        rows = paired[paired["profile"] == profile].sort_values("optimization_seed")
        axis.plot(
            x,
            rows["runtime_speedup"],
            marker="o",
            linewidth=2,
            color=colors[profile],
            label=profile.title(),
        )
    axis.axhline(10.0, color="#B04A4A", linestyle="--", linewidth=1.2, label="Acceptance floor")
    axis.set_title("Frontier runtime speedup")
    axis.set_ylabel("Hybrid / stress runtime")

    seed_labels = [str(seed)[-4:] for seed in OPTIMIZATION_SEEDS]
    for axis in axes.ravel():
        axis.set_xticks(x, seed_labels)
        axis.set_xlabel("Optimization seed suffix")
        axis.grid(axis="y", color="#E1E5E8", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        (
            "Tail-stress robustness across independent optimization samples"
            if path_model == "mean"
            else "Tail-second-moment robustness across independent optimization samples"
        ),
        x=0.045,
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
        "--path-model",
        choices=("mean", "second_moment"),
        default="mean",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("artifacts/stress_path_seed_robustness"),
    )
    args = parser.parse_args()
    outputs, metadata = run_experiment(args.solver, args.path_model)
    prefix: Path = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(prefix.with_name(prefix.name + f"_{name}.csv"), index=False)
    chart = prefix.with_name(prefix.name + ".png")
    plot_results(outputs, chart, path_model=args.path_model)
    print(outputs["paired"].round(4).to_string(index=False))
    print(f"\nall_pairs_pass: {bool(metadata['all_pairs_pass'])}")
    print(f"artifacts: {prefix.parent / (prefix.name + '*')}")


if __name__ == "__main__":
    main()
