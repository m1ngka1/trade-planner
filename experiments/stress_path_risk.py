"""Test a low-dimensional adverse event path against scenario CVaR.

The experiment converts the worst full-basket scenario tail into one centered
cross-date return factor.  Its coefficient is matched automatically to the
scenario tail that exceeds covariance-implied expected shortfall.  No user
coefficient or desired execution curve is supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np
import pandas as pd

from trade_planner import (
    BarraFactorRiskModel,
    CompositeCostModel,
    DEFAULT_RISK_PREFERENCES,
    ExpectedReturnAlphaModel,
    ParticipationCapModel,
    RebalanceRiskMeasure,
    RiskAversion,
    TCALinearBpsCost,
    TCAQuadraticParticipationImpact,
    TailSecondMomentPathRiskModel,
    TailStressPathRiskModel,
    TradePlanner,
    TradePlannerConfig,
    build_rebalance_frontier,
    default_constraints,
    evaluate_rebalance_schedule,
    infer_execution_cost_matrices,
    infer_execution_costs,
    tail_return_scenarios,
    tail_stress_return_path,
    weighted_loss_var_cvar,
)
from trade_planner.calibration import (
    _economic_lambda_scale,
    _scenario_tail_overlay_fraction,
    _security_covariance,
)

from experiments.rebalance_economic_calibration import (
    EVALUATION_SEEDS,
    N_SCENARIOS,
    _behavior_metrics,
    _scenario_pnl,
    _stress_residual_returns,
    economic_fixture,
)


STRESS_MULTIPLIERS = (0.0, 0.1, 0.3, 1.0, 2.0, 3.0, 10.0, 30.0, 100.0)


def run_experiment(
    solver: str = "OSQP",
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    ctx, classifications = economic_fixture()
    impact_matrix, linear_matrix = infer_execution_cost_matrices(ctx)
    impact_bps, linear_bps = infer_execution_costs(ctx)
    parent_target = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    parent_gross = float(np.sum(np.abs(parent_target * ctx.price[0])))

    started = perf_counter()
    variance_frontier = build_rebalance_frontier(
        ctx,
        solver=solver,
        risk_measure=RebalanceRiskMeasure.VARIANCE,
    )
    variance_runtime = perf_counter() - started
    started = perf_counter()
    hybrid_frontier = build_rebalance_frontier(
        ctx,
        solver=solver,
        risk_measure=RebalanceRiskMeasure.HYBRID_DOWNSIDE,
    )
    hybrid_runtime = perf_counter() - started
    stress_frontier, stress_results, stress_configs, stress_metadata = _build_stress_frontier(
        ctx,
        solver=solver,
        impact_matrix=impact_matrix,
        linear_matrix=linear_matrix,
        impact_bps=impact_bps,
        linear_bps=linear_bps,
    )

    variance_plans = {
        profile.value: variance_frontier.select(profile)
        for profile in RiskAversion
    }
    hybrid_plans = {
        profile.value: hybrid_frontier.select(profile)
        for profile in RiskAversion
    }
    stress_candidates = {
        profile.value: _select_candidate(
            stress_frontier,
            profile,
            parent_gross,
        )
        for profile in RiskAversion
    }

    schedules: dict[str, pd.DataFrame] = {}
    for profile, plan in variance_plans.items():
        schedules[f"variance_{profile}"] = plan.result.schedule
    for profile, plan in hybrid_plans.items():
        schedules[f"hybrid_96_{profile}"] = plan.result.schedule
    for profile, candidate in stress_candidates.items():
        schedules[f"stress_{profile}"] = stress_results[candidate].schedule

    evaluation_returns = [
        _stress_residual_returns(ctx, N_SCENARIOS, seed)
        + ctx.expected_return[None, :, :]
        for seed in EVALUATION_SEEDS
    ]
    evaluation_weights = np.full(N_SCENARIOS, 1.0 / N_SCENARIOS)
    trial_rows: list[dict[str, object]] = []
    profiles: list[pd.DataFrame] = []
    schedule_records: list[pd.DataFrame] = []
    for trial, schedule in schedules.items():
        economics = evaluate_rebalance_schedule(
            ctx,
            schedule,
            impact_bps_at_10pct_adv=impact_matrix,
            linear_cost_bps=linear_matrix,
        )
        behavior, profile_frame, _ = _behavior_metrics(
            ctx,
            classifications,
            schedule,
        )
        replicated_cvars = []
        replicated_profitable = []
        for returns in evaluation_returns:
            pnl = _scenario_pnl(ctx, schedule, economics, returns)
            _, cvar = weighted_loss_var_cvar(pnl, evaluation_weights)
            replicated_cvars.append(cvar)
            replicated_profitable.append(float(np.mean(pnl > 0.0)))
        if trial.startswith("variance_"):
            frontier_runtime = variance_runtime
            selected_multiplier = _selected_multiplier(
                variance_frontier.frontier,
                variance_frontier.results,
                variance_plans[trial.removeprefix("variance_")].result,
            )
        elif trial.startswith("hybrid_96_"):
            frontier_runtime = hybrid_runtime
            selected_multiplier = _selected_multiplier(
                hybrid_frontier.frontier,
                hybrid_frontier.results,
                hybrid_plans[trial.removeprefix("hybrid_96_")].result,
            )
        else:
            frontier_runtime = stress_metadata["frontier_runtime_seconds"]
            selected_multiplier = float(
                stress_frontier.loc[
                    stress_frontier["candidate"]
                    == stress_candidates[trial.removeprefix("stress_")],
                    "lambda_multiplier",
                ].iloc[0]
            )
        trial_rows.append(
            {
                "trial": trial,
                "model": trial.rsplit("_", 1)[0],
                "profile": trial.rsplit("_", 1)[1],
                "frontier_runtime_seconds": frontier_runtime,
                "selected_multiplier": selected_multiplier,
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
        profiles.append(profile_frame.assign(trial=trial))
        schedule_records.append(schedule.assign(trial=trial))

    trials = pd.DataFrame(trial_rows)
    trials["decision"] = "baseline"
    trials["decision_reason"] = "Comparison baseline."
    for index, row in trials[trials["model"] == "stress"].iterrows():
        baseline_trial = (
            "variance_high"
            if row["profile"] == "high"
            else f"hybrid_96_{row['profile']}"
        )
        baseline = trials.loc[trials["trial"] == baseline_trial].iloc[0]
        decision, reason = _decision_for_stress(
            row,
            baseline,
            parent_gross=parent_gross,
            hybrid_runtime=hybrid_runtime,
        )
        trials.loc[index, "decision"] = decision
        trials.loc[index, "decision_reason"] = reason

    variance_output = variance_frontier.frontier.copy()
    variance_output["frontier_model"] = "variance"
    hybrid_output = hybrid_frontier.frontier.copy()
    hybrid_output["frontier_model"] = "hybrid_96"
    stress_output = stress_frontier.copy()
    stress_output["frontier_model"] = "stress"
    frontier_output = pd.concat(
        [variance_output, hybrid_output, stress_output],
        ignore_index=True,
        sort=False,
    )
    outputs = {
        "trials": trials,
        "frontier": frontier_output,
        "profiles": pd.concat(profiles, ignore_index=True),
        "schedules": pd.concat(schedule_records, ignore_index=True),
    }
    metadata: dict[str, object] = {
        **stress_metadata,
        "parent_gross": parent_gross,
        "variance_runtime_seconds": variance_runtime,
        "hybrid_runtime_seconds": hybrid_runtime,
    }
    return outputs, metadata


def _build_stress_frontier(
    ctx,
    *,
    solver: str,
    impact_matrix: np.ndarray,
    linear_matrix: np.ndarray,
    impact_bps: float,
    linear_bps: float,
    path_model: str = "mean",
) -> tuple[pd.DataFrame, Mapping[str, object], Mapping[str, TradePlannerConfig], dict[str, object]]:
    if path_model not in {"mean", "second_moment"}:
        raise ValueError("path_model must be 'mean' or 'second_moment'")
    risk_model = BarraFactorRiskModel()
    base_variance_weight = _economic_lambda_scale(
        ctx,
        risk_model,
        impact_bps,
        linear_bps,
    )
    excess_tail_fraction = _scenario_tail_overlay_fraction(ctx, risk_model, 0.95)
    target = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    full_covariance_variance = 0.0
    for date_index in range(len(ctx.dates)):
        target_dollars = ctx.price[date_index] * target
        covariance = _security_covariance(ctx, date_index, risk_model)
        full_covariance_variance += float(target_dollars @ covariance @ target_dollars)
    target_positions = ctx.price * target[None, :]
    if path_model == "mean":
        stress_path, regime_variance = tail_stress_return_path(ctx)
        full_stress_pnl = float(np.sum(stress_path * target_positions))
        full_stress_variance = regime_variance * full_stress_pnl**2
    else:
        tail_scenarios, tail_weights = tail_return_scenarios(ctx)
        tail_pnl = np.einsum("stn,tn->s", tail_scenarios, target_positions)
        full_stress_variance = float(np.dot(tail_weights, np.square(tail_pnl)))
    stress_variance_scale = (
        excess_tail_fraction
        * full_covariance_variance
        / max(full_stress_variance, 1e-12)
    )

    rows: list[dict[str, object]] = []
    results: dict[str, object] = {}
    configs: dict[str, TradePlannerConfig] = {}
    started = perf_counter()
    for multiplier in STRESS_MULTIPLIERS:
        covariance_weight = base_variance_weight * multiplier
        stress_weight = covariance_weight * stress_variance_scale
        candidate = (
            f"stress_{path_model}__multiplier_{multiplier:.6g}"
            f"__variance_{covariance_weight:.6g}__path_{stress_weight:.6g}"
        )
        config = TradePlannerConfig(
            participation_model=ParticipationCapModel(),
            risk_model=risk_model,
            cost_model=CompositeCostModel(
                terms=(
                    TCAQuadraticParticipationImpact(impact_matrix),
                    TCALinearBpsCost(linear_matrix),
                )
            ),
            constraints=default_constraints(),
            residual_risk_weight=0.0,
            inventory_risk_weight=covariance_weight,
            inventory_alpha_model=ExpectedReturnAlphaModel(),
            inventory_path_risk_weight=stress_weight,
            inventory_path_risk_model=(
                (
                    TailStressPathRiskModel()
                    if path_model == "mean"
                    else TailSecondMomentPathRiskModel()
                )
                if stress_weight > 0
                else None
            ),
            terminal_penalty=None,
            solver=solver,
        )
        configs[candidate] = config
        try:
            result = TradePlanner(config).solve(ctx)
            metrics = evaluate_rebalance_schedule(
                ctx,
                result.schedule,
                risk_model=risk_model,
                impact_bps_at_10pct_adv=impact_matrix,
                linear_cost_bps=linear_matrix,
            )
        except Exception as error:
            rows.append(
                {
                    "candidate": candidate,
                    "lambda_multiplier": multiplier,
                    "inventory_risk_weight": covariance_weight,
                    "inventory_path_risk_weight": stress_weight,
                    "status": type(error).__name__,
                    "failure_reason": str(error),
                }
            )
            continue
        results[candidate] = result
        rows.append(
            {
                "candidate": candidate,
                "lambda_multiplier": multiplier,
                "inventory_risk_weight": covariance_weight,
                "inventory_path_risk_weight": stress_weight,
                "status": str(result.diagnostics["status"]),
                **metrics.as_dict(),
            }
        )
    runtime = perf_counter() - started
    metadata = {
        "frontier_runtime_seconds": runtime,
        "base_variance_weight": base_variance_weight,
        "excess_tail_fraction": excess_tail_fraction,
        "full_covariance_variance": full_covariance_variance,
        "full_stress_variance": full_stress_variance,
        "stress_variance_scale": stress_variance_scale,
        "path_model": path_model,
    }
    return pd.DataFrame(rows), results, configs, metadata


def _select_candidate(
    frontier: pd.DataFrame,
    profile: RiskAversion,
    parent_gross: float,
) -> str:
    valid = frontier[frontier["status"].isin(("optimal", "optimal_inaccurate"))].copy()
    risk_column = "loss_cvar_95_dollars"
    preference = DEFAULT_RISK_PREFERENCES[profile]
    minimum = float(valid[risk_column].min())
    maximum = float(valid[risk_column].max())
    budget = minimum + preference.risk_frontier_fraction * (maximum - minimum)
    eligible = valid[valid[risk_column] <= budget + abs(budget) * 1e-8].copy()
    best_pnl = float(eligible["expected_net_pnl_dollars"].max())
    tied = eligible[
        eligible["expected_net_pnl_dollars"] >= best_pnl - parent_gross / 10_000.0
    ]
    selected = tied.sort_values(
        [risk_column, "pnl_vol_dollars", "expected_net_pnl_dollars", "impact_cost_dollars"],
        ascending=[True, True, False, True],
    ).iloc[0]
    return str(selected["candidate"])


def _selected_multiplier(
    frontier: pd.DataFrame,
    results: Mapping[str, object],
    selected_result: object,
) -> float:
    candidate = next(
        candidate for candidate, result in results.items() if result is selected_result
    )
    return float(frontier.loc[frontier["candidate"] == candidate, "lambda_multiplier"].iloc[0])


def _decision_for_stress(
    row: pd.Series,
    baseline: pd.Series,
    *,
    parent_gross: float,
    hybrid_runtime: float,
) -> tuple[str, str]:
    gates = {
        "runtime_2x": hybrid_runtime / row["frontier_runtime_seconds"] >= 2.0,
        "pnl_within_1bp": (
            row["expected_net_pnl_dollars"]
            >= baseline["expected_net_pnl_dollars"] - parent_gross / 10_000.0
        ),
        "oos_cvar_within_0.25pct": (
            row["scenario_loss_cvar_95_mean_dollars"]
            <= 1.0025 * baseline["scenario_loss_cvar_95_mean_dollars"]
        ),
        "volatility_within_0.25pct": (
            row["pnl_vol_dollars"] <= 1.0025 * baseline["pnl_vol_dollars"]
        ),
        "urgent_not_later": (
            row["urgent_first_trade_day"] <= baseline["urgent_first_trade_day"]
        ),
        "small_not_earlier": (
            row["small_first_trade_day"] >= baseline["small_first_trade_day"]
        ),
        "factor_within_1pp": (
            row["early_factor_imbalance_pct"]
            <= baseline["early_factor_imbalance_pct"] + 1.0
        ),
        "ramp_within_10pct": (
            row["late_early_gross_ratio"]
            >= 0.90 * baseline["late_early_gross_ratio"]
        ),
    }
    if row["profile"] == "high":
        gates["material_cvar_improvement"] = (
            row["scenario_loss_cvar_95_mean_dollars"]
            <= 0.9975 * baseline["scenario_loss_cvar_95_mean_dollars"]
        )
    failed = [name for name, passed in gates.items() if not passed]
    return (
        ("keep", "All acceptance gates passed.")
        if not failed
        else ("discard", "Failed: " + ", ".join(failed) + ".")
    )


def plot_results(outputs: dict[str, pd.DataFrame], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials = outputs["trials"].set_index("trial")
    profiles = outputs["profiles"]
    frontier = outputs["frontier"]
    colors = {"variance": "#D97732", "hybrid_96": "#7C5C9E", "stress": "#2F6B9A"}
    labels = {"variance": "Variance", "hybrid_96": "Hybrid 96", "stress": "Tail stress"}
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.6))

    axis = axes[0, 0]
    for model in labels:
        solved = frontier[
            (frontier["frontier_model"] == model)
            & frontier["status"].isin(("optimal", "optimal_inaccurate"))
        ].sort_values("pnl_vol_dollars")
        axis.plot(
            solved["pnl_vol_dollars"] / 1e6,
            solved["expected_net_pnl_dollars"] / 1e3,
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            color=colors[model],
            label=labels[model],
        )
    axis.set_title("Expected net P&L versus P&L volatility")
    axis.set_xlabel("P&L volatility ($m)")
    axis.set_ylabel("Expected net P&L ($000)")
    axis.legend(frameon=False)

    axis = axes[0, 1]
    medium_trials = ["variance_medium", "hybrid_96_medium", "stress_medium"]
    for trial in medium_trials:
        model = trial.removesuffix("_medium")
        curve = profiles[profiles["trial"] == trial].sort_values("date")
        axis.plot(
            np.arange(1, len(curve) + 1),
            curve["daily_gross_pct"],
            marker="o",
            linewidth=2,
            color=colors[model],
            label=labels[model],
        )
    axis.set_title("Medium-profile daily gross volume")
    axis.set_xlabel("Planner day")
    axis.set_ylabel("Daily gross (% of parent)")
    axis.legend(frameon=False)

    axis = axes[1, 0]
    for trial in medium_trials:
        model = trial.removesuffix("_medium")
        curve = profiles[profiles["trial"] == trial].sort_values("date")
        axis.plot(
            np.arange(1, len(curve) + 1),
            curve["max_factor_imbalance_pct"],
            marker="o",
            linewidth=2,
            color=colors[model],
            label=labels[model],
        )
    axis.set_title("Country/sector/industry balance")
    axis.set_xlabel("Planner day")
    axis.set_ylabel("Maximum factor imbalance (%)")

    axis = axes[1, 1]
    rows = trials.loc[medium_trials]
    x = np.arange(len(rows))
    values = rows["scenario_loss_cvar_95_mean_dollars"].to_numpy(float) / 1e6
    errors = rows["scenario_loss_cvar_95_std_dollars"].to_numpy(float) / 1e6
    bars = axis.bar(
        x,
        values,
        yerr=errors,
        capsize=3,
        color=[colors[trial.removesuffix("_medium")] for trial in medium_trials],
        alpha=0.75,
    )
    for bar, trial in zip(bars, medium_trials):
        row = trials.loc[trial]
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.12,
            f"{row['frontier_runtime_seconds']:.1f}s · net ${row['expected_net_pnl_dollars']/1e3:.0f}k",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xticks(x, [labels[trial.removesuffix("_medium")] for trial in medium_trials])
    axis.set_title("Independent 95% loss CVaR")
    axis.set_ylabel("Mean loss CVaR ($m, lower is better)")

    for axis in axes.ravel():
        axis.grid(axis="y", color="#E1E5E8", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Scenario-derived tail stress path",
        x=0.045,
        y=0.995,
        ha="left",
        fontsize=16,
    )
    fig.tight_layout(rect=(0.025, 0.02, 0.995, 0.95), h_pad=2.5, w_pad=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="OSQP")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("artifacts/stress_path_risk"),
    )
    args = parser.parse_args()
    outputs, metadata = run_experiment(args.solver)
    prefix: Path = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(prefix.with_name(prefix.name + f"_{name}.csv"), index=False)
    chart = prefix.with_name(prefix.name + ".png")
    plot_results(outputs, chart)
    print(outputs["trials"].round(4).to_string(index=False))
    print("\nAutomatic stress calibration:")
    for key in (
        "excess_tail_fraction",
        "full_covariance_variance",
        "full_stress_variance",
        "stress_variance_scale",
        "frontier_runtime_seconds",
    ):
        print(f"{key}: {metadata[key]:.8g}")
    print(f"artifacts: {prefix.parent / (prefix.name + '*')}")


if __name__ == "__main__":
    main()
