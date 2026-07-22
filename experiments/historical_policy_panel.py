"""Generate and select a chronological policy panel from a historical bundle."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.historical_replay import (
    BASELINE_STRATEGY,
    CHALLENGER_STRATEGY,
    run_historical_experiment,
)
from experiments.liquidity_forecast_walkforward import plot_results
from trade_planner import (
    HistoricalReplayBundle,
    InvestmentPolicyCoefficients,
    RiskAversion,
    build_monotone_policy_ladder,
    calibrate_alpha_decay_walk_forward,
    calibrate_risk_profiles_walk_forward,
    load_historical_replay_bundle,
)


HISTORICAL_POLICY_AGGRESSIVENESS = (0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00)
CAP_TOLERANCE_SHARES = 0.05
DIRECTION_TOLERANCE_SHARES = 0.001
COMPLETION_TOLERANCE_SHARES = 0.001
EARLY_FACTOR_TOLERANCE_PP = 1.0
RANK_RAMP_TOLERANCE = 0.10
NONDECREASING_TRANSITION_TOLERANCE = 1.0
P95_PARTICIPATION_TOLERANCE = 0.005
MAX_PARTICIPATION_TOLERANCE = 0.01


def run_historical_policy_panel(
    bundle: HistoricalReplayBundle,
    *,
    risk_aversion: str = "medium",
    solver: str = "CLARABEL",
    alpha_calibration: str = "walk_forward",
    policies: pd.DataFrame | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Solve every frozen policy, select chronologically, then replay selection."""

    parsed_aversion = RiskAversion.parse(risk_aversion)
    alpha_mode = str(alpha_calibration).strip().lower()
    if alpha_mode not in {"none", "walk_forward"}:
        raise ValueError("alpha_calibration must be 'none' or 'walk_forward'")
    policy_ladder = (
        build_monotone_policy_ladder(HISTORICAL_POLICY_AGGRESSIVENESS)
        if policies is None
        else policies.copy()
    )
    if alpha_mode == "walk_forward":
        alpha_result = calibrate_alpha_decay_walk_forward(bundle)
        planning_bundle = replace(bundle, events=alpha_result.events)
    else:
        alpha_result = None
        planning_bundle = bundle

    policy_trial_rows: list[dict[str, object]] = []
    policy_schedule_frames: list[pd.DataFrame] = []
    policy_profile_frames: list[pd.DataFrame] = []
    policy_exposure_frames: list[pd.DataFrame] = []
    policy_coefficient_frames: list[pd.DataFrame] = []
    policy_frontier_frames: list[pd.DataFrame] = []
    for policy_row in policy_ladder.to_dict("records"):
        policy = InvestmentPolicyCoefficients.from_mapping(policy_row)
        policy_outputs, _ = run_historical_experiment(
            planning_bundle,
            risk_aversion=parsed_aversion.value,
            solver=solver,
            alpha_calibration="none",
            policy_coefficients=policy,
        )
        _append_policy_trials(
            policy_trial_rows,
            policy_outputs,
            policy,
        )
        for output_name, collection in (
            ("schedules", policy_schedule_frames),
            ("profiles", policy_profile_frames),
            ("exposures", policy_exposure_frames),
            ("coefficients", policy_coefficient_frames),
            ("frontiers", policy_frontier_frames),
        ):
            candidate = policy_outputs[output_name].loc[
                policy_outputs[output_name]["strategy"].eq(CHALLENGER_STRATEGY)
            ].copy()
            collection.append(
                candidate.assign(
                    policy_id=policy.policy_id,
                    policy_aggressiveness=policy.policy_aggressiveness,
                )
            )

    event_manifest = pd.DataFrame(
        [
            {
                "event_id": str(event.event_id),
                "as_of": pd.Timestamp(event.as_of),
                "information_cutoff": pd.Timestamp(event.information_cutoff),
                "realized_available_at": pd.Timestamp(event.realized_available_at),
            }
            for event in planning_bundle.events
        ]
    )
    policy_trials = pd.DataFrame(policy_trial_rows)
    calibration = calibrate_risk_profiles_walk_forward(
        event_manifest,
        policy_trials,
        policy_ladder,
    )
    selected_profile = calibration.selections.loc[
        calibration.selections["risk_aversion"].eq(parsed_aversion.value)
    ]
    selected_policy_by_event = {
        str(row["event_id"]): InvestmentPolicyCoefficients.from_mapping(row)
        for row in selected_profile.to_dict("records")
    }
    selected_outputs, selected_metadata = run_historical_experiment(
        planning_bundle,
        risk_aversion=parsed_aversion.value,
        solver=solver,
        alpha_calibration="none",
        policy_coefficients=selected_policy_by_event,
    )
    if alpha_result is not None:
        selected_outputs["alpha_audit"] = alpha_result.audit
        selected_outputs["alpha_predictions"] = alpha_result.predictions
        selected_outputs["alpha_summary"] = alpha_result.summary
        selected_outputs["alpha_coefficients"] = alpha_result.coefficients
    selected_outputs.update(
        {
            "policy_events": event_manifest,
            "policy_ladder": calibration.policies,
            "policy_trials": policy_trials,
            "policy_selections": calibration.selections,
            "policy_evaluations": calibration.policy_evaluations,
            "policy_summary": calibration.summary,
            "policy_schedules": pd.concat(
                policy_schedule_frames,
                ignore_index=True,
            ),
            "policy_profiles": pd.concat(
                policy_profile_frames,
                ignore_index=True,
            ),
            "policy_exposures": pd.concat(
                policy_exposure_frames,
                ignore_index=True,
            ),
            "policy_coefficients": pd.concat(
                policy_coefficient_frames,
                ignore_index=True,
            ),
            "policy_frontiers": pd.concat(
                policy_frontier_frames,
                ignore_index=True,
            ),
        }
    )
    status_counts = selected_profile["status"].value_counts().to_dict()
    metadata = {
        **selected_metadata,
        "alpha_calibration": alpha_mode,
        "alpha_calibrated_event_count": (
            int(alpha_result.audit["status"].eq("calibrated").sum())
            if alpha_result is not None
            else 0
        ),
        "automatic_policy_calibration": True,
        "policy_candidate_count": len(policy_ladder),
        "policy_trial_count": len(policy_trials),
        "policy_selection_profile": parsed_aversion.value,
        "policy_selection_status_counts": status_counts,
    }
    return selected_outputs, metadata


def _append_policy_trials(
    rows: list[dict[str, object]],
    outputs: dict[str, pd.DataFrame],
    policy: InvestmentPolicyCoefficients,
) -> None:
    trials = outputs["trials"]
    paired = outputs["paired"].set_index("event_id")
    baseline = trials.loc[
        trials["strategy"].eq(BASELINE_STRATEGY)
    ].set_index("event_id")
    candidate = trials.loc[
        trials["strategy"].eq(CHALLENGER_STRATEGY)
    ].set_index("event_id")
    for event_id, candidate_row in candidate.iterrows():
        baseline_row = baseline.loc[event_id]
        paired_row = paired.loc[event_id]
        hard_pass = bool(
            candidate_row["max_cap_excess_shares"] <= CAP_TOLERANCE_SHARES
            and candidate_row["max_wrong_direction_shares"]
            <= DIRECTION_TOLERANCE_SHARES
            and candidate_row["terminal_completion_error_shares_audit"]
            <= COMPLETION_TOLERANCE_SHARES
        )
        behavior_pass = bool(
            paired_row["urgent_start_delta_days"] <= 0.0
            and paired_row["small_start_delta_days"] >= 0.0
            and paired_row["early_factor_imbalance_delta_pp"]
            <= EARLY_FACTOR_TOLERANCE_PP
            and candidate_row["late_early_gross_ratio"]
            >= max(1.0, 0.90 * baseline_row["late_early_gross_ratio"])
            and candidate_row["daily_gross_spearman"]
            >= baseline_row["daily_gross_spearman"] - RANK_RAMP_TOLERANCE
            and candidate_row["nondecreasing_transitions"]
            >= baseline_row["nondecreasing_transitions"]
            - NONDECREASING_TRANSITION_TOLERANCE
            and candidate_row["p95_realized_participation_rate"]
            <= baseline_row["p95_realized_participation_rate"]
            + P95_PARTICIPATION_TOLERANCE
            and candidate_row["max_realized_participation_rate"]
            <= baseline_row["max_realized_participation_rate"]
            + MAX_PARTICIPATION_TOLERANCE
        )
        rows.append(
            {
                "event_id": str(event_id),
                "policy_id": policy.policy_id,
                "net_pnl_bps": float(candidate_row["net_pnl_bps"]),
                "within_event_max_drawdown_bps": float(
                    candidate_row["within_event_max_drawdown_bps"]
                ),
                "hard_pass": hard_pass,
                "behavior_pass": behavior_pass,
                "impact_cost_dollars": float(candidate_row["impact_cost_dollars"]),
                "early_factor_imbalance_pct": float(
                    candidate_row["early_factor_imbalance_pct"]
                ),
                "late_early_gross_ratio": float(
                    candidate_row["late_early_gross_ratio"]
                ),
                "urgent_first_trade_day": float(
                    candidate_row["urgent_first_trade_day"]
                ),
                "small_first_trade_day": float(
                    candidate_row["small_first_trade_day"]
                ),
                "max_cap_excess_shares": float(
                    candidate_row["max_cap_excess_shares"]
                ),
                "max_wrong_direction_shares": float(
                    candidate_row["max_wrong_direction_shares"]
                ),
                "terminal_completion_error_shares": float(
                    candidate_row["terminal_completion_error_shares_audit"]
                ),
            }
        )


def plot_historical_policy_panel(
    outputs: dict[str, pd.DataFrame],
    output: Path,
) -> None:
    """Render candidate economics, operational pass rates, and selections."""

    import matplotlib.pyplot as plt

    policies = outputs["policy_ladder"].set_index("policy_id")
    trials = outputs["policy_trials"].merge(
        policies[["policy_aggressiveness"]],
        left_on="policy_id",
        right_index=True,
        validate="many_to_one",
    )
    selections = outputs["policy_selections"]
    by_policy = trials.groupby("policy_aggressiveness", as_index=False).agg(
        mean_net_pnl_bps=("net_pnl_bps", "mean"),
        pnl_vol_bps=("net_pnl_bps", "std"),
        hard_pass_rate=("hard_pass", "mean"),
        behavior_pass_rate=("behavior_pass", "mean"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(
        by_policy["policy_aggressiveness"],
        by_policy["mean_net_pnl_bps"],
        marker="o",
        label="mean net P&L",
    )
    axes[0, 0].plot(
        by_policy["policy_aggressiveness"],
        by_policy["pnl_vol_bps"].fillna(0.0),
        marker="o",
        label="event P&L volatility",
    )
    axes[0, 0].set_title("Candidate policy economics")
    axes[0, 0].set_xlabel("policy aggressiveness")
    axes[0, 0].set_ylabel("bp/event")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(
        by_policy["policy_aggressiveness"],
        by_policy["hard_pass_rate"],
        marker="o",
        label="hard pass",
    )
    axes[0, 1].plot(
        by_policy["policy_aggressiveness"],
        by_policy["behavior_pass_rate"],
        marker="o",
        label="behavior pass",
    )
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].set_title("Operational candidate eligibility")
    axes[0, 1].set_xlabel("policy aggressiveness")
    axes[0, 1].set_ylabel("event pass rate")
    axes[0, 1].legend(frameon=False)

    for profile, group in selections.groupby("risk_aversion", sort=False):
        axes[1, 0].step(
            np.arange(1, len(group) + 1),
            group["policy_aggressiveness"],
            where="mid",
            marker="o",
            label=profile.title(),
        )
    axes[1, 0].set_title("Chronological profile selections")
    axes[1, 0].set_xlabel("event")
    axes[1, 0].set_ylabel("policy aggressiveness")
    axes[1, 0].legend(frameon=False)

    paired = outputs["paired"].sort_values("as_of")
    axes[1, 1].bar(
        np.arange(1, len(paired) + 1),
        paired["realized_net_pnl_delta_bps"],
        color=np.where(
            paired["realized_net_pnl_delta_bps"] >= 0.0,
            "#70A288",
            "#B04A4A",
        ),
    )
    axes[1, 1].axhline(0.0, color="#59636E", linewidth=1)
    axes[1, 1].set_title("Selected challenger versus baseline")
    axes[1, 1].set_xlabel("event")
    axes[1, 1].set_ylabel("realized net P&L delta (bp)")
    figure.suptitle(
        "Historical automatic-policy panel — replay plumbing evidence",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--role",
        choices=("development", "holdout", "backtest"),
        required=True,
    )
    parser.add_argument(
        "--risk-aversion",
        choices=("high", "medium", "low"),
        default="medium",
    )
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument(
        "--alpha-calibration",
        choices=("walk_forward", "none"),
        default="walk_forward",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("artifacts/historical_policy_development"),
    )
    args = parser.parse_args()
    bundle = load_historical_replay_bundle(
        args.bundle,
        expected_role=args.role,
    )
    outputs, metadata = run_historical_policy_panel(
        bundle,
        risk_aversion=args.risk_aversion,
        solver=args.solver,
        alpha_calibration=args.alpha_calibration,
    )
    prefix: Path = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(prefix.with_name(prefix.name + f"_{name}.csv"), index=False)
    plot_results(outputs, prefix.with_suffix(".png"))
    plot_historical_policy_panel(
        outputs,
        prefix.with_name(prefix.name + "_policy.png"),
    )
    print(outputs["summary"].round(4).to_string(index=False))
    print("\nSelected replay gates:")
    print(outputs["gates"].to_string(index=False))
    print(f"\ndecision: {metadata['decision']}")
    print(f"reason: {metadata['decision_reason']}")
    print(f"policy candidates: {metadata['policy_candidate_count']}")
    print(f"policy trials: {metadata['policy_trial_count']}")
    print("Historical smoke data is plumbing evidence, not profitability evidence.")
    print(f"artifacts: {prefix.parent / (prefix.name + '*')}")


if __name__ == "__main__":
    main()
