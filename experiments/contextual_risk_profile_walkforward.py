"""Controlled mechanics for event-conditioned High/Medium/Low policies."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trade_planner import (
    build_monotone_policy_ladder,
    calibrate_contextual_risk_profiles_walk_forward,
    calibrate_risk_profiles_walk_forward,
)


N_EVENTS = 48
MIN_TRAINING_EVENTS = 12
SEED = 20260725
AGGRESSIVENESS = (0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00)
UNSAFE_AGGRESSIVENESS = 0.35
BEHAVIOR_FAILING_AGGRESSIVENESS = 0.65
FEATURE_COLUMNS = (
    "capacity_pressure",
    "alpha_cost_ratio",
    "liquidity_ramp_strength",
    "factor_concentration",
    "forecast_uncertainty_ratio",
)


def controlled_contextual_policy_panel(
    *,
    n_events: int = N_EVENTS,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return a frozen feature-dependent event-policy result panel."""

    if n_events < 24:
        raise ValueError("n_events must be at least twenty-four")
    policies = build_monotone_policy_ladder(AGGRESSIVENESS)
    policy_ids = policies.set_index("policy_aggressiveness")["policy_id"].to_dict()
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {
            "capacity_pressure": rng.uniform(0.25, 0.95, n_events),
            "alpha_cost_ratio": np.exp(rng.normal(0.0, 0.55, n_events)),
            "liquidity_ramp_strength": rng.uniform(0.80, 2.50, n_events),
            "factor_concentration": rng.uniform(0.05, 0.30, n_events),
            "forecast_uncertainty_ratio": rng.uniform(0.15, 1.10, n_events),
        }
    )
    standardized = (features - features.mean()) / features.std(ddof=0)
    opportunity = (
        0.65 * standardized["capacity_pressure"]
        + 0.80 * standardized["alpha_cost_ratio"]
        - 0.55 * standardized["liquidity_ramp_strength"]
        - 0.45 * standardized["factor_concentration"]
        - 0.60 * standardized["forecast_uncertainty_ratio"]
    )
    base_pnl = {
        0.05: 1.80,
        0.20: 2.20,
        0.35: 5.80,
        0.50: 3.00,
        0.65: 5.50,
        0.80: 3.00,
        1.00: 2.70,
    }
    opportunity_loading = {
        0.05: -0.15,
        0.20: 0.05,
        0.35: 0.70,
        0.50: 0.55,
        0.65: 1.20,
        0.80: 1.50,
        1.00: 2.10,
    }
    shock_loading = {
        0.05: 0.25,
        0.20: 0.40,
        0.35: 0.30,
        0.50: 0.75,
        0.65: 0.50,
        0.80: 1.15,
        1.00: 1.55,
    }
    drawdown_level = {
        0.05: 0.30,
        0.20: 0.45,
        0.35: 0.30,
        0.50: 0.75,
        0.65: 0.40,
        0.80: 1.20,
        1.00: 1.65,
    }
    common_shock = rng.normal(0.0, 1.0, n_events)
    first_cutoff = pd.Timestamp("2021-01-04 16:00")
    event_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    for event_index in range(n_events):
        event_id = f"context_event_{event_index + 1:02d}"
        cutoff = first_cutoff + pd.offsets.BDay(10 * event_index)
        feature_row = features.iloc[event_index].to_dict()
        event_rows.append(
            {
                "event_id": event_id,
                "as_of": cutoff,
                "information_cutoff": cutoff,
                "realized_available_at": cutoff + pd.offsets.BDay(5),
                **feature_row,
                "opportunity_score": float(opportunity.iloc[event_index]),
            }
        )
        for policy_index, aggressiveness in enumerate(AGGRESSIVENESS):
            policy_noise = 0.08 * np.sin(
                0.71 * event_index + 0.83 * policy_index
            )
            pnl = (
                base_pnl[aggressiveness]
                + opportunity_loading[aggressiveness]
                * opportunity.iloc[event_index]
                + shock_loading[aggressiveness] * common_shock[event_index]
                + policy_noise
            )
            drawdown = max(
                0.0,
                drawdown_level[aggressiveness]
                + 0.12 * max(-common_shock[event_index], 0.0)
                + 0.04 * np.cos(event_index + policy_index),
            )
            hard_pass = not (
                aggressiveness == UNSAFE_AGGRESSIVENESS
                and event_index % 7 == 0
            )
            behavior_pass = not (
                aggressiveness == BEHAVIOR_FAILING_AGGRESSIVENESS
                and event_index % 4 == 0
            )
            trial_rows.append(
                {
                    "event_id": event_id,
                    "policy_id": policy_ids[aggressiveness],
                    "net_pnl_bps": float(pnl),
                    "within_event_max_drawdown_bps": drawdown,
                    "hard_pass": hard_pass,
                    "behavior_pass": behavior_pass,
                }
            )
    return pd.DataFrame(event_rows), pd.DataFrame(trial_rows), policies


def run_contextual_risk_profile_mechanics(
    *,
    full_suite_verified: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Run the predeclared contextual chronology and investment checks."""

    events, trials, policies = controlled_contextual_policy_panel()
    contextual = calibrate_contextual_risk_profiles_walk_forward(
        events,
        trials,
        policies,
        feature_columns=FEATURE_COLUMNS,
        min_training_events=MIN_TRAINING_EVENTS,
    )
    unconditional = calibrate_risk_profiles_walk_forward(
        events,
        trials,
        policies,
        min_training_events=MIN_TRAINING_EVENTS,
    )

    probe_index = 24
    probe_id = str(events.iloc[probe_index]["event_id"])
    perturbed_trials = trials.copy()
    future_ids = set(events.iloc[probe_index:]["event_id"].astype(str))
    perturb_mask = perturbed_trials["event_id"].isin(future_ids)
    perturbed_trials.loc[perturb_mask, "net_pnl_bps"] += 100.0
    perturbed_trials.loc[
        perturb_mask,
        "within_event_max_drawdown_bps",
    ] += 50.0
    perturbed = calibrate_contextual_risk_profiles_walk_forward(
        events,
        perturbed_trials,
        policies,
        feature_columns=FEATURE_COLUMNS,
        min_training_events=MIN_TRAINING_EVENTS,
    )
    selection_columns = [
        "risk_aversion",
        "selected_policy_id",
        "training_event_ids",
        "predicted_net_pnl_bps",
        "prediction_standard_error_bps",
        "net_pnl_lower_bound_bps",
        "contextual_ridge_multiplier",
        "contextual_cv_rmse_bps",
        *[f"current_feature_{column}" for column in FEATURE_COLUMNS],
    ]
    original_probe = contextual.selections.loc[
        contextual.selections["event_id"].eq(probe_id),
        selection_columns,
    ].sort_values("risk_aversion").reset_index(drop=True)
    perturbed_probe = perturbed.selections.loc[
        perturbed.selections["event_id"].eq(probe_id),
        selection_columns,
    ].sort_values("risk_aversion").reset_index(drop=True)
    leakage_invariant = bool(
        original_probe[
            ["risk_aversion", "selected_policy_id", "training_event_ids"]
        ].equals(
            perturbed_probe[
                ["risk_aversion", "selected_policy_id", "training_event_ids"]
            ]
        )
        and np.allclose(
            original_probe.drop(
                columns=["risk_aversion", "selected_policy_id", "training_event_ids"]
            ).to_numpy(float),
            perturbed_probe.drop(
                columns=["risk_aversion", "selected_policy_id", "training_event_ids"]
            ).to_numpy(float),
            rtol=0.0,
            atol=1e-13,
            equal_nan=True,
        )
    )

    contextual_scored = contextual.selections.loc[
        contextual.selections["status"].str.startswith("contextual")
    ].copy()
    unconditional_scored = unconditional.selections.loc[
        unconditional.selections["status"].str.startswith("calibrated")
    ].copy()
    warmup_pass = bool(
        contextual.selections.loc[
            contextual.selections["event_id"].isin(
                events.iloc[:MIN_TRAINING_EVENTS]["event_id"]
            ),
            "status",
        ].eq("fallback_contextual_warmup").all()
    )
    excluded_policy_ids = set(
        policies.loc[
            policies["policy_aggressiveness"].isin(
                [UNSAFE_AGGRESSIVENESS, BEHAVIOR_FAILING_AGGRESSIVENESS]
            ),
            "policy_id",
        ]
    )
    unsafe_excluded = bool(
        ~contextual_scored["selected_policy_id"].isin(excluded_policy_ids).any()
    )
    ordered = contextual_scored.pivot(
        index="event_id",
        columns="risk_aversion",
        values="policy_aggressiveness",
    ).reindex(columns=["high", "medium", "low"])
    monotonic = bool(
        (
            ordered["high"].le(ordered["medium"] + 1e-12)
            & ordered["medium"].le(ordered["low"] + 1e-12)
        ).all()
    )
    comparison = _profile_comparison(contextual, unconditional)
    medium_low_improvement = float(
        comparison.loc[
            comparison["risk_aversion"].isin(["medium", "low"]),
            "mean_pnl_delta_bps",
        ].mean()
    )
    no_profile_loss = bool(comparison["mean_pnl_delta_bps"].ge(-0.25).all())
    volatility_preserved = bool(comparison["pnl_vol_delta_bps"].le(0.50).all())
    positive_pnl = bool(
        contextual.summary["mean_selected_net_pnl_bps"].gt(0.0).all()
    )
    all_hard = bool(contextual.summary["all_selected_hard_pass"].all())

    low = contextual_scored.loc[
        contextual_scored["risk_aversion"].eq("low")
    ].merge(
        events[["event_id", "opportunity_score"]],
        on="event_id",
        validate="many_to_one",
    )
    opportunity_median = float(low["opportunity_score"].median())
    high_opportunity_aggressiveness = float(
        low.loc[
            low["opportunity_score"].gt(opportunity_median),
            "policy_aggressiveness",
        ].mean()
    )
    low_opportunity_aggressiveness = float(
        low.loc[
            low["opportunity_score"].le(opportunity_median),
            "policy_aggressiveness",
        ].mean()
    )
    opportunity_adaptation = (
        high_opportunity_aggressiveness
        - low_opportunity_aggressiveness
    )
    gates = pd.DataFrame(
        [
            (
                "current_and_future_outcomes_are_inaccessible",
                "exactly invariant",
                leakage_invariant,
                leakage_invariant,
            ),
            (
                "twelve_event_warmup_uses_fallback",
                "first twelve events",
                warmup_pass,
                warmup_pass,
            ),
            (
                "hard_and_behavior_failures_are_excluded",
                "never selected",
                unsafe_excluded,
                unsafe_excluded,
            ),
            (
                "profile_aggressiveness_is_monotonic",
                "high <= medium <= low",
                monotonic,
                monotonic,
            ),
            (
                "medium_low_pnl_improves_050bp",
                ">= 0.50 bp/event",
                medium_low_improvement,
                medium_low_improvement >= 0.50,
            ),
            (
                "no_profile_loses_more_than_025bp",
                ">= -0.25 bp/event",
                float(comparison["mean_pnl_delta_bps"].min()),
                no_profile_loss,
            ),
            (
                "profile_volatility_within_050bp",
                "<= +0.50 bp",
                float(comparison["pnl_vol_delta_bps"].max()),
                volatility_preserved,
            ),
            (
                "positive_pnl_and_all_hard_pass",
                "positive P&L and 100% hard pass",
                bool(positive_pnl and all_hard),
                bool(positive_pnl and all_hard),
            ),
            (
                "low_profile_adapts_to_opportunity",
                ">= 0.15 aggressiveness difference",
                opportunity_adaptation,
                opportunity_adaptation >= 0.15,
            ),
            (
                "full_repository_test_suite",
                "all tests pass",
                bool(full_suite_verified),
                bool(full_suite_verified),
            ),
        ],
        columns=("gate", "threshold", "observed", "passed"),
    )
    decision = (
        "keep"
        if bool(gates["passed"].all())
        else "discard"
        if full_suite_verified
        else "pending_or_discard"
    )
    reason = (
        "All predeclared contextual policy mechanics and repository checks passed."
        if decision == "keep"
        else "Failed: "
        + ", ".join(gates.loc[~gates["passed"], "gate"].astype(str))
        + "."
    )
    contextual_summary = contextual.summary.copy()
    contextual_summary["decision"] = decision
    contextual_summary["decision_reason"] = reason
    outputs = {
        "events": events,
        "trials": trials,
        "policies": policies,
        "contextual_selections": contextual.selections,
        "contextual_policy_evaluations": contextual.policy_evaluations,
        "contextual_summary": contextual_summary,
        "unconditional_selections": unconditional.selections,
        "unconditional_policy_evaluations": unconditional.policy_evaluations,
        "unconditional_summary": unconditional.summary,
        "comparison": comparison,
        "gates": gates,
    }
    metadata: dict[str, object] = {
        "decision": decision,
        "decision_reason": reason,
        "cohort_role": "controlled_mechanics_only",
        "event_count": len(events),
        "warmup_event_count": MIN_TRAINING_EVENTS,
        "seed": SEED,
        "feature_columns": FEATURE_COLUMNS,
        "probe_event_id": probe_id,
        "medium_low_mean_pnl_improvement_bps": medium_low_improvement,
        "low_opportunity_aggressiveness_difference": opportunity_adaptation,
        "full_suite_verified": bool(full_suite_verified),
    }
    return outputs, metadata


def _profile_comparison(
    contextual: object,
    unconditional: object,
) -> pd.DataFrame:
    contextual_summary = contextual.summary.set_index("risk_aversion")
    unconditional_summary = unconditional.summary.set_index("risk_aversion")
    rows = []
    for profile in ("high", "medium", "low"):
        contextual_row = contextual_summary.loc[profile]
        unconditional_row = unconditional_summary.loc[profile]
        rows.append(
            {
                "risk_aversion": profile,
                "contextual_mean_pnl_bps": contextual_row[
                    "mean_selected_net_pnl_bps"
                ],
                "unconditional_mean_pnl_bps": unconditional_row[
                    "mean_selected_net_pnl_bps"
                ],
                "mean_pnl_delta_bps": contextual_row[
                    "mean_selected_net_pnl_bps"
                ]
                - unconditional_row["mean_selected_net_pnl_bps"],
                "contextual_pnl_vol_bps": contextual_row["selected_pnl_vol_bps"],
                "unconditional_pnl_vol_bps": unconditional_row[
                    "selected_pnl_vol_bps"
                ],
                "pnl_vol_delta_bps": contextual_row["selected_pnl_vol_bps"]
                - unconditional_row["selected_pnl_vol_bps"],
                "contextual_mean_aggressiveness": contextual_row[
                    "mean_policy_aggressiveness"
                ],
                "unconditional_mean_aggressiveness": unconditional_row[
                    "mean_policy_aggressiveness"
                ],
            }
        )
    return pd.DataFrame(rows)


def plot_contextual_results(
    outputs: dict[str, pd.DataFrame],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contextual = outputs["contextual_selections"].loc[
        outputs["contextual_selections"]["status"].str.startswith("contextual")
    ].copy()
    unconditional = outputs["unconditional_selections"].loc[
        outputs["unconditional_selections"]["status"].str.startswith("calibrated")
    ].copy()
    events = outputs["events"]
    comparison = outputs["comparison"].set_index("risk_aversion")
    colors = {"high": "#2F6B9A", "medium": "#D88C2D", "low": "#8B4E9F"}
    figure, axes = plt.subplots(2, 3, figsize=(17.0, 9.0))

    axis = axes[0, 0]
    merged = contextual.merge(
        unconditional[["event_id", "risk_aversion", "selected_net_pnl_bps"]],
        on=["event_id", "risk_aversion"],
        suffixes=("_contextual", "_unconditional"),
        validate="one_to_one",
    )
    for profile, group in merged.groupby("risk_aversion", sort=False):
        ordered = group.sort_values("as_of")
        delta = (
            ordered["selected_net_pnl_bps_contextual"]
            - ordered["selected_net_pnl_bps_unconditional"]
        ).cumsum()
        axis.plot(
            np.arange(1, len(ordered) + 1),
            delta,
            marker="o",
            linewidth=2,
            color=colors[profile],
            label=profile.title(),
        )
    axis.axhline(0.0, color="#59636E", linewidth=0.8)
    axis.set_title("Cumulative P&L vs unconditional selector")
    axis.set_ylabel("bp")
    axis.legend(frameon=False)

    axis = axes[0, 1]
    for profile, group in contextual.groupby("risk_aversion", sort=False):
        ordered = group.sort_values("as_of")
        axis.step(
            np.arange(1, len(ordered) + 1),
            ordered["policy_aggressiveness"],
            where="mid",
            linewidth=2,
            color=colors[profile],
            label=profile.title(),
        )
    axis.set_title("Contextual policy through time")
    axis.set_ylabel("Aggressiveness")

    axis = axes[0, 2]
    low = contextual.loc[contextual["risk_aversion"].eq("low")].merge(
        events[["event_id", "opportunity_score"]],
        on="event_id",
        validate="many_to_one",
    )
    axis.scatter(
        low["opportunity_score"],
        low["policy_aggressiveness"],
        color=colors["low"],
        alpha=0.8,
    )
    axis.set_title("Low policy responds to opportunity")
    axis.set_xlabel("Alpha/capacity opportunity score")
    axis.set_ylabel("Selected aggressiveness")

    axis = axes[1, 0]
    profiles = ["high", "medium", "low"]
    x = np.arange(len(profiles))
    width = 0.34
    axis.bar(
        x - width / 2,
        comparison.loc[profiles, "unconditional_mean_pnl_bps"],
        width,
        color="#8A929A",
        label="Unconditional",
    )
    axis.bar(
        x + width / 2,
        comparison.loc[profiles, "contextual_mean_pnl_bps"],
        width,
        color="#70A288",
        label="Contextual",
    )
    axis.set_xticks(x, [item.title() for item in profiles])
    axis.set_title("Out-of-sample selected P&L")
    axis.set_ylabel("bp/event")
    axis.legend(frameon=False)

    axis = axes[1, 1]
    axis.bar(
        x - width / 2,
        comparison.loc[profiles, "unconditional_pnl_vol_bps"],
        width,
        color="#8A929A",
        label="Unconditional",
    )
    axis.bar(
        x + width / 2,
        comparison.loc[profiles, "contextual_pnl_vol_bps"],
        width,
        color="#B04A4A",
        label="Contextual",
    )
    axis.set_xticks(x, [item.title() for item in profiles])
    axis.set_title("Out-of-sample P&L volatility")
    axis.set_ylabel("bp")

    axis = axes[1, 2]
    low_ordered = low.sort_values("as_of")
    axis.step(
        np.arange(1, len(low_ordered) + 1),
        low_ordered["contextual_ridge_multiplier"],
        where="mid",
        color="#537895",
        linewidth=2,
        label="Ridge multiplier",
    )
    uncertainty_axis = axis.twinx()
    uncertainty_axis.plot(
        np.arange(1, len(low_ordered) + 1),
        low_ordered["prediction_standard_error_bps"],
        color="#D88C2D",
        linewidth=1.7,
        label="Prediction error",
    )
    axis.set_yscale("log")
    axis.set_title("Automatic regularization and uncertainty")
    axis.set_ylabel("Ridge multiplier")
    uncertainty_axis.set_ylabel("Prediction error (bp)")

    figure.suptitle(
        "Contextual High/Medium/Low policy calibration — controlled mechanics only",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=175, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("artifacts/contextual_risk_profile_mechanics"),
    )
    parser.add_argument("--full-suite-verified", action="store_true")
    args = parser.parse_args()
    outputs, metadata = run_contextual_risk_profile_mechanics(
        full_suite_verified=args.full_suite_verified
    )
    prefix: Path = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(prefix.with_name(prefix.name + f"_{name}.csv"), index=False)
    chart = prefix.with_suffix(".png")
    plot_contextual_results(outputs, chart)
    print(outputs["comparison"].round(4).to_string(index=False))
    print("\nAcceptance gates:")
    print(outputs["gates"].to_string(index=False))
    print(f"\ndecision: {metadata['decision']}")
    print(f"reason: {metadata['decision_reason']}")
    print("This controlled panel is not evidence of real profitability.")
    print(f"artifacts: {prefix.parent / (prefix.name + '*')}")


if __name__ == "__main__":
    main()
