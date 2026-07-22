"""Controlled mechanics for automatic High/Medium/Low coefficient selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trade_planner import (
    build_monotone_policy_ladder,
    calibrate_risk_profiles_walk_forward,
)


N_EVENTS = 28
SEED = 20260724
AGGRESSIVENESS = (0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00)
UNSAFE_AGGRESSIVENESS = 0.35
BEHAVIOR_FAILING_AGGRESSIVENESS = 0.65


def controlled_policy_panel(
    *,
    n_events: int = N_EVENTS,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return a frozen event-policy panel with known efficient and failed rows."""

    if n_events < 12:
        raise ValueError("n_events must be at least twelve")
    policies = build_monotone_policy_ladder(AGGRESSIVENESS)
    policy_by_aggressiveness = policies.set_index("policy_aggressiveness")[
        "policy_id"
    ].to_dict()
    mean_pnl = {
        0.05: 1.40,
        0.20: 2.00,
        0.35: 7.00,
        0.50: 3.25,
        0.65: 6.50,
        0.80: 4.45,
        1.00: 4.00,
    }
    pnl_vol = {
        0.05: 0.25,
        0.20: 0.45,
        0.35: 0.30,
        0.50: 0.75,
        0.65: 0.50,
        0.80: 1.25,
        1.00: 1.80,
    }
    drawdown_level = {
        0.05: 0.35,
        0.20: 0.50,
        0.35: 0.30,
        0.50: 0.80,
        0.65: 0.40,
        0.80: 1.30,
        1.00: 1.90,
    }
    rng = np.random.default_rng(seed)
    common_shock = rng.normal(0.0, 1.0, size=n_events)
    common_shock = (common_shock - common_shock.mean()) / common_shock.std(ddof=1)
    first_cutoff = pd.Timestamp("2021-01-04 16:00")
    event_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    for event_index in range(n_events):
        event_id = f"policy_event_{event_index + 1:02d}"
        cutoff = first_cutoff + pd.offsets.BDay(10 * event_index)
        event_rows.append(
            {
                "event_id": event_id,
                "as_of": cutoff,
                "information_cutoff": cutoff,
                "realized_available_at": cutoff + pd.offsets.BDay(5),
            }
        )
        for policy_index, aggressiveness in enumerate(AGGRESSIVENESS):
            idiosyncratic = 0.08 * np.sin(
                0.73 * event_index + 0.91 * policy_index
            )
            pnl = (
                mean_pnl[aggressiveness]
                + pnl_vol[aggressiveness] * common_shock[event_index]
                + idiosyncratic
            )
            drawdown = max(
                0.0,
                drawdown_level[aggressiveness]
                + 0.12 * max(-common_shock[event_index], 0.0)
                + 0.03 * np.cos(event_index + policy_index),
            )
            hard_pass = not (
                aggressiveness == UNSAFE_AGGRESSIVENESS and event_index % 7 == 0
            )
            behavior_pass = not (
                aggressiveness == BEHAVIOR_FAILING_AGGRESSIVENESS
                and event_index % 4 == 0
            )
            trial_rows.append(
                {
                    "event_id": event_id,
                    "policy_id": policy_by_aggressiveness[aggressiveness],
                    "net_pnl_bps": pnl,
                    "within_event_max_drawdown_bps": drawdown,
                    "hard_pass": hard_pass,
                    "behavior_pass": behavior_pass,
                }
            )
    return pd.DataFrame(event_rows), pd.DataFrame(trial_rows), policies


def run_automatic_risk_profile_mechanics(
    *,
    full_suite_verified: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Run the predeclared chronology, exclusion, and investment checks."""

    events, trials, policies = controlled_policy_panel()
    result = calibrate_risk_profiles_walk_forward(events, trials, policies)
    probe_index = 16
    probe_event_id = str(events.iloc[probe_index]["event_id"])
    perturbed_trials = trials.copy()
    future_ids = set(events.iloc[probe_index:]["event_id"].astype(str))
    perturb_mask = perturbed_trials["event_id"].isin(future_ids)
    perturbed_trials.loc[perturb_mask, "net_pnl_bps"] += 50.0
    perturbed_trials.loc[
        perturb_mask,
        "within_event_max_drawdown_bps",
    ] += 25.0
    perturbed = calibrate_risk_profiles_walk_forward(
        events,
        perturbed_trials,
        policies,
    )
    selection_columns = [
        "risk_aversion",
        "selected_policy_id",
        "training_event_ids",
        "net_pnl_lower_bound_bps",
        "realized_risk_bps",
        "realized_risk_budget_bps",
    ]
    original_probe = result.selections.loc[
        result.selections["event_id"].eq(probe_event_id),
        selection_columns,
    ].sort_values("risk_aversion").reset_index(drop=True)
    perturbed_probe = perturbed.selections.loc[
        perturbed.selections["event_id"].eq(probe_event_id),
        selection_columns,
    ].sort_values("risk_aversion").reset_index(drop=True)
    selection_invariant = bool(
        original_probe[
            ["risk_aversion", "selected_policy_id", "training_event_ids"]
        ].equals(
            perturbed_probe[
                ["risk_aversion", "selected_policy_id", "training_event_ids"]
            ]
        )
        and np.allclose(
            original_probe[
                [
                    "net_pnl_lower_bound_bps",
                    "realized_risk_bps",
                    "realized_risk_budget_bps",
                ]
            ].to_numpy(float),
            perturbed_probe[
                [
                    "net_pnl_lower_bound_bps",
                    "realized_risk_bps",
                    "realized_risk_budget_bps",
                ]
            ].to_numpy(float),
            rtol=0.0,
            atol=1e-14,
        )
    )
    calibrated = result.selections.loc[
        result.selections["status"].str.startswith("calibrated")
    ].copy()
    unsafe_policy = policies.loc[
        policies["policy_aggressiveness"].eq(UNSAFE_AGGRESSIVENESS),
        "policy_id",
    ].iloc[0]
    behavior_policy = policies.loc[
        policies["policy_aggressiveness"].eq(BEHAVIOR_FAILING_AGGRESSIVENESS),
        "policy_id",
    ].iloc[0]
    unsafe_excluded = bool(
        ~calibrated["selected_policy_id"].isin(
            [unsafe_policy, behavior_policy]
        ).any()
    )
    ordered = (
        calibrated.pivot(
            index="event_id",
            columns="risk_aversion",
            values="policy_aggressiveness",
        )
        .reindex(columns=["high", "medium", "low"])
    )
    monotonic = bool(
        (
            ordered["high"].le(ordered["medium"] + 1e-12)
            & ordered["medium"].le(ordered["low"] + 1e-12)
        ).all()
    )
    summary = result.summary.copy()
    positive_pnl = bool(summary["mean_selected_net_pnl_bps"].gt(0.0).all())
    fallback_preserved = bool(summary["mean_net_pnl_delta_bps"].ge(-1.0).all())
    warmup = result.selections.loc[
        result.selections["event_id"].isin(events.iloc[:8]["event_id"])
    ]
    warmup_pass = bool(warmup["status"].eq("fallback_warmup").all())
    gates = pd.DataFrame(
        [
            {
                "gate": "current_and_future_outcomes_are_inaccessible",
                "threshold": "exactly invariant",
                "observed": selection_invariant,
                "passed": selection_invariant,
            },
            {
                "gate": "hard_and_behavior_failures_are_excluded",
                "threshold": "never selected",
                "observed": unsafe_excluded,
                "passed": unsafe_excluded,
            },
            {
                "gate": "profile_aggressiveness_is_monotonic",
                "threshold": "high <= medium <= low",
                "observed": monotonic,
                "passed": monotonic,
            },
            {
                "gate": "selected_mean_net_pnl_is_positive",
                "threshold": "> 0 for every profile",
                "observed": positive_pnl,
                "passed": positive_pnl,
            },
            {
                "gate": "fallback_pnl_preserved_within_1bp",
                "threshold": ">= -1 bp/event",
                "observed": float(summary["mean_net_pnl_delta_bps"].min()),
                "passed": fallback_preserved,
            },
            {
                "gate": "eight_event_warmup_uses_fallback",
                "threshold": "first eight events",
                "observed": warmup_pass,
                "passed": warmup_pass,
            },
            {
                "gate": "full_repository_test_suite",
                "threshold": "all tests pass",
                "observed": bool(full_suite_verified),
                "passed": bool(full_suite_verified),
            },
        ]
    )
    decision = "keep" if bool(gates["passed"].all()) else "pending_or_discard"
    reason = (
        "All predeclared investment-policy mechanics and repository checks passed."
        if decision == "keep"
        else "At least one predeclared policy-selection gate is not verified."
    )
    summary["decision"] = decision
    summary["decision_reason"] = reason
    outputs = {
        "summary": summary,
        "gates": gates,
        "selections": result.selections,
        "policy_evaluations": result.policy_evaluations,
        "policies": result.policies,
        "trials": trials,
    }
    metadata: dict[str, object] = {
        "decision": decision,
        "decision_reason": reason,
        "cohort_role": "controlled_mechanics_only",
        "event_count": len(events),
        "warmup_event_count": 8,
        "seed": SEED,
        "probe_event_id": probe_event_id,
        "unsafe_policy_id": unsafe_policy,
        "behavior_failing_policy_id": behavior_policy,
        "full_suite_verified": bool(full_suite_verified),
    }
    return outputs, metadata


def plot_automatic_risk_profile_results(
    outputs: dict[str, pd.DataFrame],
    output: Path,
) -> None:
    """Render policy frontier, selections, and realized economics."""

    import matplotlib.pyplot as plt

    evaluations = outputs["policy_evaluations"]
    selections = outputs["selections"].loc[
        outputs["selections"]["status"].str.startswith("calibrated")
    ].copy()
    final_event = selections["event_id"].iloc[-1]
    final_medium = evaluations.loc[
        evaluations["event_id"].eq(final_event)
        & evaluations["risk_aversion"].eq("medium")
    ].copy()
    colors = {"high": "#2F6B9A", "medium": "#D88C2D", "low": "#8B4E9F"}
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    axis = axes[0, 0]
    for _, row in final_medium.iterrows():
        color = (
            "#B04A4A"
            if not bool(row["operationally_eligible"])
            else "#70A288"
        )
        axis.scatter(
            row["realized_risk_bps"],
            row["net_pnl_lower_bound_bps"],
            color=color,
            s=70,
        )
        axis.annotate(
            f"{row['policy_aggressiveness']:.2f}",
            (row["realized_risk_bps"], row["net_pnl_lower_bound_bps"]),
            xytext=(4, 4),
            textcoords="offset points",
        )
    axis.axhline(0.0, color="#59636E", linewidth=1)
    axis.set_title("Latest medium-profile investment frontier")
    axis.set_xlabel("realized risk measure (bp)")
    axis.set_ylabel("net-P&L lower confidence bound (bp/event)")

    axis = axes[0, 1]
    for profile, group in selections.groupby("risk_aversion", sort=False):
        ordered = group.sort_values("as_of")
        axis.step(
            np.arange(1, len(ordered) + 1),
            ordered["policy_aggressiveness"],
            where="mid",
            marker="o",
            label=profile.title(),
            color=colors[profile],
        )
    axis.set_title("Chronological automatic policy selections")
    axis.set_xlabel("post-warmup event")
    axis.set_ylabel("policy aggressiveness")
    axis.legend(frameon=False)

    axis = axes[1, 0]
    for profile, group in selections.groupby("risk_aversion", sort=False):
        ordered = group.sort_values("as_of")
        cumulative_delta = (
            ordered["selected_net_pnl_bps"]
            - ordered["fallback_net_pnl_bps"]
        ).cumsum()
        axis.plot(
            np.arange(1, len(ordered) + 1),
            cumulative_delta,
            marker="o",
            label=profile.title(),
            color=colors[profile],
        )
    axis.axhline(0.0, color="#59636E", linewidth=1)
    axis.set_title("Cumulative P&L versus fixed fallback")
    axis.set_xlabel("post-warmup event")
    axis.set_ylabel("selected minus fallback (bp)")
    axis.legend(frameon=False)

    axis = axes[1, 1]
    summary = outputs["summary"].set_index("risk_aversion")
    profiles = ["high", "medium", "low"]
    x = np.arange(len(profiles))
    width = 0.34
    axis.bar(
        x - width / 2,
        summary.loc[profiles, "mean_selected_net_pnl_bps"],
        width,
        label="selected P&L",
        color="#70A288",
    )
    axis.bar(
        x + width / 2,
        summary.loc[profiles, "selected_pnl_vol_bps"],
        width,
        label="P&L volatility",
        color="#B04A4A",
    )
    axis.set_xticks(x, [profile.title() for profile in profiles])
    axis.set_title("Out-of-sample return and swing")
    axis.set_ylabel("bp/event")
    axis.legend(frameon=False)
    figure.suptitle(
        "Automatic High/Medium/Low calibration — controlled mechanics only",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("artifacts/automatic_risk_profile_mechanics"),
    )
    parser.add_argument("--full-suite-verified", action="store_true")
    args = parser.parse_args()
    outputs, metadata = run_automatic_risk_profile_mechanics(
        full_suite_verified=args.full_suite_verified
    )
    prefix: Path = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(prefix.with_name(prefix.name + f"_{name}.csv"), index=False)
    chart = prefix.with_suffix(".png")
    plot_automatic_risk_profile_results(outputs, chart)
    print(outputs["summary"].round(4).to_string(index=False))
    print("\nAcceptance gates:")
    print(outputs["gates"].to_string(index=False))
    print(f"\ndecision: {metadata['decision']}")
    print(f"reason: {metadata['decision_reason']}")
    print("This controlled panel is not evidence of real profitability.")
    print(f"artifacts: {prefix.parent / (prefix.name + '*')}")


if __name__ == "__main__":
    main()
