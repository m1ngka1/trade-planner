from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.automatic_risk_profile_walkforward import (
    BEHAVIOR_FAILING_AGGRESSIVENESS,
    UNSAFE_AGGRESSIVENESS,
    controlled_policy_panel,
)
from trade_planner import (
    InvestmentPolicyCoefficients,
    build_monotone_policy_ladder,
    calibrate_risk_profiles_walk_forward,
)


def test_monotone_ladder_moves_all_coefficients_coherently() -> None:
    policies = build_monotone_policy_ladder([1.0, 0.05, 0.50])

    assert policies["policy_aggressiveness"].tolist() == [0.05, 0.50, 1.0]
    assert policies["risk_frontier_fraction"].is_monotonic_increasing
    assert policies["liquidity_quantile"].is_monotonic_increasing
    assert policies["liquidity_shape_fraction"].is_monotonic_decreasing
    assert policies["alpha_confidence"].is_monotonic_decreasing
    assert policies["factor_stress_fraction"].is_monotonic_decreasing
    np.testing.assert_allclose(
        policies.loc[policies["policy_aggressiveness"].eq(0.50)].iloc[0][
            [
                "risk_frontier_fraction",
                "liquidity_quantile",
                "liquidity_shape_fraction",
                "alpha_confidence",
                "factor_stress_fraction",
            ]
        ].to_numpy(float),
        [0.50, 0.30, 0.50, 0.75, 0.50],
    )


def test_walk_forward_selector_is_profitable_feasible_and_monotonic() -> None:
    events, trials, policies = controlled_policy_panel()

    result = calibrate_risk_profiles_walk_forward(events, trials, policies)
    calibrated = result.selections.loc[
        result.selections["status"].str.startswith("calibrated")
    ]
    excluded = set(
        policies.loc[
            policies["policy_aggressiveness"].isin(
                [UNSAFE_AGGRESSIVENESS, BEHAVIOR_FAILING_AGGRESSIVENESS]
            ),
            "policy_id",
        ]
    )
    ordered = calibrated.pivot(
        index="event_id",
        columns="risk_aversion",
        values="policy_aggressiveness",
    )

    assert result.selections.iloc[: 8 * 3]["status"].eq("fallback_warmup").all()
    assert not calibrated["selected_policy_id"].isin(excluded).any()
    assert (ordered["high"] <= ordered["medium"]).all()
    assert (ordered["medium"] <= ordered["low"]).all()
    assert result.summary["mean_selected_net_pnl_bps"].gt(0.0).all()
    assert result.summary["mean_net_pnl_delta_bps"].ge(-1.0).all()
    assert result.summary["all_selected_hard_pass"].all()
    assert result.summary["selected_behavior_pass_rate"].eq(1.0).all()
    selected_row = calibrated.iloc[-1].to_dict()
    consumable = InvestmentPolicyCoefficients.from_mapping(selected_row)
    assert consumable.policy_id == selected_row["selected_policy_id"]
    assert consumable.policy_aggressiveness == selected_row[
        "policy_aggressiveness"
    ]


def test_current_and_future_outcomes_do_not_change_current_policy() -> None:
    events, trials, policies = controlled_policy_panel()
    probe_index = 16
    probe_id = str(events.iloc[probe_index]["event_id"])
    original = calibrate_risk_profiles_walk_forward(events, trials, policies)
    perturbed_trials = trials.copy()
    future_ids = set(events.iloc[probe_index:]["event_id"].astype(str))
    mask = perturbed_trials["event_id"].isin(future_ids)
    perturbed_trials.loc[mask, "net_pnl_bps"] -= 100.0
    perturbed_trials.loc[mask, "within_event_max_drawdown_bps"] += 50.0

    rerun = calibrate_risk_profiles_walk_forward(events, perturbed_trials, policies)
    columns = [
        "risk_aversion",
        "selected_policy_id",
        "training_event_ids",
        "net_pnl_lower_bound_bps",
        "realized_risk_bps",
        "realized_risk_budget_bps",
    ]
    expected = original.selections.loc[
        original.selections["event_id"].eq(probe_id),
        columns,
    ].sort_values("risk_aversion").reset_index(drop=True)
    actual = rerun.selections.loc[
        rerun.selections["event_id"].eq(probe_id),
        columns,
    ].sort_values("risk_aversion").reset_index(drop=True)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_outcome_unavailable_at_cutoff_is_not_in_training_history() -> None:
    events, trials, policies = controlled_policy_panel(n_events=14)
    delayed_index = 6
    current_index = 9
    current_cutoff = pd.Timestamp(events.iloc[current_index]["information_cutoff"])
    delayed_id = str(events.iloc[delayed_index]["event_id"])
    events.loc[delayed_index, "realized_available_at"] = (
        current_cutoff + pd.Timedelta(days=2)
    )

    result = calibrate_risk_profiles_walk_forward(events, trials, policies)
    current_id = str(events.iloc[current_index]["event_id"])
    current = result.selections.loc[result.selections["event_id"].eq(current_id)]

    assert current["status"].str.startswith("calibrated").all()
    assert current["training_event_count"].eq(8).all()
    assert current["training_event_ids"].str.split("|").apply(
        lambda values: delayed_id not in values
    ).all()


def test_negative_history_returns_executable_but_not_profitable_policy() -> None:
    events, trials, policies = controlled_policy_panel(n_events=14)
    trials["net_pnl_bps"] -= 20.0

    result = calibrate_risk_profiles_walk_forward(
        events,
        trials,
        policies,
        min_training_events=4,
    )
    calibrated = result.selections.loc[
        result.selections["status"].str.startswith("calibrated")
    ]

    assert calibrated["status"].eq("calibrated_no_profitable_policy").all()
    assert ~calibrated["economically_viable"].any()
    assert calibrated["selected_policy_id"].isin(policies["policy_id"]).all()


def test_incoherent_policy_ladder_is_rejected() -> None:
    events, trials, policies = controlled_policy_panel(n_events=12)
    policies = policies.copy()
    policies.loc[policies.index[-1], "liquidity_shape_fraction"] = 0.99

    with pytest.raises(ValueError, match="liquidity_shape_fraction must be non-increasing"):
        calibrate_risk_profiles_walk_forward(events, trials, policies)
