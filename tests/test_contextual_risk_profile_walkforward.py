from __future__ import annotations

import pytest

from experiments.contextual_risk_profile_walkforward import (
    FEATURE_COLUMNS,
    controlled_contextual_policy_panel,
    run_contextual_risk_profile_mechanics,
)
from trade_planner import calibrate_contextual_risk_profiles_walk_forward


def test_contextual_selector_is_chronological_and_auditable() -> None:
    events, trials, policies = controlled_contextual_policy_panel(n_events=24)

    result = calibrate_contextual_risk_profiles_walk_forward(
        events,
        trials,
        policies,
        feature_columns=FEATURE_COLUMNS,
        min_training_events=12,
    )
    scored = result.selections.loc[
        result.selections["status"].str.startswith("contextual")
    ]

    assert result.selections.iloc[: 12 * 3]["status"].eq(
        "fallback_contextual_warmup"
    ).all()
    assert scored["contextual_ridge_multiplier"].isin(
        [0.01, 0.10, 1.0, 10.0, 100.0]
    ).all()
    for row in scored.itertuples(index=False):
        assert str(row.event_id) not in str(row.training_event_ids).split("|")
    assert result.policy_evaluations[
        "training_feature_mean_alpha_cost_ratio"
    ].notna().all()


def test_contextual_selector_rejects_missing_or_nonfinite_features() -> None:
    events, trials, policies = controlled_contextual_policy_panel(n_events=24)
    missing = events.drop(columns=[FEATURE_COLUMNS[0]])
    with pytest.raises(ValueError, match="missing contextual features"):
        calibrate_contextual_risk_profiles_walk_forward(
            missing,
            trials,
            policies,
            feature_columns=FEATURE_COLUMNS,
        )

    nonfinite = events.copy()
    nonfinite.loc[0, FEATURE_COLUMNS[0]] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        calibrate_contextual_risk_profiles_walk_forward(
            nonfinite,
            trials,
            policies,
            feature_columns=FEATURE_COLUMNS,
        )


def test_contextual_mechanics_records_only_the_volatility_failure() -> None:
    outputs, metadata = run_contextual_risk_profile_mechanics(
        full_suite_verified=True
    )
    failed = outputs["gates"].loc[
        ~outputs["gates"]["passed"],
        "gate",
    ].tolist()

    assert failed == ["profile_volatility_within_050bp"]
    assert metadata["decision"] == "discard"
    assert metadata["medium_low_mean_pnl_improvement_bps"] >= 0.50
    assert metadata["low_opportunity_aggressiveness_difference"] >= 0.15
