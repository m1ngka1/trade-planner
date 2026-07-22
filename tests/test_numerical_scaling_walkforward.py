from __future__ import annotations

from experiments.numerical_scaling_walkforward import (
    run_numerical_scaling_screen,
)


def test_fixed_scaling_screen_solves_and_honors_profit_floor() -> None:
    outputs, metadata = run_numerical_scaling_screen()
    candidate = outputs["trials"].loc[
        outputs["trials"]["strategy"].eq("forecast_liquidity")
    ]
    paired = outputs["paired"]

    assert metadata["event_start"] == 0
    assert metadata["n_events"] == 2
    assert metadata["numerical_scaling"] == "per_name"
    assert metadata["verify_hard_constraints"] is True
    assert len(candidate) == 2
    assert candidate["forecast_profit_floor_slack_dollars"].min() >= -1.0
    assert candidate["max_cap_excess_shares"].max() <= 0.05
    assert candidate["max_wrong_direction_shares"].max() <= 0.001
    assert candidate["terminal_completion_error_shares_audit"].max() <= 0.001
    assert paired["urgent_start_delta_days"].max() <= 0.0
