from __future__ import annotations

from experiments.automatic_risk_profile_walkforward import (
    run_automatic_risk_profile_mechanics,
)


def test_predeclared_automatic_risk_profile_mechanics_gates_pass() -> None:
    outputs, metadata = run_automatic_risk_profile_mechanics(
        full_suite_verified=False
    )

    mechanics = outputs["gates"].loc[
        outputs["gates"]["gate"].ne("full_repository_test_suite")
    ]
    assert mechanics["passed"].all()
    assert metadata["decision"] == "pending_or_discard"
    assert outputs["summary"]["mean_selected_net_pnl_bps"].gt(0.0).all()
