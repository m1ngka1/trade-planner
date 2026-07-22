from __future__ import annotations

from experiments.alpha_decay_walkforward import run_alpha_decay_mechanics


def test_predeclared_alpha_decay_mechanics_gates_pass() -> None:
    outputs, metadata = run_alpha_decay_mechanics(full_suite_verified=False)

    estimator_gates = outputs["gates"].loc[
        outputs["gates"]["gate"].ne("full_repository_test_suite")
    ]
    assert estimator_gates["passed"].all()
    assert metadata["decision"] == "pending_or_discard"
    assert outputs["summary"].iloc[0]["rmse_improvement_fraction"] >= 0.10
