from __future__ import annotations

import pytest

from experiments.systematic_first_risk_walkforward import (
    run_systematic_first_risk_experiment,
)


def test_systematic_first_mechanics_use_only_spent_events() -> None:
    outputs, metadata = run_systematic_first_risk_experiment(n_events=2)

    assert set(outputs["summary"]["strategy"]) == {
        "static_open_loop",
        "full_specific_control",
        "systematic_first",
    }
    assert outputs["trials"].shape[0] == 6
    assert outputs["risk_decomposition"]["event_id"].nunique() == 2
    assert metadata["challenger_specific_risk_fraction"] == 0.50
    assert metadata["sealed_events_untouched"] is True
    assert outputs["coefficients"].loc[
        outputs["coefficients"]["strategy"].eq("systematic_first"),
        "specific_risk_fraction",
    ].eq(0.50).all()


def test_systematic_first_rejects_sealed_synthetic_events() -> None:
    with pytest.raises(ValueError, match="events 109-120 remain sealed"):
        run_systematic_first_risk_experiment(n_events=2, event_start=11)
