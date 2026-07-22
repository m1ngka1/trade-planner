from __future__ import annotations

from experiments import raw_alpha_opportunity_walkforward as experiment
from experiments.liquidity_forecast_walkforward import LIQUIDITY_CALIBRATION_SEED
from experiments.profit_floor_walkforward import (
    PROFIT_FLOOR_EVENT_SEEDS,
    PROFIT_FLOOR_LIQUIDITY_SEEDS,
    PROFIT_FLOOR_SCENARIO_SEEDS,
)
from experiments.risk_scaled_liquidity_walkforward import (
    RISK_SCALED_EVENT_SEEDS,
    RISK_SCALED_LIQUIDITY_SEEDS,
    RISK_SCALED_SCENARIO_SEEDS,
)


def test_raw_alpha_cohort_is_fresh_and_maps_to_events_97_through_120() -> None:
    prior = set(
        PROFIT_FLOOR_EVENT_SEEDS
        + PROFIT_FLOOR_SCENARIO_SEEDS
        + PROFIT_FLOOR_LIQUIDITY_SEEDS
        + RISK_SCALED_EVENT_SEEDS
        + RISK_SCALED_SCENARIO_SEEDS
        + RISK_SCALED_LIQUIDITY_SEEDS
        + (LIQUIDITY_CALIBRATION_SEED,)
    )
    new = (
        experiment.RAW_ALPHA_EVENT_SEEDS
        + experiment.RAW_ALPHA_SCENARIO_SEEDS
        + experiment.RAW_ALPHA_LIQUIDITY_SEEDS
    )

    assert len(new) == len(set(new))
    assert prior.isdisjoint(new)
    assert experiment.RAW_ALPHA_EVENT_INDEX_OFFSET + 1 == 97
    assert (
        experiment.RAW_ALPHA_EVENT_INDEX_OFFSET
        + len(experiment.RAW_ALPHA_EVENT_SEEDS)
        == 120
    )


def test_raw_alpha_wrapper_freezes_optimizer_policy(monkeypatch) -> None:
    captured = {}

    def fake_run_experiment(**kwargs):
        captured.update(kwargs)
        return {}, {}

    monkeypatch.setattr(experiment, "run_experiment", fake_run_experiment)
    experiment.run_raw_alpha_experiment(n_events=2, event_start=3)

    assert captured["coefficient_policy"] == "baseline_locked"
    assert captured["alpha_policy"] == "raw"
    assert captured["factor_policy"] == "minimax_factor_stress"
    assert captured["liquidity_shape_policy"] == "risk_scaled"
    assert captured["event_index_offset"] == 96
    assert captured["development_event_count"] == 12
