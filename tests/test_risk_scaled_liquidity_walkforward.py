from __future__ import annotations

from experiments.liquidity_forecast_walkforward import (
    LIQUIDITY_CALIBRATION_SEED,
    LIQUIDITY_EVENT_SEEDS,
    LIQUIDITY_SCENARIO_SEEDS,
    REALIZED_LIQUIDITY_SEEDS,
)
from experiments.profit_floor_walkforward import (
    PROFIT_FLOOR_EVENT_SEEDS,
    PROFIT_FLOOR_LIQUIDITY_SEEDS,
    PROFIT_FLOOR_SCENARIO_SEEDS,
)
from experiments.risk_scaled_liquidity_walkforward import (
    RISK_SCALED_EVENT_INDEX_OFFSET,
    RISK_SCALED_EVENT_SEEDS,
    RISK_SCALED_LIQUIDITY_SEEDS,
    RISK_SCALED_SCENARIO_SEEDS,
)


def test_risk_scaled_cohort_is_fresh_and_maps_to_events_73_through_96() -> None:
    prior = set(
        LIQUIDITY_EVENT_SEEDS
        + LIQUIDITY_SCENARIO_SEEDS
        + REALIZED_LIQUIDITY_SEEDS
        + PROFIT_FLOOR_EVENT_SEEDS
        + PROFIT_FLOOR_SCENARIO_SEEDS
        + PROFIT_FLOOR_LIQUIDITY_SEEDS
        + (LIQUIDITY_CALIBRATION_SEED,)
    )
    new = (
        RISK_SCALED_EVENT_SEEDS
        + RISK_SCALED_SCENARIO_SEEDS
        + RISK_SCALED_LIQUIDITY_SEEDS
    )

    assert len(new) == len(set(new))
    assert prior.isdisjoint(new)
    assert RISK_SCALED_EVENT_INDEX_OFFSET + 1 == 73
    assert RISK_SCALED_EVENT_INDEX_OFFSET + len(RISK_SCALED_EVENT_SEEDS) == 96
