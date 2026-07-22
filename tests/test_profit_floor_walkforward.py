from __future__ import annotations

from experiments.liquidity_forecast_walkforward import (
    LIQUIDITY_CALIBRATION_SEED,
    LIQUIDITY_EVENT_SEEDS,
    LIQUIDITY_SCENARIO_SEEDS,
    REALIZED_LIQUIDITY_SEEDS,
)
from experiments.profit_floor_walkforward import (
    PROFIT_FLOOR_EVENT_INDEX_OFFSET,
    PROFIT_FLOOR_EVENT_SEEDS,
    PROFIT_FLOOR_LIQUIDITY_SEEDS,
    PROFIT_FLOOR_SCENARIO_SEEDS,
)


def test_profit_floor_cohort_is_fresh_and_maps_to_events_49_through_72() -> None:
    old = set(
        LIQUIDITY_EVENT_SEEDS
        + LIQUIDITY_SCENARIO_SEEDS
        + REALIZED_LIQUIDITY_SEEDS
        + (LIQUIDITY_CALIBRATION_SEED,)
    )
    new = (
        PROFIT_FLOOR_EVENT_SEEDS
        + PROFIT_FLOOR_SCENARIO_SEEDS
        + PROFIT_FLOOR_LIQUIDITY_SEEDS
    )

    assert len(new) == len(set(new))
    assert old.isdisjoint(new)
    assert PROFIT_FLOOR_EVENT_INDEX_OFFSET + 1 == 49
    assert PROFIT_FLOOR_EVENT_INDEX_OFFSET + len(PROFIT_FLOOR_EVENT_SEEDS) == 72
