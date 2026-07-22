from __future__ import annotations

import numpy as np

from experiments.liquidity_forecast_walkforward import (
    DATE_LOG_LIQUIDITY_STD,
    EVENT_LOG_LIQUIDITY_STD,
    LIQUIDITY_CALIBRATION_SEED,
    LIQUIDITY_EVENT_SEEDS,
    LIQUIDITY_SCENARIO_SEEDS,
    NAME_LOG_LIQUIDITY_STD,
    REALIZED_LIQUIDITY_SEEDS,
    calibrate_liquidity_distribution,
    forecast_adv_for_risk_profile,
    simulate_liquidity_multipliers,
)
from experiments.rebalance_economic_calibration import (
    EVENT_LIQUIDITY_CURVES,
    economic_fixture,
)
from trade_planner import RiskAversion


def test_fresh_liquidity_seeds_are_disjoint() -> None:
    seeds = (
        LIQUIDITY_EVENT_SEEDS
        + LIQUIDITY_SCENARIO_SEEDS
        + REALIZED_LIQUIDITY_SEEDS
        + (LIQUIDITY_CALIBRATION_SEED,)
    )

    assert len(seeds) == len(set(seeds))


def test_liquidity_population_is_reproducible_and_event_shaped() -> None:
    ctx, _ = economic_fixture()

    first = simulate_liquidity_multipliers(ctx, n_events=3, seed=123)
    second = simulate_liquidity_multipliers(ctx, n_events=3, seed=123)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (3, len(ctx.dates), len(ctx.symbols))
    assert np.all(first > 0.0)
    assert float(np.mean(first[:, -1])) > float(np.mean(first[:, 0]))


def test_calibration_recovers_log_liquidity_population() -> None:
    ctx, _ = economic_fixture()
    calibration = calibrate_liquidity_distribution(ctx)
    expected_log_mean = np.log(
        EVENT_LIQUIDITY_CURVES["medium_event_liquidity"]
    )[:, None]
    expected_log_std = np.sqrt(
        EVENT_LOG_LIQUIDITY_STD**2
        + DATE_LOG_LIQUIDITY_STD**2
        + NAME_LOG_LIQUIDITY_STD**2
    )

    np.testing.assert_allclose(
        calibration["log_mean"],
        np.broadcast_to(expected_log_mean, calibration["log_mean"].shape),
        atol=0.012,
    )
    np.testing.assert_allclose(
        calibration["log_std"],
        expected_log_std,
        atol=0.007,
    )


def test_risk_label_selects_monotone_liquidity_buffer() -> None:
    ctx, _ = economic_fixture()
    calibration = calibrate_liquidity_distribution(ctx)

    high = forecast_adv_for_risk_profile(ctx, calibration, RiskAversion.HIGH)
    medium = forecast_adv_for_risk_profile(ctx, calibration, RiskAversion.MEDIUM)
    low = forecast_adv_for_risk_profile(ctx, calibration, RiskAversion.LOW)

    assert np.all(high < medium)
    assert np.all(medium < low)
    assert float(np.mean(medium[-1])) > float(np.mean(medium[0]))
