from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from experiments.liquidity_forecast_walkforward import (
    BaselineRelativeCVaRRiskModel,
    BaselineRelativeSecondMomentRiskModel,
    DATE_LOG_LIQUIDITY_STD,
    EVENT_LOG_LIQUIDITY_STD,
    LIQUIDITY_CALIBRATION_SEED,
    LIQUIDITY_EVENT_SEEDS,
    LIQUIDITY_SCENARIO_SEEDS,
    NAME_LOG_LIQUIDITY_STD,
    REALIZED_LIQUIDITY_SEEDS,
    alpha_confidence_for_risk_profile,
    calibrate_liquidity_distribution,
    capacity_slack_fraction,
    factor_stress_fraction_for_risk_profile,
    forecast_adv_for_risk_profile,
    regret_weight_for_risk_profile,
    run_experiment,
    risk_scaled_liquidity_forecast,
    simulate_liquidity_multipliers,
    specific_risk_fraction_for_risk_profile,
)
from experiments.rebalance_economic_calibration import (
    EVENT_LIQUIDITY_CURVES,
    economic_fixture,
)
from trade_planner import BarraFactorRiskModel, RiskAversion


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


def test_research_quantile_override_matches_the_same_population_quantile() -> None:
    ctx, _ = economic_fixture()
    calibration = calibrate_liquidity_distribution(ctx)

    medium_at_median = forecast_adv_for_risk_profile(
        ctx,
        calibration,
        RiskAversion.MEDIUM,
        quantile=0.50,
    )
    low_default = forecast_adv_for_risk_profile(
        ctx,
        calibration,
        RiskAversion.LOW,
    )

    np.testing.assert_array_equal(medium_at_median, low_default)


def test_baseline_locked_policy_reuses_automatic_risk_price() -> None:
    outputs, metadata = run_experiment(
        n_events=2,
        risk_aversion="medium",
        coefficient_policy="baseline_locked",
    )
    coefficients = outputs["coefficients"].pivot(
        index="event_id",
        columns="strategy",
        values="inventory_risk_weight",
    )

    np.testing.assert_array_equal(
        coefficients["forecast_liquidity"],
        coefficients["static_open_loop"],
    )
    assert metadata["coefficient_policy"] == "baseline_locked"


def test_risk_label_and_capacity_set_optional_alpha_hurdle() -> None:
    ctx, classifications = economic_fixture()
    slack = dict(zip(ctx.symbols, capacity_slack_fraction(ctx)))
    urgent = [slack[symbol] for symbol in classifications.index[
        classifications["urgency"].eq("urgent")
    ]]
    small = [slack[symbol] for symbol in classifications.index[
        classifications["urgency"].eq("small")
    ]]

    assert alpha_confidence_for_risk_profile(RiskAversion.HIGH) == 0.975
    assert alpha_confidence_for_risk_profile(RiskAversion.MEDIUM) == 0.75
    assert alpha_confidence_for_risk_profile(RiskAversion.LOW) == 0.50
    assert factor_stress_fraction_for_risk_profile(RiskAversion.HIGH) == 0.95
    assert factor_stress_fraction_for_risk_profile(RiskAversion.MEDIUM) == 0.50
    assert factor_stress_fraction_for_risk_profile(RiskAversion.LOW) == 0.0
    assert specific_risk_fraction_for_risk_profile(RiskAversion.HIGH) == 0.95
    assert specific_risk_fraction_for_risk_profile(RiskAversion.MEDIUM) == 0.50
    assert specific_risk_fraction_for_risk_profile(RiskAversion.LOW) == 0.25
    assert regret_weight_for_risk_profile(RiskAversion.HIGH) == 0.95
    assert regret_weight_for_risk_profile(RiskAversion.MEDIUM) == 0.50
    assert regret_weight_for_risk_profile(RiskAversion.LOW) == 0.0
    assert float(np.mean(small)) > float(np.mean(urgent))


def test_barra_specific_risk_multiplier_changes_only_specific_variance() -> None:
    ctx, _ = economic_fixture()
    position = cp.Constant(np.linspace(-100.0, 100.0, len(ctx.symbols)))

    factor_only = BarraFactorRiskModel(
        specific_variance_multiplier=0.0
    ).objective(position, ctx, 0).value
    half_specific = BarraFactorRiskModel(
        specific_variance_multiplier=0.5
    ).objective(position, ctx, 0).value
    full_specific = BarraFactorRiskModel(
        specific_variance_multiplier=1.0
    ).objective(position, ctx, 0).value

    assert half_specific == pytest.approx(
        0.5 * (factor_only + full_specific),
        rel=1e-12,
    )
    with pytest.raises(ValueError, match="finite and non-negative"):
        BarraFactorRiskModel(specific_variance_multiplier=-0.1)


def test_risk_scaled_liquidity_shape_uses_existing_risk_budget_fraction() -> None:
    flat = np.ones((2, 2), dtype=float)
    full = np.full((2, 2), 4.0, dtype=float)

    medium = risk_scaled_liquidity_forecast(
        flat,
        full,
        RiskAversion.MEDIUM,
    )
    low = risk_scaled_liquidity_forecast(flat, full, RiskAversion.LOW)

    np.testing.assert_allclose(medium, 2.0)
    np.testing.assert_array_equal(low, flat)


def test_baseline_relative_cvar_is_zero_for_the_reference_inventory() -> None:
    ctx, _ = economic_fixture()
    baseline = np.arange(
        len(ctx.dates) * len(ctx.symbols),
        dtype=float,
    ).reshape(len(ctx.dates), len(ctx.symbols))
    model = BaselineRelativeCVaRRiskModel(baseline)
    positions = tuple(cp.Constant(row) for row in baseline)

    problem = cp.Problem(cp.Minimize(model.objective(positions, ctx)))
    problem.solve(solver="OSQP")

    assert problem.status == "optimal"
    assert problem.value == pytest.approx(0.0, abs=1e-6)


def test_baseline_relative_cvar_rejects_misaligned_reference() -> None:
    ctx, _ = economic_fixture()
    model = BaselineRelativeCVaRRiskModel(np.zeros((1, len(ctx.symbols))))
    positions = tuple(
        cp.Constant(np.zeros(len(ctx.symbols))) for _ in ctx.dates
    )

    with pytest.raises(ValueError, match="align with planner dates"):
        model.objective(positions, ctx)


def test_baseline_relative_second_moment_is_zero_for_reference_inventory() -> None:
    ctx, _ = economic_fixture()
    baseline = np.arange(
        len(ctx.dates) * len(ctx.symbols),
        dtype=float,
    ).reshape(len(ctx.dates), len(ctx.symbols))
    model = BaselineRelativeSecondMomentRiskModel(baseline)
    positions = tuple(cp.Constant(row) for row in baseline)

    objective = model.objective(positions, ctx)

    assert objective.is_convex()
    assert objective.value == pytest.approx(0.0, abs=1e-9)
