from __future__ import annotations

from dataclasses import replace

import cvxpy as cp
import numpy as np
import pytest

from experiments.alpha_confidence_walkforward import _calibrated_alpha_uncertainty
from experiments.forecast_error_risk_walkforward import (
    ForecastErrorPathRiskModel,
    calibrated_persistent_directional_scale,
    estimate_persistent_directional_scale,
)
from experiments.rebalance_economic_calibration import economic_fixture


def test_persistent_directional_scale_removes_independent_sampling_noise() -> None:
    rng = np.random.default_rng(12345)
    target_sign = np.array([1.0, -1.0, 1.0])
    independent = rng.standard_normal((40_000, 4, 3))
    persistent = rng.normal(0.0, 0.40, size=(40_000, 1, 1))
    standardized_errors = (
        independent + persistent * target_sign[None, None, :]
    )

    estimate = estimate_persistent_directional_scale(
        standardized_errors,
        target_sign,
    )

    assert estimate == pytest.approx(0.40, abs=0.015)


def test_forecast_error_path_objective_matches_dollar_variance() -> None:
    ctx, _ = economic_fixture()
    uncertainty = _calibrated_alpha_uncertainty(ctx)
    ctx = replace(ctx, expected_return_uncertainty=uncertainty)
    scale = 0.35
    model = ForecastErrorPathRiskModel(scale)
    cumulative = tuple(cp.Variable(len(ctx.symbols)) for _ in ctx.dates)
    values = []
    for date_index, variable in enumerate(cumulative):
        value = np.linspace(-2.0, 3.0, len(ctx.symbols)) * (date_index + 1.0)
        variable.value = value
        values.append(value)

    objective_value = float(model.objective(cumulative, ctx).value)
    cumulative_dollars = np.asarray(values) * ctx.price
    uncertainty_dollars = uncertainty * cumulative_dollars
    target_sign = np.sign(
        ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    )
    expected = float(
        np.sum(np.square(uncertainty_dollars))
        + np.square(scale * np.sum(uncertainty_dollars * target_sign[None, :]))
    )

    assert objective_value == pytest.approx(expected)


def test_forecast_error_path_requires_point_in_time_uncertainty() -> None:
    ctx, _ = economic_fixture()
    model = ForecastErrorPathRiskModel(0.25)
    cumulative = tuple(cp.Variable(len(ctx.symbols)) for _ in ctx.dates)

    with pytest.raises(ValueError, match="expected_return_uncertainty"):
        model.objective(cumulative, ctx)


def test_synthetic_history_calibration_recovers_population_scale() -> None:
    ctx, _ = economic_fixture()

    estimate = calibrated_persistent_directional_scale(ctx)

    assert estimate == pytest.approx(0.55, abs=0.02)
