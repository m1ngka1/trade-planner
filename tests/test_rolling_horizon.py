from __future__ import annotations

from dataclasses import replace

import cvxpy as cp
import numpy as np

from experiments.alpha_confidence_walkforward import (
    EVENT_SEEDS,
    SCENARIO_SEEDS,
    _build_event,
    _build_event_with_truth,
    _calibrated_alpha_uncertainty,
)
from experiments.rebalance_economic_calibration import economic_fixture
from experiments.rolling_horizon_walkforward import (
    InventoryPathRevisionRiskModel,
    ScheduleRevisionCost,
    _automatic_revision_cost_bps,
    _executable_lot_trades,
    _forecast_vintages,
    _material_replan_decision,
    _slice_context,
)
from trade_planner import RiskAversion


def test_truth_returning_event_builder_preserves_public_fixture() -> None:
    ctx, _ = economic_fixture()
    uncertainty = _calibrated_alpha_uncertainty(ctx)
    public_event, public_rmse = _build_event(
        ctx,
        uncertainty,
        0,
        EVENT_SEEDS[0],
        SCENARIO_SEEDS[0],
    )
    research_event, research_rmse, latent = _build_event_with_truth(
        ctx,
        uncertainty,
        0,
        EVENT_SEEDS[0],
        SCENARIO_SEEDS[0],
    )

    assert public_event.event_id == research_event.event_id
    assert public_rmse == research_rmse
    np.testing.assert_array_equal(
        public_event.ctx.expected_return,
        research_event.ctx.expected_return,
    )
    np.testing.assert_array_equal(
        public_event.realized_returns,
        research_event.realized_returns,
    )
    assert latent.shape == public_event.ctx.expected_return.shape


def test_forecast_vintages_preserve_initial_snapshot_and_reduce_uncertainty() -> None:
    latent = np.zeros((4, 2))
    initial = np.array(
        [
            [0.04, -0.02],
            [0.03, -0.01],
            [0.02, 0.01],
            [0.01, 0.02],
        ]
    )
    uncertainty = np.full_like(initial, 0.02)

    forecasts, uncertainties = _forecast_vintages(
        initial_forecast=initial,
        latent_expected_return=latent,
        initial_uncertainty=uncertainty,
        retention=0.65,
        innovation_scale=0.20,
        seed=123,
    )

    np.testing.assert_array_equal(forecasts[0], initial)
    np.testing.assert_array_equal(uncertainties[0], uncertainty)
    assert len(forecasts) == len(initial)
    assert np.mean(uncertainties[-1]) < np.mean(uncertainties[0])
    assert not np.shares_memory(forecasts[0], initial)


def test_revision_cost_is_in_dollars_and_avoids_shift_double_counting() -> None:
    ctx, _ = economic_fixture()
    reference = np.zeros((len(ctx.dates), len(ctx.symbols)))
    term = ScheduleRevisionCost(reference_trades=reference, revision_cost_bps=2.0)
    trade = cp.Variable(len(ctx.symbols))
    trade.value = np.ones(len(ctx.symbols))

    value = float(term.objective(trade, ctx, 0).value)
    expected = 0.5 * 2.0 / 10_000.0 * float(np.sum(ctx.price[0]))

    assert np.isclose(value, expected)


def test_inventory_revision_cost_prices_changed_dollar_days() -> None:
    ctx, _ = economic_fixture()
    reference = np.zeros((len(ctx.dates), len(ctx.symbols)))
    reference_scale_dollars = 50_000_000.0
    model = InventoryPathRevisionRiskModel(
        reference_cumulative_trades=reference,
        revision_cost_bps=1.5,
        reference_scale_dollars=reference_scale_dollars,
    )
    cumulative = tuple(cp.Variable(len(ctx.symbols)) for _ in ctx.dates)
    for date_index, variable in enumerate(cumulative):
        variable.value = np.full(len(ctx.symbols), date_index + 1.0)

    value = float(model.objective(cumulative, ctx).value)
    expected = 1.5 / 10_000.0 / reference_scale_dollars * sum(
        float(np.sum((ctx.price[date_index] * (date_index + 1.0)) ** 2))
        for date_index in range(len(ctx.dates))
    )

    assert np.isclose(value, expected)


def test_revision_hurdle_is_automatic_and_monotone_by_risk_profile() -> None:
    ctx, _ = economic_fixture()
    uncertainty = _calibrated_alpha_uncertainty(ctx)
    ctx = replace(ctx, expected_return_uncertainty=uncertainty)
    remaining = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)

    high = _automatic_revision_cost_bps(ctx, remaining, RiskAversion.HIGH)
    medium = _automatic_revision_cost_bps(ctx, remaining, RiskAversion.MEDIUM)
    low = _automatic_revision_cost_bps(ctx, remaining, RiskAversion.LOW)

    assert high > medium > low >= 1.0


def test_slice_context_changes_only_remaining_horizon_and_target() -> None:
    ctx, _ = economic_fixture()
    original_target = ctx.orders["target_shares"].to_numpy(float).copy()
    remaining = 0.6 * original_target

    sliced = _slice_context(
        ctx,
        start=3,
        remaining_target=remaining,
        expected_return=ctx.expected_return,
        expected_return_uncertainty=np.ones_like(ctx.expected_return) * 1e-4,
    )

    assert sliced.dates.equals(ctx.dates[3:])
    assert sliced.price.shape == (len(ctx.dates) - 3, len(ctx.symbols))
    assert sliced.factor_exposure.shape[0] == len(ctx.dates) - 3
    assert sliced.return_residual_scenarios.shape[1] == len(ctx.dates) - 3
    np.testing.assert_array_equal(
        sliced.orders["target_shares"].reindex(ctx.symbols).to_numpy(float),
        remaining,
    )
    np.testing.assert_array_equal(
        ctx.orders["target_shares"].to_numpy(float),
        original_target,
    )


def test_executable_lots_preserve_future_completion_capacity() -> None:
    executed = _executable_lot_trades(
        raw_trades=np.array([3.4, -2.4]),
        remaining_target=np.array([10.0, -4.0]),
        caps=np.array([5.0, 3.0]),
        future_capacity=np.array([5.0, 2.0]),
    )

    # The first raw trade would leave seven shares for only five shares of
    # future capacity, so hard completion makes five shares urgent today.
    np.testing.assert_array_equal(executed, np.array([5.0, -2.0]))


def test_defensive_replan_rejects_profit_only_change() -> None:
    kwargs = {
        "expected_pnl_gain_bps": 2.5,
        "forecast_vol_reduction_bps": -0.5,
        "forecast_cvar_reduction_bps": -0.5,
        "threshold_bps": 2.0,
    }

    assert _material_replan_decision(**kwargs, allow_profit_case=True)
    assert not _material_replan_decision(**kwargs, allow_profit_case=False)
    assert _material_replan_decision(
        expected_pnl_gain_bps=-1.0,
        forecast_vol_reduction_bps=2.5,
        forecast_cvar_reduction_bps=0.5,
        threshold_bps=2.0,
        allow_profit_case=False,
    )
