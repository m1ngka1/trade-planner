from __future__ import annotations

from dataclasses import replace

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from trade_planner import (
    CompositeCostModel,
    ExpectedReturnAlphaModel,
    ParticipationCapModel,
    PlannerContext,
    QuadraticParticipationImpact,
    StaticCovarianceRiskModel,
    TCALinearBpsCost,
    TCAQuadraticParticipationImpact,
    TradePlanner,
    TradePlannerConfig,
    default_constraints,
)


@pytest.mark.parametrize("solver", ["CLARABEL"])
def test_per_name_scaling_preserves_heterogeneous_share_optimum(
    solver: str,
) -> None:
    if solver not in cp.installed_solvers():
        pytest.skip(f"{solver} is not installed")
    ctx = _context(targets=[1_000_000.0, -10.0], adv=[3_000_000.0, 100.0])
    base = _config(solver="OSQP", numerical_scaling="none")

    unscaled = TradePlanner(base).solve(ctx)
    scaled = TradePlanner(
        replace(
            base,
            solver=solver,
            numerical_scaling="per_name",
            verify_hard_constraints=True,
        )
    ).solve(ctx)

    np.testing.assert_allclose(
        scaled.schedule["trade_shares"],
        unscaled.schedule["trade_shares"],
        rtol=2e-6,
        atol=0.02,
    )
    assert scaled.diagnostics["objective"] == pytest.approx(
        unscaled.diagnostics["objective"],
        rel=2e-6,
        abs=0.01,
    )
    assert scaled.diagnostics["numerical_scaling"] == "per_name"
    assert scaled.diagnostics["solver_name"] == solver
    assert scaled.diagnostics["decision_scale_min_shares"] == 10.0
    assert scaled.diagnostics["decision_scale_max_shares"] == 1_000_000.0
    assert scaled.diagnostics["max_cap_excess_shares"] <= 0.05
    assert scaled.diagnostics["max_wrong_direction_shares"] <= 0.001
    assert scaled.diagnostics["max_abs_terminal_residual"] <= 0.001


def test_custom_constraint_still_receives_share_expression() -> None:
    class FirstDayShareFloor:
        def constraints(self, ctx, state):
            return [state.trades[0, 0] >= 400.0]

    ctx = _context(targets=[1_000.0], adv=[5_000.0])
    config = replace(
        _config(solver="OSQP", numerical_scaling="per_name"),
        constraints=default_constraints() + (FirstDayShareFloor(),),
        verify_hard_constraints=True,
    )

    result = TradePlanner(config).solve(ctx)

    first_trade = float(result.schedule.iloc[0]["trade_shares"])
    assert first_trade >= 400.0 - 1e-4
    assert first_trade < 1_000.0


def test_scaled_risk_alpha_and_tca_objective_is_economically_identical() -> None:
    ctx = _context(targets=[20_000.0, -200.0], adv=[100_000.0, 2_000.0])
    ctx = replace(
        ctx,
        expected_return=np.tile(np.array([0.001, -0.002])[None, :], (4, 1)),
    )
    impact = np.tile(np.array([8.0, 12.0])[None, :], (4, 1))
    linear = np.tile(np.array([1.0, 2.0])[None, :], (4, 1))
    common = dict(
        participation_model=ParticipationCapModel(),
        risk_model=StaticCovarianceRiskModel(
            covariance=np.array([[0.0004, 0.0001], [0.0001, 0.0009]])
        ),
        cost_model=CompositeCostModel(
            terms=(
                TCAQuadraticParticipationImpact(impact),
                TCALinearBpsCost(linear),
            )
        ),
        constraints=default_constraints(),
        residual_risk_weight=0.0,
        inventory_risk_weight=1e-7,
        inventory_alpha_model=ExpectedReturnAlphaModel(),
        solver="CLARABEL",
    )
    unscaled = TradePlanner(TradePlannerConfig(**common, numerical_scaling="none"))
    scaled = TradePlanner(
        TradePlannerConfig(**common, numerical_scaling="per_name")
    )
    target = ctx.orders["target_shares"].to_numpy(float)
    caps = common["participation_model"].caps(ctx)
    reference = caps / caps.sum(axis=0)[None, :] * target[None, :]
    unscaled_variable, unscaled_state = unscaled._new_decision_state(
        target=target,
        caps=caps,
        ctx=ctx,
    )
    scaled_variable, scaled_state = scaled._new_decision_state(
        target=target,
        caps=caps,
        ctx=ctx,
    )
    unscaled_variable.value = reference
    scaled_variable.value = reference / scaled_state.share_scale[None, :]

    unscaled_value = sum(
        term.value
        for term in unscaled._objective_terms(
            unscaled._objective_context(ctx, unscaled_state.share_scale),
            unscaled_state,
        )
    )
    scaled_value = sum(
        term.value
        for term in scaled._objective_terms(
            scaled._objective_context(ctx, scaled_state.share_scale),
            scaled_state,
        )
    )

    assert scaled_value == pytest.approx(unscaled_value, rel=1e-12, abs=1e-8)


def test_zero_target_scale_uses_available_cap() -> None:
    planner = TradePlanner(_config(solver="OSQP", numerical_scaling="per_name"))
    scale = planner._share_scale(
        np.array([1_000.0, 0.0]),
        np.array([[100.0, 40.0], [200.0, 50.0]]),
    )

    np.testing.assert_array_equal(scale, [1_000.0, 50.0])


def test_raw_share_certificate_detects_each_hard_violation() -> None:
    planner = TradePlanner(
        replace(
            _config(solver="OSQP", numerical_scaling="per_name"),
            verify_hard_constraints=True,
        )
    )
    certificate = planner._hard_constraint_certificate(
        trades=np.array([[60.1, 0.002], [40.0, -10.0]]),
        target=np.array([100.0, -10.0]),
        caps=np.array([[60.0, 20.0], [60.0, 20.0]]),
    )

    assert certificate["max_cap_excess_shares"] == pytest.approx(0.1)
    assert certificate["max_wrong_direction_shares"] == pytest.approx(0.002)
    assert certificate["max_abs_terminal_residual"] == pytest.approx(0.1)
    assert {item[0] for item in planner._certificate_violations(certificate)} == {
        "max_cap_excess_shares",
        "max_wrong_direction_shares",
        "max_abs_terminal_residual",
    }


@pytest.mark.parametrize("value", ["global", "target", "PER_NAME"])
def test_unknown_numerical_scaling_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="numerical_scaling"):
        _config(solver="OSQP", numerical_scaling=value)


def _config(*, solver: str, numerical_scaling: str) -> TradePlannerConfig:
    return TradePlannerConfig(
        participation_model=ParticipationCapModel(),
        risk_model=StaticCovarianceRiskModel(covariance=np.eye(2)),
        cost_model=CompositeCostModel(
            terms=(QuadraticParticipationImpact(impact_bps_at_10pct_adv=5.0),)
        ),
        constraints=default_constraints(),
        residual_risk_weight=0.0,
        inventory_risk_weight=0.0,
        solver=solver,
        numerical_scaling=numerical_scaling,
    )


def _context(*, targets: list[float], adv: list[float]) -> PlannerContext:
    dates = pd.bdate_range("2026-08-03", periods=4)
    symbols = [f"S{index}" for index in range(len(targets))]
    shape = (len(dates), len(symbols))
    return PlannerContext(
        symbols=symbols,
        dates=dates,
        orders=pd.DataFrame({"target_shares": targets}, index=symbols),
        panel=pd.DataFrame(index=pd.MultiIndex.from_product([dates, symbols])),
        price=np.tile(
            np.linspace(10.0, 100.0, len(symbols))[None, :],
            (len(dates), 1),
        ),
        adv_shares=np.tile(np.asarray(adv, dtype=float)[None, :], (len(dates), 1)),
        is_open=np.ones(shape, dtype=bool),
        base_participation=np.full(shape, 0.50),
        event_days=pd.DataFrame(np.inf, index=dates, columns=symbols),
    )
