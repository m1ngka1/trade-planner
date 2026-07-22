from __future__ import annotations

import unittest

import cvxpy as cp
import numpy as np
import pandas as pd

from trade_planner import (
    BarraFactorRiskModel,
    CompositeCostModel,
    ExpectedReturnAlphaModel,
    LinearBpsCost,
    ParticipationCapModel,
    PlannerContext,
    QuadraticParticipationImpact,
    RebalanceRiskMeasure,
    RiskAversion,
    TradePlanner,
    TradePlannerConfig,
    build_context_from_provider,
    build_rebalance_frontier,
    calibrate_rebalance_plan,
    centered_return_scenarios,
    default_constraints,
    infer_execution_cost_matrices,
    infer_execution_costs,
    reduce_return_scenarios,
    tail_return_scenarios,
    tail_stress_return_path,
    weighted_loss_var_cvar,
)
from trade_planner.examples import ToyProvider


@unittest.skipUnless("CLARABEL" in cp.installed_solvers(), "CLARABEL is not installed")
class RebalanceCalibrationTests(unittest.TestCase):
    def test_weighted_loss_cvar_integrates_only_the_requested_tail_mass(self) -> None:
        loss_var, loss_cvar = weighted_loss_var_cvar(
            np.array([-10.0, -5.0, 0.0, 5.0]),
            np.full(4, 0.25),
            confidence=0.75,
        )

        self.assertEqual(loss_var, 5.0)
        self.assertEqual(loss_cvar, 10.0)

    def test_tail_preserving_reduction_is_centered_deterministic_and_keeps_cvar(self) -> None:
        ctx = _economic_context()
        rng = np.random.default_rng(101)
        scenarios = rng.normal(
            scale=0.004,
            size=(200, len(ctx.dates), len(ctx.symbols)),
        )
        target_sign = np.sign(ctx.orders["target_shares"].to_numpy(float))
        scenarios[:20, 2:5, :] -= 0.025 * target_sign[None, None, :]
        ctx = PlannerContext(
            **{**ctx.__dict__, "return_residual_scenarios": scenarios}
        )

        reduced, reduced_weights = reduce_return_scenarios(
            ctx,
            max_scenarios=40,
        )
        repeated, repeated_weights = reduce_return_scenarios(
            ctx,
            max_scenarios=40,
        )

        self.assertEqual(reduced.shape, (40, len(ctx.dates), len(ctx.symbols)))
        np.testing.assert_allclose(reduced, repeated)
        np.testing.assert_allclose(reduced_weights, repeated_weights)
        self.assertAlmostEqual(float(np.sum(reduced_weights)), 1.0)
        np.testing.assert_allclose(
            np.einsum("s,stn->tn", reduced_weights, reduced),
            0.0,
            atol=1e-12,
        )

        full, full_weights = centered_return_scenarios(ctx)
        target = ctx.orders["target_shares"].to_numpy(float)
        positions = ctx.price * target[None, :]
        full_pnl = np.einsum("stn,tn->s", full, positions)
        reduced_pnl = np.einsum("stn,tn->s", reduced, positions)
        _, full_cvar = weighted_loss_var_cvar(full_pnl, full_weights)
        _, reduced_cvar = weighted_loss_var_cvar(reduced_pnl, reduced_weights)
        self.assertLess(abs(reduced_cvar / full_cvar - 1.0), 0.05)

    def test_tail_stress_path_uses_exact_weighted_mass_and_is_adverse(self) -> None:
        ctx = _economic_context()
        signs = np.sign(ctx.orders["target_shares"].to_numpy(float))
        scenarios = np.zeros((4, len(ctx.dates), len(ctx.symbols)), dtype=float)
        scenarios[0] = -0.04 * signs[None, :]
        scenarios[1] = -0.02 * signs[None, :]
        weights = np.array([0.05, 0.10, 0.35, 0.50])
        ctx = PlannerContext(
            **{
                **ctx.__dict__,
                "return_residual_scenarios": scenarios,
                "return_scenario_weights": weights,
            }
        )

        stress_path, regime_variance = tail_stress_return_path(
            ctx,
            tail_probability=0.10,
        )
        tail_scenarios, tail_weights = tail_return_scenarios(
            ctx,
            tail_probability=0.10,
        )

        centered, _ = centered_return_scenarios(ctx)
        np.testing.assert_allclose(tail_scenarios, centered[:2])
        np.testing.assert_allclose(tail_weights, [0.5, 0.5])
        np.testing.assert_allclose(stress_path, 0.5 * centered[0] + 0.5 * centered[1])
        self.assertAlmostEqual(regime_variance, 0.10 / 0.90)
        target = ctx.orders["target_shares"].to_numpy(float)
        stress_pnl = float(np.sum(stress_path * ctx.price * target[None, :]))
        self.assertLess(stress_pnl, 0.0)

    def test_expected_alpha_moves_profitable_flow_earlier(self) -> None:
        ctx = _economic_context()
        common = dict(
            participation_model=ParticipationCapModel(),
            risk_model=BarraFactorRiskModel(),
            cost_model=CompositeCostModel(
                terms=(QuadraticParticipationImpact(impact_bps_at_10pct_adv=8.0),)
            ),
            constraints=default_constraints(),
            residual_risk_weight=0.0,
            inventory_risk_weight=0.0,
            solver="CLARABEL",
        )
        without_alpha = TradePlanner(TradePlannerConfig(**common)).solve(ctx).schedule
        with_alpha = TradePlanner(
            TradePlannerConfig(
                **common,
                inventory_alpha_model=ExpectedReturnAlphaModel(),
            )
        ).solve(ctx).schedule

        self.assertLess(_average_execution_day(with_alpha, ctx), _average_execution_day(without_alpha, ctx))

    def test_risk_labels_select_monotone_pnl_budgets_from_one_frontier(self) -> None:
        ctx = _economic_context()
        frontier = build_rebalance_frontier(
            ctx,
            lambda_multipliers=(0.0, 0.01, 0.1, 1.0, 10.0, 100.0),
        )
        high = frontier.select(RiskAversion.HIGH)
        medium = frontier.select("medium")
        low = frontier.select(RiskAversion.LOW)

        self.assertLessEqual(high.risk_budget_dollars, medium.risk_budget_dollars)
        self.assertLessEqual(medium.risk_budget_dollars, low.risk_budget_dollars)
        self.assertLessEqual(high.metrics.pnl_vol_dollars, high.risk_budget_dollars + 1e-6)
        self.assertLessEqual(medium.metrics.pnl_vol_dollars, medium.risk_budget_dollars + 1e-6)
        self.assertLessEqual(low.metrics.pnl_vol_dollars, low.risk_budget_dollars + 1e-6)
        self.assertGreaterEqual(
            low.metrics.expected_net_pnl_dollars,
            high.metrics.expected_net_pnl_dollars - 1e-6,
        )
        valid = frontier.frontier[frontier.frontier["status"].str.startswith("optimal")]
        parent_gross = float(
            np.sum(
                np.abs(
                    ctx.orders["target_shares"].to_numpy(float) * ctx.price[0]
                )
            )
        )
        self.assertGreaterEqual(
            low.metrics.expected_net_pnl_dollars,
            float(valid["expected_net_pnl_dollars"].max()) - parent_gross / 10_000.0 - 1e-6,
        )

    def test_tca_inputs_set_cost_coefficients_without_user_numbers(self) -> None:
        ctx = _economic_context()
        orders = ctx.orders.copy()
        orders["impact_bps_at_10pct_adv"] = [6.0, 9.0, 15.0, 20.0]
        orders["linear_cost_bps"] = [0.5, 0.8, 1.2, 2.0]
        ctx = PlannerContext(**{**ctx.__dict__, "orders": orders})

        impact, linear = infer_execution_costs(ctx)

        self.assertEqual(impact, 9.0)
        self.assertEqual(linear, 0.8)

    def test_calibration_rejects_missing_alpha_forecast(self) -> None:
        ctx = _economic_context()
        ctx = PlannerContext(**{**ctx.__dict__, "expected_return": None})
        with self.assertRaisesRegex(ValueError, "expected_return"):
            build_rebalance_frontier(ctx, lambda_multipliers=(0.0, 1.0))

    def test_realistic_dollar_scale_keeps_nonzero_risk_frontier_solvable(self) -> None:
        ctx = _economic_context()
        frontier = build_rebalance_frontier(
            ctx,
            solver="OSQP",
            lambda_multipliers=(0.0, 0.1, 1.0, 10.0),
        )

        self.assertTrue(frontier.frontier["status"].isin(("optimal", "optimal_inaccurate")).all())
        for result in frontier.results.values():
            self.assertLessEqual(
                float(
                    np.max(
                        np.abs(result.schedule["trade_shares"].to_numpy(float))
                        - result.schedule["cap_shares"].to_numpy(float)
                    )
                ),
                0.1,
            )

    def test_unprofitable_forecast_is_flagged_instead_of_claimed_profitable(self) -> None:
        ctx = _economic_context()
        ctx = PlannerContext(
            **{**ctx.__dict__, "expected_return": np.zeros_like(ctx.expected_return)}
        )

        plan = build_rebalance_frontier(
            ctx,
            lambda_multipliers=(0.0, 0.1, 1.0, 10.0),
        ).select("medium")

        self.assertFalse(plan.economically_viable)
        self.assertLess(plan.metrics.expected_net_pnl_dollars, 0.0)

    def test_scenarios_automatically_select_hybrid_downside_calibration(self) -> None:
        ctx = _economic_context()
        rng = np.random.default_rng(11)
        scenarios = rng.normal(
            scale=0.004,
            size=(120, len(ctx.dates), len(ctx.symbols)),
        )
        target_sign = np.sign(ctx.orders["target_shares"].to_numpy(float))
        scenarios[:12, 3:5, :] -= 0.04 * target_sign[None, None, :]
        ctx = PlannerContext(
            **{**ctx.__dict__, "return_residual_scenarios": scenarios}
        )

        frontier = build_rebalance_frontier(
            ctx,
            solver="CLARABEL",
            lambda_multipliers=(0.0, 1.0, 3.0),
        )
        plan = frontier.select("medium")

        self.assertIs(frontier.risk_measure, RebalanceRiskMeasure.HYBRID_DOWNSIDE)
        self.assertEqual(frontier.risk_metric_column, "loss_cvar_95_dollars")
        self.assertGreater(frontier.scenario_tail_overlay_fraction, 0.0)
        self.assertEqual(frontier.optimization_scenario_count, 96)
        self.assertTrue(
            any(
                config.inventory_risk_weight > 0
                and config.inventory_path_risk_weight > 0
                for config in frontier.configs.values()
            )
        )
        self.assertIs(plan.risk_measure, RebalanceRiskMeasure.HYBRID_DOWNSIDE)
        medium_plan = calibrate_rebalance_plan(
            ctx,
            risk_aversion="medium",
            solver="CLARABEL",
            lambda_multipliers=(0.0, 1.0, 3.0),
        )
        self.assertIs(
            medium_plan.risk_measure,
            RebalanceRiskMeasure.HYBRID_DOWNSIDE,
        )
        high_plan = calibrate_rebalance_plan(
            ctx,
            risk_aversion="high",
            solver="CLARABEL",
            lambda_multipliers=(0.0, 1.0, 3.0),
        )
        self.assertIs(high_plan.risk_measure, RebalanceRiskMeasure.VARIANCE)
        low_plan = calibrate_rebalance_plan(
            ctx,
            risk_aversion="low",
            solver="CLARABEL",
            lambda_multipliers=(0.0, 1.0, 3.0),
        )
        self.assertIs(
            low_plan.risk_measure,
            RebalanceRiskMeasure.TAIL_SECOND_MOMENT,
        )
        self.assertEqual(low_plan.optimization_scenario_count, 12)

    def test_auto_low_falls_back_to_covariance_without_scenarios(self) -> None:
        plan = calibrate_rebalance_plan(
            _economic_context(),
            risk_aversion="low",
            solver="CLARABEL",
            lambda_multipliers=(0.0, 1.0, 3.0),
        )

        self.assertIs(plan.risk_measure, RebalanceRiskMeasure.VARIANCE)
        self.assertIsNone(plan.optimization_scenario_count)

    def test_provider_economic_inputs_are_aligned_into_context(self) -> None:
        class AlphaProvider(ToyProvider):
            def load_expected_return(self, symbols, dates):
                return pd.DataFrame(
                    {
                        symbol: np.full(len(dates), (index + 1) / 10_000.0)
                        for index, symbol in enumerate(symbols)
                    },
                    index=dates,
                )

            def load_impact_bps_at_10pct_adv(self, symbols, dates):
                return pd.DataFrame(
                    {
                        symbol: np.full(len(dates), 7.0 + index)
                        for index, symbol in enumerate(symbols)
                    },
                    index=dates,
                )

            def load_linear_cost_bps(self, symbols, dates):
                return pd.DataFrame(
                    {
                        symbol: np.full(len(dates), 0.5 + index / 10.0)
                        for index, symbol in enumerate(symbols)
                    },
                    index=dates,
                )

            def load_return_residual_scenarios(self, symbols, dates):
                return np.arange(
                    4 * len(dates) * len(symbols),
                    dtype=float,
                ).reshape(4, len(dates), len(symbols)) / 1_000_000.0

            def load_return_scenario_weights(self, symbols, dates):
                return np.array([0.1, 0.2, 0.3, 0.4])

        orders = pd.DataFrame(
            {"target_shares": [1_000.0, -800.0]},
            index=["BBB", "AAA"],
        )
        ctx = build_context_from_provider(
            orders,
            "2026-07-01",
            "2026-07-03",
            AlphaProvider(),
        )

        self.assertEqual(ctx.expected_return.shape, (3, 2))
        np.testing.assert_allclose(ctx.expected_return[0], [0.0001, 0.0002])
        impact, linear = infer_execution_cost_matrices(ctx)
        self.assertEqual(impact.shape, (3, 2))
        self.assertEqual(linear.shape, (3, 2))
        np.testing.assert_allclose(impact[0], [7.0, 8.0])
        np.testing.assert_allclose(linear[0], [0.5, 0.6])
        self.assertEqual(ctx.return_residual_scenarios.shape, (4, 3, 2))
        np.testing.assert_allclose(
            ctx.return_scenario_weights,
            [0.1, 0.2, 0.3, 0.4],
        )


def _economic_context() -> PlannerContext:
    dates = pd.bdate_range("2026-07-01", periods=6)
    symbols = ["URGENT_BUY", "URGENT_SELL", "SMALL_BUY", "SMALL_SELL"]
    targets = np.array([55_000.0, -55_000.0, 10_000.0, -10_000.0])
    prices = np.array([80.0, 82.0, 45.0, 47.0])
    shape = (len(dates), len(symbols))
    exposure = np.array(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    expected_return = np.tile(
        np.array([0.0008, -0.0008, 0.0005, -0.0005])[None, :],
        (len(dates), 1),
    )
    expected_return[-1] = 0.0
    return PlannerContext(
        symbols=symbols,
        dates=dates,
        orders=pd.DataFrame({"target_shares": targets}, index=symbols),
        panel=pd.DataFrame(index=pd.MultiIndex.from_product([dates, symbols])),
        price=np.tile(prices[None, :], (len(dates), 1)),
        adv_shares=np.full(shape, 100_000.0),
        is_open=np.ones(shape, dtype=bool),
        base_participation=np.full(shape, 0.10),
        event_days=pd.DataFrame(np.inf, index=dates, columns=symbols),
        factor_names=["country", "sector", "industry", "offset"],
        factor_exposure=np.tile(exposure[None, :, :], (len(dates), 1, 1)),
        factor_covariance=np.tile(
            (np.eye(exposure.shape[1]) * 0.0001)[None, :, :],
            (len(dates), 1, 1),
        ),
        specific_variance=np.full(shape, 0.000025),
        expected_return=expected_return,
    )


def _average_execution_day(schedule: pd.DataFrame, ctx: PlannerContext) -> float:
    trades = (
        schedule.pivot(index="date", columns="symbol", values="trade_shares")
        .reindex(index=ctx.dates, columns=ctx.symbols)
        .to_numpy(float)
    )
    daily = np.sum(np.abs(trades) * ctx.price, axis=1)
    days = np.arange(1, len(daily) + 1, dtype=float)
    return float(np.dot(days, daily) / np.sum(daily))


if __name__ == "__main__":
    unittest.main()
