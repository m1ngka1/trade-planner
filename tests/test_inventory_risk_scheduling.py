from __future__ import annotations

import unittest

import cvxpy as cp
import numpy as np
import pandas as pd

from trade_planner.config import TradePlannerConfig
from trade_planner.config import default_rebalance_aware_config
from trade_planner.constraints import default_constraints
from trade_planner.context import PlannerContext
from trade_planner.costs import CompositeCostModel, QuadraticParticipationImpact
from trade_planner.participation import ParticipationCapModel
from trade_planner.planner import TradePlanner
from trade_planner.risk import BarraFactorRiskModel, StaticCovarianceRiskModel


@unittest.skipUnless("CLARABEL" in cp.installed_solvers(), "CLARABEL is not installed")
class InventoryRiskSchedulingTests(unittest.TestCase):
    def test_inventory_risk_backloads_while_quadratic_impact_smooths(self) -> None:
        ctx = _context(targets=[60.0], adv=[200.0], periods=6)
        impact = CompositeCostModel(
            terms=(QuadraticParticipationImpact(impact_bps_at_10pct_adv=5.0),)
        )
        risk = StaticCovarianceRiskModel(covariance=np.array([[0.0001]]))

        inventory_schedule = _planner(
            risk_model=risk,
            cost_model=impact,
            inventory_risk_weight=1.0,
            residual_risk_weight=0.0,
        ).solve(ctx).schedule
        residual_schedule = _planner(
            risk_model=risk,
            cost_model=impact,
            inventory_risk_weight=0.0,
            residual_risk_weight=1.0,
        ).solve(ctx).schedule

        inventory_volume = inventory_schedule["trade_shares"].to_numpy(float)
        residual_volume = residual_schedule["trade_shares"].to_numpy(float)
        day = np.arange(len(ctx.dates), dtype=float)

        self.assertTrue(np.all(np.diff(inventory_volume) >= -1e-5), inventory_volume)
        self.assertGreater(inventory_volume[-1], inventory_volume[0] + 1.0)
        self.assertGreater(
            np.dot(day, inventory_volume) / inventory_volume.sum(),
            np.dot(day, residual_volume) / residual_volume.sum() + 1.0,
        )

    def test_urgent_name_obeys_latest_start_floor_while_flexible_name_waits(self) -> None:
        ctx = _context(targets=[55.0, 20.0], adv=[100.0, 100.0], periods=6)
        result = _planner(
            risk_model=StaticCovarianceRiskModel(covariance=np.eye(2) * 0.0001),
            cost_model=CompositeCostModel(terms=()),
            inventory_risk_weight=1.0,
            residual_risk_weight=0.0,
        ).solve(ctx)

        trades = _trade_matrix(result.schedule, ctx)
        caps = ctx.base_participation * ctx.adv_shares
        targets = np.abs(ctx.orders["target_shares"].to_numpy(float))
        cumulative = np.cumsum(np.abs(trades), axis=0)
        future_capacity = np.flip(np.cumsum(np.flip(caps, axis=0), axis=0), axis=0) - caps
        latest_start_floors = np.maximum(targets[None, :] - future_capacity, 0.0)

        self.assertTrue(np.all(cumulative + 1e-5 >= latest_start_floors))
        self.assertGreaterEqual(trades[0, 0], 5.0 - 1e-5)
        self.assertLess(abs(trades[0, 1]), 1e-3)
        self.assertGreater(np.sum(trades[-2:, 1]), 19.99)

    def test_barra_inventory_risk_balances_early_country_sector_exposure(self) -> None:
        symbols = ["URGENT_BUY", "FLEX_HEDGE", "FLEX_UNMATCHED"]
        ctx = _context(
            targets=[35.0, -35.0, 35.0],
            adv=[100.0, 300.0, 300.0],
            periods=4,
            symbols=symbols,
        )
        # The urgent buy and flexible sell share country/sector exposure, so
        # matched signed dollar inventory offsets.  The unmatched buy cannot.
        static_exposure = np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        )
        ctx = PlannerContext(
            **{
                **ctx.__dict__,
                "factor_names": ["country_US", "sector_tech", "country_HK", "sector_health"],
                "factor_exposure": np.tile(static_exposure[None, :, :], (len(ctx.dates), 1, 1)),
                "factor_covariance": np.tile(
                    (np.eye(static_exposure.shape[1]) * 0.01)[None, :, :],
                    (len(ctx.dates), 1, 1),
                ),
                # Positive specific risk removes the zero-risk degeneracy of a
                # perfectly factor-matched pair and still makes factor balance
                # the dominant early-basket consideration.
                "specific_variance": np.full((len(ctx.dates), len(symbols)), 1e-6),
            }
        )
        result = _planner(
            risk_model=BarraFactorRiskModel(),
            cost_model=CompositeCostModel(terms=()),
            inventory_risk_weight=1.0,
            residual_risk_weight=0.0,
        ).solve(ctx)

        trades = _trade_matrix(result.schedule, ctx)
        cumulative = np.cumsum(trades, axis=0)
        first_day_factor_dollars = static_exposure.T @ cumulative[0]

        self.assertGreaterEqual(trades[0, 0], 5.0 - 1e-5)
        self.assertAlmostEqual(cumulative[0, 1], -cumulative[0, 0], delta=1e-3)
        self.assertLess(abs(cumulative[0, 2]), 1e-3)
        self.assertLess(np.linalg.norm(first_day_factor_dollars), 1e-3)

    def test_zero_inventory_weight_preserves_residual_only_schedule(self) -> None:
        ctx = _context(targets=[35.0, -25.0], adv=[100.0, 100.0], periods=5)
        risk = StaticCovarianceRiskModel(covariance=np.eye(2) * 0.0001)
        impact = CompositeCostModel(
            terms=(QuadraticParticipationImpact(impact_bps_at_10pct_adv=5.0),)
        )

        omitted = _planner(
            risk_model=risk,
            cost_model=impact,
            residual_risk_weight=1.0,
        ).solve(ctx)
        explicit_zero = _planner(
            risk_model=risk,
            cost_model=impact,
            inventory_risk_weight=0.0,
            residual_risk_weight=1.0,
        ).solve(ctx)

        np.testing.assert_allclose(
            omitted.schedule["trade_shares"],
            explicit_zero.schedule["trade_shares"],
            atol=1e-7,
        )

    def test_zero_risk_weights_skip_risk_model_calls(self) -> None:
        class UnexpectedRiskModel:
            def objective(self, position_shares, ctx, date_index):
                raise AssertionError("zero-weight risk model should not be evaluated")

        ctx = _context(targets=[20.0], adv=[100.0], periods=3)
        result = _planner(
            risk_model=UnexpectedRiskModel(),
            cost_model=CompositeCostModel(
                terms=(QuadraticParticipationImpact(impact_bps_at_10pct_adv=5.0),)
            ),
            inventory_risk_weight=0.0,
            residual_risk_weight=0.0,
        ).solve(ctx)

        self.assertLess(result.diagnostics["max_abs_terminal_residual"], 1e-5)

    def test_negative_inventory_risk_weight_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "inventory_risk_weight must be non-negative"):
            _planner(
                risk_model=StaticCovarianceRiskModel(),
                cost_model=CompositeCostModel(terms=()),
                inventory_risk_weight=-1.0,
                residual_risk_weight=0.0,
            )

    def test_negative_residual_risk_weight_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "residual_risk_weight must be non-negative"):
            _planner(
                risk_model=StaticCovarianceRiskModel(),
                cost_model=CompositeCostModel(terms=()),
                inventory_risk_weight=0.0,
                residual_risk_weight=-1.0,
            )

    def test_rebalance_default_uses_physical_caps_and_inventory_risk(self) -> None:
        config = default_rebalance_aware_config()

        self.assertEqual(tuple(config.participation_model.modifiers), ())
        self.assertEqual(config.residual_risk_weight, 0.0)
        self.assertEqual(config.inventory_risk_weight, 1.0)
        self.assertEqual(config.solver, "CLARABEL")


def _planner(
    *,
    risk_model,
    cost_model: CompositeCostModel,
    residual_risk_weight: float,
    inventory_risk_weight: float = 0.0,
) -> TradePlanner:
    return TradePlanner(
        TradePlannerConfig(
            participation_model=ParticipationCapModel(),
            risk_model=risk_model,
            cost_model=cost_model,
            constraints=default_constraints(),
            residual_risk_weight=residual_risk_weight,
            solver="CLARABEL",
            inventory_risk_weight=inventory_risk_weight,
        )
    )


def _context(
    *,
    targets: list[float],
    adv: list[float],
    periods: int,
    symbols: list[str] | None = None,
) -> PlannerContext:
    dates = pd.bdate_range("2026-07-01", periods=periods)
    if symbols is None:
        symbols = [f"S{i}" for i in range(len(targets))]
    shape = (periods, len(symbols))
    return PlannerContext(
        symbols=symbols,
        dates=dates,
        orders=pd.DataFrame({"target_shares": targets}, index=symbols),
        panel=pd.DataFrame(index=pd.MultiIndex.from_product([dates, symbols])),
        price=np.ones(shape),
        adv_shares=np.tile(np.asarray(adv, dtype=float)[None, :], (periods, 1)),
        is_open=np.ones(shape, dtype=bool),
        base_participation=np.full(shape, 0.10),
        event_days=pd.DataFrame(np.inf, index=dates, columns=symbols),
    )


def _trade_matrix(schedule: pd.DataFrame, ctx: PlannerContext) -> np.ndarray:
    return (
        schedule.pivot(index="date", columns="symbol", values="trade_shares")
        .reindex(index=ctx.dates, columns=ctx.symbols)
        .to_numpy(float)
    )


if __name__ == "__main__":
    unittest.main()
