from __future__ import annotations

import unittest

import cvxpy as cp
import numpy as np
import pandas as pd

from trade_planner.config import TradePlannerConfig
from trade_planner.constraints import default_constraints
from trade_planner.context import PlannerContext
from trade_planner.costs import CompositeCostModel, QuadraticParticipationImpact
from trade_planner.participation import (
    AdaptiveAnnouncementParticipation,
    AnnouncementParticipationCurve,
    AnnouncementParticipationModifier,
    ParticipationCapModel,
    announcement_participation_rates,
)
from trade_planner.planner import TradePlanner
from trade_planner.risk import StaticCovarianceRiskModel


class AnnouncementParticipationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.date_range("2026-07-01", periods=15, freq="D")
        self.announcement = self.dates[9]

    def test_given_step_scenario(self) -> None:
        rates = announcement_participation_rates(self.dates, self.announcement)

        self.assertAlmostEqual(rates.iloc[4], 0.025)   # day 5
        self.assertAlmostEqual(rates.iloc[9], 0.025)   # day 10 / announcement
        self.assertAlmostEqual(rates.iloc[11], 0.15)   # day 12

    def test_logistic_transition_preserves_announcement_day_rate(self) -> None:
        curve = AnnouncementParticipationCurve(transition="logistic", transition_days=3)
        rates = curve.rates(self.dates, self.announcement)

        self.assertAlmostEqual(rates.loc[self.announcement], 0.025)
        self.assertGreater(rates.iloc[10], 0.025)
        self.assertLess(rates.iloc[10], 0.15)
        self.assertAlmostEqual(rates.iloc[12], 0.15)

    def test_higher_volatility_reduces_rate_on_both_sides(self) -> None:
        curve = AnnouncementParticipationCurve(
            pre_volatility_sensitivity=1.0,
            post_volatility_sensitivity=1.0,
            reference_volatility=0.20,
        )
        volatility = np.full(len(self.dates), 0.20)
        volatility[[4, 11]] = 0.40
        rates = curve.rates(self.dates, self.announcement, volatility=volatility)

        self.assertAlmostEqual(rates.iloc[4], 0.0125)
        self.assertAlmostEqual(rates.iloc[11], 0.075)

    def test_modifier_can_raise_base_rate_after_announcement(self) -> None:
        symbols = ["AAA"]
        base = np.full((len(self.dates), 1), 0.025)
        ctx = PlannerContext(
            symbols=symbols,
            dates=self.dates,
            orders=pd.DataFrame({"target_shares": [100.0]}, index=symbols),
            panel=pd.DataFrame(index=pd.MultiIndex.from_product([self.dates, symbols])),
            price=np.ones((len(self.dates), 1)),
            adv_shares=np.full((len(self.dates), 1), 100.0),
            is_open=np.ones((len(self.dates), 1), dtype=bool),
            base_participation=base,
            event_days=pd.DataFrame(np.inf, index=self.dates, columns=symbols),
        )
        model = ParticipationCapModel(
            modifiers=[AnnouncementParticipationModifier(self.announcement)]
        )

        caps = model.caps(ctx)

        self.assertAlmostEqual(caps[9, 0], 2.5)
        self.assertAlmostEqual(caps[11, 0], 15.0)

    def test_announcement_at_window_edges(self) -> None:
        first = announcement_participation_rates(self.dates, self.dates[0])
        last = announcement_participation_rates(self.dates, self.dates[-1])

        self.assertAlmostEqual(first.iloc[0], 0.025)
        self.assertTrue(np.allclose(first.iloc[1:], 0.15))
        self.assertTrue(np.allclose(last, 0.025))

    def test_adaptive_model_infers_different_pre_event_fractions(self) -> None:
        ctx = self._adaptive_context(targets=[40.0, -35.0])
        modifier = AdaptiveAnnouncementParticipation(
            pre_event_flex=0.0,
            capacity_buffer=0.0,
            balance_sides=False,
        )

        summary = modifier.allocation_summary(ctx)

        self.assertAlmostEqual(summary.loc["AAA", "mandatory_pre_fraction"], 0.25)
        self.assertAlmostEqual(summary.loc["BBB", "mandatory_pre_fraction"], 5.0 / 35.0)
        self.assertAlmostEqual(summary.loc["AAA", "pre_event_cap_fraction"], 0.25)
        self.assertAlmostEqual(summary.loc["BBB", "pre_event_cap_fraction"], 5.0 / 35.0)

    def test_adaptive_model_balances_long_and_short_pre_event_capacity(self) -> None:
        ctx = self._adaptive_context(targets=[40.0, -35.0])
        modifier = AdaptiveAnnouncementParticipation(
            pre_event_flex=0.0,
            capacity_buffer=0.0,
            balance_sides=True,
        )

        summary = modifier.allocation_summary(ctx)

        self.assertAlmostEqual(
            summary.loc["AAA", "pre_event_cap_fraction"],
            summary.loc["BBB", "pre_event_cap_fraction"],
        )
        self.assertGreater(
            summary.loc["BBB", "pre_event_cap_fraction"],
            summary.loc["BBB", "mandatory_pre_fraction"],
        )

    def test_event_after_horizon_does_not_force_other_side_before_event(self) -> None:
        ctx = self._adaptive_context(targets=[40.0, -35.0])
        ctx.event_days.loc[:, "BBB"] = [6.0, 5.0, 4.0, 3.0, 2.0]
        modifier = AdaptiveAnnouncementParticipation(
            pre_event_flex=0.0,
            capacity_buffer=0.0,
            balance_sides=True,
        )

        summary = modifier.allocation_summary(ctx)

        self.assertAlmostEqual(summary.loc["AAA", "pre_event_cap_fraction"], 0.25)
        self.assertAlmostEqual(summary.loc["BBB", "pre_event_cap_fraction"], 1.0)

    def test_adaptive_model_uses_safer_pre_dates_and_full_post_rate(self) -> None:
        ctx = self._adaptive_context(targets=[40.0, -35.0])
        model = ParticipationCapModel(
            modifiers=(
                AdaptiveAnnouncementParticipation(
                    pre_event_flex=0.0,
                    capacity_buffer=0.0,
                    balance_sides=False,
                ),
            )
        )

        rates = model.caps(ctx) / ctx.adv_shares

        self.assertGreater(rates[0, 0], rates[2, 0])
        self.assertLess(rates[2, 0], 0.025)
        self.assertAlmostEqual(rates[3, 0], 0.15)
        self.assertAlmostEqual(rates[4, 0], 0.15)

    def test_adaptive_model_solves_with_free_qp_and_conic_backends(self) -> None:
        ctx = self._adaptive_context(targets=[40.0, -35.0])
        participation_model = ParticipationCapModel(
            modifiers=(AdaptiveAnnouncementParticipation(),)
        )
        installed = set(cp.installed_solvers())

        for solver in ("CLARABEL", "OSQP"):
            if solver not in installed:
                continue
            with self.subTest(solver=solver):
                result = TradePlanner(
                    TradePlannerConfig(
                        participation_model=participation_model,
                        risk_model=StaticCovarianceRiskModel(covariance=np.eye(2) * 0.01),
                        cost_model=CompositeCostModel(
                            terms=(QuadraticParticipationImpact(),)
                        ),
                        constraints=default_constraints(),
                        solver=solver,
                    )
                ).solve(ctx)
                self.assertIn(result.diagnostics["status"], {"optimal", "optimal_inaccurate"})
                self.assertLess(result.diagnostics["max_abs_terminal_residual"], 1e-5)

    @staticmethod
    def _adaptive_context(targets: list[float]) -> PlannerContext:
        dates = pd.bdate_range("2026-07-01", periods=5)
        symbols = ["AAA", "BBB"]
        shape = (len(dates), len(symbols))
        event_days = pd.DataFrame(
            np.tile(np.array([2.0, 1.0, 0.0, np.inf, np.inf])[:, None], (1, len(symbols))),
            index=dates,
            columns=symbols,
        )
        return PlannerContext(
            symbols=symbols,
            dates=dates,
            orders=pd.DataFrame({"target_shares": targets}, index=symbols),
            panel=pd.DataFrame(index=pd.MultiIndex.from_product([dates, symbols])),
            price=np.ones(shape),
            adv_shares=np.full(shape, 100.0),
            is_open=np.ones(shape, dtype=bool),
            base_participation=np.full(shape, 0.15),
            event_days=event_days,
        )


if __name__ == "__main__":
    unittest.main()
