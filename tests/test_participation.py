from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from trade_planner.context import PlannerContext
from trade_planner.participation import (
    AnnouncementParticipationCurve,
    AnnouncementParticipationModifier,
    ParticipationCapModel,
    announcement_participation_rates,
)


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


if __name__ == "__main__":
    unittest.main()
