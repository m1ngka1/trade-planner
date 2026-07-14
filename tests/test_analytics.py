from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from trade_planner.analytics import cumulative_side_completion
from trade_planner.context import PlannerContext


class CumulativeSideCompletionTests(unittest.TestCase):
    def test_reports_long_and_short_gross_completion_percentages(self) -> None:
        dates = pd.bdate_range("2026-07-01", periods=3)
        symbols = ["LONG", "SHORT"]
        shape = (len(dates), len(symbols))
        ctx = PlannerContext(
            symbols=symbols,
            dates=dates,
            orders=pd.DataFrame({"target_shares": [100.0, -200.0]}, index=symbols),
            panel=pd.DataFrame(index=pd.MultiIndex.from_product([dates, symbols])),
            price=np.tile(np.array([10.0, 5.0]), (len(dates), 1)),
            adv_shares=np.full(shape, 1_000.0),
            is_open=np.ones(shape, dtype=bool),
            base_participation=np.full(shape, 0.15),
            event_days=pd.DataFrame(np.inf, index=dates, columns=symbols),
        )
        trades = np.array(
            [
                [10.0, -10.0],
                [40.0, -90.0],
                [50.0, -100.0],
            ]
        )
        schedule = pd.DataFrame(
            [
                {"date": date, "symbol": symbol, "trade_shares": trades[t, s]}
                for t, date in enumerate(dates)
                for s, symbol in enumerate(symbols)
            ]
        )

        completion = cumulative_side_completion(ctx, schedule)

        self.assertAlmostEqual(completion.iloc[0]["cumulative_long_pct"], 10.0)
        self.assertAlmostEqual(completion.iloc[0]["cumulative_short_pct"], 5.0)
        self.assertAlmostEqual(completion.iloc[0]["long_short_gap_pp"], 5.0)
        self.assertAlmostEqual(completion.iloc[-1]["cumulative_long_pct"], 100.0)
        self.assertAlmostEqual(completion.iloc[-1]["cumulative_short_pct"], 100.0)
        self.assertAlmostEqual(completion.iloc[-1]["cumulative_gross_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
