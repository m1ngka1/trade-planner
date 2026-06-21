"""Synthetic example for local smoke testing."""

from __future__ import annotations

import pandas as pd

from .config import default_earnings_aware_config
from .context import build_context
from .planner import TradePlanner, TradePlannerResult


def example() -> TradePlannerResult:
    """Run a tiny synthetic example."""
    dates = pd.bdate_range("2026-07-01", periods=8)
    orders = pd.DataFrame(
        {
            "target_shares": [90_000, -65_000, 30_000],
            "price": [50.0, 80.0, 25.0],
            "adv_shares": [1_000_000, 500_000, 250_000],
            "base_participation": [0.20, 0.15, 0.20],
            "daily_vol": [0.025, 0.030, 0.035],
            "event_vol": [0.06, 0.08, 0.05],
            "earnings_date": ["2026-07-08", "2026-07-20", "2026-07-03"],
        },
        index=["AAA", "BBB", "CCC"],
    )
    ctx = build_context(orders=orders, dates=dates)
    planner = TradePlanner(default_earnings_aware_config())
    return planner.solve(ctx)


if __name__ == "__main__":
    result = example()
    print(result.diagnostics)
    print(result.schedule.round(4).to_string(index=False))
