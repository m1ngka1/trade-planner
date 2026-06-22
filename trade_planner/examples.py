"""Synthetic example for local smoke testing."""

from __future__ import annotations

import pandas as pd

from .config import default_earnings_aware_config
from .data import PlannerDataProvider, build_context_from_provider
from .planner import TradePlanner, TradePlannerResult


class ToyProvider(PlannerDataProvider):
    """Tiny in-memory provider used by the smoke-test example."""

    def _static_market(self, symbols):
        return pd.DataFrame(
            {
                "price": [50.0, 80.0, 25.0],
                "adv_shares": [1_000_000, 500_000, 250_000],
                "base_participation": [0.20, 0.15, 0.20],
            },
            index=["AAA", "BBB", "CCC"],
        ).reindex(symbols)

    def load_price(self, symbols, dates):
        return self._static_market(symbols)["price"]

    def load_adv_shares(self, symbols, dates):
        return self._static_market(symbols)["adv_shares"]

    def load_base_participation(self, symbols, dates):
        return self._static_market(symbols)["base_participation"]

    def load_event_dates(self, symbols, start_date, end_date):
        events = {
            "AAA": "2026-07-08",
            "BBB": "2026-07-20",
            "CCC": "2026-07-03",
        }
        return {symbol: events[symbol] for symbol in symbols}

    def load_event_volatility(self, symbols, dates):
        event_vol = pd.Series({"AAA": 0.06, "BBB": 0.08, "CCC": 0.05})
        return event_vol.reindex(symbols)

    def load_factor_exposure(self, symbols, dates):
        return pd.DataFrame(
            {
                "market": [1.10, 0.85, 1.35],
                "size": [-0.20, 0.10, 0.45],
                "value": [0.15, -0.25, 0.05],
            },
            index=["AAA", "BBB", "CCC"],
        ).reindex(symbols)

    def load_factor_covariance(self, factor_names, dates):
        return pd.DataFrame(
            [
                [0.0004, 0.00005, 0.00002],
                [0.00005, 0.0003, 0.00004],
                [0.00002, 0.00004, 0.00025],
            ],
            index=factor_names,
            columns=factor_names,
        )

    def load_specific_variance(self, symbols, dates):
        return pd.Series({"AAA": 0.0005, "BBB": 0.0007, "CCC": 0.0010}).reindex(symbols)


def example() -> TradePlannerResult:
    """Run a tiny synthetic example."""
    orders = pd.DataFrame(
        {
            "target_shares": [90_000, -65_000, 30_000],
        },
        index=["AAA", "BBB", "CCC"],
    )
    ctx = build_context_from_provider(
        orders=orders,
        start_date="2026-07-01",
        end_date="2026-07-10",
        provider=ToyProvider(),
    )
    planner = TradePlanner(default_earnings_aware_config())
    return planner.solve(ctx)


if __name__ == "__main__":
    result = example()
    print(result.diagnostics)
    print(result.schedule.round(4).to_string(index=False))
