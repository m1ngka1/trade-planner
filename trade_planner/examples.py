"""Synthetic example for local smoke testing."""

from __future__ import annotations

import pandas as pd

from .config import default_earnings_aware_config
from .data import FactorRiskData, build_context_from_provider
from .planner import TradePlanner, TradePlannerResult


class ToyProvider:
    """Tiny in-memory provider used by the smoke-test example."""

    def load_market_data(self, symbols, dates):
        base = pd.DataFrame(
            {
                "price": [50.0, 80.0, 25.0],
                "adv_shares": [1_000_000, 500_000, 250_000],
                "base_participation": [0.20, 0.15, 0.20],
            },
            index=["AAA", "BBB", "CCC"],
        ).reindex(symbols)
        idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        market = pd.DataFrame(index=idx)
        market["price"] = [base.loc[symbol, "price"] for _, symbol in idx]
        market["adv_shares"] = [base.loc[symbol, "adv_shares"] for _, symbol in idx]
        market["base_participation"] = [base.loc[symbol, "base_participation"] for _, symbol in idx]
        market["is_open"] = True
        return market

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

    def load_factor_risk_data(self, symbols, dates):
        static_exposure = pd.DataFrame(
            {
                "market": [1.10, 0.85, 1.35],
                "size": [-0.20, 0.10, 0.45],
                "value": [0.15, -0.25, 0.05],
            },
            index=["AAA", "BBB", "CCC"],
        ).reindex(symbols)

        idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        factor_exposure = pd.DataFrame(index=idx, columns=static_exposure.columns, dtype=float)
        for symbol in symbols:
            factor_exposure.loc[(slice(None), symbol), :] = static_exposure.loc[symbol].to_numpy(float)

        factor_covariance = pd.DataFrame(
            [
                [0.0004, 0.00005, 0.00002],
                [0.00005, 0.0003, 0.00004],
                [0.00002, 0.00004, 0.00025],
            ],
            index=static_exposure.columns,
            columns=static_exposure.columns,
        )
        specific_variance = pd.Series({"AAA": 0.0005, "BBB": 0.0007, "CCC": 0.0010}).reindex(symbols)
        return FactorRiskData(
            factor_exposure=factor_exposure,
            factor_covariance=factor_covariance,
            specific_variance=specific_variance,
        )


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
