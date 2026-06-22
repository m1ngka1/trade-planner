"""Data-provider interface for building planner context from minimal orders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import pandas as pd

from .context import PlannerContext, as_datetime_index, build_context
from .types import Array


@dataclass(frozen=True)
class FactorRiskData:
    """
    Barra-style factor risk inputs for the planner date range.

    factor_exposure:
        DataFrame indexed by (date, symbol) with factor columns, or indexed by
        symbol for static exposures.

    factor_covariance:
        Either a date -> factor covariance mapping, a static factor covariance
        DataFrame/array, or a T x K x K array.

    specific_variance:
        Specific return variance by date-symbol, or static symbol Series.
    """

    factor_exposure: pd.DataFrame
    factor_covariance: pd.DataFrame | Mapping[pd.Timestamp | str, pd.DataFrame | Array] | Array
    specific_variance: pd.DataFrame | pd.Series | Array


class PlannerDataProvider(Protocol):
    """Provider adapter for Bloomberg/Barra/Axioma/internal data loaders."""

    def load_market_data(self, symbols: Sequence[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
        ...

    def load_event_dates(
        self,
        symbols: Sequence[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> Mapping[str, Sequence[pd.Timestamp | str] | pd.Timestamp | str]:
        ...

    def load_factor_risk_data(self, symbols: Sequence[str], dates: pd.DatetimeIndex) -> FactorRiskData:
        ...


def build_context_from_provider(
    orders: pd.DataFrame,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    provider: PlannerDataProvider,
    default_participation: float = 0.15,
) -> PlannerContext:
    """
    Build context when the user supplies only symbols and target shares.

    The provider is responsible for querying market data, event dates, and
    Barra-style factor risk inputs for the requested symbols and date range.
    """
    orders = orders.copy()
    orders.index = orders.index.astype(str)
    if "target_shares" not in orders.columns:
        raise ValueError("orders must contain target_shares")

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    dates = as_datetime_index(pd.bdate_range(start, end))
    symbols = list(orders.index)
    market = provider.load_market_data(symbols, dates)
    event_dates = provider.load_event_dates(symbols, start, end)
    factor_risk_data = provider.load_factor_risk_data(symbols, dates)

    event_vol_loader = getattr(provider, "load_event_volatility", None)
    if callable(event_vol_loader):
        event_vol = event_vol_loader(symbols, dates)
        orders["event_vol"] = pd.Series(event_vol, index=symbols).reindex(symbols).to_numpy(float)

    return build_context(
        orders=orders,
        dates=dates,
        market=market,
        event_dates=event_dates,
        factor_exposure=factor_risk_data.factor_exposure,
        factor_covariance=factor_risk_data.factor_covariance,
        specific_variance=factor_risk_data.specific_variance,
        default_participation=default_participation,
    )
