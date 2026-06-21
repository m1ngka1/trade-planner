"""Input normalization and event-date utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .types import Array


def as_datetime_index(dates: Sequence[pd.Timestamp | str]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()


def days_to_next_event(
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
    event_dates: Mapping[str, Sequence[pd.Timestamp | str] | pd.Timestamp | str],
) -> pd.DataFrame:
    """
    Return trading/business days from each planner date to each symbol's next event.

    If the event is on one of the planner dates, the distance is the number of
    planner rows until that date. Otherwise it falls back to business-day count.
    Missing future events are set to +inf, so plugin multipliers naturally become
    neutral far from events.
    """
    dates = dates.normalize()
    date_positions = {date: idx for idx, date in enumerate(dates)}
    result = pd.DataFrame(np.inf, index=dates, columns=list(symbols), dtype=float)

    for symbol in symbols:
        raw = event_dates.get(symbol)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        if isinstance(raw, (str, pd.Timestamp)):
            events = [pd.Timestamp(raw).normalize()]
        else:
            events = [pd.Timestamp(value).normalize() for value in raw]
        events = sorted(event for event in events if not pd.isna(event))

        for date in dates:
            future_events = [event for event in events if event >= date]
            if not future_events:
                continue
            event = future_events[0]
            if event in date_positions:
                days = date_positions[event] - date_positions[date]
            else:
                days = np.busday_count(date.date(), event.date())
            result.loc[date, symbol] = max(float(days), 0.0)
    return result


@dataclass(frozen=True)
class PlannerContext:
    """Normalized inputs shared by all model plugins."""

    symbols: list[str]
    dates: pd.DatetimeIndex
    orders: pd.DataFrame
    panel: pd.DataFrame
    price: Array
    adv_shares: Array
    is_open: Array
    base_participation: Array
    event_days: pd.DataFrame


def build_context(
    orders: pd.DataFrame,
    dates: Sequence[pd.Timestamp | str],
    market: pd.DataFrame | None = None,
    event_dates: Mapping[str, Sequence[pd.Timestamp | str] | pd.Timestamp | str] | None = None,
    default_participation: float = 0.15,
) -> PlannerContext:
    """
    Normalize order-level and date-symbol-level data into arrays.

    orders index:
        symbols

    required order columns:
        target_shares, price, adv_shares

    optional order columns:
        base_participation, daily_vol, event_vol, earnings_date

    market index:
        MultiIndex(date, symbol)

    optional market columns:
        price, adv_shares, is_open, base_participation
    """
    orders = orders.copy()
    orders.index = orders.index.astype(str)

    required = {"target_shares", "price", "adv_shares"}
    missing = required - set(orders.columns)
    if missing:
        raise ValueError(f"orders is missing required columns: {sorted(missing)}")

    symbols = list(orders.index.astype(str))
    dates_idx = as_datetime_index(dates)
    idx = pd.MultiIndex.from_product([dates_idx, symbols], names=["date", "symbol"])

    defaults = pd.DataFrame(index=idx)
    defaults["price"] = np.tile(orders["price"].to_numpy(float), len(dates_idx))
    defaults["adv_shares"] = np.tile(orders["adv_shares"].to_numpy(float), len(dates_idx))
    defaults["is_open"] = True
    if "base_participation" in orders:
        base_part = orders["base_participation"].to_numpy(float)
    else:
        base_part = np.full(len(symbols), default_participation, dtype=float)
    defaults["base_participation"] = np.tile(base_part, len(dates_idx))

    if market is None:
        panel = defaults
    else:
        market = market.copy()
        if not isinstance(market.index, pd.MultiIndex):
            raise ValueError("market must be indexed by MultiIndex(date, symbol)")
        market.index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp(d).normalize(), str(s)) for d, s in market.index],
            names=["date", "symbol"],
        )
        panel = market.reindex(idx).combine_first(defaults)

    t_count, n_names = len(dates_idx), len(symbols)
    price = panel["price"].to_numpy(float).reshape(t_count, n_names)
    adv = panel["adv_shares"].to_numpy(float).reshape(t_count, n_names)
    is_open = panel["is_open"].astype(bool).to_numpy().reshape(t_count, n_names)
    base_participation = panel["base_participation"].to_numpy(float).reshape(t_count, n_names)

    if event_dates is None and "earnings_date" in orders:
        event_dates = {
            symbol: orders.loc[symbol, "earnings_date"]
            for symbol in symbols
            if not pd.isna(orders.loc[symbol, "earnings_date"])
        }
    event_days = days_to_next_event(dates_idx, symbols, event_dates or {})

    return PlannerContext(
        symbols=symbols,
        dates=dates_idx,
        orders=orders,
        panel=panel,
        price=price,
        adv_shares=adv,
        is_open=is_open,
        base_participation=base_participation,
        event_days=event_days,
    )
