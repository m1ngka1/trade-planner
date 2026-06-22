"""Normalized planner context and date utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    factor_names: list[str] | None = None
    factor_exposure: Array | None = None
    factor_covariance: Array | None = None
    specific_variance: Array | None = None
    metadata: dict[str, object] = field(default_factory=dict)
