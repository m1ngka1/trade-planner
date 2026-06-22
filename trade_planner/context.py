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
    factor_names: list[str] | None = None
    factor_exposure: Array | None = None
    factor_covariance: Array | None = None
    specific_variance: Array | None = None


def build_context(
    orders: pd.DataFrame,
    dates: Sequence[pd.Timestamp | str],
    market: pd.DataFrame | None = None,
    event_dates: Mapping[str, Sequence[pd.Timestamp | str] | pd.Timestamp | str] | None = None,
    event_days: pd.DataFrame | None = None,
    factor_exposure: pd.DataFrame | None = None,
    factor_covariance: pd.DataFrame | Mapping[pd.Timestamp | str, pd.DataFrame | Array] | Array | None = None,
    specific_variance: pd.DataFrame | pd.Series | Array | None = None,
    default_participation: float = 0.15,
) -> PlannerContext:
    """
    Normalize order-level and date-symbol-level data into arrays.

    orders index:
        symbols

    required order columns:
        target_shares

    optional order columns:
        base_participation, daily_vol, event_vol, earnings_date

    market index:
        MultiIndex(date, symbol)

    optional market columns:
        price, adv_shares, is_open, base_participation
    """
    orders = orders.copy()
    orders.index = orders.index.astype(str)

    required = {"target_shares"}
    missing = required - set(orders.columns)
    if missing:
        raise ValueError(f"orders is missing required columns: {sorted(missing)}")

    symbols = list(orders.index.astype(str))
    dates_idx = as_datetime_index(dates)
    idx = pd.MultiIndex.from_product([dates_idx, symbols], names=["date", "symbol"])

    defaults = pd.DataFrame(index=idx)
    if "price" in orders:
        defaults["price"] = np.tile(orders["price"].to_numpy(float), len(dates_idx))
    else:
        defaults["price"] = np.nan
    if "adv_shares" in orders:
        defaults["adv_shares"] = np.tile(orders["adv_shares"].to_numpy(float), len(dates_idx))
    else:
        defaults["adv_shares"] = np.nan
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

    required_panel_fields = ["price", "adv_shares"]
    for field in required_panel_fields:
        if field not in panel or panel[field].isna().any():
            raise ValueError(
                f"{field} must be supplied either in orders or market data for every date-symbol"
            )

    t_count, n_names = len(dates_idx), len(symbols)
    price = panel["price"].to_numpy(float).reshape(t_count, n_names)
    adv = panel["adv_shares"].to_numpy(float).reshape(t_count, n_names)
    is_open = panel["is_open"].astype(bool).to_numpy().reshape(t_count, n_names)
    base_participation = panel["base_participation"].to_numpy(float).reshape(t_count, n_names)

    if event_days is None:
        if event_dates is None and "earnings_date" in orders:
            event_dates = {
                symbol: orders.loc[symbol, "earnings_date"]
                for symbol in symbols
                if not pd.isna(orders.loc[symbol, "earnings_date"])
            }
        event_days = days_to_next_event(dates_idx, symbols, event_dates or {})
    else:
        event_days = _align_date_symbol_frame(event_days, dates_idx, symbols, "event_days")

    factor_names, factor_exposure_array = _align_factor_exposure(factor_exposure, dates_idx, symbols)
    factor_covariance_array = _align_factor_covariance(
        factor_covariance,
        dates_idx,
        factor_names,
    )
    specific_variance_array = _align_specific_variance(specific_variance, dates_idx, symbols)

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
        factor_names=factor_names,
        factor_exposure=factor_exposure_array,
        factor_covariance=factor_covariance_array,
        specific_variance=specific_variance_array,
    )


def _align_date_symbol_frame(
    frame: pd.DataFrame,
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
    name: str,
) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).normalize()
    frame.columns = frame.columns.astype(str)
    aligned = frame.reindex(index=dates, columns=list(symbols))
    if aligned.isna().any().any():
        raise ValueError(f"{name} is missing values for at least one date-symbol")
    return aligned


def _align_factor_exposure(
    exposure: pd.DataFrame | None,
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
) -> tuple[list[str] | None, Array | None]:
    if exposure is None:
        return None, None

    exposure = exposure.copy()
    exposure.columns = exposure.columns.astype(str)
    factor_names = list(exposure.columns)

    if isinstance(exposure.index, pd.MultiIndex):
        exposure.index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp(d).normalize(), str(s)) for d, s in exposure.index],
            names=["date", "symbol"],
        )
        idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        aligned = exposure.reindex(idx)
        if aligned.isna().any().any():
            raise ValueError("factor_exposure is missing values for at least one date-symbol")
        array = aligned.to_numpy(float).reshape(len(dates), len(symbols), len(factor_names))
        return factor_names, array

    exposure.index = exposure.index.astype(str)
    aligned = exposure.reindex(list(symbols))
    if aligned.isna().any().any():
        raise ValueError("factor_exposure is missing values for at least one symbol")
    static = aligned.to_numpy(float)
    return factor_names, np.tile(static[None, :, :], (len(dates), 1, 1))


def _align_factor_covariance(
    covariance: pd.DataFrame | Mapping[pd.Timestamp | str, pd.DataFrame | Array] | Array | None,
    dates: pd.DatetimeIndex,
    factor_names: Sequence[str] | None,
) -> Array | None:
    if covariance is None:
        return None
    if factor_names is None:
        raise ValueError("factor_covariance was supplied without factor_exposure")

    k = len(factor_names)
    if isinstance(covariance, Mapping):
        values = []
        normalized = {pd.Timestamp(key).normalize(): value for key, value in covariance.items()}
        for date in dates:
            if date not in normalized:
                raise ValueError(f"factor_covariance missing date {date.date()}")
            values.append(_coerce_factor_covariance_matrix(normalized[date], factor_names))
        return np.stack(values)

    if isinstance(covariance, pd.DataFrame):
        if isinstance(covariance.index, pd.MultiIndex):
            covariance = covariance.copy()
            covariance.index = pd.MultiIndex.from_tuples(
                [(pd.Timestamp(d).normalize(), str(f)) for d, f in covariance.index],
                names=["date", "factor"],
            )
            values = []
            for date in dates:
                if date not in covariance.index.get_level_values("date"):
                    raise ValueError(f"factor_covariance missing date {date.date()}")
                matrix = covariance.loc[date].reindex(index=factor_names, columns=factor_names)
                values.append(matrix.to_numpy(float))
            return np.stack(values)
        return np.tile(
            _coerce_factor_covariance_matrix(covariance, factor_names)[None, :, :],
            (len(dates), 1, 1),
        )

    array = np.asarray(covariance, dtype=float)
    if array.shape == (k, k):
        return np.tile(array[None, :, :], (len(dates), 1, 1))
    if array.shape == (len(dates), k, k):
        return array
    raise ValueError(
        f"factor_covariance shape {array.shape} does not match {(k, k)} or {(len(dates), k, k)}"
    )


def _coerce_factor_covariance_matrix(value: pd.DataFrame | Array, factor_names: Sequence[str]) -> Array:
    if isinstance(value, pd.DataFrame):
        matrix = value.reindex(index=factor_names, columns=factor_names).to_numpy(float)
    else:
        matrix = np.asarray(value, dtype=float)
    expected = (len(factor_names), len(factor_names))
    if matrix.shape != expected:
        raise ValueError(f"factor covariance matrix shape {matrix.shape} does not match {expected}")
    return matrix


def _align_specific_variance(
    specific_variance: pd.DataFrame | pd.Series | Array | None,
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
) -> Array | None:
    if specific_variance is None:
        return None

    if isinstance(specific_variance, pd.Series):
        series = specific_variance.copy()
        series.index = series.index.astype(str)
        aligned = series.reindex(list(symbols))
        if aligned.isna().any():
            raise ValueError("specific_variance is missing values for at least one symbol")
        return np.tile(aligned.to_numpy(float)[None, :], (len(dates), 1))

    if isinstance(specific_variance, pd.DataFrame):
        frame = specific_variance.copy()
        if isinstance(frame.index, pd.MultiIndex):
            frame.index = pd.MultiIndex.from_tuples(
                [(pd.Timestamp(d).normalize(), str(s)) for d, s in frame.index],
                names=["date", "symbol"],
            )
            column = "specific_variance"
            if column not in frame.columns:
                raise ValueError("MultiIndex specific_variance frame must contain a specific_variance column")
            idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
            aligned = frame.reindex(idx)[column]
            if aligned.isna().any():
                raise ValueError("specific_variance is missing values for at least one date-symbol")
            return aligned.to_numpy(float).reshape(len(dates), len(symbols))

        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).normalize()
        frame.columns = frame.columns.astype(str)
        aligned = frame.reindex(index=dates, columns=list(symbols))
        if aligned.isna().any().any():
            raise ValueError("specific_variance is missing values for at least one date-symbol")
        return aligned.to_numpy(float)

    array = np.asarray(specific_variance, dtype=float)
    if array.shape == (len(symbols),):
        return np.tile(array[None, :], (len(dates), 1))
    if array.shape == (len(dates), len(symbols)):
        return array
    raise ValueError(
        f"specific_variance shape {array.shape} does not match {(len(symbols),)} or {(len(dates), len(symbols))}"
    )
