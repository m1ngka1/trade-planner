"""Participation cap models and cap modifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from .context import PlannerContext
from .types import Array


class ParticipationModifier(Protocol):
    """Plugin that multiplies the base participation cap by a T x N matrix."""

    def multiplier(self, ctx: PlannerContext) -> Array:
        ...


@dataclass(frozen=True)
class ParticipationCapModel:
    """Build per-date, per-symbol share caps."""

    modifiers: Sequence[ParticipationModifier] = ()

    def caps(self, ctx: PlannerContext) -> Array:
        cap = ctx.base_participation * ctx.adv_shares * ctx.is_open.astype(float)
        for modifier in self.modifiers:
            # A modifier may intentionally lift a conservative base rate after
            # an event, so only enforce the lower bound here.
            cap = cap * np.maximum(modifier.multiplier(ctx), 0.0)
        return np.maximum(cap, 0.0)


@dataclass(frozen=True)
class AnnouncementParticipationCurve:
    """Participation rates before and after a known announcement.

    The announcement date belongs to the pre-event regime.  ``transition`` may
    be ``"step"`` or ``"logistic"``; the latter ramps from the pre rate on the
    announcement date to the post rate over ``transition_days`` calendar days.

    Volatility modulation is optional.  Positive sensitivities apply the factor
    ``(reference_volatility / volatility) ** sensitivity``, so higher volatility
    lowers participation and falling volatility raises it.  This matches the
    desired cautious pre-event and faster post-event behavior.
    """

    pre_rate: float = 0.025
    post_rate: float = 0.15
    transition: str = "step"
    transition_days: float = 2.0
    logistic_steepness: float = 8.0
    pre_volatility_sensitivity: float = 0.0
    post_volatility_sensitivity: float = 0.0
    reference_volatility: float | None = None
    min_rate: float = 0.0
    max_rate: float = 1.0

    def __post_init__(self) -> None:
        if self.transition not in {"step", "logistic"}:
            raise ValueError("transition must be 'step' or 'logistic'")
        if self.transition_days <= 0:
            raise ValueError("transition_days must be positive")
        if self.logistic_steepness <= 0:
            raise ValueError("logistic_steepness must be positive")
        if min(self.pre_rate, self.post_rate, self.min_rate) < 0:
            raise ValueError("participation rates must be non-negative")
        if self.max_rate < self.min_rate:
            raise ValueError("max_rate must be greater than or equal to min_rate")
        if self.reference_volatility is not None and self.reference_volatility <= 0:
            raise ValueError("reference_volatility must be positive")
        if min(self.pre_volatility_sensitivity, self.post_volatility_sensitivity) < 0:
            raise ValueError("volatility sensitivities must be non-negative")

    def rates(
        self,
        dates: Sequence[pd.Timestamp | str] | pd.DatetimeIndex,
        announcement_date: pd.Timestamp | str,
        *,
        volatility: float | Sequence[float] | pd.Series | None = None,
    ) -> pd.Series:
        """Return a date-indexed participation-rate series."""
        index = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()
        if index.empty:
            return pd.Series(dtype=float, index=index, name="participation_rate")

        announcement = pd.Timestamp(announcement_date).normalize()
        days_after = (index - announcement).days.to_numpy(dtype=float)
        post_weight = self._post_weight(days_after)
        base_rates = self.pre_rate + (self.post_rate - self.pre_rate) * post_weight

        vol = _aligned_volatility(volatility, index)
        if vol is not None:
            reference = self.reference_volatility
            if reference is None:
                valid = vol[np.isfinite(vol) & (vol > 0)]
                reference = float(np.median(valid)) if valid.size else 1.0
            sensitivity = np.where(
                days_after <= 0,
                self.pre_volatility_sensitivity,
                self.post_volatility_sensitivity,
            )
            valid = np.isfinite(vol) & (vol > 0)
            factor = np.ones_like(base_rates)
            factor[valid] = np.power(reference / vol[valid], sensitivity[valid])
            base_rates = base_rates * factor

        return pd.Series(
            np.clip(base_rates, self.min_rate, self.max_rate),
            index=index,
            name="participation_rate",
        )

    def _post_weight(self, days_after: np.ndarray) -> np.ndarray:
        if self.transition == "step":
            return (days_after > 0).astype(float)

        x = np.clip(days_after / self.transition_days, 0.0, 1.0)
        raw = 1.0 / (1.0 + np.exp(-self.logistic_steepness * (x - 0.5)))
        lower = 1.0 / (1.0 + np.exp(self.logistic_steepness * 0.5))
        upper = 1.0 / (1.0 + np.exp(-self.logistic_steepness * 0.5))
        weight = (raw - lower) / (upper - lower)
        weight[days_after <= 0] = 0.0
        weight[days_after >= self.transition_days] = 1.0
        return weight


def announcement_participation_rates(
    dates: Sequence[pd.Timestamp | str] | pd.DatetimeIndex,
    announcement_date: pd.Timestamp | str,
    *,
    pre_rate: float = 0.025,
    post_rate: float = 0.15,
    transition: str = "step",
    transition_days: float = 2.0,
    volatility: float | Sequence[float] | pd.Series | None = None,
    reference_volatility: float | None = None,
    pre_volatility_sensitivity: float = 0.0,
    post_volatility_sensitivity: float = 0.0,
) -> pd.Series:
    """Convenience wrapper for :class:`AnnouncementParticipationCurve`."""
    return AnnouncementParticipationCurve(
        pre_rate=pre_rate,
        post_rate=post_rate,
        transition=transition,
        transition_days=transition_days,
        reference_volatility=reference_volatility,
        pre_volatility_sensitivity=pre_volatility_sensitivity,
        post_volatility_sensitivity=post_volatility_sensitivity,
    ).rates(dates, announcement_date, volatility=volatility)


@dataclass(frozen=True)
class AnnouncementParticipationModifier:
    """Apply absolute announcement participation rates inside the planner.

    ``announcement_dates`` may be one date for every symbol or a mapping keyed
    by symbol.  Optional volatility can be supplied as a scalar, a ``T x N``
    array, or a date-by-symbol DataFrame.
    """

    announcement_dates: pd.Timestamp | str | Mapping[str, pd.Timestamp | str]
    curve: AnnouncementParticipationCurve = AnnouncementParticipationCurve()
    volatility: float | Array | pd.DataFrame | None = None

    def multiplier(self, ctx: PlannerContext) -> Array:
        volatility = _volatility_matrix(self.volatility, ctx)
        rates = np.empty_like(ctx.base_participation, dtype=float)
        for column, symbol in enumerate(ctx.symbols):
            announcement = (
                self.announcement_dates.get(symbol)
                if isinstance(self.announcement_dates, Mapping)
                else self.announcement_dates
            )
            if announcement is None:
                rates[:, column] = ctx.base_participation[:, column]
                continue
            symbol_volatility = None if volatility is None else volatility[:, column]
            rates[:, column] = self.curve.rates(
                ctx.dates,
                announcement,
                volatility=symbol_volatility,
            ).to_numpy()

        denominator = np.asarray(ctx.base_participation, dtype=float)
        return np.divide(rates, denominator, out=np.zeros_like(rates), where=denominator > 0)


@dataclass(frozen=True)
class LogisticEarningsParticipation:
    """
    Smoothly reduce participation as earnings approaches.

    h(d) = h_min + (1 - h_min) / (1 + exp(-steepness * (d - midpoint_days)))
    """

    h_min: float = 0.25
    midpoint_days: float = 5.0
    steepness: float = 1.0

    def multiplier(self, ctx: PlannerContext) -> Array:
        d = ctx.event_days.to_numpy(float)
        finite = np.isfinite(d)
        z = np.zeros_like(d, dtype=float)
        z[finite] = 1.0 / (1.0 + np.exp(-self.steepness * (d[finite] - self.midpoint_days)))
        z[~finite] = 1.0
        return self.h_min + (1.0 - self.h_min) * z


@dataclass(frozen=True)
class PiecewiseEarningsParticipation:
    """
    Step-rule participation modifier.

    thresholds are interpreted as (max_days_to_event, multiplier), sorted from
    nearest to farthest. Example: ((5, 0.25), (10, 0.5)) means 25% cap within
    five days, 50% cap within ten days, and 100% otherwise.
    """

    thresholds: Sequence[tuple[float, float]] = ((5.0, 0.25), (10.0, 0.5))

    def multiplier(self, ctx: PlannerContext) -> Array:
        d = ctx.event_days.to_numpy(float)
        out = np.ones_like(d, dtype=float)
        for max_days, value in sorted(self.thresholds, key=lambda item: item[0], reverse=True):
            out[d <= max_days] = value
        out[~np.isfinite(d)] = 1.0
        return out


def _aligned_volatility(
    volatility: float | Sequence[float] | pd.Series | None,
    dates: pd.DatetimeIndex,
) -> np.ndarray | None:
    if volatility is None:
        return None
    if np.isscalar(volatility):
        return np.full(len(dates), float(volatility), dtype=float)
    if isinstance(volatility, pd.Series):
        series = volatility.copy()
        series.index = pd.DatetimeIndex(pd.to_datetime(series.index)).normalize()
        return series.reindex(dates).to_numpy(dtype=float)
    array = np.asarray(volatility, dtype=float)
    if array.ndim != 1 or len(array) != len(dates):
        raise ValueError("volatility must be scalar or have one value per date")
    return array


def _volatility_matrix(volatility: float | Array | pd.DataFrame | None, ctx: PlannerContext) -> Array | None:
    if volatility is None:
        return None
    if np.isscalar(volatility):
        return np.full_like(ctx.base_participation, float(volatility), dtype=float)
    if isinstance(volatility, pd.DataFrame):
        frame = volatility.copy()
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).normalize()
        return frame.reindex(index=ctx.dates, columns=ctx.symbols).to_numpy(dtype=float)
    array = np.asarray(volatility, dtype=float)
    if array.shape != ctx.base_participation.shape:
        raise ValueError(f"volatility must have shape {ctx.base_participation.shape}")
    return array
