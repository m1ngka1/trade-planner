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
class AdaptiveAnnouncementParticipation:
    """Infer per-name pre-announcement capacity from liquidity and urgency.

    Users provide targets, regular participation caps, and days to the next
    announcement; they do not provide a pre-event completion percentage for
    every name.  For each symbol, the unavoidable pre-event amount is

    ``max(abs(target) - regular post-event capacity, 0)``.

    A small portfolio-level ``pre_event_flex`` adds scheduling freedom, while
    ``capacity_buffer`` avoids numerically tight capacity-equals-target edges.
    When both long and short horizons cross an announcement,
    ``balance_sides=True`` gives both sides the same aggregate pre-event capacity
    fraction where liquidity makes that possible. Optional capacity is assigned
    to safer names, then each name's capacity is water-filled toward dates
    farther from its announcement.
    The announcement date receives the lowest weight and regular participation
    resumes immediately afterward.

    The calculation is NumPy-only and therefore independent of the CVXPY
    solver used by :class:`~trade_planner.planner.TradePlanner`.
    """

    pre_event_flex: float = 0.05
    capacity_buffer: float = 0.02
    risk_window_days: float = 5.0
    announcement_weight: float = 0.05
    risk_shape: float = 2.0
    balance_sides: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.pre_event_flex <= 1.0:
            raise ValueError("pre_event_flex must be between zero and one")
        if not 0.0 <= self.capacity_buffer <= 1.0:
            raise ValueError("capacity_buffer must be between zero and one")
        if self.risk_window_days <= 0:
            raise ValueError("risk_window_days must be positive")
        if not 0.0 <= self.announcement_weight <= 1.0:
            raise ValueError("announcement_weight must be between zero and one")
        if self.risk_shape < 0:
            raise ValueError("risk_shape must be non-negative")

    def multiplier(self, ctx: PlannerContext) -> Array:
        """Return adaptive multipliers for the context's regular share caps."""
        base_caps = _base_caps(ctx)
        targets = np.abs(ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float))
        event_days = ctx.event_days.to_numpy(float)
        multipliers = np.ones_like(base_caps)

        mandatory = np.zeros_like(targets)
        max_pre_capacity = np.zeros_like(targets)
        safe_name_priority = np.ones_like(targets)
        crosses_announcement = np.zeros_like(targets, dtype=bool)
        pre_masks: list[np.ndarray] = []
        daily_priorities: list[np.ndarray] = []

        for column, target in enumerate(targets):
            pre_mask = _pre_announcement_mask(event_days[:, column])
            pre_masks.append(pre_mask)
            if not np.any(pre_mask):
                daily_priorities.append(np.empty(0, dtype=float))
                continue

            post_mask = ~pre_mask
            crosses_announcement[column] = bool(np.any(post_mask))
            pre_capacity = base_caps[pre_mask, column]
            post_capacity = float(np.sum(base_caps[post_mask, column]))
            mandatory[column] = max(target - post_capacity, 0.0)
            max_pre_capacity[column] = min(target, float(np.sum(pre_capacity)))

            priority = self._daily_priority(event_days[pre_mask, column])
            daily_priorities.append(priority)
            if np.sum(pre_capacity) > 0:
                safe_name_priority[column] = float(np.average(priority, weights=pre_capacity))

        if self.balance_sides:
            pre_budgets = self._balanced_pre_budgets(
                ctx=ctx,
                targets=targets,
                mandatory=mandatory,
                max_pre_capacity=max_pre_capacity,
                safe_name_priority=safe_name_priority,
                crosses_announcement=crosses_announcement,
            )
        else:
            discretionary = np.maximum(targets - mandatory, 0.0)
            pre_budgets = np.minimum(
                max_pre_capacity,
                mandatory + self.pre_event_flex * discretionary,
            )

        for column, pre_budget in enumerate(pre_budgets):
            pre_mask = pre_masks[column]
            if not np.any(pre_mask):
                continue
            pre_capacity = base_caps[pre_mask, column]
            buffered_budget = min(
                float(np.sum(pre_capacity)),
                float(pre_budget) * (1.0 + self.capacity_buffer),
            )
            allocated = _weighted_capped_allocation(
                total=buffered_budget,
                capacity=pre_capacity,
                priority=daily_priorities[column],
            )
            multipliers[pre_mask, column] = np.divide(
                allocated,
                pre_capacity,
                out=np.zeros_like(allocated),
                where=pre_capacity > 0,
            )
        return multipliers

    def allocation_summary(self, ctx: PlannerContext) -> pd.DataFrame:
        """Explain the inferred pre-event allowance for every symbol.

        Fraction columns are decimal fractions of each parent order.  Rate
        columns are decimal fractions of ADV, consistent with planner schedule
        fields such as ``cap_pct_adv``.
        """
        base_caps = _base_caps(ctx)
        caps = base_caps * self.multiplier(ctx)
        signed_targets = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
        targets = np.abs(signed_targets)
        event_days = ctx.event_days.to_numpy(float)
        records: list[dict[str, float | str | bool]] = []

        for column, symbol in enumerate(ctx.symbols):
            pre_mask = _pre_announcement_mask(event_days[:, column])
            has_announcement = bool(np.any(pre_mask))
            post_mask = ~pre_mask if has_announcement else np.zeros(len(ctx.dates), dtype=bool)
            post_capacity = float(np.sum(base_caps[post_mask, column]))
            mandatory_pre = max(targets[column] - post_capacity, 0.0) if has_announcement else 0.0
            pre_cap = float(np.sum(caps[pre_mask, column]))
            horizon_cap = float(np.sum(caps[:, column]))
            target = float(targets[column])

            pre_rates = np.divide(
                caps[pre_mask, column],
                ctx.adv_shares[pre_mask, column],
                out=np.zeros(np.count_nonzero(pre_mask), dtype=float),
                where=ctx.adv_shares[pre_mask, column] > 0,
            )
            post_rates = np.divide(
                caps[post_mask, column],
                ctx.adv_shares[post_mask, column],
                out=np.zeros(np.count_nonzero(post_mask), dtype=float),
                where=ctx.adv_shares[post_mask, column] > 0,
            )
            records.append(
                {
                    "symbol": symbol,
                    "side": (
                        "long"
                        if signed_targets[column] > 0
                        else "short"
                        if signed_targets[column] < 0
                        else "flat"
                    ),
                    "target_abs_shares": target,
                    "mandatory_pre_shares": mandatory_pre,
                    "mandatory_pre_fraction": mandatory_pre / target if target > 0 else 0.0,
                    "pre_event_cap_shares": pre_cap,
                    "pre_event_cap_fraction": pre_cap / target if target > 0 else 0.0,
                    "max_pre_participation_rate": float(np.max(pre_rates)) if pre_rates.size else 0.0,
                    "max_post_participation_rate": float(np.max(post_rates)) if post_rates.size else 0.0,
                    "horizon_capacity_ratio": horizon_cap / target if target > 0 else np.inf,
                    "capacity_feasible": target <= horizon_cap + 1e-8,
                    "announcement_in_horizon": bool(np.any(np.isclose(event_days[:, column], 0.0))),
                }
            )
        return pd.DataFrame.from_records(records).set_index("symbol")

    def _daily_priority(self, days_to_announcement: np.ndarray) -> np.ndarray:
        distance = np.clip(days_to_announcement / self.risk_window_days, 0.0, 1.0)
        return self.announcement_weight + (1.0 - self.announcement_weight) * np.power(
            distance,
            self.risk_shape,
        )

    def _balanced_pre_budgets(
        self,
        ctx: PlannerContext,
        targets: np.ndarray,
        mandatory: np.ndarray,
        max_pre_capacity: np.ndarray,
        safe_name_priority: np.ndarray,
        crosses_announcement: np.ndarray,
    ) -> np.ndarray:
        prices = np.asarray(ctx.price[0], dtype=float)
        signed_targets = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
        budgets = np.minimum(mandatory, max_pre_capacity)
        active_masks = [
            mask
            for mask in (
                (signed_targets > 0) & crosses_announcement,
                (signed_targets < 0) & crosses_announcement,
            )
            if np.any(mask) and float(np.sum(targets[mask] * prices[mask])) > 0
        ]
        if not active_masks:
            return budgets

        side_stats: list[tuple[np.ndarray, float, float, float]] = []
        for mask in active_masks:
            total_notional = float(np.sum(targets[mask] * prices[mask]))
            mandatory_fraction = float(np.sum(mandatory[mask] * prices[mask])) / total_notional
            max_fraction = float(np.sum(max_pre_capacity[mask] * prices[mask])) / total_notional
            side_stats.append((mask, total_notional, mandatory_fraction, max_fraction))

        if len(side_stats) == 2:
            common_floor = max(item[2] for item in side_stats)
            common_ceiling = min(item[3] for item in side_stats)
            desired_common = common_floor + self.pre_event_flex * (1.0 - common_floor)
            desired_common = min(desired_common, common_ceiling)
        else:
            mandatory_fraction = side_stats[0][2]
            desired_common = mandatory_fraction + self.pre_event_flex * (1.0 - mandatory_fraction)

        for mask, total_notional, mandatory_fraction, max_fraction in side_stats:
            desired_fraction = min(max(mandatory_fraction, desired_common), max_fraction)
            indices = np.flatnonzero(mask)
            mandatory_notional = float(np.sum(mandatory[indices] * prices[indices]))
            extra_notional = max(desired_fraction * total_notional - mandatory_notional, 0.0)
            extra_capacity_notional = np.maximum(
                max_pre_capacity[indices] - mandatory[indices],
                0.0,
            ) * prices[indices]
            extra = _weighted_capped_allocation(
                total=extra_notional,
                capacity=extra_capacity_notional,
                priority=safe_name_priority[indices],
            )
            budgets[indices] += np.divide(
                extra,
                prices[indices],
                out=np.zeros_like(extra),
                where=prices[indices] > 0,
            )
        return budgets


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


def _base_caps(ctx: PlannerContext) -> Array:
    return (
        np.asarray(ctx.base_participation, dtype=float)
        * np.asarray(ctx.adv_shares, dtype=float)
        * np.asarray(ctx.is_open, dtype=bool).astype(float)
    )


def _pre_announcement_mask(days_to_announcement: np.ndarray) -> np.ndarray:
    """Identify dates up to and including the first upcoming announcement."""
    days = np.asarray(days_to_announcement, dtype=float)
    finite = np.isfinite(days)
    if not np.any(finite):
        return np.zeros_like(finite)
    event_rows = np.flatnonzero(finite & np.isclose(days, 0.0))
    if event_rows.size:
        return np.arange(len(days)) <= int(event_rows[0])
    return finite


def _weighted_capped_allocation(
    total: float,
    capacity: np.ndarray,
    priority: np.ndarray,
) -> np.ndarray:
    """Water-fill a scalar budget by priority without exceeding capacities."""
    capacity = np.maximum(np.asarray(capacity, dtype=float), 0.0)
    if total <= 0 or not np.any(capacity > 0):
        return np.zeros_like(capacity)

    total = min(float(total), float(np.sum(capacity)))
    weights = capacity * np.maximum(np.asarray(priority, dtype=float), 1e-9)
    high = float(
        np.max(
            np.divide(
                capacity,
                weights,
                out=np.zeros_like(capacity),
                where=weights > 0,
            )
        )
    )
    low = 0.0
    for _ in range(64):
        scale = 0.5 * (low + high)
        if float(np.sum(np.minimum(capacity, scale * weights))) < total:
            low = scale
        else:
            high = scale

    allocated = np.minimum(capacity, high * weights)
    residual = total - float(np.sum(allocated))
    if residual > 1e-8:
        room = capacity - allocated
        room_total = float(np.sum(room))
        if room_total > 0:
            allocated += residual * room / room_total
    return allocated
