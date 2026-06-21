"""Participation cap models and cap modifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

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
            cap = cap * np.clip(modifier.multiplier(ctx), 0.0, 1.0)
        return np.maximum(cap, 0.0)


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
