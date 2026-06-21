"""Risk models and residual-risk overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

from .context import PlannerContext
from .types import Array
from .utils import as_psd


class RiskOverlay(Protocol):
    """Plugin that adds a PSD covariance adjustment for one date."""

    def covariance_addition(self, ctx: PlannerContext, date_index: int) -> Array:
        ...


@dataclass(frozen=True)
class StaticCovarianceRiskModel:
    """Return a base covariance plus optional date-dependent overlays."""

    covariance: pd.DataFrame | Array | None = None
    overlays: Sequence[RiskOverlay] = ()

    def base_covariance(self, ctx: PlannerContext) -> Array:
        n_names = len(ctx.symbols)
        if self.covariance is None:
            if "daily_vol" in ctx.orders:
                vol = ctx.orders["daily_vol"].reindex(ctx.symbols).to_numpy(float)
            else:
                vol = np.full(n_names, 0.02, dtype=float)
            return np.diag(vol**2)
        if isinstance(self.covariance, pd.DataFrame):
            matrix = self.covariance.reindex(index=ctx.symbols, columns=ctx.symbols).to_numpy(float)
        else:
            matrix = np.asarray(self.covariance, dtype=float)
        if matrix.shape != (n_names, n_names):
            raise ValueError(f"covariance shape {matrix.shape} does not match ({n_names}, {n_names})")
        return as_psd(matrix)

    def covariance_for_date(self, ctx: PlannerContext, date_index: int) -> Array:
        matrix = self.base_covariance(ctx).copy()
        for overlay in self.overlays:
            matrix = matrix + overlay.covariance_addition(ctx, date_index)
        return as_psd(matrix)


@dataclass(frozen=True)
class ExponentialEarningsRiskOverlay:
    """
    Add single-name event variance that grows as earnings approaches.

    addition_i(d) = event_vol_i^2 * exp(-d / tau_days)
    """

    event_vol_column: str = "event_vol"
    tau_days: float = 5.0

    def covariance_addition(self, ctx: PlannerContext, date_index: int) -> Array:
        n_names = len(ctx.symbols)
        if self.event_vol_column in ctx.orders:
            event_vol = ctx.orders[self.event_vol_column].reindex(ctx.symbols).fillna(0.0).to_numpy(float)
        else:
            event_vol = np.zeros(n_names, dtype=float)

        d = ctx.event_days.iloc[date_index].to_numpy(float)
        weight = np.zeros(n_names, dtype=float)
        finite = np.isfinite(d)
        weight[finite] = np.exp(-d[finite] / self.tau_days)
        return np.diag((event_vol**2) * weight)
