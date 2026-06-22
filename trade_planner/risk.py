"""Risk models and residual-risk overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import cvxpy as cp
import numpy as np
import pandas as pd

from .context import PlannerContext
from .types import Array
from .utils import as_psd


class RiskModel(Protocol):
    """Plugin that returns the residual-risk expression for one date."""

    def objective(
        self,
        residual_shares: cp.Expression,
        ctx: PlannerContext,
        date_index: int,
    ) -> cp.Expression:
        ...


class RiskOverlay(Protocol):
    """Plugin that adds a PSD covariance adjustment for one date."""

    def covariance_addition(self, ctx: PlannerContext, date_index: int) -> Array:
        ...


@dataclass(frozen=True)
class StaticCovarianceRiskModel:
    """Full security covariance model, retained as a simple fallback."""

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

    def objective(
        self,
        residual_shares: cp.Expression,
        ctx: PlannerContext,
        date_index: int,
    ) -> cp.Expression:
        residual_dollars = cp.multiply(ctx.price[date_index], residual_shares)
        return cp.quad_form(residual_dollars, self.covariance_for_date(ctx, date_index))


class SpecificRiskOverlay(Protocol):
    """Plugin that adds date-dependent specific return variance."""

    def specific_variance_addition(self, ctx: PlannerContext, date_index: int) -> Array:
        ...


@dataclass(frozen=True)
class BarraFactorRiskModel:
    """
    Barra-style factor risk model.

    For residual shares r:
        w = price * r
        f = B.T @ w
        risk = f.T @ Sigma_f @ f + sum_i specific_variance_i * w_i^2

    B is the security-by-factor exposure matrix. The covariance inputs are
    return covariance/variance, so dollar residuals convert share residuals
    into portfolio risk dollars.
    """

    specific_overlays: Sequence[SpecificRiskOverlay] = ()

    def objective(
        self,
        residual_shares: cp.Expression,
        ctx: PlannerContext,
        date_index: int,
    ) -> cp.Expression:
        if ctx.factor_exposure is None or ctx.factor_covariance is None or ctx.specific_variance is None:
            raise ValueError(
                "BarraFactorRiskModel requires factor_exposure, factor_covariance, "
                "and specific_variance in PlannerContext"
            )

        residual_dollars = cp.multiply(ctx.price[date_index], residual_shares)
        exposure = ctx.factor_exposure[date_index]
        factor_covariance = as_psd(ctx.factor_covariance[date_index])
        factor_dollars = exposure.T @ residual_dollars

        specific_variance = ctx.specific_variance[date_index].copy()
        for overlay in self.specific_overlays:
            specific_variance = specific_variance + overlay.specific_variance_addition(ctx, date_index)
        specific_variance = np.maximum(specific_variance, 0.0)

        factor_risk = cp.quad_form(factor_dollars, factor_covariance)
        specific_risk = cp.sum(cp.multiply(specific_variance, cp.square(residual_dollars)))
        return factor_risk + specific_risk


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

    def specific_variance_addition(self, ctx: PlannerContext, date_index: int) -> Array:
        return np.diag(self.covariance_addition(ctx, date_index)).copy()
