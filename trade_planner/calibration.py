"""Investment-driven calibration for rebalance execution schedules.

The optimizer objective is kept in economic units:

* alpha and transaction-cost terms are expected dollars;
* accumulated-inventory risk is daily P&L variance in dollars squared; and
* the inventory-risk coefficient is selected from a solved risk/profit frontier.

Users therefore choose a risk-aversion label instead of hand-tuning unrelated
numerical coefficients. Market-impact and spread estimates come from TCA-style
context fields when available and use a volatility fallback otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import erf, sqrt
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .alpha import ExpectedReturnAlphaModel
from .config import TradePlannerConfig
from .constraints import default_constraints
from .context import PlannerContext
from .costs import (
    CompositeCostModel,
    LinearBpsCost,
    QuadraticParticipationImpact,
    TCALinearBpsCost,
    TCAQuadraticParticipationImpact,
)
from .participation import ParticipationCapModel
from .planner import TradePlanner, TradePlannerResult
from .risk import BarraFactorRiskModel
from .utils import safe_numeric


class RiskAversion(str, Enum):
    """Desk-level risk preference; HIGH means the smallest P&L-risk budget."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def parse(cls, value: RiskAversion | str) -> RiskAversion:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as error:
            allowed = ", ".join(member.value for member in cls)
            raise ValueError(f"risk_aversion must be one of: {allowed}") from error


@dataclass(frozen=True)
class RiskPreference:
    """Fraction of the feasible risk range made available to a desk profile."""

    risk_frontier_fraction: float
    description: str


DEFAULT_RISK_PREFERENCES: Mapping[RiskAversion, RiskPreference] = {
    RiskAversion.HIGH: RiskPreference(
        risk_frontier_fraction=0.15,
        description="Stay close to the minimum feasible accumulated P&L risk.",
    ),
    RiskAversion.MEDIUM: RiskPreference(
        risk_frontier_fraction=0.50,
        description="Use half of the feasible P&L-risk range when alpha pays for it.",
    ),
    RiskAversion.LOW: RiskPreference(
        risk_frontier_fraction=1.00,
        description="Allow the full feasible P&L-risk range and maximize expected net P&L.",
    ),
}


@dataclass(frozen=True)
class RebalanceEconomicMetrics:
    expected_alpha_dollars: float
    impact_cost_dollars: float
    linear_cost_dollars: float
    expected_net_pnl_dollars: float
    pnl_vol_dollars: float
    loss_var_95_dollars: float
    probability_profitable: float

    def as_dict(self) -> dict[str, float]:
        return {
            "expected_alpha_dollars": self.expected_alpha_dollars,
            "impact_cost_dollars": self.impact_cost_dollars,
            "linear_cost_dollars": self.linear_cost_dollars,
            "expected_net_pnl_dollars": self.expected_net_pnl_dollars,
            "pnl_vol_dollars": self.pnl_vol_dollars,
            "loss_var_95_dollars": self.loss_var_95_dollars,
            "probability_profitable": self.probability_profitable,
        }


@dataclass(frozen=True)
class CalibratedRebalancePlan:
    risk_aversion: RiskAversion
    config: TradePlannerConfig
    result: TradePlannerResult
    metrics: RebalanceEconomicMetrics
    frontier: pd.DataFrame
    risk_budget_dollars: float
    impact_bps_at_10pct_adv: float
    linear_cost_bps: float
    economically_viable: bool
    heterogeneous_tca: bool


@dataclass(frozen=True)
class RebalanceFrontier:
    """Solved schedules spanning expected-profit and P&L-risk preferences."""

    ctx: PlannerContext
    frontier: pd.DataFrame
    configs: Mapping[str, TradePlannerConfig]
    results: Mapping[str, TradePlannerResult]
    impact_bps_at_10pct_adv: float
    linear_cost_bps: float
    impact_bps_matrix: np.ndarray
    linear_cost_bps_matrix: np.ndarray
    heterogeneous_tca: bool

    def select(
        self,
        risk_aversion: RiskAversion | str,
        *,
        preferences: Mapping[RiskAversion, RiskPreference] = DEFAULT_RISK_PREFERENCES,
        minimum_edge_bps: float = 1.0,
    ) -> CalibratedRebalancePlan:
        preference_name = RiskAversion.parse(risk_aversion)
        preference = preferences[preference_name]
        if not 0.0 <= preference.risk_frontier_fraction <= 1.0:
            raise ValueError("risk_frontier_fraction must be between zero and one")
        if minimum_edge_bps < 0:
            raise ValueError("minimum_edge_bps must be non-negative")

        valid = self.frontier[self.frontier["status"].isin(("optimal", "optimal_inaccurate"))].copy()
        if valid.empty:
            raise RuntimeError("rebalance calibration frontier contains no solved candidates")
        min_risk = float(valid["pnl_vol_dollars"].min())
        max_risk = float(valid["pnl_vol_dollars"].max())
        budget = min_risk + preference.risk_frontier_fraction * (max_risk - min_risk)
        eligible = valid[valid["pnl_vol_dollars"] <= budget + max(1e-8, budget * 1e-8)].copy()
        if eligible.empty:
            eligible = valid.nsmallest(1, "pnl_vol_dollars")
        parent_target = self.ctx.orders["target_shares"].reindex(self.ctx.symbols).to_numpy(float)
        parent_gross = float(np.sum(np.abs(parent_target * self.ctx.price[0])))
        materiality_dollars = minimum_edge_bps / 10_000.0 * parent_gross
        best_expected_pnl = float(eligible["expected_net_pnl_dollars"].max())
        economically_tied = eligible[
            eligible["expected_net_pnl_dollars"] >= best_expected_pnl - materiality_dollars
        ]
        # Do not spend P&L risk for forecast differences smaller than the desk's
        # materiality threshold. Solver noise and tiny alpha changes should not
        # move a production schedule along the frontier.
        selected_row = economically_tied.sort_values(
            ["pnl_vol_dollars", "expected_net_pnl_dollars", "impact_cost_dollars"],
            ascending=[True, False, True],
        ).iloc[0]
        candidate = str(selected_row["candidate"])
        metrics = RebalanceEconomicMetrics(
            **{
                field: float(selected_row[field])
                for field in RebalanceEconomicMetrics.__dataclass_fields__
            }
        )
        return CalibratedRebalancePlan(
            risk_aversion=preference_name,
            config=self.configs[candidate],
            result=self.results[candidate],
            metrics=metrics,
            frontier=self.frontier.copy(),
            risk_budget_dollars=budget,
            impact_bps_at_10pct_adv=self.impact_bps_at_10pct_adv,
            linear_cost_bps=self.linear_cost_bps,
            economically_viable=metrics.expected_net_pnl_dollars > 0.0,
            heterogeneous_tca=self.heterogeneous_tca,
        )


def calibrate_rebalance_plan(
    ctx: PlannerContext,
    risk_aversion: RiskAversion | str = RiskAversion.MEDIUM,
    *,
    solver: str = "OSQP",
    lambda_multipliers: Iterable[float] | None = None,
    heterogeneous_tca: bool = True,
) -> CalibratedRebalancePlan:
    """Solve a data-scaled frontier and select the best plan inside a risk budget."""

    return build_rebalance_frontier(
        ctx,
        solver=solver,
        lambda_multipliers=lambda_multipliers,
        heterogeneous_tca=heterogeneous_tca,
    ).select(risk_aversion)


def build_rebalance_frontier(
    ctx: PlannerContext,
    *,
    solver: str = "OSQP",
    lambda_multipliers: Iterable[float] | None = None,
    heterogeneous_tca: bool = True,
) -> RebalanceFrontier:
    """Solve the expected-net-P&L versus accumulated-P&L-risk frontier."""

    _validate_economic_context(ctx)
    impact_matrix, linear_matrix = infer_execution_cost_matrices(ctx)
    impact_bps, linear_bps = infer_execution_costs(ctx)
    risk_model = BarraFactorRiskModel()
    base_lambda = _economic_lambda_scale(ctx, risk_model, impact_bps, linear_bps)
    if lambda_multipliers is None:
        lambda_multipliers = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1_000.0)
    multipliers = sorted({float(value) for value in lambda_multipliers})
    if not multipliers or multipliers[0] < 0 or not np.all(np.isfinite(multipliers)):
        raise ValueError("lambda_multipliers must contain finite non-negative values")

    rows: list[dict[str, object]] = []
    configs: dict[str, TradePlannerConfig] = {}
    results: dict[str, TradePlannerResult] = {}
    for multiplier in multipliers:
        risk_weight = base_lambda * multiplier
        candidate = f"multiplier_{multiplier:.6g}__lambda_{risk_weight:.6g}"
        if heterogeneous_tca:
            cost_terms = (
                TCAQuadraticParticipationImpact(impact_matrix),
                TCALinearBpsCost(linear_matrix),
            )
            evaluation_impact: float | np.ndarray = impact_matrix
            evaluation_linear: float | np.ndarray = linear_matrix
        else:
            cost_terms = (
                QuadraticParticipationImpact(impact_bps_at_10pct_adv=impact_bps),
                LinearBpsCost(bps=linear_bps),
            )
            evaluation_impact = impact_bps
            evaluation_linear = linear_bps
        config = TradePlannerConfig(
            participation_model=ParticipationCapModel(),
            risk_model=risk_model,
            cost_model=CompositeCostModel(terms=cost_terms),
            constraints=default_constraints(),
            residual_risk_weight=0.0,
            inventory_risk_weight=risk_weight,
            inventory_alpha_model=ExpectedReturnAlphaModel(),
            terminal_penalty=None,
            solver=solver,
        )
        configs[candidate] = config
        try:
            result = TradePlanner(config).solve(ctx)
            metrics = evaluate_rebalance_schedule(
                ctx,
                result.schedule,
                risk_model=risk_model,
                impact_bps_at_10pct_adv=evaluation_impact,
                linear_cost_bps=evaluation_linear,
            )
        except Exception as error:
            rows.append(
                {
                    "candidate": candidate,
                    "lambda_multiplier": multiplier,
                    "inventory_risk_weight": risk_weight,
                    "status": type(error).__name__,
                    "failure_reason": str(error),
                }
            )
            continue
        results[candidate] = result
        rows.append(
            {
                "candidate": candidate,
                "lambda_multiplier": multiplier,
                "inventory_risk_weight": risk_weight,
                "status": str(result.diagnostics["status"]),
                **metrics.as_dict(),
            }
        )

    return RebalanceFrontier(
        ctx=ctx,
        frontier=pd.DataFrame(rows),
        configs=configs,
        results=results,
        impact_bps_at_10pct_adv=impact_bps,
        linear_cost_bps=linear_bps,
        impact_bps_matrix=impact_matrix,
        linear_cost_bps_matrix=linear_matrix,
        heterogeneous_tca=heterogeneous_tca,
    )


def infer_execution_costs(ctx: PlannerContext) -> tuple[float, float]:
    """Infer scalar impact and linear-cost inputs from TCA data or risk data."""

    target = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    weights = np.abs(target * ctx.price[0])
    impact_matrix, linear_matrix = infer_execution_cost_matrices(ctx)
    repeated_weights = np.tile(weights[None, :], (len(ctx.dates), 1)).reshape(-1)
    impact = _weighted_median(impact_matrix.reshape(-1), repeated_weights)
    linear = _weighted_median(linear_matrix.reshape(-1), repeated_weights)
    if impact is None or linear is None:
        raise ValueError("execution-cost forecasts contain no usable target-weighted values")
    return impact, linear


def infer_execution_cost_matrices(ctx: PlannerContext) -> tuple[np.ndarray, np.ndarray]:
    """Return finite date-by-name TCA forecasts, using economic fallbacks."""

    impact = _explicit_cost_matrix(
        ctx,
        context_field="impact_bps_at_10pct_adv",
        order_column="impact_bps_at_10pct_adv",
        metadata_key="impact_bps_at_10pct_adv",
    )
    if impact is None:
        risk_model = BarraFactorRiskModel()
        rows = []
        for date_index in range(len(ctx.dates)):
            daily_vol = np.sqrt(
                np.maximum(
                    np.diag(_security_covariance(ctx, date_index, risk_model)),
                    0.0,
                )
            )
            rows.append(10_000.0 * 0.10 * daily_vol * sqrt(0.10))
        impact = np.asarray(rows, dtype=float)
    linear = _explicit_cost_matrix(
        ctx,
        context_field="linear_cost_bps",
        order_column="linear_cost_bps",
        metadata_key="linear_cost_bps",
    )
    if linear is None:
        linear = np.ones((len(ctx.dates), len(ctx.symbols)), dtype=float)
    return (
        _validated_cost_matrix(impact, ctx, "impact_bps_at_10pct_adv", 0.1, 100.0),
        _validated_cost_matrix(linear, ctx, "linear_cost_bps", 0.0, 100.0),
    )


def evaluate_rebalance_schedule(
    ctx: PlannerContext,
    schedule: pd.DataFrame,
    *,
    risk_model: BarraFactorRiskModel | None = None,
    impact_bps_at_10pct_adv: float | np.ndarray,
    linear_cost_bps: float | np.ndarray,
) -> RebalanceEconomicMetrics:
    """Evaluate expected net P&L and holding-P&L volatility for one schedule."""

    risk_model = risk_model or BarraFactorRiskModel()
    impact_matrix = _as_cost_matrix(
        impact_bps_at_10pct_adv,
        ctx,
        "impact_bps_at_10pct_adv",
    )
    linear_matrix = _as_cost_matrix(linear_cost_bps, ctx, "linear_cost_bps")
    trades = (
        schedule.pivot_table(
            index="date",
            columns="symbol",
            values="trade_shares",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=ctx.dates, columns=ctx.symbols, fill_value=0.0)
        .to_numpy(float)
    )
    cumulative = np.cumsum(trades, axis=0)
    expected_alpha = 0.0
    risk_variance = 0.0
    impact_cost = 0.0
    linear_cost = 0.0
    for date_index in range(len(ctx.dates)):
        price = np.asarray(ctx.price[date_index], dtype=float)
        position_dollars = price * cumulative[date_index]
        if ctx.expected_return is not None:
            expected_alpha += float(
                np.dot(position_dollars, np.asarray(ctx.expected_return[date_index], dtype=float))
            )
        covariance = _security_covariance(ctx, date_index, risk_model)
        risk_variance += float(position_dollars @ covariance @ position_dollars)

        adv = safe_numeric(ctx.adv_shares[date_index])
        eta = (
            (impact_matrix[date_index] / 10_000.0)
            * price
            / safe_numeric(0.10 * adv)
        )
        impact_cost += float(np.sum(eta * np.square(trades[date_index])))
        linear_cost += float(
            np.sum(
                (linear_matrix[date_index] / 10_000.0)
                * price
                * np.abs(trades[date_index])
            )
        )

    net = expected_alpha - impact_cost - linear_cost
    pnl_vol = sqrt(max(risk_variance, 0.0))
    probability_profitable = (
        0.5 * (1.0 + erf(net / (pnl_vol * sqrt(2.0))))
        if pnl_vol > 0
        else float(net > 0)
    )
    return RebalanceEconomicMetrics(
        expected_alpha_dollars=expected_alpha,
        impact_cost_dollars=impact_cost,
        linear_cost_dollars=linear_cost,
        expected_net_pnl_dollars=net,
        pnl_vol_dollars=pnl_vol,
        loss_var_95_dollars=1.645 * pnl_vol - net,
        probability_profitable=probability_profitable,
    )


def _validate_economic_context(ctx: PlannerContext) -> None:
    if ctx.expected_return is None:
        raise ValueError(
            "automatic rebalance calibration requires probability-weighted expected_return data"
        )
    if np.asarray(ctx.expected_return).shape != (len(ctx.dates), len(ctx.symbols)):
        raise ValueError(
            "expected_return shape must match planner dates and symbols: "
            f"{(len(ctx.dates), len(ctx.symbols))}"
        )
    if not np.all(np.isfinite(ctx.expected_return)):
        raise ValueError("expected_return must contain finite values")
    if ctx.factor_exposure is None or ctx.factor_covariance is None or ctx.specific_variance is None:
        raise ValueError("automatic rebalance calibration requires complete factor risk data")


def _economic_lambda_scale(
    ctx: PlannerContext,
    risk_model: BarraFactorRiskModel,
    impact_bps: float,
    linear_bps: float,
) -> float:
    target = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    full_alpha = 0.0
    full_risk_variance = 0.0
    for date_index in range(len(ctx.dates)):
        target_dollars = np.asarray(ctx.price[date_index], dtype=float) * target
        full_alpha += float(np.dot(target_dollars, ctx.expected_return[date_index]))
        covariance = _security_covariance(ctx, date_index, risk_model)
        full_risk_variance += float(target_dollars @ covariance @ target_dollars)
    gross = float(np.sum(np.abs(ctx.price[0] * target)))
    cost_scale = gross * (impact_bps + linear_bps) / 10_000.0
    economic_dollars = max(abs(full_alpha), cost_scale, gross / 10_000.0, 1e-6)
    return economic_dollars / max(full_risk_variance, 1e-12)


def _security_covariance(
    ctx: PlannerContext,
    date_index: int,
    risk_model: BarraFactorRiskModel,
) -> np.ndarray:
    if ctx.factor_exposure is None or ctx.factor_covariance is None or ctx.specific_variance is None:
        raise ValueError("Barra schedule economics requires complete factor risk data")
    factor_idx = risk_model.factor_indices(ctx)
    exposure = np.asarray(ctx.factor_exposure[date_index][:, factor_idx], dtype=float)
    factor_covariance = np.asarray(
        ctx.factor_covariance[date_index][np.ix_(factor_idx, factor_idx)],
        dtype=float,
    )
    specific = np.asarray(ctx.specific_variance[date_index], dtype=float).copy()
    for overlay in risk_model.specific_overlays:
        specific += overlay.specific_variance_addition(ctx, date_index)
    return exposure @ factor_covariance @ exposure.T + np.diag(np.maximum(specific, 0.0))


def _explicit_cost_matrix(
    ctx: PlannerContext,
    *,
    context_field: str,
    order_column: str,
    metadata_key: str,
) -> np.ndarray | None:
    raw = getattr(ctx, context_field)
    if raw is None and order_column in ctx.orders:
        raw = ctx.orders[order_column].reindex(ctx.symbols).to_numpy(float)
    if raw is None:
        raw = ctx.metadata.get(metadata_key)
    if raw is None:
        return None
    values = np.asarray(raw, dtype=float)
    shape = (len(ctx.dates), len(ctx.symbols))
    if values.ndim == 0:
        return np.full(shape, float(values), dtype=float)
    if values.shape == (len(ctx.symbols),):
        return np.tile(values[None, :], (len(ctx.dates), 1))
    if values.shape == shape:
        return values.copy()
    raise ValueError(
        f"{context_field} must be scalar, one value per symbol, or date by symbol"
    )


def _validated_cost_matrix(
    values: np.ndarray,
    ctx: PlannerContext,
    name: str,
    lower: float,
    upper: float,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    expected_shape = (len(ctx.dates), len(ctx.symbols))
    if matrix.shape != expected_shape:
        raise ValueError(f"{name} shape {matrix.shape} does not match {expected_shape}")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return np.clip(matrix, lower, upper)


def _as_cost_matrix(
    values: float | np.ndarray,
    ctx: PlannerContext,
    name: str,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    expected_shape = (len(ctx.dates), len(ctx.symbols))
    if matrix.ndim == 0:
        matrix = np.full(expected_shape, float(matrix), dtype=float)
    if matrix.shape != expected_shape:
        raise ValueError(f"{name} shape {matrix.shape} does not match {expected_shape}")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return matrix


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float | None:
    valid = np.isfinite(values) & (values >= 0) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return None
    ordered = np.argsort(values[valid])
    selected_values = values[valid][ordered]
    selected_weights = weights[valid][ordered]
    cutoff = 0.5 * float(np.sum(selected_weights))
    index = int(np.searchsorted(np.cumsum(selected_weights), cutoff, side="left"))
    return float(selected_values[min(index, len(selected_values) - 1)])
