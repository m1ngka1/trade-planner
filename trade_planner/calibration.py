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
from statistics import NormalDist
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
from .downside import (
    ScenarioCVaRRiskModel,
    TailSecondMomentPathRiskModel,
    centered_return_scenarios,
    reduce_return_scenarios,
    tail_return_scenarios,
    weighted_loss_var_cvar,
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


class RebalanceRiskMeasure(str, Enum):
    """Risk measure used to build and select the execution frontier."""

    AUTO = "auto"
    VARIANCE = "variance"
    DOWNSIDE_CVAR = "downside_cvar"
    HYBRID_DOWNSIDE = "hybrid_downside"
    TAIL_SECOND_MOMENT = "tail_second_moment"

    @classmethod
    def parse(cls, value: RebalanceRiskMeasure | str) -> RebalanceRiskMeasure:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as error:
            allowed = ", ".join(member.value for member in cls)
            raise ValueError(f"risk_measure must be one of: {allowed}") from error


@dataclass(frozen=True)
class RiskPreference:
    """Fraction of the feasible risk range made available to a desk profile."""

    risk_frontier_fraction: float
    description: str


DEFAULT_RISK_PREFERENCES: Mapping[RiskAversion, RiskPreference] = {
    RiskAversion.HIGH: RiskPreference(
        risk_frontier_fraction=0.05,
        description="Stay inside the lowest five percent of feasible P&L risk.",
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

# Scenario objectives add one slack variable per path.  The recorded economic
# benchmark found 96 tail-preserving representatives retained the selected
# medium-risk schedule and independent CVaR while reducing frontier time by
# almost four times.  Passing None explicitly remains available for audits.
DEFAULT_MAX_OPTIMIZATION_SCENARIOS = 96


@dataclass(frozen=True)
class RebalanceEconomicMetrics:
    expected_alpha_dollars: float
    impact_cost_dollars: float
    linear_cost_dollars: float
    expected_net_pnl_dollars: float
    pnl_vol_dollars: float
    loss_var_95_dollars: float
    loss_cvar_95_dollars: float
    probability_profitable: float

    def as_dict(self) -> dict[str, float]:
        return {
            "expected_alpha_dollars": self.expected_alpha_dollars,
            "impact_cost_dollars": self.impact_cost_dollars,
            "linear_cost_dollars": self.linear_cost_dollars,
            "expected_net_pnl_dollars": self.expected_net_pnl_dollars,
            "pnl_vol_dollars": self.pnl_vol_dollars,
            "loss_var_95_dollars": self.loss_var_95_dollars,
            "loss_cvar_95_dollars": self.loss_cvar_95_dollars,
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
    risk_measure: RebalanceRiskMeasure
    scenario_tail_overlay_fraction: float
    optimization_scenario_count: int | None


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
    risk_measure: RebalanceRiskMeasure
    risk_metric_column: str
    scenario_tail_overlay_fraction: float
    optimization_scenario_count: int | None

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
        min_risk = float(valid[self.risk_metric_column].min())
        max_risk = float(valid[self.risk_metric_column].max())
        budget = min_risk + preference.risk_frontier_fraction * (max_risk - min_risk)
        eligible = valid[
            valid[self.risk_metric_column]
            <= budget + max(1e-8, abs(budget) * 1e-8)
        ].copy()
        if eligible.empty:
            eligible = valid.nsmallest(1, self.risk_metric_column)
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
            [
                self.risk_metric_column,
                "pnl_vol_dollars",
                "expected_net_pnl_dollars",
                "impact_cost_dollars",
            ],
            ascending=[True, True, False, True],
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
            risk_measure=self.risk_measure,
            scenario_tail_overlay_fraction=self.scenario_tail_overlay_fraction,
            optimization_scenario_count=self.optimization_scenario_count,
        )


def calibrate_rebalance_plan(
    ctx: PlannerContext,
    risk_aversion: RiskAversion | str = RiskAversion.MEDIUM,
    *,
    solver: str = "OSQP",
    lambda_multipliers: Iterable[float] | None = None,
    heterogeneous_tca: bool = True,
    risk_measure: RebalanceRiskMeasure | str = RebalanceRiskMeasure.AUTO,
    cvar_confidence: float = 0.95,
    max_optimization_scenarios: int | None = DEFAULT_MAX_OPTIMIZATION_SCENARIOS,
) -> CalibratedRebalancePlan:
    """Solve a data-scaled frontier and select the best plan inside a risk budget."""

    parsed_aversion = RiskAversion.parse(risk_aversion)
    parsed_measure = RebalanceRiskMeasure.parse(risk_measure)
    if parsed_measure is RebalanceRiskMeasure.AUTO and ctx.return_residual_scenarios is not None:
        if parsed_aversion is RiskAversion.HIGH:
            # The high-aversion experiment found no material independent-tail
            # improvement. Keep this profile on stable covariance risk.
            parsed_measure = RebalanceRiskMeasure.VARIANCE
        elif parsed_aversion is RiskAversion.LOW:
            # Across five independent optimization samples, the tail second
            # moment preserved low-profile mechanics and improved out-of-sample
            # risk at a fraction of scenario-CVaR runtime.
            parsed_measure = RebalanceRiskMeasure.TAIL_SECOND_MOMENT
        else:
            parsed_measure = RebalanceRiskMeasure.HYBRID_DOWNSIDE

    return build_rebalance_frontier(
        ctx,
        solver=solver,
        lambda_multipliers=lambda_multipliers,
        heterogeneous_tca=heterogeneous_tca,
        risk_measure=parsed_measure,
        cvar_confidence=cvar_confidence,
        max_optimization_scenarios=max_optimization_scenarios,
    ).select(parsed_aversion)


def build_rebalance_frontier(
    ctx: PlannerContext,
    *,
    solver: str = "OSQP",
    lambda_multipliers: Iterable[float] | None = None,
    heterogeneous_tca: bool = True,
    risk_measure: RebalanceRiskMeasure | str = RebalanceRiskMeasure.AUTO,
    cvar_confidence: float = 0.95,
    max_optimization_scenarios: int | None = DEFAULT_MAX_OPTIMIZATION_SCENARIOS,
) -> RebalanceFrontier:
    """Solve the expected-net-P&L versus accumulated-P&L-risk frontier."""

    if not np.isclose(cvar_confidence, 0.95):
        raise ValueError("rebalance calibration metrics currently require cvar_confidence=0.95")
    _validate_economic_context(ctx)
    impact_matrix, linear_matrix = infer_execution_cost_matrices(ctx)
    impact_bps, linear_bps = infer_execution_costs(ctx)
    risk_model = BarraFactorRiskModel()
    resolved_risk_measure = _resolve_risk_measure(ctx, risk_measure)
    scenario_tail_overlay_fraction = 0.0
    optimization_scenario_count: int | None = None
    tail_probability = min(2.0 * (1.0 - cvar_confidence), 0.5)
    base_variance_weight = _economic_lambda_scale(
        ctx,
        risk_model,
        impact_bps,
        linear_bps,
    )
    base_cvar_weight = 0.0
    tail_second_moment_scale = 0.0
    if resolved_risk_measure in {
        RebalanceRiskMeasure.DOWNSIDE_CVAR,
        RebalanceRiskMeasure.HYBRID_DOWNSIDE,
        RebalanceRiskMeasure.TAIL_SECOND_MOMENT,
    }:
        if not 0.0 < cvar_confidence < 1.0:
            raise ValueError("cvar_confidence must be strictly between zero and one")
        scenarios, _ = centered_return_scenarios(ctx)
        optimization_scenario_count = len(scenarios)
        if resolved_risk_measure is RebalanceRiskMeasure.TAIL_SECOND_MOMENT:
            tail_scenarios, _ = tail_return_scenarios(
                ctx,
                tail_probability=tail_probability,
            )
            optimization_scenario_count = len(tail_scenarios)
            scenario_tail_overlay_fraction = _scenario_tail_overlay_fraction(
                ctx,
                risk_model,
                cvar_confidence,
            )
            tail_second_moment_scale = _tail_second_moment_variance_scale(
                ctx,
                risk_model,
                cvar_confidence,
            )
        elif max_optimization_scenarios is not None:
            reduced_scenarios, _ = reduce_return_scenarios(
                ctx,
                max_scenarios=max_optimization_scenarios,
            )
            optimization_scenario_count = len(reduced_scenarios)
        if resolved_risk_measure in {
            RebalanceRiskMeasure.DOWNSIDE_CVAR,
            RebalanceRiskMeasure.HYBRID_DOWNSIDE,
        }:
            base_cvar_weight = _economic_cvar_weight_scale(
                ctx,
                impact_bps,
                linear_bps,
                cvar_confidence,
            )
            if resolved_risk_measure is RebalanceRiskMeasure.HYBRID_DOWNSIDE:
                scenario_tail_overlay_fraction = _scenario_tail_overlay_fraction(
                    ctx,
                    risk_model,
                    cvar_confidence,
                )
        risk_metric_column = "loss_cvar_95_dollars"
    else:
        risk_metric_column = "pnl_vol_dollars"
    if lambda_multipliers is None:
        lambda_multipliers = _default_lambda_multipliers(resolved_risk_measure)
    multipliers = sorted({float(value) for value in lambda_multipliers})
    if not multipliers or multipliers[0] < 0 or not np.all(np.isfinite(multipliers)):
        raise ValueError("lambda_multipliers must contain finite non-negative values")

    rows: list[dict[str, object]] = []
    configs: dict[str, TradePlannerConfig] = {}
    results: dict[str, TradePlannerResult] = {}
    for multiplier in multipliers:
        variance_weight = (
            base_variance_weight * multiplier
            if resolved_risk_measure
            in {
                RebalanceRiskMeasure.VARIANCE,
                RebalanceRiskMeasure.HYBRID_DOWNSIDE,
                RebalanceRiskMeasure.TAIL_SECOND_MOMENT,
            }
            else 0.0
        )
        if resolved_risk_measure is RebalanceRiskMeasure.TAIL_SECOND_MOMENT:
            path_risk_weight = (
                base_variance_weight * multiplier * tail_second_moment_scale
            )
        else:
            path_risk_weight = (
                base_cvar_weight
                * multiplier
                * (
                    scenario_tail_overlay_fraction
                    if resolved_risk_measure is RebalanceRiskMeasure.HYBRID_DOWNSIDE
                    else 1.0
                )
                if resolved_risk_measure
                in {
                    RebalanceRiskMeasure.DOWNSIDE_CVAR,
                    RebalanceRiskMeasure.HYBRID_DOWNSIDE,
                }
                else 0.0
            )
        candidate = (
            f"{resolved_risk_measure.value}__multiplier_{multiplier:.6g}"
            f"__variance_{variance_weight:.6g}__path_{path_risk_weight:.6g}"
        )
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
            inventory_risk_weight=variance_weight,
            inventory_alpha_model=ExpectedReturnAlphaModel(),
            inventory_path_risk_weight=path_risk_weight,
            inventory_path_risk_model=(
                (
                    TailSecondMomentPathRiskModel(
                        tail_probability=tail_probability,
                    )
                    if resolved_risk_measure is RebalanceRiskMeasure.TAIL_SECOND_MOMENT
                    else ScenarioCVaRRiskModel(
                        confidence=cvar_confidence,
                        max_scenarios=max_optimization_scenarios,
                    )
                )
                if path_risk_weight > 0
                else None
            ),
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
                cvar_confidence=cvar_confidence,
            )
        except Exception as error:
            rows.append(
                {
                    "candidate": candidate,
                    "risk_measure": resolved_risk_measure.value,
                    "scenario_tail_overlay_fraction": scenario_tail_overlay_fraction,
                    "optimization_scenario_count": optimization_scenario_count,
                    "lambda_multiplier": multiplier,
                    "inventory_risk_weight": variance_weight,
                    "inventory_path_risk_weight": path_risk_weight,
                    "status": type(error).__name__,
                    "failure_reason": str(error),
                }
            )
            continue
        results[candidate] = result
        rows.append(
            {
                "candidate": candidate,
                "risk_measure": resolved_risk_measure.value,
                "scenario_tail_overlay_fraction": scenario_tail_overlay_fraction,
                "optimization_scenario_count": optimization_scenario_count,
                "lambda_multiplier": multiplier,
                "inventory_risk_weight": variance_weight,
                "inventory_path_risk_weight": path_risk_weight,
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
        risk_measure=resolved_risk_measure,
        risk_metric_column=risk_metric_column,
        scenario_tail_overlay_fraction=scenario_tail_overlay_fraction,
        optimization_scenario_count=optimization_scenario_count,
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
    cvar_confidence: float = 0.95,
) -> RebalanceEconomicMetrics:
    """Evaluate expected net P&L and holding-P&L volatility for one schedule."""

    if not np.isclose(cvar_confidence, 0.95):
        raise ValueError("rebalance economic metrics currently require cvar_confidence=0.95")
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
    if ctx.return_residual_scenarios is not None:
        scenarios, weights = centered_return_scenarios(ctx)
        residual_pnl = np.einsum(
            "stn,tn->s",
            scenarios,
            cumulative * ctx.price,
        )
        scenario_net_pnl = net + residual_pnl
        loss_var, loss_cvar = weighted_loss_var_cvar(
            scenario_net_pnl,
            weights,
            confidence=cvar_confidence,
        )
        probability_profitable = float(np.sum(weights[scenario_net_pnl > 0]))
    else:
        probability_profitable = (
            0.5 * (1.0 + erf(net / (pnl_vol * sqrt(2.0))))
            if pnl_vol > 0
            else float(net > 0)
        )
        normal = NormalDist()
        normal_quantile = normal.inv_cdf(cvar_confidence)
        normal_expected_shortfall_multiplier = (
            normal.pdf(normal_quantile) / (1.0 - cvar_confidence)
        )
        loss_var = normal_quantile * pnl_vol - net
        loss_cvar = normal_expected_shortfall_multiplier * pnl_vol - net
    return RebalanceEconomicMetrics(
        expected_alpha_dollars=expected_alpha,
        impact_cost_dollars=impact_cost,
        linear_cost_dollars=linear_cost,
        expected_net_pnl_dollars=net,
        pnl_vol_dollars=pnl_vol,
        loss_var_95_dollars=loss_var,
        loss_cvar_95_dollars=loss_cvar,
        probability_profitable=probability_profitable,
    )


def _resolve_risk_measure(
    ctx: PlannerContext,
    risk_measure: RebalanceRiskMeasure | str,
) -> RebalanceRiskMeasure:
    parsed = RebalanceRiskMeasure.parse(risk_measure)
    if parsed is RebalanceRiskMeasure.AUTO:
        return (
            RebalanceRiskMeasure.HYBRID_DOWNSIDE
            if ctx.return_residual_scenarios is not None
            else RebalanceRiskMeasure.VARIANCE
        )
    return parsed


def _default_lambda_multipliers(
    risk_measure: RebalanceRiskMeasure,
) -> tuple[float, ...]:
    """Use a smaller grid for scenario models, which are costlier to solve."""

    if risk_measure in {
        RebalanceRiskMeasure.HYBRID_DOWNSIDE,
        RebalanceRiskMeasure.TAIL_SECOND_MOMENT,
    }:
        return (0.0, 0.1, 0.3, 1.0, 2.0, 3.0, 10.0, 30.0, 100.0)
    if risk_measure is RebalanceRiskMeasure.DOWNSIDE_CVAR:
        return (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
    return (
        0.0,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
        0.1,
        0.3,
        1.0,
        3.0,
        10.0,
        30.0,
        100.0,
        300.0,
        1_000.0,
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


def _economic_cvar_weight_scale(
    ctx: PlannerContext,
    impact_bps: float,
    linear_bps: float,
    confidence: float,
) -> float:
    """Scale a dollar-CVaR penalty from the basket's own economics."""

    target = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    full_alpha = 0.0
    full_positions = []
    for date_index in range(len(ctx.dates)):
        target_dollars = np.asarray(ctx.price[date_index], dtype=float) * target
        full_positions.append(target_dollars)
        full_alpha += float(np.dot(target_dollars, ctx.expected_return[date_index]))
    gross = float(np.sum(np.abs(ctx.price[0] * target)))
    cost_scale = gross * (impact_bps + linear_bps) / 10_000.0
    economic_dollars = max(abs(full_alpha), cost_scale, gross / 10_000.0, 1e-6)

    scenarios, weights = centered_return_scenarios(ctx)
    full_residual_pnl = np.einsum(
        "stn,tn->s",
        scenarios,
        np.asarray(full_positions, dtype=float),
    )
    _, full_cvar = weighted_loss_var_cvar(
        full_residual_pnl,
        weights,
        confidence=confidence,
    )
    return economic_dollars / max(full_cvar, gross / 10_000.0, 1e-12)


def _tail_second_moment_variance_scale(
    ctx: PlannerContext,
    risk_model: BarraFactorRiskModel,
    confidence: float,
) -> float:
    """Match tail-path variance to the scenario excess beyond covariance risk."""

    target = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    target_positions = np.asarray(ctx.price, dtype=float) * target[None, :]
    full_covariance_variance = 0.0
    for date_index in range(len(ctx.dates)):
        position_dollars = target_positions[date_index]
        covariance = _security_covariance(ctx, date_index, risk_model)
        full_covariance_variance += float(
            position_dollars @ covariance @ position_dollars
        )

    tail_probability = min(2.0 * (1.0 - confidence), 0.5)
    tail_scenarios, tail_weights = tail_return_scenarios(
        ctx,
        tail_probability=tail_probability,
    )
    tail_pnl = np.einsum("stn,tn->s", tail_scenarios, target_positions)
    tail_second_moment = float(np.dot(tail_weights, np.square(tail_pnl)))
    excess_fraction = _scenario_tail_overlay_fraction(ctx, risk_model, confidence)
    return (
        excess_fraction
        * full_covariance_variance
        / max(tail_second_moment, 1e-12)
    )


def _scenario_tail_overlay_fraction(
    ctx: PlannerContext,
    risk_model: BarraFactorRiskModel,
    confidence: float,
) -> float:
    """Price only scenario tail risk beyond covariance-implied expected shortfall."""

    target = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    target_positions = []
    full_variance = 0.0
    for date_index in range(len(ctx.dates)):
        position_dollars = np.asarray(ctx.price[date_index], dtype=float) * target
        target_positions.append(position_dollars)
        covariance = _security_covariance(ctx, date_index, risk_model)
        full_variance += float(position_dollars @ covariance @ position_dollars)

    scenarios, weights = centered_return_scenarios(ctx)
    residual_pnl = np.einsum(
        "stn,tn->s",
        scenarios,
        np.asarray(target_positions, dtype=float),
    )
    _, scenario_cvar = weighted_loss_var_cvar(
        residual_pnl,
        weights,
        confidence=confidence,
    )
    normal = NormalDist()
    normal_quantile = normal.inv_cdf(confidence)
    normal_cvar_multiplier = normal.pdf(normal_quantile) / (1.0 - confidence)
    normal_cvar = normal_cvar_multiplier * sqrt(max(full_variance, 0.0))
    if normal_cvar <= 0:
        return 0.0
    excess_fraction = scenario_cvar / normal_cvar - 1.0
    return float(np.clip(excess_fraction, 0.0, 1.0))


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
