"""Economic calibration experiment for optimizer-derived rebalance schedules.

The experiment keeps the existing shape/feasibility ideas but evaluates plans
in money terms. It compares expected alpha capture, impact, accumulated P&L
volatility, factor imbalance, and Monte Carlo net P&L for high/medium/low risk
preferences selected from one solved frontier.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from trade_planner import (
    BarraFactorRiskModel,
    CompositeCostModel,
    ExpectedReturnAlphaModel,
    LinearBpsCost,
    ParticipationCapModel,
    PlannerContext,
    QuadraticParticipationImpact,
    RiskAversion,
    TCALinearBpsCost,
    TCAQuadraticParticipationImpact,
    TradePlanner,
    TradePlannerConfig,
    build_rebalance_frontier,
    days_to_next_event,
    default_constraints,
    evaluate_rebalance_schedule,
    infer_execution_cost_matrices,
    infer_execution_costs,
)


DATES = pd.bdate_range("2026-07-01", periods=10)
FACTOR_NAMES = (
    "country_HK",
    "country_JP",
    "country_US",
    "sector_financials",
    "sector_it",
    "sector_healthcare",
    "industry_banks",
    "industry_software",
    "industry_biotech",
)
RNG_SEED = 20260723
N_SCENARIOS = 5_000
EVENT_LIQUIDITY_CURVES = {
    "medium_event_liquidity": np.array([0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.50, 1.80]),
    "strong_event_liquidity": np.array([0.40, 0.50, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60, 2.00, 2.50]),
}


def economic_fixture() -> tuple[PlannerContext, pd.DataFrame]:
    """Build a deterministic basket with liquidity, alpha, and factor trade-offs."""

    templates = (
        ("HK", "financials", "banks", 78.0, 1_200_000.0),
        ("JP", "it", "software", 54.0, 1_500_000.0),
        ("US", "healthcare", "biotech", 112.0, 900_000.0),
        ("HK", "it", "software", 66.0, 1_100_000.0),
    )
    rows: list[dict[str, object]] = []
    # Urgent flow creates factor exposure; medium flow is the flexible hedge.
    signs_by_group = {
        "urgent": (1.0, -1.0, 1.0, -1.0),
        "medium": (-1.0, 1.0, -1.0, 1.0),
        "small": (1.0, -1.0, 1.0, -1.0),
    }
    capacity_days = {"urgent": 8.5, "medium": 4.5, "small": 1.0}
    for urgency in ("urgent", "medium", "small"):
        for index, (country, sector, industry, price, adv) in enumerate(templates):
            sign = signs_by_group[urgency][index]
            rows.append(
                {
                    "symbol": f"{urgency[:1].upper()}{index}_{country}_{sector}",
                    "urgency": urgency,
                    "country": country,
                    "sector": sector,
                    "industry": industry,
                    "price": price,
                    "adv_shares": adv,
                    "target_shares": sign * capacity_days[urgency] * 0.10 * adv,
                    "impact_bps_at_10pct_adv": 6.0 + 2.0 * index + (2.0 if urgency == "small" else 0.0),
                    "linear_cost_bps": 0.8 + 0.15 * index,
                }
            )
    classifications = pd.DataFrame(rows).set_index("symbol")
    symbols = list(classifications.index)
    shape = (len(DATES), len(symbols))
    exposure = np.asarray(
        [
            [
                float(row.country == "HK"),
                float(row.country == "JP"),
                float(row.country == "US"),
                float(row.sector == "financials"),
                float(row.sector == "it"),
                float(row.sector == "healthcare"),
                float(row.industry == "banks"),
                float(row.industry == "software"),
                float(row.industry == "biotech"),
            ]
            for row in classifications.itertuples()
        ],
        dtype=float,
    )
    factor_daily_vol = np.array([0.006, 0.006, 0.005, 0.008, 0.009, 0.008, 0.010, 0.011, 0.012])
    factor_covariance = np.diag(np.square(factor_daily_vol))
    specific_variance = np.full(shape, 0.010**2)

    target_sign = np.sign(classifications["target_shares"].to_numpy(float))
    # Expected rebalance anticipation return accelerates near the event. The
    # forecast is already probability weighted, so no separate alpha knob is
    # needed in the optimizer.
    alpha_curve_bps = np.array([0.0, 0.0, 0.0, 0.4, 1.2, 2.5, 4.5, 7.5, 12.0, 0.0])
    confidence = np.where(classifications["urgency"].to_numpy(str) == "small", 0.75, 1.0)
    expected_return = (
        alpha_curve_bps[:, None]
        / 10_000.0
        * target_sign[None, :]
        * confidence[None, :]
    )
    prices = classifications["price"].to_numpy(float)
    adv = classifications["adv_shares"].to_numpy(float)
    impact_time_multiplier = np.array([1.35, 1.25, 1.15, 1.05, 1.00, 0.95, 0.90, 0.85, 0.75, 0.65])
    impact_bps_matrix = (
        impact_time_multiplier[:, None]
        * classifications["impact_bps_at_10pct_adv"].to_numpy(float)[None, :]
    )
    linear_time_multiplier = np.array([1.10, 1.08, 1.06, 1.04, 1.02, 1.00, 0.98, 0.96, 0.92, 0.88])
    linear_cost_bps_matrix = (
        linear_time_multiplier[:, None]
        * classifications["linear_cost_bps"].to_numpy(float)[None, :]
    )
    event_dates = {symbol: DATES[-1] for symbol in symbols}
    orders = classifications[
        ["target_shares", "impact_bps_at_10pct_adv", "linear_cost_bps"]
    ].copy()
    ctx = PlannerContext(
        symbols=symbols,
        dates=DATES,
        orders=orders,
        panel=pd.DataFrame(index=pd.MultiIndex.from_product([DATES, symbols])),
        price=np.tile(prices[None, :], (len(DATES), 1)),
        adv_shares=np.tile(adv[None, :], (len(DATES), 1)),
        is_open=np.ones(shape, dtype=bool),
        base_participation=np.full(shape, 0.10),
        event_days=days_to_next_event(DATES, symbols, event_dates),
        factor_names=list(FACTOR_NAMES),
        factor_exposure=np.tile(exposure[None, :, :], (len(DATES), 1, 1)),
        factor_covariance=np.tile(factor_covariance[None, :, :], (len(DATES), 1, 1)),
        specific_variance=specific_variance,
        expected_return=expected_return,
        impact_bps_at_10pct_adv=impact_bps_matrix,
        linear_cost_bps=linear_cost_bps_matrix,
    )
    return ctx, classifications


def _planner(
    *,
    inventory_risk_weight: float,
    impact_bps: float | np.ndarray,
    linear_bps: float | np.ndarray,
    alpha: bool,
    solver: str,
) -> TradePlanner:
    if np.asarray(impact_bps).ndim == 0 and np.asarray(linear_bps).ndim == 0:
        cost_terms = (
            QuadraticParticipationImpact(impact_bps_at_10pct_adv=float(impact_bps)),
            LinearBpsCost(bps=float(linear_bps)),
        )
    else:
        cost_terms = (
            TCAQuadraticParticipationImpact(np.asarray(impact_bps, dtype=float)),
            TCALinearBpsCost(np.asarray(linear_bps, dtype=float)),
        )
    return TradePlanner(
        TradePlannerConfig(
            participation_model=ParticipationCapModel(),
            risk_model=BarraFactorRiskModel(),
            cost_model=CompositeCostModel(terms=cost_terms),
            constraints=default_constraints(),
            residual_risk_weight=0.0,
            inventory_risk_weight=inventory_risk_weight,
            inventory_alpha_model=ExpectedReturnAlphaModel() if alpha else None,
            solver=solver,
        )
    )


def run_experiment(
    solver: str = "OSQP",
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    ctx, classifications = economic_fixture()
    frontier = build_rebalance_frontier(ctx, solver=solver)
    selections = {
        profile.value: frontier.select(profile)
        for profile in (RiskAversion.HIGH, RiskAversion.MEDIUM, RiskAversion.LOW)
    }
    impact_bps, linear_bps = infer_execution_costs(ctx)
    impact_matrix, linear_matrix = infer_execution_cost_matrices(ctx)
    medium_weight = selections["medium"].config.inventory_risk_weight

    trial_schedules: dict[str, pd.DataFrame] = {
        f"profile_{name}": plan.result.schedule
        for name, plan in selections.items()
    }
    trial_contexts = {trial: ctx for trial in trial_schedules}
    scalar_frontier = build_rebalance_frontier(
        ctx,
        solver=solver,
        heterogeneous_tca=False,
    )
    trial_schedules["medium_scalar_tca"] = scalar_frontier.select("medium").result.schedule
    trial_contexts["medium_scalar_tca"] = ctx
    # Ablations isolate whether profitability comes from alpha and whether early
    # balance comes from factor risk rather than a hard-coded schedule.
    failed_trials: dict[str, str] = {}
    try:
        trial_schedules["reference_fixed_weight_no_alpha"] = _planner(
            inventory_risk_weight=1.0,
            impact_bps=20.0,
            linear_bps=1.0,
            alpha=False,
            solver=solver,
        ).solve(ctx).schedule
        trial_contexts["reference_fixed_weight_no_alpha"] = ctx
    except Exception as error:
        # A fixed unitless coefficient can itself be numerically inappropriate
        # when a realistic basket replaces the unit-scaled shape fixture.
        failed_trials["reference_fixed_weight_no_alpha"] = (
            f"{type(error).__name__}: {error}"
        )
    trial_schedules["medium_weight_no_alpha"] = _planner(
        inventory_risk_weight=medium_weight,
        impact_bps=impact_matrix,
        linear_bps=linear_matrix,
        alpha=False,
        solver=solver,
    ).solve(ctx).schedule
    trial_contexts["medium_weight_no_alpha"] = ctx
    no_factor_ctx = replace(ctx, factor_covariance=np.zeros_like(ctx.factor_covariance))
    trial_schedules["medium_weight_no_factor"] = _planner(
        inventory_risk_weight=medium_weight,
        impact_bps=impact_matrix,
        linear_bps=linear_matrix,
        alpha=True,
        solver=solver,
    ).solve(no_factor_ctx).schedule
    # Evaluate this ablation against the original full covariance so the risk
    # omitted by the solve is not also omitted from measurement.
    trial_contexts["medium_weight_no_factor"] = ctx

    # New hypothesis: if forecast closing-auction/rebalance liquidity rises
    # toward the event, date-varying ADV should create late capacity and lower
    # impact without a hard-coded volume curve. Test a moderate and strong
    # forecast separately so the P&L cost of forcing too much lateness is clear.
    for trial, multipliers in EVENT_LIQUIDITY_CURVES.items():
        liquidity_ctx = replace(
            ctx,
            adv_shares=ctx.adv_shares * multipliers[:, None],
        )
        liquidity_frontier = build_rebalance_frontier(liquidity_ctx, solver=solver)
        trial_schedules[trial] = liquidity_frontier.select("medium").result.schedule
        trial_contexts[trial] = liquidity_ctx

    returns = _simulated_returns(ctx, N_SCENARIOS, RNG_SEED)
    trial_rows: list[dict[str, object]] = []
    profiles: list[pd.DataFrame] = []
    exposures: list[pd.DataFrame] = []
    scenario_summaries: list[dict[str, object]] = []
    schedule_records: list[pd.DataFrame] = []
    for trial, schedule in trial_schedules.items():
        trial_ctx = trial_contexts[trial]
        trial_impact_matrix, trial_linear_matrix = infer_execution_cost_matrices(trial_ctx)
        economics = evaluate_rebalance_schedule(
            trial_ctx,
            schedule,
            impact_bps_at_10pct_adv=trial_impact_matrix,
            linear_cost_bps=trial_linear_matrix,
        )
        behavior, profile, exposure = _behavior_metrics(trial_ctx, classifications, schedule)
        scenario_pnl = _scenario_pnl(trial_ctx, schedule, economics, returns)
        scenario_summary = {
            "trial": trial,
            "scenario_mean_pnl_dollars": float(np.mean(scenario_pnl)),
            "scenario_pnl_std_dollars": float(np.std(scenario_pnl, ddof=1)),
            "scenario_p05_pnl_dollars": float(np.quantile(scenario_pnl, 0.05)),
            "scenario_median_pnl_dollars": float(np.median(scenario_pnl)),
            "scenario_p95_pnl_dollars": float(np.quantile(scenario_pnl, 0.95)),
            "scenario_probability_profitable": float(np.mean(scenario_pnl > 0)),
        }
        scenario_summaries.append(scenario_summary)
        trial_rows.append(
            {
                "trial": trial,
                "idea": _idea_for(trial),
                "status": "optimal",
                **economics.as_dict(),
                **behavior,
                **{key: value for key, value in scenario_summary.items() if key != "trial"},
            }
        )
        profiles.append(profile.assign(trial=trial))
        exposures.append(exposure.assign(trial=trial))
        schedule_records.append(schedule.assign(trial=trial))

    for trial, failure_reason in failed_trials.items():
        trial_rows.append(
            {
                "trial": trial,
                "idea": _idea_for(trial),
                "status": "crash",
                "failure_reason": failure_reason,
            }
        )

    trials = pd.DataFrame(trial_rows)
    parent_gross = float(
        np.sum(
            np.abs(
                ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
                * ctx.price[0]
            )
        )
    )
    trials["decision"] = trials.apply(
        lambda row: _decision_for(row, trials, parent_gross),
        axis=1,
    )
    selected_candidates = {
        profile: _candidate_for_result(frontier.results, plan.result)
        for profile, plan in selections.items()
    }
    frontier_output = frontier.frontier.copy()
    frontier_output["selected_profile"] = ""
    for profile, candidate in selected_candidates.items():
        frontier_output.loc[
            frontier_output["candidate"] == candidate,
            "selected_profile",
        ] = profile

    outputs = {
        "trials": trials,
        "frontier": frontier_output,
        "profiles": pd.concat(profiles, ignore_index=True),
        "exposures": pd.concat(exposures, ignore_index=True),
        "scenario_summary": pd.DataFrame(scenario_summaries),
        "schedules": pd.concat(schedule_records, ignore_index=True),
    }
    metadata: dict[str, object] = {
        "ctx": ctx,
        "classifications": classifications,
        "selections": selections,
        "impact_bps": impact_bps,
        "linear_bps": linear_bps,
        "scenario_returns": returns,
    }
    return outputs, metadata


def _behavior_metrics(
    ctx: PlannerContext,
    classifications: pd.DataFrame,
    schedule: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    trades = _trade_matrix(ctx, schedule)
    abs_notional = np.abs(trades) * ctx.price
    parent_gross = float(
        np.sum(np.abs(ctx.orders["target_shares"].to_numpy(float) * ctx.price[0]))
    )
    daily_gross = np.sum(abs_notional, axis=1)
    cumulative = np.cumsum(trades, axis=0)
    factor_matrix = np.asarray(ctx.factor_exposure[0], dtype=float)
    exposure_rows: list[dict[str, object]] = []
    imbalance_curve = []
    for date_index, date in enumerate(ctx.dates):
        position_dollars = cumulative[date_index] * ctx.price[date_index]
        gross = float(np.sum(np.abs(position_dollars)))
        factor_dollars = factor_matrix.T @ position_dollars
        normalized = 100.0 * np.abs(factor_dollars) / gross if gross > 1e-8 else np.zeros(len(factor_dollars))
        imbalance_curve.append(float(np.max(normalized)))
        for factor, dollars, pct in zip(ctx.factor_names or [], factor_dollars, normalized):
            exposure_rows.append(
                {
                    "date": date,
                    "factor": factor,
                    "factor_dollars": float(dollars),
                    "normalized_abs_exposure_pct": float(pct),
                }
            )

    urgency = classifications["urgency"].reindex(ctx.symbols).to_numpy(str)
    group_daily = {
        group: 100.0 * np.sum(abs_notional[:, urgency == group], axis=1) / parent_gross
        for group in ("urgent", "medium", "small")
    }
    first_trade = {}
    for group in ("urgent", "medium", "small"):
        group_flow = np.sum(np.abs(trades[:, urgency == group]), axis=1)
        group_target = float(
            np.sum(
                np.abs(
                    ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)[
                        urgency == group
                    ]
                )
            )
        )
        # Ignore solver dust: a group starts when at least 0.5% of its parent
        # shares trade on one date.
        active = np.flatnonzero(group_flow >= max(1e-5, 0.005 * group_target))
        first_trade[group] = float(active[0] + 1) if active.size else np.nan
    day_numbers = np.arange(1, len(ctx.dates) + 1, dtype=float)
    daily_pct = 100.0 * daily_gross / parent_gross
    profile = pd.DataFrame(
        {
            "date": ctx.dates,
            "daily_gross_pct": daily_pct,
            "cumulative_gross_pct": np.cumsum(daily_pct),
            "urgent_daily_gross_pct": group_daily["urgent"],
            "medium_daily_gross_pct": group_daily["medium"],
            "small_daily_gross_pct": group_daily["small"],
            "max_factor_imbalance_pct": imbalance_curve,
        }
    )
    behavior = {
        "daily_gross_spearman": float(
            pd.Series(day_numbers).corr(pd.Series(daily_gross), method="spearman")
        ),
        "late_early_gross_ratio": float(
            np.mean(daily_gross[-3:]) / max(np.mean(daily_gross[:3]), 1e-12)
        ),
        "completion_day5_pct": float(np.sum(daily_pct[:5])),
        "nondecreasing_transitions": float(np.sum(np.diff(daily_gross) >= -1e-5)),
        "urgent_first_trade_day": first_trade["urgent"],
        "medium_first_trade_day": first_trade["medium"],
        "small_first_trade_day": first_trade["small"],
        "early_factor_imbalance_pct": float(np.max(imbalance_curve[1:4])),
    }
    return behavior, profile, pd.DataFrame(exposure_rows)


def _simulated_returns(ctx: PlannerContext, n_scenarios: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = np.empty((n_scenarios, len(ctx.dates), len(ctx.symbols)), dtype=float)
    factor_exposure = np.asarray(ctx.factor_exposure, dtype=float)
    for date_index in range(len(ctx.dates)):
        covariance = (
            factor_exposure[date_index]
            @ ctx.factor_covariance[date_index]
            @ factor_exposure[date_index].T
            + np.diag(ctx.specific_variance[date_index])
        )
        draws[:, date_index, :] = rng.multivariate_normal(
            mean=ctx.expected_return[date_index],
            cov=covariance,
            size=n_scenarios,
        )
    return draws


def _scenario_pnl(
    ctx: PlannerContext,
    schedule: pd.DataFrame,
    economics,
    returns: np.ndarray,
) -> np.ndarray:
    trades = _trade_matrix(ctx, schedule)
    position_dollars = np.cumsum(trades, axis=0) * ctx.price
    gross_holding_pnl = np.einsum("stn,tn->s", returns, position_dollars)
    costs = economics.impact_cost_dollars + economics.linear_cost_dollars
    return gross_holding_pnl - costs


def _trade_matrix(ctx: PlannerContext, schedule: pd.DataFrame) -> np.ndarray:
    return (
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


def _candidate_for_result(
    results: Mapping[str, object],
    selected_result: object,
) -> str:
    for candidate, result in results.items():
        if result is selected_result:
            return str(candidate)
    raise KeyError("selected calibration result is missing from frontier")


def _idea_for(trial: str) -> str:
    if trial.startswith("profile_"):
        return "Select the highest expected net P&L inside the desk risk budget."
    if trial == "reference_fixed_weight_no_alpha":
        return "Use the prior fixed synthetic coefficient without an alpha forecast."
    if trial == "medium_weight_no_alpha":
        return "Ablate expected rebalance alpha at the medium calibrated risk weight."
    if trial == "medium_scalar_tca":
        return "Ablate date-by-name TCA and solve with target-weighted basket medians."
    if trial == "medium_event_liquidity":
        return "Use a moderate data-driven rise in forecast event liquidity instead of a schedule rule."
    if trial == "strong_event_liquidity":
        return "Stress-test a strong rise in forecast event liquidity and measure the alpha sacrificed."
    return "Ablate country, sector, and industry covariance at the medium risk weight."


def _decision_for(row: pd.Series, trials: pd.DataFrame, parent_gross: float) -> str:
    trial = str(row["trial"])
    if row.get("status") == "crash":
        return "discard"
    if trial.startswith("profile_"):
        return "keep"
    if trial == "reference_fixed_weight_no_alpha":
        return "baseline_only"
    medium = trials.loc[trials["trial"] == "profile_medium"].iloc[0]
    if trial == "medium_weight_no_alpha":
        return "discard" if row["expected_net_pnl_dollars"] < medium["expected_net_pnl_dollars"] else "inconclusive"
    if trial == "medium_weight_no_factor":
        return "discard" if row["early_factor_imbalance_pct"] > medium["early_factor_imbalance_pct"] else "inconclusive"
    if trial == "medium_scalar_tca":
        heterogeneous_is_economically_tied = (
            medium["expected_net_pnl_dollars"]
            >= row["expected_net_pnl_dollars"] - parent_gross / 10_000.0
        )
        heterogeneous_has_no_more_risk = (
            medium["pnl_vol_dollars"] <= row["pnl_vol_dollars"]
        )
        return (
            "discard"
            if heterogeneous_is_economically_tied and heterogeneous_has_no_more_risk
            else "inconclusive"
        )
    if trial in EVENT_LIQUIDITY_CURVES:
        economically_tied = (
            row["expected_net_pnl_dollars"]
            >= medium["expected_net_pnl_dollars"] - parent_gross / 10_000.0
        )
        improves_risk_and_ramp = (
            row["pnl_vol_dollars"] < medium["pnl_vol_dollars"]
            and row["late_early_gross_ratio"] > medium["late_early_gross_ratio"]
        )
        return "keep" if economically_tied and improves_risk_and_ramp else "discard"
    return "inconclusive"


def plot_results(outputs: dict[str, pd.DataFrame], metadata: dict[str, object], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials = outputs["trials"].set_index("trial")
    frontier = outputs["frontier"]
    profiles = outputs["profiles"]
    returns = metadata["scenario_returns"]
    ctx = metadata["ctx"]
    schedules = outputs["schedules"]
    colors = {"high": "#2F6B9A", "medium": "#D97732", "low": "#70A288"}
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0))

    axis = axes[0, 0]
    solved = frontier[frontier["status"].isin(("optimal", "optimal_inaccurate"))]
    axis.plot(
        solved["pnl_vol_dollars"] / 1_000.0,
        solved["expected_net_pnl_dollars"] / 1_000.0,
        color="#8A929A",
        linewidth=1.4,
        marker="o",
        markersize=3.5,
    )
    for profile, color in colors.items():
        row = trials.loc[f"profile_{profile}"]
        axis.scatter(
            row["pnl_vol_dollars"] / 1_000.0,
            row["expected_net_pnl_dollars"] / 1_000.0,
            s=70,
            color=color,
            label=profile.title(),
            zorder=4,
        )
    event_row = trials.loc["medium_event_liquidity"]
    axis.scatter(
        event_row["pnl_vol_dollars"] / 1_000.0,
        event_row["expected_net_pnl_dollars"] / 1_000.0,
        s=105,
        marker="*",
        color="#7C5C9E",
        label="Medium + event liquidity",
        zorder=5,
    )
    scalar_row = trials.loc["medium_scalar_tca"]
    axis.scatter(
        scalar_row["pnl_vol_dollars"] / 1_000.0,
        scalar_row["expected_net_pnl_dollars"] / 1_000.0,
        s=80,
        marker="X",
        color="#59636E",
        label="Medium + scalar TCA",
        zorder=5,
    )
    axis.set_title("Expected net P&L versus accumulated P&L risk")
    axis.set_xlabel("P&L volatility ($000)")
    axis.set_ylabel("Expected net P&L ($000)")
    axis.legend(frameon=False)

    axis = axes[0, 1]
    compare = [
        "profile_high",
        "profile_medium",
        "medium_scalar_tca",
        "medium_event_liquidity",
        "profile_low",
    ]
    style = {
        "profile_high": (colors["high"], "High risk aversion"),
        "profile_medium": (colors["medium"], "Medium risk aversion"),
        "medium_scalar_tca": ("#59636E", "Medium + scalar TCA"),
        "medium_event_liquidity": ("#7C5C9E", "Medium + event liquidity"),
        "profile_low": (colors["low"], "Low risk aversion"),
    }
    for trial in compare:
        curve = profiles[profiles["trial"] == trial].sort_values("date")
        if curve.empty:
            continue
        axis.plot(
            np.arange(1, len(curve) + 1),
            curve["daily_gross_pct"],
            marker="o",
            linewidth=2,
            color=style[trial][0],
            label=style[trial][1],
        )
    axis.set_title("Optimizer-derived daily gross volume")
    axis.set_xlabel("Planner day")
    axis.set_ylabel("Daily gross (% of parent basket)")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 0]
    for trial, color, label in (
        ("profile_medium", colors["medium"], "Medium calibrated"),
        ("medium_weight_no_factor", "#8A929A", "Same plan without factor risk"),
    ):
        curve = profiles[profiles["trial"] == trial].sort_values("date")
        axis.plot(
            np.arange(1, len(curve) + 1),
            curve["max_factor_imbalance_pct"],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
    axis.set_title("Country/sector/industry imbalance ablation")
    axis.set_xlabel("Planner day")
    axis.set_ylabel("Maximum absolute factor exposure (% of held gross)")
    axis.legend(frameon=False)

    axis = axes[1, 1]
    pnl_samples = []
    labels = []
    box_colors = []
    for profile, color in colors.items():
        trial = f"profile_{profile}"
        schedule = schedules[schedules["trial"] == trial].drop(columns="trial")
        row = trials.loc[trial]
        cost_proxy = type(
            "CostProxy",
            (),
            {
                "impact_cost_dollars": row["impact_cost_dollars"],
                "linear_cost_dollars": row["linear_cost_dollars"],
            },
        )()
        samples = _scenario_pnl(ctx, schedule, cost_proxy, returns) / 1_000.0
        pnl_samples.append(samples)
        labels.append(profile.title())
        box_colors.append(color)
    boxes = axis.boxplot(pnl_samples, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, color in zip(boxes["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    axis.axhline(0.0, color="#6B7280", linewidth=1.0)
    axis.set_title(f"Net P&L distribution · {N_SCENARIOS:,} common scenarios")
    axis.set_ylabel("Net P&L ($000)")

    for axis in axes.ravel():
        axis.grid(axis="y", color="#E1E5E8", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Investment-driven rebalance calibration",
        x=0.055,
        y=0.99,
        ha="left",
        fontsize=16,
    )
    fig.tight_layout(rect=(0.03, 0.02, 0.995, 0.96), h_pad=3.0, w_pad=2.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="OSQP")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("artifacts/rebalance_economic_calibration"),
    )
    args = parser.parse_args()
    outputs, metadata = run_experiment(args.solver)
    prefix: Path = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(prefix.with_name(prefix.name + f"_{name}.csv"), index=False)
    chart = prefix.with_name(prefix.name + ".png")
    plot_results(outputs, metadata, chart)
    print(outputs["trials"].round(4).to_string(index=False))
    print(f"\nimpact_bps_at_10pct_adv: {metadata['impact_bps']:.4f}")
    print(f"linear_cost_bps: {metadata['linear_bps']:.4f}")
    print(f"artifacts: {prefix.parent / (prefix.name + '*')}")


if __name__ == "__main__":
    main()
