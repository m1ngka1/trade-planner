"""Recorded CLARABEL experiment for optimizer-derived daily volume behavior.

The fixed benchmark separates two questions so one effect cannot hide another:

* ``urgency_ramp`` checks that capacity-heavy names start first, flexible names
  wait, and aggregate volume rises toward a common event/deadline.
* ``factor_balance`` checks that accumulated-inventory risk pulls flexible
  country/sector hedges forward while leaving the final parent basket intact.

Every tested parameter combination is written to CSV and every daily volume
profile is plotted.  The experiment is deterministic and uses only synthetic
data; it is intended to calibrate model behavior, not estimate production
transaction costs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

from trade_planner import (
    AdaptiveAnnouncementParticipation,
    BarraFactorRiskModel,
    CompositeCostModel,
    LinearBpsCost,
    ParticipationCapModel,
    PlannerContext,
    QuadraticParticipationImpact,
    TradePlanner,
    TradePlannerConfig,
    days_to_next_event,
    default_constraints,
)


DATES = pd.bdate_range("2026-07-01", periods=10)
EVENT_DATE = DATES[-1]
PRICE = 1.0
ADV = 100.0
PARTICIPATION = 0.10
CAPACITY = ADV * PARTICIPATION
FACTOR_NAMES = (
    "country_HK",
    "country_JP",
    "sector_financials",
    "sector_it",
    "industry_banks",
    "industry_software",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    inventory_weight: float
    residual_weight: float
    factor_weight: float
    impact_bps: float
    adaptive_caps: bool = False


@dataclass(frozen=True)
class Fixture:
    name: str
    ctx: PlannerContext
    classifications: pd.DataFrame


def _factor_row(country: str, sector: str) -> list[float]:
    return [
        float(country == "HK"),
        float(country == "JP"),
        float(sector == "financials"),
        float(sector == "it"),
        float(sector == "financials"),
        float(sector == "it"),
    ]


def _build_fixture(name: str, rows: list[dict[str, object]]) -> Fixture:
    classifications = pd.DataFrame(rows).set_index("symbol")
    symbols = list(classifications.index.astype(str))
    targets = classifications["target_shares"].to_numpy(float)
    factor_exposure = np.asarray(
        [
            _factor_row(str(row.country), str(row.sector))
            for row in classifications.itertuples()
        ],
        dtype=float,
    )
    shape = (len(DATES), len(symbols))
    factor_count = len(FACTOR_NAMES)
    event_dates = {symbol: EVENT_DATE for symbol in symbols}
    ctx = PlannerContext(
        symbols=symbols,
        dates=DATES,
        orders=pd.DataFrame({"target_shares": targets}, index=symbols),
        panel=pd.DataFrame(index=pd.MultiIndex.from_product([DATES, symbols])),
        price=np.full(shape, PRICE),
        adv_shares=np.full(shape, ADV),
        is_open=np.ones(shape, dtype=bool),
        base_participation=np.full(shape, PARTICIPATION),
        event_days=days_to_next_event(DATES, symbols, event_dates),
        factor_names=list(FACTOR_NAMES),
        factor_exposure=np.tile(factor_exposure[None, :, :], (len(DATES), 1, 1)),
        factor_covariance=np.tile(
            (np.eye(factor_count) * 0.0005)[None, :, :],
            (len(DATES), 1, 1),
        ),
        specific_variance=np.full(shape, 0.0001),
    )
    return Fixture(name=name, ctx=ctx, classifications=classifications)


def urgency_ramp_fixture() -> Fixture:
    rows: list[dict[str, object]] = []
    templates = (
        ("HK", "financials", 1.0),
        ("HK", "financials", -1.0),
        ("JP", "it", 1.0),
        ("JP", "it", -1.0),
    )
    for urgency, target in (("urgent", 85.0), ("medium", 45.0), ("small", 10.0)):
        for index, (country, sector, sign) in enumerate(templates):
            rows.append(
                {
                    "symbol": f"{urgency[:1].upper()}{index}_{country}_{sector}",
                    "target_shares": sign * target,
                    "urgency": urgency,
                    "country": country,
                    "sector": sector,
                }
            )
    return _build_fixture("urgency_ramp", rows)


def factor_balance_fixture() -> Fixture:
    # Urgent positions deliberately create HK/financials versus JP/IT risk.
    # The smaller flexible positions are exact factor offsets early, but cannot
    # erase the intentionally imbalanced terminal parent basket.
    rows = [
        {
            "symbol": "URG_HK_FIN_LONG",
            "target_shares": 85.0,
            "urgency": "urgent",
            "country": "HK",
            "sector": "financials",
        },
        {
            "symbol": "URG_JP_IT_SHORT",
            "target_shares": -85.0,
            "urgency": "urgent",
            "country": "JP",
            "sector": "it",
        },
        {
            "symbol": "FLEX_HK_FIN_SHORT",
            "target_shares": -45.0,
            "urgency": "medium",
            "country": "HK",
            "sector": "financials",
        },
        {
            "symbol": "FLEX_JP_IT_LONG",
            "target_shares": 45.0,
            "urgency": "medium",
            "country": "JP",
            "sector": "it",
        },
    ]
    return _build_fixture("factor_balance", rows)


def candidates() -> list[Candidate]:
    out = [
        Candidate("baseline_adaptive_residual", 0.0, 1.0, 1.0, 5.0, True),
        Candidate("baseline_physical_residual", 0.0, 1.0, 1.0, 5.0, False),
    ]
    for inventory_weight in (0.1, 1.0, 10.0):
        for factor_weight in (0.0, 1.0, 10.0):
            for impact_bps in (1.0, 5.0, 20.0):
                out.append(
                    Candidate(
                        name=(
                            f"inventory_{inventory_weight:g}__factor_{factor_weight:g}"
                            f"__impact_{impact_bps:g}"
                        ),
                        inventory_weight=inventory_weight,
                        residual_weight=0.0,
                        factor_weight=factor_weight,
                        impact_bps=impact_bps,
                    )
                )
    return out


def _planner(candidate: Candidate, fixture: Fixture, solver: str) -> tuple[TradePlanner, PlannerContext]:
    ctx = replace(
        fixture.ctx,
        factor_covariance=fixture.ctx.factor_covariance * candidate.factor_weight,
    )
    participation_model = ParticipationCapModel()
    if candidate.adaptive_caps:
        participation_model = ParticipationCapModel(
            modifiers=(AdaptiveAnnouncementParticipation(balance_sides=True),)
        )
    cost_terms = [LinearBpsCost(bps=1.0)]
    if candidate.impact_bps > 0:
        cost_terms.insert(0, QuadraticParticipationImpact(candidate.impact_bps))
    planner = TradePlanner(
        TradePlannerConfig(
            participation_model=participation_model,
            risk_model=BarraFactorRiskModel(),
            cost_model=CompositeCostModel(terms=tuple(cost_terms)),
            constraints=default_constraints(),
            residual_risk_weight=candidate.residual_weight,
            inventory_risk_weight=candidate.inventory_weight,
            solver=solver,
        )
    )
    return planner, ctx


def _trade_matrix(schedule: pd.DataFrame, ctx: PlannerContext) -> np.ndarray:
    return (
        schedule.pivot(index="date", columns="symbol", values="trade_shares")
        .reindex(index=ctx.dates, columns=ctx.symbols)
        .to_numpy(float)
    )


def _spearman(values: np.ndarray) -> float:
    left = pd.Series(np.arange(len(values), dtype=float)).rank().to_numpy(float)
    right = pd.Series(values).rank().to_numpy(float)
    if np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _evaluate(
    candidate: Candidate,
    fixture: Fixture,
    ctx: PlannerContext,
    schedule: pd.DataFrame,
    objective: float,
    status: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades = _trade_matrix(schedule, ctx)
    signed_targets = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    target_abs = np.abs(signed_targets)
    direction = np.sign(signed_targets)
    prices = ctx.price[0]
    caps = ctx.base_participation * ctx.adv_shares * ctx.is_open.astype(float)
    abs_trades = np.abs(trades)
    cumulative_signed = np.cumsum(trades, axis=0)
    cumulative_completed = cumulative_signed * direction[None, :]
    total_gross = float(np.sum(target_abs * prices))
    daily_gross = np.sum(abs_trades * prices[None, :], axis=1)
    daily_gross_pct = 100.0 * daily_gross / total_gross
    cumulative_gross_pct = np.cumsum(daily_gross_pct)

    future_capacity = np.flip(np.cumsum(np.flip(caps, axis=0), axis=0), axis=0) - caps
    latest_start_floors = np.maximum(target_abs[None, :] - future_capacity, 0.0)
    max_floor_violation = float(np.max(np.maximum(latest_start_floors - cumulative_completed, 0.0)))
    max_capacity_violation = float(np.max(np.maximum(abs_trades - caps, 0.0)))
    max_direction_violation = float(np.max(np.maximum(-(trades * direction[None, :]), 0.0)))
    terminal_residual = float(np.max(np.abs(signed_targets - cumulative_signed[-1])))

    classifications = fixture.classifications.reindex(ctx.symbols)
    group_daily: dict[str, np.ndarray] = {}
    for urgency in ("urgent", "medium", "small"):
        mask = classifications["urgency"].to_numpy(str) == urgency
        group_daily[urgency] = (
            np.sum(abs_trades[:, mask] * prices[None, mask], axis=1) / total_gross * 100.0
            if np.any(mask)
            else np.zeros(len(ctx.dates))
        )

    material = np.maximum(1.0, 0.0001 * target_abs)
    first_trade_days: list[float] = []
    median_days: list[float] = []
    for column in range(len(ctx.symbols)):
        active = np.flatnonzero(abs_trades[:, column] >= material[column])
        first_trade_days.append(float(active[0] + 1) if active.size else np.nan)
        weights = abs_trades[:, column]
        median_day = float(
            np.searchsorted(np.cumsum(weights), 0.5 * np.sum(weights), side="left") + 1
        )
        median_days.append(median_day)

    name_summary = classifications.copy()
    name_summary["candidate"] = candidate.name
    name_summary["fixture"] = fixture.name
    name_summary["target_abs_shares"] = target_abs
    name_summary["capacity_days_required"] = target_abs / np.sum(caps, axis=0) * len(ctx.dates)
    name_summary["analytic_latest_start_day"] = np.maximum(
        1,
        len(ctx.dates) - np.ceil(target_abs / CAPACITY).astype(int) + 1,
    )
    name_summary["first_trade_day"] = first_trade_days
    name_summary["median_execution_day"] = median_days
    name_summary = name_summary.reset_index(names="symbol")

    def group_stat(group: str, column: str, how: str) -> float:
        values = name_summary.loc[name_summary["urgency"] == group, column]
        if values.empty:
            return np.nan
        return float(getattr(values, how)())

    factor_matrix = ctx.factor_exposure[0]
    exposure_rows: list[dict[str, object]] = []
    early_max_imbalance = 0.0
    for date_index, date in enumerate(ctx.dates):
        dollars = prices * cumulative_signed[date_index]
        cumulative_gross = float(np.sum(np.abs(dollars)))
        factor_dollars = factor_matrix.T @ dollars
        for factor_index, factor in enumerate(ctx.factor_names or []):
            normalized = (
                100.0 * abs(float(factor_dollars[factor_index])) / cumulative_gross
                if cumulative_gross > 1e-8
                else 0.0
            )
            exposure_rows.append(
                {
                    "candidate": candidate.name,
                    "fixture": fixture.name,
                    "date": date,
                    "factor": factor,
                    "factor_dollars": float(factor_dollars[factor_index]),
                    "normalized_abs_exposure_pct": normalized,
                }
            )
            if 1 <= date_index <= 3:
                early_max_imbalance = max(early_max_imbalance, normalized)

    profiles = pd.DataFrame(
        {
            "candidate": candidate.name,
            "fixture": fixture.name,
            "date": ctx.dates,
            "daily_gross_pct": daily_gross_pct,
            "cumulative_gross_pct": cumulative_gross_pct,
            "urgent_daily_gross_pct": group_daily["urgent"],
            "medium_daily_gross_pct": group_daily["medium"],
            "small_daily_gross_pct": group_daily["small"],
        }
    )
    schedule_records = schedule.copy()
    schedule_records.insert(0, "fixture", fixture.name)
    schedule_records.insert(0, "candidate", candidate.name)

    urgent_first_max = group_stat("urgent", "first_trade_day", "max")
    medium_first_min = group_stat("medium", "first_trade_day", "min")
    medium_first_max = group_stat("medium", "first_trade_day", "max")
    small_first_min = group_stat("small", "first_trade_day", "min")
    urgent_median = group_stat("urgent", "median_execution_day", "median")
    medium_median = group_stat("medium", "median_execution_day", "median")
    small_median = group_stat("small", "median_execution_day", "median")
    small_target = float(
        np.sum(target_abs[classifications["urgency"].to_numpy(str) == "small"])
    )
    small_day5 = (
        100.0
        * float(np.sum(abs_trades[:5, classifications["urgency"].to_numpy(str) == "small"]))
        / small_target
        if small_target > 0
        else np.nan
    )

    global_ok = (
        status in {"optimal", "optimal_inaccurate"}
        and terminal_residual <= 1e-5
        and max_capacity_violation <= 1e-6
        and max_direction_violation <= 1e-6
        and max_floor_violation <= 1e-5
    )
    ramp_ok = bool(
        fixture.name != "urgency_ramp"
        or (
            _spearman(daily_gross) >= 0.80
            and np.mean(daily_gross[-3:]) / max(np.mean(daily_gross[:3]), 1e-12) >= 2.0
            and cumulative_gross_pct[4] <= 35.0
            and int(np.sum(np.diff(daily_gross) >= -1e-5)) >= 7
        )
    )
    urgency_ok = bool(
        fixture.name != "urgency_ramp"
        or (
            urgent_first_max <= 2
            and 5 <= medium_first_min <= medium_first_max <= 6
            and small_first_min >= 8
            and small_day5 <= 5.0
            and urgent_median < medium_median < small_median
            and small_median - urgent_median >= 4
        )
    )
    balance_ok = bool(fixture.name != "factor_balance" or early_max_imbalance <= 10.0)

    row: dict[str, object] = {
        "candidate": candidate.name,
        "fixture": fixture.name,
        "solver": "CLARABEL",
        "status": status,
        "inventory_weight": candidate.inventory_weight,
        "residual_weight": candidate.residual_weight,
        "factor_weight": candidate.factor_weight,
        "impact_bps": candidate.impact_bps,
        "adaptive_caps": candidate.adaptive_caps,
        "objective": objective,
        "max_terminal_residual": terminal_residual,
        "max_capacity_violation": max_capacity_violation,
        "max_direction_violation": max_direction_violation,
        "max_latest_start_floor_violation": max_floor_violation,
        "early_factor_imbalance_pct": early_max_imbalance,
        "urgent_first_trade_max_day": urgent_first_max,
        "medium_first_trade_min_day": medium_first_min,
        "medium_first_trade_max_day": medium_first_max,
        "small_first_trade_min_day": small_first_min,
        "small_completion_day5_pct": small_day5,
        "urgent_median_execution_day": urgent_median,
        "medium_median_execution_day": medium_median,
        "small_median_execution_day": small_median,
        "urgency_median_gap_days": small_median - urgent_median,
        "daily_gross_spearman": _spearman(daily_gross),
        "late_early_gross_ratio": float(
            np.mean(daily_gross[-3:]) / max(np.mean(daily_gross[:3]), 1e-12)
        ),
        "completion_day5_pct": float(cumulative_gross_pct[4]),
        "nondecreasing_transitions": int(np.sum(np.diff(daily_gross) >= -1e-5)),
        "peak_daily_gross_pct": float(np.max(daily_gross_pct)),
        "passes_global": global_ok,
        "passes_balance": balance_ok,
        "passes_urgency": urgency_ok,
        "passes_ramp": ramp_ok,
        "passes_fixture": global_ok and balance_ok and urgency_ok and ramp_ok,
    }
    return row, profiles, schedule_records, pd.DataFrame(exposure_rows), name_summary


def run_experiment(solver: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], str | None]:
    if solver not in cp.installed_solvers():
        raise RuntimeError(f"{solver} is not installed; installed solvers: {cp.installed_solvers()}")
    fixtures = (urgency_ramp_fixture(), factor_balance_fixture())
    rows: list[dict[str, object]] = []
    profiles: list[pd.DataFrame] = []
    schedules: list[pd.DataFrame] = []
    exposures: list[pd.DataFrame] = []
    names: list[pd.DataFrame] = []
    candidate_map = {candidate.name: candidate for candidate in candidates()}

    for candidate in candidate_map.values():
        for fixture in fixtures:
            try:
                planner, ctx = _planner(candidate, fixture, solver)
                result = planner.solve(ctx)
                evaluated = _evaluate(
                    candidate,
                    fixture,
                    ctx,
                    result.schedule,
                    float(result.diagnostics["objective"]),
                    str(result.diagnostics["status"]),
                )
            except Exception as error:  # preserve failed combinations as evidence
                rows.append(
                    {
                        "candidate": candidate.name,
                        "fixture": fixture.name,
                        "solver": solver,
                        "status": type(error).__name__,
                        "inventory_weight": candidate.inventory_weight,
                        "residual_weight": candidate.residual_weight,
                        "factor_weight": candidate.factor_weight,
                        "impact_bps": candidate.impact_bps,
                        "adaptive_caps": candidate.adaptive_caps,
                        "failure_reason": str(error),
                        "passes_fixture": False,
                    }
                )
                continue
            row, profile, schedule, exposure, name_summary = evaluated
            rows.append(row)
            profiles.append(profile)
            schedules.append(schedule)
            exposures.append(exposure)
            names.append(name_summary)

    runs = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for candidate_name, group in runs.groupby("candidate", sort=False):
        by_fixture = group.set_index("fixture")
        if not {"urgency_ramp", "factor_balance"}.issubset(by_fixture.index):
            continue
        ramp = by_fixture.loc["urgency_ramp"]
        balance = by_fixture.loc["factor_balance"]
        candidate = candidate_map[candidate_name]
        no_balance_name = (
            f"inventory_{candidate.inventory_weight:g}__factor_0__impact_{candidate.impact_bps:g}"
        )
        no_balance = runs[
            (runs["candidate"] == no_balance_name)
            & (runs["fixture"] == "factor_balance")
        ]
        no_balance_imbalance = (
            float(no_balance.iloc[0]["early_factor_imbalance_pct"])
            if not no_balance.empty and "early_factor_imbalance_pct" in no_balance
            else np.nan
        )
        balance_improvement = (
            100.0
            * (no_balance_imbalance - float(balance.get("early_factor_imbalance_pct", np.nan)))
            / no_balance_imbalance
            if no_balance_imbalance > 0
            else np.nan
        )
        passes_all = bool(
            bool(ramp.get("passes_fixture", False))
            and bool(balance.get("passes_fixture", False))
            and candidate.factor_weight > 0
            and balance_improvement >= 75.0
        )
        summary_rows.append(
            {
                "candidate": candidate_name,
                "inventory_weight": candidate.inventory_weight,
                "residual_weight": candidate.residual_weight,
                "factor_weight": candidate.factor_weight,
                "impact_bps": candidate.impact_bps,
                "ramp_spearman": ramp.get("daily_gross_spearman", np.nan),
                "late_early_ratio": ramp.get("late_early_gross_ratio", np.nan),
                "completion_day5_pct": ramp.get("completion_day5_pct", np.nan),
                "nondecreasing_transitions": ramp.get("nondecreasing_transitions", np.nan),
                "urgent_first_trade_max_day": ramp.get("urgent_first_trade_max_day", np.nan),
                "medium_first_trade_min_day": ramp.get("medium_first_trade_min_day", np.nan),
                "small_first_trade_min_day": ramp.get("small_first_trade_min_day", np.nan),
                "urgency_median_gap_days": ramp.get("urgency_median_gap_days", np.nan),
                "early_factor_imbalance_pct": balance.get("early_factor_imbalance_pct", np.nan),
                "no_balance_imbalance_pct": no_balance_imbalance,
                "balance_improvement_pct": balance_improvement,
                "peak_daily_gross_pct": ramp.get("peak_daily_gross_pct", np.nan),
                "passes_all": passes_all,
            }
        )
    summary = pd.DataFrame(summary_rows)
    passing = summary[summary["passes_all"]].copy()
    selected: str | None = None
    if not passing.empty:
        passing = passing.sort_values(
            ["peak_daily_gross_pct", "inventory_weight", "factor_weight", "impact_bps"]
        )
        selected = str(passing.iloc[0]["candidate"])

    outputs = {
        "summary": summary,
        "profiles": pd.concat(profiles, ignore_index=True),
        "schedules": pd.concat(schedules, ignore_index=True),
        "exposures": pd.concat(exposures, ignore_index=True),
        "names": pd.concat(names, ignore_index=True),
    }
    return runs, outputs, selected


def _plot_all_profiles(
    profiles: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
    *,
    fixture: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = profiles[profiles["fixture"] == fixture]
    candidate_names = list(dict.fromkeys(frame["candidate"]))
    columns = 4
    rows = int(np.ceil(len(candidate_names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(18, 3.2 * rows), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes).ravel()
    colors = {"urgent": "#24557A", "medium": "#D97732", "small": "#70A288"}
    passing = set(summary.loc[summary["passes_all"], "candidate"])
    for axis, candidate_name in zip(axes_array, candidate_names):
        curve = frame[frame["candidate"] == candidate_name].sort_values("date")
        bottom = np.zeros(len(curve))
        for urgency in ("urgent", "medium", "small"):
            values = curve[f"{urgency}_daily_gross_pct"].to_numpy(float)
            axis.bar(
                np.arange(1, len(curve) + 1),
                values,
                bottom=bottom,
                color=colors[urgency],
                label=urgency.title(),
            )
            bottom += values
        marker = "PASS" if candidate_name in passing else ""
        axis.set_title(f"{candidate_name}\n{marker}", fontsize=8.2, color="#202124")
        axis.grid(axis="y", color="#E1E5E8", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7)
    for axis in axes_array[len(candidate_names) :]:
        axis.set_visible(False)
    axes_array[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    fig.suptitle(
        f"Every tested daily volume profile · {fixture} fixture",
        x=0.06,
        y=0.998,
        ha="left",
        fontsize=16,
        fontweight="semibold",
    )
    fig.supxlabel("Planner day (common event/deadline on day 10)")
    fig.supylabel("Daily gross (% of parent basket)")
    fig.tight_layout(rect=(0.035, 0.02, 0.995, 0.98))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_selected(
    profiles: pd.DataFrame,
    exposures: pd.DataFrame,
    selected: str,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    compare = ["baseline_adaptive_residual", "baseline_physical_residual", selected]
    selected_row = profiles[(profiles["candidate"] == selected) & (profiles["fixture"] == "urgency_ramp")]
    inventory_weight = float(selected.split("__")[0].split("_")[-1])
    impact_bps = float(selected.split("__")[-1].split("_")[-1])
    no_balance = f"inventory_{inventory_weight:g}__factor_0__impact_{impact_bps:g}"

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axis = axes[0, 0]
    for candidate_name in compare:
        curve = profiles[
            (profiles["candidate"] == candidate_name)
            & (profiles["fixture"] == "urgency_ramp")
        ].sort_values("date")
        axis.plot(
            np.arange(1, len(curve) + 1),
            curve["daily_gross_pct"],
            marker="o",
            linewidth=2,
            label=candidate_name,
        )
    axis.set_title("Daily gross profile")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[0, 1]
    bottom = np.zeros(len(selected_row))
    for urgency, color in (("urgent", "#24557A"), ("medium", "#D97732"), ("small", "#70A288")):
        values = selected_row[f"{urgency}_daily_gross_pct"].to_numpy(float)
        axis.bar(np.arange(1, len(values) + 1), values, bottom=bottom, color=color, label=urgency.title())
        bottom += values
    axis.set_title("Selected profile by urgency")
    axis.legend(frameon=False)

    for axis, candidate_name in zip(axes[1], (no_balance, selected)):
        subset = exposures[
            (exposures["candidate"] == candidate_name)
            & (exposures["fixture"] == "factor_balance")
        ]
        for factor in FACTOR_NAMES[:4]:
            curve = subset[subset["factor"] == factor].sort_values("date")
            axis.plot(
                np.arange(1, len(curve) + 1),
                curve["normalized_abs_exposure_pct"],
                marker="o",
                linewidth=1.8,
                label=factor,
            )
        axis.set_title("No factor balance" if candidate_name == no_balance else "Selected factor balance")
        axis.legend(frameon=False, fontsize=8)

    for axis in axes.ravel():
        axis.set_xlabel("Planner day")
        axis.grid(axis="y", color="#E1E5E8", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Daily gross (% of parent basket)")
    axes[0, 1].set_ylabel("Daily gross (% of parent basket)")
    axes[1, 0].set_ylabel("Absolute factor exposure (% of cumulative gross)")
    axes[1, 1].set_ylabel("Absolute factor exposure (% of cumulative gross)")
    fig.suptitle(
        f"Optimizer-derived daily volume behavior · selected {selected}",
        x=0.055,
        y=0.99,
        ha="left",
        fontsize=15,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0.03, 0.02, 0.995, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("artifacts/daily_volume_behavior"),
    )
    args = parser.parse_args()
    runs, outputs, selected = run_experiment(args.solver)
    prefix: Path = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(prefix.with_name(prefix.name + "_runs.csv"), index=False)
    for key, frame in outputs.items():
        frame.to_csv(prefix.with_name(prefix.name + f"_{key}.csv"), index=False)
    _plot_all_profiles(
        outputs["profiles"],
        outputs["summary"],
        prefix.with_name(prefix.name + "_all_profiles.png"),
        fixture="urgency_ramp",
    )
    _plot_all_profiles(
        outputs["profiles"],
        outputs["summary"],
        prefix.with_name(prefix.name + "_all_profiles_factor_balance.png"),
        fixture="factor_balance",
    )
    if selected is not None:
        _plot_selected(
            outputs["profiles"],
            outputs["exposures"],
            selected,
            prefix.with_name(prefix.name + "_selected.png"),
        )
    print(runs.groupby(["fixture", "passes_fixture"], dropna=False).size().to_string())
    print("\nPassing candidates:")
    print(outputs["summary"].loc[outputs["summary"]["passes_all"]].round(4).to_string(index=False))
    print(f"\nselected: {selected}")
    print(f"artifacts: {prefix.parent / (prefix.name + '_*')}")
    if selected is None:
        raise SystemExit("No candidate passed all fixed behavior gates")


if __name__ == "__main__":
    main()
