"""Reproducible experiment for adaptive announcement participation policies.

It compares a single fixed pre-announcement rate with adaptive policies that
infer each name's pre-event capacity budget from order size and the liquidity
available after its announcement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trade_planner import (
    AdaptiveAnnouncementParticipation,
    AnnouncementParticipationCurve,
    AnnouncementParticipationModifier,
    CompositeCostModel,
    LinearBpsCost,
    ParticipationCapModel,
    PlannerContext,
    QuadraticParticipationImpact,
    StaticCovarianceRiskModel,
    TradePlanner,
    TradePlannerConfig,
    cumulative_side_completion,
    days_to_next_event,
    default_constraints,
)


def synthetic_basket(seed: int = 17, n_names: int = 40) -> tuple[PlannerContext, dict[str, pd.Timestamp]]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-07-01", periods=15)
    symbols = [f"S{i:03d}" for i in range(n_names)]
    event_positions = rng.integers(4, 11, size=n_names)
    announcement_dates = {symbol: dates[event_positions[i]] for i, symbol in enumerate(symbols)}
    event_days = days_to_next_event(dates, symbols, announcement_dates)

    # Keep the numerical scale compact so all open-source conic/QP solvers can
    # be compared without solver-specific tolerances dominating the experiment.
    adv = rng.uniform(400.0, 1_500.0, size=n_names)
    price = rng.uniform(20.0, 160.0, size=n_names)
    base_rate = 0.15
    desired_mandatory_fraction = rng.uniform(0.0, 0.38, size=n_names)
    post_days = len(dates) - event_positions - 1
    post_capacity = post_days * base_rate * adv
    target_abs = post_capacity / np.maximum(1.0 - desired_mandatory_fraction, 0.55)
    target_abs *= rng.uniform(0.82, 1.0, size=n_names)
    target_abs = np.minimum(target_abs, 0.95 * len(dates) * base_rate * adv)

    signs = np.where(np.arange(n_names) % 2 == 0, 1.0, -1.0)
    # Equalize aggregate target notional by side so the completion-gap metric
    # measures pacing rather than a deliberately directional basket.
    notionals = target_abs * price
    long_total = float(np.sum(notionals[signs > 0]))
    short_total = float(np.sum(notionals[signs < 0]))
    if long_total > short_total:
        target_abs[signs > 0] *= short_total / long_total
    else:
        target_abs[signs < 0] *= long_total / short_total
    targets = signs * target_abs

    orders = pd.DataFrame({"target_shares": targets}, index=symbols)
    shape = (len(dates), n_names)
    price_matrix = np.tile(price, (len(dates), 1))
    adv_matrix = np.tile(adv, (len(dates), 1))
    open_matrix = np.ones(shape, dtype=bool)
    base_participation = np.full(shape, base_rate)
    panel = pd.DataFrame(index=pd.MultiIndex.from_product([dates, symbols]))
    context = PlannerContext(
        symbols=symbols,
        dates=dates,
        orders=orders,
        panel=panel,
        price=price_matrix,
        adv_shares=adv_matrix,
        is_open=open_matrix,
        base_participation=base_participation,
        event_days=event_days,
    )
    return context, announcement_dates


def planner_for(participation_model: ParticipationCapModel, ctx: PlannerContext) -> TradePlanner:
    n_names = len(ctx.symbols)
    market_beta = np.ones(n_names)
    covariance = 0.00025 * np.outer(market_beta, market_beta) + np.eye(n_names) * 0.00008
    return TradePlanner(
        TradePlannerConfig(
            participation_model=participation_model,
            risk_model=StaticCovarianceRiskModel(covariance=covariance),
            cost_model=CompositeCostModel(
                terms=(
                    QuadraticParticipationImpact(impact_bps_at_10pct_adv=5.0),
                    LinearBpsCost(bps=1.0),
                )
            ),
            constraints=default_constraints(),
            residual_risk_weight=1.0,
            solver="CLARABEL",
        )
    )


def metrics(ctx: PlannerContext, schedule: pd.DataFrame) -> dict[str, float]:
    ordered = schedule.copy()
    symbol_index = {symbol: i for i, symbol in enumerate(ctx.symbols)}
    ordered["symbol_index"] = ordered["symbol"].map(symbol_index)
    ordered["side"] = np.where(
        ordered["symbol_index"].map(lambda i: ctx.orders.iloc[i]["target_shares"]) > 0,
        "long",
        "short",
    )
    ordered["reference_notional"] = np.abs(ordered["trade_shares"]) * ordered["symbol_index"].map(
        lambda i: ctx.price[0, i]
    )
    finite = np.isfinite(ordered["days_to_earnings"].to_numpy(float))
    days = ordered["days_to_earnings"].to_numpy(float)
    notionals = ordered["reference_notional"].to_numpy(float)
    total = float(np.sum(notionals))
    near = finite & (days <= 2)
    event_weight = np.zeros_like(days)
    event_weight[finite] = np.exp(-days[finite] / 3.0)
    daily_gross = ordered.groupby("date")["reference_notional"].sum()
    completion = cumulative_side_completion(ctx, schedule)
    return {
        "pre_event_trade_pct": 100.0 * float(np.sum(notionals[finite])) / total,
        "near_event_trade_pct": 100.0 * float(np.sum(notionals[near])) / total,
        "event_risk_score_pct": 100.0 * float(np.sum(notionals * event_weight)) / total,
        "peak_daily_gross_pct": 100.0 * float(daily_gross.max()) / total,
        "max_side_gap_pp": float(np.max(np.abs(completion["long_short_gap_pp"]))),
    }


def run_experiment() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    ctx, announcements = synthetic_basket()
    fixed_rate = 0.10
    candidates = {
        "fixed_10pct_adv": ParticipationCapModel(
            modifiers=(
                AnnouncementParticipationModifier(
                    announcements,
                    curve=AnnouncementParticipationCurve(pre_rate=fixed_rate, post_rate=0.15),
                ),
            )
        ),
        "adaptive_uniform": ParticipationCapModel(
            modifiers=(
                AdaptiveAnnouncementParticipation(
                    risk_shape=0.0,
                    announcement_weight=1.0,
                    balance_sides=False,
                ),
            )
        ),
        "adaptive_risk_weighted": ParticipationCapModel(
            modifiers=(AdaptiveAnnouncementParticipation(balance_sides=False),)
        ),
        "adaptive_side_balanced": ParticipationCapModel(
            modifiers=(AdaptiveAnnouncementParticipation(balance_sides=True),)
        ),
    }
    rows = []
    completion_curves: dict[str, pd.DataFrame] = {}
    for name, model in candidates.items():
        try:
            result = planner_for(model, ctx).solve(ctx)
        except Exception as error:  # experiment records model failures as evidence
            rows.append({"candidate": name, "status": type(error).__name__})
            continue
        rows.append({"candidate": name, "status": "keepable", **metrics(ctx, result.schedule)})
        completion_curves[name] = cumulative_side_completion(ctx, result.schedule)
    return pd.DataFrame(rows).set_index("candidate"), completion_curves


def run() -> pd.DataFrame:
    return run_experiment()[0]


def plot_completion_comparison(
    results: pd.DataFrame,
    completion_curves: dict[str, pd.DataFrame],
    output: Path,
) -> None:
    """Plot fixed and retained adaptive cumulative long/short completion."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = [
        ("fixed_10pct_adv", "Fixed 10% ADV before announcement"),
        ("adaptive_side_balanced", "Adaptive side-balanced policy"),
    ]
    colors = {"long": "#2F6B9A", "short": "#D97732"}
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), sharex=True, sharey=True)
    for axis, (key, label) in zip(axes, policies):
        curve = completion_curves[key]
        axis.plot(
            curve.index,
            curve["cumulative_long_pct"],
            color=colors["long"],
            linewidth=2.4,
            marker="o",
            markersize=3.8,
            label="Long",
        )
        axis.plot(
            curve.index,
            curve["cumulative_short_pct"],
            color=colors["short"],
            linewidth=2.4,
            linestyle="--",
            marker="s",
            markersize=3.5,
            markerfacecolor="white",
            label="Short",
        )
        row = results.loc[key]
        axis.set_title(
            f"{label}\nNear-event flow {row['near_event_trade_pct']:.1f}% · "
            f"max side gap {row['max_side_gap_pp']:.1f} pp",
            fontsize=10.5,
            color="#2B2B2B",
            pad=12,
        )
        axis.set_ylim(0, 104)
        axis.grid(axis="y", color="#D9DEE3", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", rotation=35, labelsize=8)
        axis.tick_params(axis="y", labelsize=9)
        axis.set_xlabel("Planner date", color="#4A4A4A")

    axes[0].set_ylabel("Cumulative completion (% of side target gross)", color="#4A4A4A")
    axes[0].legend(frameon=False, loc="upper left", ncol=2)
    fig.suptitle(
        "Cumulative long and short gross-notional completion",
        x=0.065,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="semibold",
        color="#202124",
    )
    fig.text(
        0.065,
        0.925,
        "Synthetic 40-name basket · start-date prices · 15 planner dates · CLARABEL",
        ha="left",
        fontsize=9.5,
        color="#62676D",
    )
    fig.tight_layout(rect=(0.04, 0.03, 0.99, 0.89), w_pad=3.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/participation_refinement.png"),
    )
    args = parser.parse_args()
    experiment_results, curves = run_experiment()
    print(experiment_results.round(4).to_string())
    plot_completion_comparison(experiment_results, curves, args.output)
    metrics_output = args.output.with_name(f"{args.output.stem}_metrics.csv")
    curves_output = args.output.with_name(f"{args.output.stem}_cumulative.csv")
    experiment_results.to_csv(metrics_output)
    pd.concat(curves, names=["candidate", "date"]).to_csv(curves_output)
    print(f"\nchart: {args.output}")
    print(f"metrics: {metrics_output}")
    print(f"curves: {curves_output}")
