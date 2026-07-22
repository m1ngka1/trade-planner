"""Spent-development test of systematic-first inventory-risk pricing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.alpha_confidence_walkforward import (
    _build_event_with_truth,
    _calibrated_alpha_uncertainty,
)
from experiments.liquidity_forecast_walkforward import (
    run_experiment,
    specific_risk_fraction_for_risk_profile,
)
from experiments.raw_alpha_opportunity_walkforward import (
    RAW_ALPHA_DEVELOPMENT_EVENTS,
    RAW_ALPHA_EVENT_INDEX_OFFSET,
    RAW_ALPHA_EVENT_SEEDS,
    RAW_ALPHA_LIQUIDITY_SEEDS,
    RAW_ALPHA_SCENARIO_SEEDS,
)
from experiments.rebalance_economic_calibration import economic_fixture
from trade_planner import RiskAversion


PNL_RECOVERY_MATERIALITY_BPS = 0.50
CONTROL_VOLATILITY_TOLERANCE_BPS = 1.00
CONTROL_FACTOR_TOLERANCE_PP = 0.75
CONTROL_RAMP_RETENTION = 0.90


def run_systematic_first_risk_experiment(
    *,
    solver: str = "OSQP",
    n_events: int = RAW_ALPHA_DEVELOPMENT_EVENTS,
    event_start: int = 0,
    risk_aversion: str = "medium",
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Compare full and profile-scaled specific risk on spent events only."""

    if event_start + n_events > RAW_ALPHA_DEVELOPMENT_EVENTS:
        raise ValueError(
            "systematic-first mechanics may use only spent events 97-108; "
            "events 109-120 remain sealed"
        )
    parsed_aversion = RiskAversion.parse(risk_aversion)
    scaled_specific_fraction = specific_risk_fraction_for_risk_profile(
        parsed_aversion
    )
    common = {
        "solver": solver,
        "n_events": n_events,
        "event_start": event_start,
        "risk_aversion": parsed_aversion.value,
        "coefficient_policy": "baseline_locked",
        "alpha_policy": "raw",
        "factor_policy": "minimax_factor_stress",
        "liquidity_shape_policy": "risk_scaled",
        "event_seeds": RAW_ALPHA_EVENT_SEEDS,
        "scenario_seeds": RAW_ALPHA_SCENARIO_SEEDS,
        "realized_liquidity_seeds": RAW_ALPHA_LIQUIDITY_SEEDS,
        "event_index_offset": RAW_ALPHA_EVENT_INDEX_OFFSET,
        "development_event_count": RAW_ALPHA_DEVELOPMENT_EVENTS,
    }
    control, control_metadata = run_experiment(
        **common,
        specific_risk_fraction=1.0,
    )
    challenger, challenger_metadata = run_experiment(
        **common,
        specific_risk_fraction=scaled_specific_fraction,
    )

    combined = _combine_outputs(control, challenger)
    combined["risk_decomposition"] = _risk_decomposition(
        combined["schedules"],
        event_start=event_start,
        n_events=n_events,
        challenger_specific_fraction=scaled_specific_fraction,
    )
    gates = _systematic_first_gates(combined, challenger["gates"])
    passed = bool(gates["passed"].all())
    decision = "keep_for_real_replay" if passed else "discard"
    reason = (
        "All predeclared systematic-first economics, swing, behavior, "
        "liquidity, and hard-execution gates passed."
        if passed
        else "Failed: "
        + ", ".join(gates.loc[~gates["passed"], "gate"].astype(str))
        + "."
    )
    combined["gates"] = gates
    combined["summary"].loc[
        combined["summary"]["strategy"].eq("systematic_first"),
        ["decision", "decision_reason"],
    ] = [decision, reason]
    combined["summary"].loc[
        combined["summary"]["strategy"].eq("full_specific_control"),
        ["decision", "decision_reason"],
    ] = ["control", "Current 100%-specific-risk challenger."]

    metadata: dict[str, object] = {
        "decision": decision,
        "decision_reason": reason,
        "solver": solver,
        "n_events": n_events,
        "event_start": event_start,
        "risk_aversion": parsed_aversion.value,
        "control_specific_risk_fraction": 1.0,
        "challenger_specific_risk_fraction": scaled_specific_fraction,
        "protected_event_start": RAW_ALPHA_DEVELOPMENT_EVENTS,
        "sealed_events_untouched": event_start + n_events
        <= RAW_ALPHA_DEVELOPMENT_EVENTS,
        "control_metadata_decision": control_metadata["decision"],
        "challenger_standard_decision": challenger_metadata["decision"],
    }
    return combined, metadata


def _combine_outputs(
    control: dict[str, pd.DataFrame],
    challenger: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for name in (
        "trials",
        "schedules",
        "daily",
        "profiles",
        "exposures",
        "coefficients",
        "frontiers",
    ):
        outputs[name] = pd.concat(
            [
                _strategy_slice(
                    control[name], "static_open_loop", "static_open_loop"
                ),
                _strategy_slice(
                    control[name], "forecast_liquidity", "full_specific_control"
                ),
                _strategy_slice(
                    challenger[name], "forecast_liquidity", "systematic_first"
                ),
            ],
            ignore_index=True,
        )
    outputs["summary"] = pd.concat(
        [
            _strategy_slice(
                control["summary"], "static_open_loop", "static_open_loop"
            ),
            _strategy_slice(
                control["summary"], "forecast_liquidity", "full_specific_control"
            ),
            _strategy_slice(
                challenger["summary"], "forecast_liquidity", "systematic_first"
            ),
        ],
        ignore_index=True,
    )
    paired = challenger["paired"].copy()
    control_trial = control["trials"].loc[
        control["trials"]["strategy"].eq("forecast_liquidity"),
        ["event_id", "net_pnl_bps", "net_pnl_dollars"],
    ].rename(
        columns={
            "net_pnl_bps": "control_net_pnl_bps",
            "net_pnl_dollars": "control_net_pnl_dollars",
        }
    )
    challenger_trial = challenger["trials"].loc[
        challenger["trials"]["strategy"].eq("forecast_liquidity"),
        ["event_id", "net_pnl_bps", "net_pnl_dollars"],
    ].rename(
        columns={
            "net_pnl_bps": "challenger_net_pnl_bps",
            "net_pnl_dollars": "challenger_net_pnl_dollars",
        }
    )
    paired = paired.merge(control_trial, on="event_id", validate="one_to_one")
    paired = paired.merge(
        challenger_trial,
        on="event_id",
        validate="one_to_one",
    )
    paired["pnl_recovery_vs_control_bps"] = (
        paired["challenger_net_pnl_bps"] - paired["control_net_pnl_bps"]
    )
    paired["pnl_recovery_vs_control_dollars"] = (
        paired["challenger_net_pnl_dollars"]
        - paired["control_net_pnl_dollars"]
    )
    outputs["paired"] = paired
    outputs["liquidity"] = challenger["liquidity"].copy()
    outputs["control_gates"] = control["gates"].copy()
    return outputs


def _strategy_slice(
    frame: pd.DataFrame,
    source: str,
    target: str,
) -> pd.DataFrame:
    selected = frame.loc[frame["strategy"].eq(source)].copy()
    selected["strategy"] = target
    if "selected_plan_source" in selected:
        selected["selected_plan_source"] = target
    return selected


def _systematic_first_gates(
    outputs: dict[str, pd.DataFrame],
    standard_gates: pd.DataFrame,
) -> pd.DataFrame:
    summary = outputs["summary"].set_index("strategy")
    baseline = summary.loc["static_open_loop"]
    control = summary.loc["full_specific_control"]
    challenger = summary.loc["systematic_first"]
    recovery = float(
        challenger["mean_net_pnl_bps"] - control["mean_net_pnl_bps"]
    )
    incremental = pd.DataFrame(
        [
            (
                "pnl_recovery_vs_full_specific",
                recovery >= PNL_RECOVERY_MATERIALITY_BPS,
                PNL_RECOVERY_MATERIALITY_BPS,
                recovery,
                "challenger improves mean P&L by at least 0.50 bp/event",
            ),
            (
                "volatility_within_1bp_of_control",
                challenger["pnl_vol_bps"]
                <= control["pnl_vol_bps"] + CONTROL_VOLATILITY_TOLERANCE_BPS,
                control["pnl_vol_bps"] + CONTROL_VOLATILITY_TOLERANCE_BPS,
                challenger["pnl_vol_bps"],
                "challenger P&L volatility <= control + 1.00 bp",
            ),
            (
                "factor_within_075pp_of_control",
                challenger["mean_early_factor_imbalance_pct"]
                <= control["mean_early_factor_imbalance_pct"]
                + CONTROL_FACTOR_TOLERANCE_PP,
                control["mean_early_factor_imbalance_pct"]
                + CONTROL_FACTOR_TOLERANCE_PP,
                challenger["mean_early_factor_imbalance_pct"],
                "challenger early factor imbalance <= control + 0.75 pp",
            ),
            (
                "ramp_preserves_90pct_of_control",
                challenger["mean_late_early_gross_ratio"]
                >= CONTROL_RAMP_RETENTION
                * control["mean_late_early_gross_ratio"],
                CONTROL_RAMP_RETENTION
                * control["mean_late_early_gross_ratio"],
                challenger["mean_late_early_gross_ratio"],
                "challenger late/early ratio >= 90% of control",
            ),
            (
                "volatility_below_static_by_005bp",
                challenger["pnl_vol_bps"] <= baseline["pnl_vol_bps"] - 0.05,
                baseline["pnl_vol_bps"] - 0.05,
                challenger["pnl_vol_bps"],
                "challenger P&L volatility <= baseline - 0.05 bp",
            ),
        ],
        columns=("gate", "passed", "baseline_or_limit", "candidate", "criterion"),
    )
    gates = pd.concat(
        [standard_gates.copy(), incremental],
        ignore_index=True,
    )
    gates["passed"] = gates["passed"].astype(bool)
    return gates


def _risk_decomposition(
    schedules: pd.DataFrame,
    *,
    event_start: int,
    n_events: int,
    challenger_specific_fraction: float,
) -> pd.DataFrame:
    base_ctx, _ = economic_fixture()
    alpha_uncertainty = _calibrated_alpha_uncertainty(base_ctx)
    rows: list[dict[str, object]] = []
    selected = zip(
        RAW_ALPHA_EVENT_SEEDS[event_start : event_start + n_events],
        RAW_ALPHA_SCENARIO_SEEDS[event_start : event_start + n_events],
    )
    for cohort_index, (event_seed, scenario_seed) in enumerate(
        selected,
        start=event_start,
    ):
        event, _, _ = _build_event_with_truth(
            base_ctx,
            alpha_uncertainty,
            RAW_ALPHA_EVENT_INDEX_OFFSET + cohort_index,
            event_seed,
            scenario_seed,
        )
        ctx = event.ctx
        for strategy in (
            "static_open_loop",
            "full_specific_control",
            "systematic_first",
        ):
            schedule = schedules.loc[
                schedules["event_id"].eq(event.event_id)
                & schedules["strategy"].eq(strategy)
            ].copy()
            schedule["date"] = pd.to_datetime(schedule["date"])
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
            specific_fraction = (
                challenger_specific_fraction
                if strategy == "systematic_first"
                else 1.0
            )
            for date_index, date in enumerate(ctx.dates):
                position_dollars = ctx.price[date_index] * cumulative[date_index]
                factor_dollars = (
                    ctx.factor_exposure[date_index].T @ position_dollars
                )
                raw_factor = float(
                    factor_dollars
                    @ ctx.factor_covariance[date_index]
                    @ factor_dollars
                )
                raw_specific = float(
                    np.sum(
                        ctx.specific_variance[date_index]
                        * np.square(position_dollars)
                    )
                )
                rows.append(
                    {
                        "event_id": event.event_id,
                        "date": pd.Timestamp(date),
                        "day_index": date_index + 1,
                        "strategy": strategy,
                        "specific_risk_fraction": specific_fraction,
                        "raw_factor_variance_dollars2": raw_factor,
                        "raw_specific_variance_dollars2": raw_specific,
                        "priced_specific_variance_dollars2": (
                            specific_fraction * raw_specific
                        ),
                        "total_priced_variance_dollars2": (
                            raw_factor + specific_fraction * raw_specific
                        ),
                        "raw_specific_variance_share": raw_specific
                        / max(raw_factor + raw_specific, 1e-12),
                    }
                )
    return pd.DataFrame(rows)


def plot_systematic_first_results(
    outputs: dict[str, pd.DataFrame],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "static_open_loop": "#8A929A",
        "full_specific_control": "#2F6B9A",
        "systematic_first": "#D17C2F",
    }
    labels = {
        "static_open_loop": "Flat ADV baseline",
        "full_specific_control": "100% specific risk",
        "systematic_first": "Systematic first",
    }
    trials = outputs["trials"]
    summary = outputs["summary"].set_index("strategy")
    paired = outputs["paired"].sort_values("as_of")
    profiles = outputs["profiles"]
    decomposition = outputs["risk_decomposition"]
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.2))

    axis = axes[0, 0]
    for strategy, group in trials.groupby("strategy", sort=False):
        ordered = group.sort_values("as_of")
        axis.plot(
            np.arange(1, len(ordered) + 1),
            ordered["net_pnl_bps"].cumsum(),
            marker="o",
            linewidth=2,
            color=colors[strategy],
            label=labels[strategy],
        )
    axis.axhline(0.0, color="#59636E", linewidth=0.8)
    axis.set_title("Cumulative realized net P&L")
    axis.set_ylabel("Cumulative bps of parent gross")
    axis.legend(frameon=False)

    axis = axes[0, 1]
    axis.bar(
        np.arange(1, len(paired) + 1),
        paired["pnl_recovery_vs_control_bps"],
        color=np.where(
            paired["pnl_recovery_vs_control_bps"] >= 0.0,
            "#70A288",
            "#B04A4A",
        ),
    )
    axis.axhline(0.0, color="#59636E", linewidth=0.8)
    axis.set_title("P&L recovered vs 100% specific risk")
    axis.set_ylabel("bp per event")

    axis = axes[0, 2]
    measures = [
        "pnl_vol_bps",
        "loss_cvar_95_bps",
        "mean_within_event_drawdown_bps",
    ]
    x = np.arange(len(measures))
    width = 0.25
    for offset, strategy in zip((-width, 0.0, width), colors):
        axis.bar(
            x + offset,
            [summary.loc[strategy, measure] for measure in measures],
            width,
            color=colors[strategy],
            label=labels[strategy],
        )
    axis.set_xticks(x, ["P&L vol", "Loss CVaR", "Within-event DD"])
    axis.set_title("Realized swing and downside")
    axis.set_ylabel("bps")

    axis = axes[1, 0]
    risk_mean = decomposition.groupby("strategy").agg(
        factor=("raw_factor_variance_dollars2", "mean"),
        specific=("raw_specific_variance_dollars2", "mean"),
        priced_specific=("priced_specific_variance_dollars2", "mean"),
    )
    order = list(colors)
    positions = np.arange(len(order))
    factor = risk_mean.loc[order, "factor"] / 1e12
    priced_specific = risk_mean.loc[order, "priced_specific"] / 1e12
    axis.bar(positions, factor, color="#537895", label="Factor variance")
    axis.bar(
        positions,
        priced_specific,
        bottom=factor,
        color="#D6A45B",
        label="Priced specific variance",
    )
    axis.set_xticks(positions, ["Baseline", "Control", "Systematic\nfirst"])
    axis.set_title("Average inventory risk priced by objective")
    axis.set_ylabel("$tn variance")
    axis.legend(frameon=False)

    axis = axes[1, 1]
    mean_profiles = profiles.groupby(
        ["strategy", "day_index"], as_index=False
    )["daily_gross_pct"].mean()
    for strategy, group in mean_profiles.groupby("strategy", sort=False):
        axis.plot(
            group["day_index"],
            group["daily_gross_pct"],
            marker="o",
            linewidth=2,
            color=colors[strategy],
            label=labels[strategy],
        )
    axis.set_title("Optimizer-derived daily volume")
    axis.set_xlabel("Planner day")
    axis.set_ylabel("Parent gross traded (%)")

    axis = axes[1, 2]
    factor_balance = summary.loc[order, "mean_early_factor_imbalance_pct"]
    axis.bar(positions, factor_balance, color=[colors[item] for item in order])
    axis.set_xticks(positions, ["Baseline", "Control", "Systematic\nfirst"])
    axis.set_title("Mean early factor imbalance")
    axis.set_ylabel("% of parent gross")

    fig.suptitle(
        "Systematic-first risk: preserve factor balance without overpricing diversified risk",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="OSQP")
    parser.add_argument("--n-events", type=int, default=RAW_ALPHA_DEVELOPMENT_EVENTS)
    parser.add_argument("--event-start", type=int, default=0)
    parser.add_argument(
        "--risk-aversion",
        choices=("high", "medium", "low"),
        default="medium",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("artifacts/systematic_first_risk_spent"),
    )
    args = parser.parse_args()
    outputs, metadata = run_systematic_first_risk_experiment(
        solver=args.solver,
        n_events=args.n_events,
        event_start=args.event_start,
        risk_aversion=args.risk_aversion,
    )
    prefix: Path = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(prefix.with_name(prefix.name + f"_{name}.csv"), index=False)
    chart = prefix.with_name(prefix.name + ".png")
    plot_systematic_first_results(outputs, chart)
    print(outputs["summary"].round(4).to_string(index=False))
    print("\nAcceptance gates:")
    print(outputs["gates"].to_string(index=False))
    print(f"\ndecision: {metadata['decision']}")
    print(f"reason: {metadata['decision_reason']}")
    print(
        "specific-risk fraction: "
        f"{100.0 * metadata['challenger_specific_risk_fraction']:.1f}%"
    )
    print(f"sealed events untouched: {metadata['sealed_events_untouched']}")
    print(f"artifacts: {prefix.parent / (prefix.name + '*')}")


if __name__ == "__main__":
    main()
