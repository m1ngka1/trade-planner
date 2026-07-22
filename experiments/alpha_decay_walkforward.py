"""Controlled mechanics screen for point-in-time alpha-decay calibration."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from trade_planner import (
    HistoricalReplayBundle,
    PlannerContext,
    PointInTimeRebalanceEvent,
    calibrate_alpha_decay_walk_forward,
)


N_EVENTS = 14
N_DATES = 6
SEED = 20260723


def synthetic_alpha_decay_bundle(
    *,
    n_events: int = N_EVENTS,
    seed: int = SEED,
) -> HistoricalReplayBundle:
    """Build deterministic events with a known conditional holding-alpha path."""

    if n_events < 6:
        raise ValueError("n_events must be at least six")
    rng = np.random.default_rng(seed)
    symbols = [f"NAME_{index:02d}" for index in range(8)]
    base_sign = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    countries = np.array(["HK", "US", "JP", "SG", "HK", "US", "JP", "SG"])
    sectors = np.array(
        ["technology", "financials", "healthcare", "industrials"] * 2
    )
    industries = np.array(
        ["software", "banks", "biotech", "capital_goods"] * 2
    )
    urgencies = np.array(
        ["urgent", "urgent", "medium", "medium", "small", "small", "medium", "urgent"]
    )
    confidence = np.array([0.95, 0.90, 0.80, 0.75, 0.60, 0.65, 0.85, 0.92])
    crowding = np.array([0.2, 0.8, 0.4, 0.6, 0.1, 0.9, 0.3, 0.7])
    price_by_name = np.array([48.0, 72.0, 55.0, 91.0, 64.0, 83.0, 44.0, 106.0])
    adv_by_name = np.array(
        [1_200_000, 900_000, 1_500_000, 1_100_000, 1_300_000, 850_000, 1_650_000, 950_000],
        dtype=float,
    )
    urgency_capacity_days = {"urgent": 4.2, "medium": 2.3, "small": 0.7}
    progress = np.arange(N_DATES, dtype=float) / (N_DATES - 1)

    events: list[PointInTimeRebalanceEvent] = []
    classifications: dict[str, pd.DataFrame] = {}
    forecast_quantiles: dict[str, dict[float, np.ndarray]] = {}
    first_cutoff = pd.Timestamp("2020-01-02 16:00")
    for event_index in range(n_events):
        event_id = f"alpha_event_{event_index + 1:02d}"
        cutoff = first_cutoff + pd.offsets.BDay(10 * event_index)
        dates = pd.bdate_range(cutoff.normalize() + pd.offsets.BDay(1), periods=N_DATES)
        event_sign = base_sign * (-1.0 if event_index % 3 == 2 else 1.0)
        rebalance_type = "add_delete" if event_index % 2 == 0 else "weight_change"
        event_countries = countries.copy()
        event_sectors = sectors.copy()
        event_industries = industries.copy()
        if event_index == n_events - 1:
            # The final event deliberately contains labels absent from training.
            event_countries[0] = "AU"
            event_sectors[0] = "energy"
            event_industries[0] = "oil_gas"
        targets = np.asarray(
            [
                event_sign[index]
                * urgency_capacity_days[str(urgencies[index])]
                * 0.10
                * adv_by_name[index]
                for index in range(len(symbols))
            ],
            dtype=float,
        )
        order_frame = pd.DataFrame(
            {"target_shares": targets},
            index=pd.Index(symbols, name="symbol"),
        )
        classification = pd.DataFrame(
            {
                "country": event_countries,
                "sector": event_sectors,
                "industry": event_industries,
                "urgency": urgencies,
                "rebalance_type": rebalance_type,
                "prediction_confidence": confidence,
                "crowding": crowding,
            },
            index=pd.Index(symbols, name="symbol"),
        )
        classifications[event_id] = classification

        event_adv = adv_by_name * (1.0 + 0.025 * np.sin(event_index))
        adv = np.tile(event_adv[None, :], (N_DATES, 1))
        price = np.tile(price_by_name[None, :], (N_DATES, 1))
        participation = np.full_like(adv, 0.10)
        is_open = np.ones_like(adv, dtype=bool)
        capacity_pressure = np.abs(targets) / np.sum(participation * adv, axis=0)
        side_effect = np.where(event_sign > 0, 0.000025, -0.000020)
        country_effect = np.asarray(
            [{"HK": 0.000035, "US": -0.000025, "JP": 0.000020, "SG": 0.0}.get(value, 0.0) for value in event_countries]
        )
        sector_effect = np.asarray(
            [
                {
                    "technology": 0.000055,
                    "financials": -0.000045,
                    "healthcare": 0.000025,
                    "industrials": -0.000010,
                }.get(value, 0.0)
                for value in event_sectors
            ]
        )
        urgency_effect = np.asarray(
            [{"urgent": 0.000045, "medium": 0.0, "small": -0.000035}[value] for value in urgencies]
        )
        type_effect = 0.000020 if rebalance_type == "add_delete" else -0.000010
        conditional_level = (
            side_effect
            + country_effect
            + sector_effect
            + urgency_effect
            + 0.000030 * (confidence - 0.75)
            - 0.000025 * crowding
            + 0.000025 * capacity_pressure
            + type_effect
        )
        true_directional = (
            -0.000105
            + 0.000310 * progress[:, None]
            + 0.000090 * np.square(progress[:, None])
            + conditional_level[None, :]
        )
        raw_directional = (
            0.000025
            + 0.45 * true_directional
            + 0.000075 * (1.0 - progress[:, None])
            + rng.normal(0.0, 0.000060, size=true_directional.shape)
        )
        realized_directional = true_directional + rng.normal(
            0.0,
            0.000090,
            size=true_directional.shape,
        )
        expected_return = raw_directional * event_sign[None, :]
        realized_return = realized_directional * event_sign[None, :]
        event_days = pd.DataFrame(
            np.tile(np.arange(N_DATES - 1, -1, -1)[:, None], (1, len(symbols))),
            index=dates,
            columns=symbols,
            dtype=float,
        )
        shape = (N_DATES, len(symbols))
        ctx = PlannerContext(
            symbols=symbols,
            dates=dates,
            orders=order_frame,
            panel=pd.DataFrame(
                index=pd.MultiIndex.from_product(
                    [dates, symbols],
                    names=["date", "symbol"],
                )
            ),
            price=price,
            adv_shares=adv,
            is_open=is_open,
            base_participation=participation,
            event_days=event_days,
            expected_return=expected_return,
            expected_return_uncertainty=np.full(shape, 0.000150),
            impact_bps_at_10pct_adv=np.full(shape, 8.0),
            linear_cost_bps=np.full(shape, 1.0),
            metadata={
                "historical_bundle_role": "development",
                "historical_event_id": event_id,
                "information_cutoff": cutoff.isoformat(),
                "controlled_mechanics_fixture": True,
            },
        )
        event = PointInTimeRebalanceEvent(
            event_id=event_id,
            as_of=cutoff,
            information_cutoff=cutoff,
            ctx=ctx,
            realized_returns=realized_return,
            realized_impact_bps_at_10pct_adv=np.full(shape, 8.5),
            realized_linear_cost_bps=np.full(shape, 1.1),
            realized_available_at=dates[-1] + pd.Timedelta(hours=20),
            realized_adv_shares=adv * (
                0.92 + rng.uniform(0.0, 0.16, size=shape)
            ),
        )
        events.append(event)
        forecast_quantiles[event_id] = {
            0.10: adv * 0.80,
            0.25: adv * 0.90,
            0.50: adv.copy(),
        }
    return HistoricalReplayBundle(
        role="development",
        events=tuple(events),
        classifications=classifications,
        forecast_adv_quantiles=forecast_quantiles,
        source_hashes={"controlled_fixture": f"seed-{seed}"},
    )


def run_alpha_decay_mechanics(
    *,
    full_suite_verified: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Run the predeclared recovery, chronology, and robustness checks."""

    bundle = synthetic_alpha_decay_bundle()
    result = calibrate_alpha_decay_walk_forward(bundle)
    summary = result.summary.copy()
    metrics = summary.iloc[0]
    probe_index = 8
    probe_event_id = bundle.event_ids[probe_index]
    perturbed_events = tuple(
        replace(
            event,
            realized_returns=(
                np.asarray(event.realized_returns, dtype=float) + 0.05
                if index >= probe_index
                else event.realized_returns
            ),
        )
        for index, event in enumerate(bundle.events)
    )
    perturbed = calibrate_alpha_decay_walk_forward(
        replace(bundle, events=perturbed_events)
    )
    probe_columns = [
        "calibrated_directional_return",
        "predictive_uncertainty",
        "ridge_multiplier",
        "ridge_penalty",
    ]
    original_probe = result.predictions.loc[
        result.predictions["event_id"].eq(probe_event_id),
        probe_columns,
    ].reset_index(drop=True)
    perturbed_probe = perturbed.predictions.loc[
        perturbed.predictions["event_id"].eq(probe_event_id),
        probe_columns,
    ].reset_index(drop=True)
    leakage_pass = bool(
        np.allclose(
            original_probe.to_numpy(float),
            perturbed_probe.to_numpy(float),
            rtol=0.0,
            atol=1e-15,
            equal_nan=True,
        )
        and result.audit.loc[
            result.audit["event_id"].eq(probe_event_id),
            "eligible_training_event_ids",
        ].iloc[0]
        == perturbed.audit.loc[
            perturbed.audit["event_id"].eq(probe_event_id),
            "eligible_training_event_ids",
        ].iloc[0]
    )
    last_event = bundle.event_ids[-1]
    unseen_rows = result.predictions.loc[result.predictions["event_id"].eq(last_event)]
    unseen_pass = bool(
        unseen_rows["country"].eq("AU").any()
        and np.isfinite(
            unseen_rows[
                ["calibrated_directional_return", "predictive_uncertainty"]
            ].to_numpy(float)
        ).all()
    )
    warmup = result.audit.iloc[:4]
    fallback_pass = bool(warmup["status"].eq("raw_fallback").all())
    gate_rows = [
        {
            "gate": "calibrated_rmse_improves_at_least_10pct",
            "threshold": ">= 0.10",
            "observed": float(metrics["rmse_improvement_fraction"]),
            "passed": bool(metrics["rmse_improvement_fraction"] >= 0.10),
        },
        {
            "gate": "directional_sign_accuracy_not_lower",
            "threshold": "calibrated >= raw",
            "observed": float(
                metrics["calibrated_sign_accuracy"] - metrics["raw_sign_accuracy"]
            ),
            "passed": bool(
                metrics["calibrated_sign_accuracy"] >= metrics["raw_sign_accuracy"]
            ),
        },
        {
            "gate": "predictive_interval_80_coverage",
            "threshold": "0.65 to 0.95",
            "observed": float(metrics["predictive_interval_80_coverage"]),
            "passed": bool(
                0.65 <= metrics["predictive_interval_80_coverage"] <= 0.95
            ),
        },
        {
            "gate": "current_and_future_realized_returns_are_inaccessible",
            "threshold": "exactly invariant",
            "observed": leakage_pass,
            "passed": leakage_pass,
        },
        {
            "gate": "unseen_classifications_predict",
            "threshold": "finite without refit",
            "observed": unseen_pass,
            "passed": unseen_pass,
        },
        {
            "gate": "four_event_warmup_uses_raw_forecast",
            "threshold": "first four events",
            "observed": fallback_pass,
            "passed": fallback_pass,
        },
        {
            "gate": "full_repository_test_suite",
            "threshold": "all tests pass",
            "observed": bool(full_suite_verified),
            "passed": bool(full_suite_verified),
        },
    ]
    gates = pd.DataFrame(gate_rows)
    decision = "keep" if bool(gates["passed"].all()) else "pending_or_discard"
    reason = (
        "All predeclared estimator mechanics and repository checks passed."
        if decision == "keep"
        else "At least one predeclared estimator or repository gate is not verified."
    )
    summary["decision"] = decision
    summary["decision_reason"] = reason
    outputs = {
        "summary": summary,
        "gates": gates,
        "audit": result.audit,
        "predictions": result.predictions,
        "coefficients": result.coefficients,
    }
    metadata: dict[str, object] = {
        "decision": decision,
        "decision_reason": reason,
        "cohort_role": "controlled_mechanics_only",
        "event_count": len(bundle.events),
        "warmup_event_count": 4,
        "seed": SEED,
        "probe_event_id": probe_event_id,
        "unseen_label_event_id": last_event,
        "full_suite_verified": bool(full_suite_verified),
    }
    return outputs, metadata


def plot_alpha_decay_results(outputs: dict[str, pd.DataFrame], output: Path) -> None:
    """Render accuracy, timing-shape, and calibration diagnostics."""

    import matplotlib.pyplot as plt

    predictions = outputs["predictions"]
    calibrated = predictions.loc[predictions["status"].eq("calibrated")].copy()
    per_event = (
        calibrated.assign(
            raw_squared=np.square(
                calibrated["raw_directional_return"]
                - calibrated["realized_directional_return"]
            ),
            calibrated_squared=np.square(
                calibrated["calibrated_directional_return"]
                - calibrated["realized_directional_return"]
            ),
        )
        .groupby("event_id", sort=False)[["raw_squared", "calibrated_squared"]]
        .mean()
        .pow(0.5)
    )
    by_progress = calibrated.groupby("progress", sort=True)[
        [
            "realized_directional_return",
            "raw_directional_return",
            "calibrated_directional_return",
        ]
    ].mean()
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(per_event.index, 10_000 * per_event["raw_squared"], marker="o", label="raw")
    axes[0, 0].plot(
        per_event.index,
        10_000 * per_event["calibrated_squared"],
        marker="o",
        label="calibrated",
    )
    axes[0, 0].set_title("Post-warmup directional RMSE by event")
    axes[0, 0].set_ylabel("basis points")
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].legend()

    for column, label in (
        ("realized_directional_return", "realized"),
        ("raw_directional_return", "raw"),
        ("calibrated_directional_return", "calibrated"),
    ):
        axes[0, 1].plot(
            by_progress.index,
            10_000 * by_progress[column],
            marker="o",
            label=label,
        )
    axes[0, 1].set_title("Learned holding-alpha shape")
    axes[0, 1].set_xlabel("horizon progress")
    axes[0, 1].set_ylabel("directional return (bp/day)")
    axes[0, 1].legend()

    realized = 10_000 * calibrated["realized_directional_return"].to_numpy(float)
    axes[1, 0].scatter(
        10_000 * calibrated["raw_directional_return"],
        realized,
        s=12,
        alpha=0.35,
        label="raw",
    )
    axes[1, 0].scatter(
        10_000 * calibrated["calibrated_directional_return"],
        realized,
        s=12,
        alpha=0.35,
        label="calibrated",
    )
    bounds = np.nanmax(np.abs(axes[1, 0].get_xlim() + axes[1, 0].get_ylim()))
    axes[1, 0].plot([-bounds, bounds], [-bounds, bounds], color="black", linewidth=1)
    axes[1, 0].set_title("Prediction versus realized directional return")
    axes[1, 0].set_xlabel("prediction (bp/day)")
    axes[1, 0].set_ylabel("realized (bp/day)")
    axes[1, 0].legend()

    audit = outputs["audit"].loc[outputs["audit"]["status"].eq("calibrated")]
    axes[1, 1].step(
        audit["event_id"],
        audit["ridge_multiplier"],
        where="mid",
        marker="o",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Automatically selected ridge multiplier")
    axes[1, 1].set_ylabel("design-scale multiplier")
    axes[1, 1].tick_params(axis="x", rotation=45)
    figure.suptitle(
        "Point-in-time alpha-decay calibration — controlled mechanics only",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("artifacts/alpha_decay_mechanics"),
    )
    parser.add_argument(
        "--full-suite-verified",
        action="store_true",
        help="Record that the repository test suite was separately run and passed.",
    )
    args = parser.parse_args()
    outputs, metadata = run_alpha_decay_mechanics(
        full_suite_verified=args.full_suite_verified
    )
    prefix: Path = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(prefix.with_name(prefix.name + f"_{name}.csv"), index=False)
    chart = prefix.with_suffix(".png")
    plot_alpha_decay_results(outputs, chart)
    print(outputs["summary"].round(6).to_string(index=False))
    print("\nAcceptance gates:")
    print(outputs["gates"].to_string(index=False))
    print(f"\ndecision: {metadata['decision']}")
    print(f"reason: {metadata['decision_reason']}")
    print("This controlled population is not evidence of real profitability.")
    print(f"artifacts: {prefix.parent / (prefix.name + '*')}")


if __name__ == "__main__":
    main()
