from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from experiments.alpha_decay_walkforward import synthetic_alpha_decay_bundle
from trade_planner import calibrate_alpha_decay_walk_forward


def test_walk_forward_calibration_recovers_signal_and_uncertainty() -> None:
    bundle = synthetic_alpha_decay_bundle()

    result = calibrate_alpha_decay_walk_forward(bundle)
    metrics = result.summary.iloc[0]

    assert result.audit.iloc[:4]["status"].eq("raw_fallback").all()
    assert result.audit.iloc[4:]["status"].eq("calibrated").all()
    assert metrics["rmse_improvement_fraction"] >= 0.10
    assert metrics["calibrated_sign_accuracy"] >= metrics["raw_sign_accuracy"]
    assert 0.65 <= metrics["predictive_interval_80_coverage"] <= 0.95
    assert result.coefficients["event_id"].nunique() == len(bundle.events) - 4
    np.testing.assert_allclose(
        result.events[0].ctx.expected_return,
        bundle.events[0].ctx.expected_return,
    )
    assert not np.allclose(
        result.events[4].ctx.expected_return,
        bundle.events[4].ctx.expected_return,
    )


def test_current_and_future_realizations_cannot_change_current_prediction() -> None:
    bundle = synthetic_alpha_decay_bundle()
    probe_index = 8
    probe_id = bundle.event_ids[probe_index]
    original = calibrate_alpha_decay_walk_forward(bundle)
    perturbed = replace(
        bundle,
        events=tuple(
            replace(
                event,
                realized_returns=(
                    np.asarray(event.realized_returns, dtype=float) + 0.10
                    if index >= probe_index
                    else event.realized_returns
                ),
            )
            for index, event in enumerate(bundle.events)
        ),
    )

    rerun = calibrate_alpha_decay_walk_forward(perturbed)
    columns = [
        "calibrated_directional_return",
        "predictive_uncertainty",
        "ridge_multiplier",
        "ridge_penalty",
    ]
    expected = original.predictions.loc[
        original.predictions["event_id"].eq(probe_id),
        columns,
    ].to_numpy(float)
    actual = rerun.predictions.loc[
        rerun.predictions["event_id"].eq(probe_id),
        columns,
    ].to_numpy(float)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)
    assert (
        original.audit.loc[
            original.audit["event_id"].eq(probe_id),
            "eligible_training_event_ids",
        ].iloc[0]
        == rerun.audit.loc[
            rerun.audit["event_id"].eq(probe_id),
            "eligible_training_event_ids",
        ].iloc[0]
    )


def test_outcome_not_available_by_cutoff_is_excluded_from_training() -> None:
    bundle = synthetic_alpha_decay_bundle(n_events=7)
    delayed_index = 4
    current_index = 5
    current_cutoff = pd.Timestamp(bundle.events[current_index].information_cutoff)
    delayed_event = replace(
        bundle.events[delayed_index],
        realized_available_at=current_cutoff + pd.Timedelta(days=2),
    )
    delayed_bundle = replace(
        bundle,
        events=tuple(
            delayed_event if index == delayed_index else event
            for index, event in enumerate(bundle.events)
        ),
    )

    result = calibrate_alpha_decay_walk_forward(delayed_bundle)
    current_id = bundle.event_ids[current_index]
    audit = result.audit.loc[result.audit["event_id"].eq(current_id)].iloc[0]

    assert audit["status"] == "calibrated"
    assert audit["eligible_training_event_count"] == 4
    assert bundle.event_ids[delayed_index] not in audit["eligible_training_event_ids"]
    assert pd.Timestamp(audit["latest_training_realized_available_at"]) <= current_cutoff


def test_unseen_classifications_use_training_schema_without_failure() -> None:
    bundle = synthetic_alpha_decay_bundle()

    result = calibrate_alpha_decay_walk_forward(bundle)
    last = result.predictions.loc[
        result.predictions["event_id"].eq(bundle.event_ids[-1])
    ]

    assert last["country"].eq("AU").any()
    assert np.isfinite(
        last[["calibrated_directional_return", "predictive_uncertainty"]].to_numpy(
            float
        )
    ).all()


def test_zero_target_is_audited_but_not_calibrated() -> None:
    bundle = synthetic_alpha_decay_bundle()
    zero_symbol = bundle.events[4].ctx.symbols[0]
    zero_events = list(bundle.events)
    event = zero_events[4]
    orders = event.ctx.orders.copy()
    orders.loc[zero_symbol, "target_shares"] = 0.0
    raw_return = np.asarray(event.ctx.expected_return, dtype=float).copy()
    zero_events[4] = replace(event, ctx=replace(event.ctx, orders=orders))

    result = calibrate_alpha_decay_walk_forward(
        replace(bundle, events=tuple(zero_events))
    )
    calibrated = result.events[4]
    symbol_index = calibrated.ctx.symbols.index(zero_symbol)
    prediction_rows = result.predictions.loc[
        result.predictions["event_id"].eq(bundle.event_ids[4])
        & result.predictions["symbol"].eq(zero_symbol)
    ]

    np.testing.assert_allclose(
        calibrated.ctx.expected_return[:, symbol_index],
        raw_return[:, symbol_index],
    )
    assert len(prediction_rows) == len(calibrated.ctx.dates)
    assert prediction_rows["target_sign"].eq(0.0).all()
