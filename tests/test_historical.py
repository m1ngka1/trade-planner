from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.historical_replay import run_historical_experiment
from experiments.historical_policy_panel import run_historical_policy_panel
from trade_planner import InvestmentPolicyCoefficients, load_historical_replay_bundle


def test_load_historical_bundle_aligns_all_point_in_time_fields(tmp_path: Path) -> None:
    _write_bundle(tmp_path)

    bundle = load_historical_replay_bundle(
        tmp_path,
        expected_role="development",
    )

    assert bundle.role == "development"
    assert bundle.event_ids == ("event_01", "event_02")
    assert set(bundle.source_hashes) == {
        "events.csv",
        "orders.csv",
        "planning.csv",
        "factor_covariance.csv",
        "realized.csv",
        "scenarios.csv",
    }
    assert all(len(value) == 64 for value in bundle.source_hashes.values())
    first = bundle.events[0]
    assert first.ctx.symbols == ["URGENT_BUY", "HEDGE_SELL", "SMALL_BUY"]
    assert first.ctx.factor_names == ["country_HK", "sector_IT", "industry_SW"]
    assert first.ctx.factor_exposure.shape == (3, 3, 3)
    assert first.ctx.return_residual_scenarios.shape == (2, 3, 3)
    np.testing.assert_allclose(first.ctx.return_scenario_weights, [0.6, 0.4])
    np.testing.assert_allclose(
        bundle.forecast_adv_for("event_01", "medium")[:, 0],
        [850.0, 1_050.0, 1_350.0],
    )
    np.testing.assert_allclose(
        bundle.forecast_adv_quantile("event_01", 0.175)[:, 0],
        np.sqrt(
            np.asarray([800.0, 1_000.0, 1_300.0])
            * np.asarray([850.0, 1_050.0, 1_350.0])
        ),
    )
    assert bundle.classifications["event_01"].loc["SMALL_BUY", "urgency"] == "small"
    assert first.ctx.metadata["historical_bundle_role"] == "development"
    assert first.ctx.metadata["source_hashes"] == dict(bundle.source_hashes)


def test_loader_preserves_optional_alpha_classifications(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    order_path = tmp_path / "orders.csv"
    orders = pd.read_csv(order_path)
    orders["rebalance_type"] = "add_delete"
    orders["prediction_confidence"] = 0.80
    orders["crowding"] = 0.35
    orders.to_csv(order_path, index=False)

    bundle = load_historical_replay_bundle(
        tmp_path,
        expected_role="development",
    )
    classifications = bundle.classifications["event_01"]

    assert classifications.loc["URGENT_BUY", "rebalance_type"] == "add_delete"
    assert classifications.loc["HEDGE_SELL", "prediction_confidence"] == 0.80
    assert classifications.loc["SMALL_BUY", "crowding"] == 0.35


def test_loader_rejects_future_planning_information(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    planning_path = tmp_path / "planning.csv"
    planning = pd.read_csv(planning_path)
    planning.loc[0, "available_at"] = "2026-01-02T09:00:00Z"
    planning.to_csv(planning_path, index=False)

    with pytest.raises(ValueError, match="contains future information"):
        load_historical_replay_bundle(tmp_path, expected_role="development")


def test_loader_rejects_incomplete_date_symbol_grid(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    planning_path = tmp_path / "planning.csv"
    planning = pd.read_csv(planning_path).iloc[1:]
    planning.to_csv(planning_path, index=False)

    with pytest.raises(ValueError, match="date-symbol grid is incomplete"):
        load_historical_replay_bundle(tmp_path, expected_role="development")


def test_role_mismatch_stops_before_detailed_files_are_required(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "event_id": "sealed_01",
                "cohort_role": "holdout",
                "as_of": "2026-01-01T16:00:00Z",
                "information_cutoff": "2026-01-01T16:00:00Z",
                "realized_available_at": "2026-01-06T20:00:00Z",
            }
        ]
    ).to_csv(tmp_path / "events.csv", index=False)

    with pytest.raises(ValueError, match="expected_role='development'"):
        load_historical_replay_bundle(tmp_path, expected_role="development")


def test_loader_requires_positive_semidefinite_factor_covariance(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    path = tmp_path / "factor_covariance.csv"
    covariance = pd.read_csv(path)
    mask = (
        covariance["event_id"].eq("event_01")
        & covariance["date"].eq("2026-01-02")
        & covariance["factor_left"].eq("country_HK")
        & covariance["factor_right"].eq("country_HK")
    )
    covariance.loc[mask, "covariance"] = -0.001
    covariance.to_csv(path, index=False)

    with pytest.raises(ValueError, match="positive semidefinite"):
        load_historical_replay_bundle(tmp_path, expected_role="development")


def test_historical_runner_produces_auditable_baseline_comparison(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path)
    bundle = load_historical_replay_bundle(tmp_path, expected_role="development")

    outputs, metadata = run_historical_experiment(
        bundle,
        risk_aversion="medium",
        solver="CLARABEL",
    )

    assert set(outputs) == {
        "trials",
        "paired",
        "summary",
        "gates",
        "schedules",
        "daily",
        "profiles",
        "exposures",
        "liquidity",
        "coefficients",
        "frontiers",
        "source_hashes",
        "alpha_audit",
        "alpha_predictions",
        "alpha_summary",
        "alpha_coefficients",
    }
    assert len(outputs["trials"]) == 4
    assert len(outputs["paired"]) == 2
    assert len(outputs["gates"]) == 16
    assert set(outputs["trials"]["strategy"]) == {
        "static_open_loop",
        "forecast_liquidity",
    }
    candidate_coefficients = outputs["coefficients"].loc[
        outputs["coefficients"]["strategy"].eq("forecast_liquidity")
    ]
    assert (candidate_coefficients["liquidity_quantile"] == 0.25).all()
    assert (candidate_coefficients["liquidity_shape_fraction"] == 0.50).all()
    assert (candidate_coefficients["alpha_confidence"] == 0.75).all()
    assert (candidate_coefficients["factor_stress_fraction"] == 0.50).all()
    assert metadata["cohort_role"] == "development"
    assert metadata["source_hashes"] == dict(bundle.source_hashes)
    assert metadata["alpha_calibration"] == "walk_forward"
    assert metadata["alpha_calibrated_event_count"] == 0
    assert outputs["alpha_audit"]["status"].eq("raw_fallback").all()


def test_historical_runner_consumes_selected_policy_vector(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    bundle = load_historical_replay_bundle(tmp_path, expected_role="development")
    policy = InvestmentPolicyCoefficients(
        policy_id="learned_medium",
        policy_aggressiveness=0.40,
        risk_frontier_fraction=0.40,
        liquidity_quantile=0.175,
        liquidity_shape_fraction=0.60,
        alpha_confidence=0.80,
        factor_stress_fraction=0.60,
    )

    outputs, metadata = run_historical_experiment(
        bundle,
        risk_aversion="medium",
        solver="CLARABEL",
        policy_coefficients=policy,
    )
    candidate = outputs["coefficients"].loc[
        outputs["coefficients"]["strategy"].eq("forecast_liquidity")
    ]

    assert metadata["policy_source"] == "supplied"
    assert metadata["policy_ids"] == ["learned_medium"]
    assert metadata["risk_frontier_fraction"] == 0.40
    assert metadata["liquidity_quantile"] == 0.175
    assert candidate["policy_id"].eq("learned_medium").all()
    assert candidate["liquidity_quantile"].eq(0.175).all()
    assert candidate["liquidity_shape_fraction"].eq(0.60).all()
    assert candidate["alpha_confidence"].eq(0.80).all()
    assert candidate["factor_stress_fraction"].eq(0.60).all()


def test_historical_policy_panel_selects_and_replays_complete_ladder(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path)
    bundle = load_historical_replay_bundle(tmp_path, expected_role="development")

    outputs, metadata = run_historical_policy_panel(
        bundle,
        risk_aversion="medium",
        solver="CLARABEL",
    )
    panel = outputs["policy_trials"]
    medium = outputs["policy_selections"].loc[
        outputs["policy_selections"]["risk_aversion"].eq("medium")
    ]
    selected_coefficients = outputs["coefficients"].loc[
        outputs["coefficients"]["strategy"].eq("forecast_liquidity"),
        ["event_id", "policy_id"],
    ].sort_values("event_id").reset_index(drop=True)
    expected_coefficients = medium[
        ["event_id", "selected_policy_id"]
    ].rename(columns={"selected_policy_id": "policy_id"}).sort_values(
        "event_id"
    ).reset_index(drop=True)

    assert len(panel) == 14
    assert panel["event_id"].nunique() == 2
    assert panel["policy_id"].nunique() == 7
    assert panel.groupby("event_id")["policy_id"].nunique().eq(7).all()
    assert panel["hard_pass"].dtype == bool
    assert panel["behavior_pass"].dtype == bool
    assert medium["status"].eq("fallback_warmup").all()
    assert medium["selected_policy_id"].eq("policy_0500").all()
    pd.testing.assert_frame_equal(selected_coefficients, expected_coefficients)
    assert outputs["policy_schedules"]["policy_id"].nunique() == 7
    assert outputs["alpha_audit"]["status"].eq("raw_fallback").all()
    assert set(outputs["source_hashes"]["file"]) == set(bundle.source_hashes)
    assert metadata["automatic_policy_calibration"] is True
    assert metadata["policy_candidate_count"] == 7
    assert metadata["policy_trial_count"] == 14
    assert metadata["policy_selection_status_counts"] == {"fallback_warmup": 2}


def _write_bundle(root: Path) -> None:
    symbols = ["URGENT_BUY", "HEDGE_SELL", "SMALL_BUY"]
    factor_names = ["country_HK", "sector_IT", "industry_SW"]
    event_specs = [
        (
            "event_01",
            pd.Timestamp("2026-01-01 16:00"),
            pd.bdate_range("2026-01-02", periods=3),
        ),
        (
            "event_02",
            pd.Timestamp("2026-01-09 16:00"),
            pd.bdate_range("2026-01-12", periods=3),
        ),
    ]
    event_rows = []
    order_rows = []
    planning_rows = []
    covariance_rows = []
    realized_rows = []
    scenario_rows = []
    targets = [100.0, -80.0, 40.0]
    countries = ["HK", "US", "HK"]
    sectors = ["IT", "FIN", "IT"]
    industries = ["SW", "BANK", "SW"]
    urgencies = ["urgent", "medium", "small"]
    exposures = np.array(
        [
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    for event_index, (event_id, cutoff, dates) in enumerate(event_specs):
        cutoff_iso = cutoff.isoformat() + "Z"
        event_rows.append(
            {
                "event_id": event_id,
                "cohort_role": "development",
                "as_of": cutoff_iso,
                "information_cutoff": cutoff_iso,
                "realized_available_at": dates[-1].strftime("%Y-%m-%dT20:00:00Z"),
            }
        )
        for symbol_index, symbol in enumerate(symbols):
            order_rows.append(
                {
                    "event_id": event_id,
                    "symbol": symbol,
                    "target_shares": targets[symbol_index],
                    "country": countries[symbol_index],
                    "sector": sectors[symbol_index],
                    "industry": industries[symbol_index],
                    "urgency": urgencies[symbol_index],
                    "available_at": cutoff_iso,
                }
            )
        for date_index, date in enumerate(dates):
            forecast_adv = [800.0, 1_000.0, 1_300.0][date_index]
            for symbol_index, symbol in enumerate(symbols):
                planning_rows.append(
                    {
                        "event_id": event_id,
                        "date": str(date.date()),
                        "symbol": symbol,
                        "price": 50.0 + 5.0 * symbol_index,
                        "adv_shares": 1_000.0 + 100.0 * symbol_index,
                        "forecast_adv_p10_shares": forecast_adv
                        + 100.0 * symbol_index,
                        "forecast_adv_p25_shares": forecast_adv
                        + 50.0
                        + 100.0 * symbol_index,
                        "forecast_adv_p50_shares": forecast_adv
                        + 100.0
                        + 100.0 * symbol_index,
                        "is_open": True,
                        "base_participation": 0.20,
                        "event_days": 2 - date_index,
                        "specific_variance": 0.0002,
                        "expected_return": (0.0002 + 0.00005 * date_index)
                        * np.sign(targets[symbol_index]),
                        "expected_return_uncertainty": 0.0001,
                        "impact_bps_at_10pct_adv": 8.0,
                        "linear_cost_bps": 1.0,
                        "available_at": cutoff_iso,
                        **{
                            f"factor:{factor}": exposures[symbol_index, factor_index]
                            for factor_index, factor in enumerate(factor_names)
                        },
                    }
                )
                realized_rows.append(
                    {
                        "event_id": event_id,
                        "date": str(date.date()),
                        "symbol": symbol,
                        "realized_return": (0.00015 + 0.00003 * date_index)
                        * np.sign(targets[symbol_index]),
                        "realized_adv_shares": forecast_adv + 50.0 * symbol_index,
                        "realized_impact_bps_at_10pct_adv": 8.5,
                        "realized_linear_cost_bps": 1.1,
                        "available_at": date.strftime("%Y-%m-%dT18:00:00Z"),
                    }
                )
                for scenario, weight, direction in (
                    ("up", 0.6, 1.0),
                    ("down", 0.4, -1.0),
                ):
                    scenario_rows.append(
                        {
                            "event_id": event_id,
                            "scenario": scenario,
                            "date": str(date.date()),
                            "symbol": symbol,
                            "residual_return": direction
                            * (0.0001 + 0.00001 * symbol_index),
                            "scenario_weight": weight,
                            "available_at": cutoff_iso,
                        }
                    )
            for left_index, left in enumerate(factor_names):
                for right_index, right in enumerate(factor_names):
                    covariance_rows.append(
                        {
                            "event_id": event_id,
                            "date": str(date.date()),
                            "factor_left": left,
                            "factor_right": right,
                            "covariance": 0.0001 if left_index == right_index else 0.0,
                            "available_at": cutoff_iso,
                        }
                    )
    pd.DataFrame(event_rows).to_csv(root / "events.csv", index=False)
    pd.DataFrame(order_rows).to_csv(root / "orders.csv", index=False)
    pd.DataFrame(planning_rows).to_csv(root / "planning.csv", index=False)
    pd.DataFrame(covariance_rows).to_csv(root / "factor_covariance.csv", index=False)
    pd.DataFrame(realized_rows).to_csv(root / "realized.csv", index=False)
    pd.DataFrame(scenario_rows).to_csv(root / "scenarios.csv", index=False)
