"""Run the frozen optimizer challenger on an auditable historical bundle."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from experiments.alpha_confidence_walkforward import LAMBDA_MULTIPLIERS
from experiments.liquidity_forecast_walkforward import (
    CapacitySlackConfidenceAlphaModel,
    MinimaxFactorStressRiskModel,
    _liquidity_decision,
    _liquidity_summary,
    alpha_confidence_for_risk_profile,
    factor_stress_fraction_for_risk_profile,
    liquidity_shape_fraction_for_risk_profile,
    plot_results,
    risk_scaled_liquidity_forecast,
)
from experiments.rebalance_economic_calibration import _behavior_metrics
from experiments.rolling_horizon_walkforward import _schedule_audit
from trade_planner import (
    AlphaDecayWalkForward,
    DEFAULT_RISK_PREFERENCES,
    HistoricalReplayBundle,
    InvestmentPolicyCoefficients,
    RebalanceRiskMeasure,
    RiskAversion,
    RiskPreference,
    TradePlanner,
    build_rebalance_frontier,
    calibrate_alpha_decay_walk_forward,
    evaluate_realized_rebalance_schedule,
    evaluate_rebalance_schedule,
    load_historical_replay_bundle,
)
from trade_planner.historical import LIQUIDITY_QUANTILE_BY_RISK


BASELINE_STRATEGY = "static_open_loop"
CHALLENGER_STRATEGY = "forecast_liquidity"


def run_historical_experiment(
    bundle: HistoricalReplayBundle,
    *,
    risk_aversion: str = "medium",
    solver: str = "CLARABEL",
    alpha_calibration: str = "walk_forward",
    policy_coefficients: (
        InvestmentPolicyCoefficients
        | Mapping[str, InvestmentPolicyCoefficients]
        | None
    ) = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Compare the flat-ADV baseline with the frozen minimax challenger.

    Conditional alpha calibration is fitted event by event before either
    strategy is solved, so both strategies receive the same leakage-safe
    expected-return surface and predictive uncertainty.
    """

    parsed_aversion = RiskAversion.parse(risk_aversion)
    alpha_mode = str(alpha_calibration).strip().lower()
    if alpha_mode not in {"none", "walk_forward"}:
        raise ValueError("alpha_calibration must be 'none' or 'walk_forward'")
    alpha_result: AlphaDecayWalkForward | None = None
    if alpha_mode == "walk_forward":
        alpha_result = calibrate_alpha_decay_walk_forward(bundle)
        planning_events = alpha_result.events
    else:
        planning_events = bundle.events
    if isinstance(policy_coefficients, Mapping):
        missing_policy_events = set(bundle.event_ids).difference(policy_coefficients)
        if missing_policy_events:
            raise ValueError(
                "policy_coefficients mapping is missing events: "
                f"{sorted(missing_policy_events)}"
            )
        if any(
            not isinstance(policy, InvestmentPolicyCoefficients)
            for policy in policy_coefficients.values()
        ):
            raise TypeError(
                "policy_coefficients mapping values must be "
                "InvestmentPolicyCoefficients"
            )
    elif policy_coefficients is not None and not isinstance(
        policy_coefficients,
        InvestmentPolicyCoefficients,
    ):
        raise TypeError(
            "policy_coefficients must be InvestmentPolicyCoefficients, an "
            "event mapping, or None"
        )
    trial_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    schedule_rows: list[pd.DataFrame] = []
    daily_rows: list[pd.DataFrame] = []
    profile_rows: list[pd.DataFrame] = []
    exposure_rows: list[pd.DataFrame] = []
    liquidity_rows: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, object]] = []
    frontier_rows: list[pd.DataFrame] = []
    applied_policies: list[InvestmentPolicyCoefficients] = []

    for event in planning_events:
        event_id = str(event.event_id)
        event_policy = _event_policy_coefficients(
            event_id,
            parsed_aversion,
            policy_coefficients,
        )
        applied_policies.append(event_policy)
        preferences = {
            **DEFAULT_RISK_PREFERENCES,
            parsed_aversion: RiskPreference(
                risk_frontier_fraction=event_policy.risk_frontier_fraction,
                description=(
                    "Automatically selected chronological investment policy."
                ),
            ),
        }
        classifications = bundle.classifications[event_id]
        baseline_frontier = build_rebalance_frontier(
            event.ctx,
            solver=solver,
            lambda_multipliers=LAMBDA_MULTIPLIERS,
            risk_measure=RebalanceRiskMeasure.VARIANCE,
            numerical_scaling="per_name",
            verify_hard_constraints=True,
        )
        baseline_plan = baseline_frontier.select(
            parsed_aversion,
            preferences=preferences,
        )

        quantile_forecast_adv = bundle.forecast_adv_quantile(
            event_id,
            event_policy.liquidity_quantile,
        )
        challenger_adv = risk_scaled_liquidity_forecast(
            event.ctx.adv_shares,
            quantile_forecast_adv,
            parsed_aversion,
            shape_fraction=event_policy.liquidity_shape_fraction,
        )
        challenger_ctx = replace(
            event.ctx,
            adv_shares=challenger_adv,
            metadata={
                **event.ctx.metadata,
                "liquidity_forecast_quantile": LIQUIDITY_QUANTILE_BY_RISK[
                    parsed_aversion.value
                ],
                "liquidity_shape_fraction": (
                    liquidity_shape_fraction_for_risk_profile(parsed_aversion)
                ),
            },
        )
        challenger_frontier = build_rebalance_frontier(
            challenger_ctx,
            solver=solver,
            lambda_multipliers=LAMBDA_MULTIPLIERS,
            risk_measure=RebalanceRiskMeasure.VARIANCE,
            inventory_alpha_model=CapacitySlackConfidenceAlphaModel(
                parsed_aversion,
                confidence=event_policy.alpha_confidence,
            ),
            numerical_scaling="per_name",
            verify_hard_constraints=True,
        )
        challenger_frontier_plan = challenger_frontier.select(
            parsed_aversion,
            preferences=preferences,
        )
        challenger_config = replace(
            challenger_frontier_plan.config,
            inventory_risk_weight=baseline_plan.config.inventory_risk_weight,
            risk_model=MinimaxFactorStressRiskModel(
                parsed_aversion,
                stress_fraction=event_policy.factor_stress_fraction,
            ),
        )
        challenger_result = TradePlanner(challenger_config).solve(challenger_ctx)
        challenger_metrics = evaluate_rebalance_schedule(
            challenger_ctx,
            challenger_result.schedule,
            risk_model=challenger_config.risk_model,
            impact_bps_at_10pct_adv=challenger_frontier.impact_bps_matrix,
            linear_cost_bps=challenger_frontier.linear_cost_bps_matrix,
        )
        strategies = {
            BASELINE_STRATEGY: (
                event.ctx,
                baseline_plan.result.schedule,
                baseline_plan.metrics,
                baseline_plan.config.inventory_risk_weight,
                baseline_frontier.frontier,
            ),
            CHALLENGER_STRATEGY: (
                challenger_ctx,
                challenger_result.schedule,
                challenger_metrics,
                challenger_config.inventory_risk_weight,
                challenger_frontier.frontier,
            ),
        }
        event_strategy_rows: dict[str, dict[str, object]] = {}
        for strategy, (
            planning_ctx,
            schedule,
            forecast_metrics,
            inventory_risk_weight,
            frontier_frame,
        ) in strategies.items():
            realized_metrics, daily = evaluate_realized_rebalance_schedule(
                event,
                schedule,
            )
            behavior, profile, exposures = _behavior_metrics(
                planning_ctx,
                classifications,
                schedule,
            )
            audit = _schedule_audit(planning_ctx, schedule)
            row = {
                "event_id": event_id,
                "as_of": pd.Timestamp(event.as_of),
                "strategy": strategy,
                "cohort_role": bundle.role,
                "risk_aversion": parsed_aversion.value,
                "forecast_expected_net_pnl_dollars": (
                    forecast_metrics.expected_net_pnl_dollars
                ),
                "selected_inventory_risk_weight": inventory_risk_weight,
                **realized_metrics.as_dict(),
                **behavior,
                **audit,
            }
            trial_rows.append(row)
            event_strategy_rows[strategy] = row
            schedule_rows.append(
                schedule.assign(event_id=event_id, strategy=strategy)
            )
            daily_rows.append(daily.assign(event_id=event_id, strategy=strategy))
            profile_rows.append(
                profile.assign(
                    event_id=event_id,
                    strategy=strategy,
                    day_index=np.arange(1, len(profile) + 1),
                )
            )
            exposure_rows.append(
                exposures.assign(event_id=event_id, strategy=strategy)
            )
            coefficient_rows.append(
                {
                    "event_id": event_id,
                    "strategy": strategy,
                    "cohort_role": bundle.role,
                    "risk_aversion": parsed_aversion.value,
                    "policy_id": event_policy.policy_id,
                    "policy_aggressiveness": event_policy.policy_aggressiveness,
                    "risk_frontier_fraction": (
                        event_policy.risk_frontier_fraction
                    ),
                    "liquidity_quantile": (
                        0.50
                        if strategy == BASELINE_STRATEGY
                        else event_policy.liquidity_quantile
                    ),
                    "liquidity_shape_fraction": (
                        0.0
                        if strategy == BASELINE_STRATEGY
                        else event_policy.liquidity_shape_fraction
                    ),
                    "alpha_confidence": (
                        0.50
                        if strategy == BASELINE_STRATEGY
                        else event_policy.alpha_confidence
                    ),
                    "factor_stress_fraction": (
                        0.0
                        if strategy == BASELINE_STRATEGY
                        else event_policy.factor_stress_fraction
                    ),
                    "inventory_risk_weight": inventory_risk_weight,
                }
            )
            frontier_rows.append(
                frontier_frame.assign(event_id=event_id, strategy=strategy)
            )

        baseline = event_strategy_rows[BASELINE_STRATEGY]
        challenger = event_strategy_rows[CHALLENGER_STRATEGY]
        paired_rows.append(
            {
                "event_id": event_id,
                "as_of": pd.Timestamp(event.as_of),
                "realized_net_pnl_delta_dollars": (
                    challenger["net_pnl_dollars"] - baseline["net_pnl_dollars"]
                ),
                "realized_net_pnl_delta_bps": (
                    challenger["net_pnl_bps"] - baseline["net_pnl_bps"]
                ),
                "realized_impact_cost_delta_dollars": (
                    challenger["impact_cost_dollars"]
                    - baseline["impact_cost_dollars"]
                ),
                "within_event_drawdown_delta_bps": (
                    challenger["within_event_max_drawdown_bps"]
                    - baseline["within_event_max_drawdown_bps"]
                ),
                "early_factor_imbalance_delta_pp": (
                    challenger["early_factor_imbalance_pct"]
                    - baseline["early_factor_imbalance_pct"]
                ),
                "late_early_ratio_delta": (
                    challenger["late_early_gross_ratio"]
                    - baseline["late_early_gross_ratio"]
                ),
                "daily_gross_spearman_delta": (
                    challenger["daily_gross_spearman"]
                    - baseline["daily_gross_spearman"]
                ),
                "nondecreasing_transitions_delta": (
                    challenger["nondecreasing_transitions"]
                    - baseline["nondecreasing_transitions"]
                ),
                "urgent_start_delta_days": (
                    challenger["urgent_first_trade_day"]
                    - baseline["urgent_first_trade_day"]
                ),
                "small_start_delta_days": (
                    challenger["small_first_trade_day"]
                    - baseline["small_first_trade_day"]
                ),
                "p95_realized_participation_delta": (
                    challenger["p95_realized_participation_rate"]
                    - baseline["p95_realized_participation_rate"]
                ),
                "max_realized_participation_delta": (
                    challenger["max_realized_participation_rate"]
                    - baseline["max_realized_participation_rate"]
                ),
            }
        )
        liquidity_rows.append(
            _historical_liquidity_frame(
                event_id=event_id,
                event=event,
                quantile_forecast_adv=quantile_forecast_adv,
                planning_adv=challenger_adv,
            )
        )

    trials = pd.DataFrame(trial_rows)
    paired = pd.DataFrame(paired_rows)
    summary = _liquidity_summary(trials)
    gate_decision, gate_reason, gates = _liquidity_decision(summary, paired)
    if bundle.role == "development":
        decision, reason = gate_decision, gate_reason
    elif bundle.role == "holdout":
        decision = (
            "holdout_pass"
            if gate_decision == "keep_for_holdout"
            else "holdout_fail"
        )
        reason = (
            "Historical holdout passed every predeclared gate."
            if decision == "holdout_pass"
            else "Historical holdout failed: "
            + gate_reason.removeprefix("Failed: ")
        )
    else:
        decision = "descriptive_only"
        reason = "Backtest bundle is descriptive and cannot authorize promotion."
    summary["decision"] = np.where(
        summary["strategy"].eq(CHALLENGER_STRATEGY),
        decision,
        "baseline",
    )
    summary["decision_reason"] = np.where(
        summary["strategy"].eq(CHALLENGER_STRATEGY),
        reason,
        "Flat-ADV automatic risk-profile baseline.",
    )
    outputs = {
        "trials": trials,
        "paired": paired,
        "summary": summary,
        "gates": gates,
        "schedules": pd.concat(schedule_rows, ignore_index=True),
        "daily": pd.concat(daily_rows, ignore_index=True),
        "profiles": pd.concat(profile_rows, ignore_index=True),
        "exposures": pd.concat(exposure_rows, ignore_index=True),
        "liquidity": pd.concat(liquidity_rows, ignore_index=True),
        "coefficients": pd.DataFrame(coefficient_rows),
        "frontiers": pd.concat(frontier_rows, ignore_index=True),
        "source_hashes": pd.DataFrame(
            [
                {"file": name, "sha256": digest}
                for name, digest in bundle.source_hashes.items()
            ]
        ),
        "alpha_audit": (
            alpha_result.audit if alpha_result is not None else pd.DataFrame()
        ),
        "alpha_predictions": (
            alpha_result.predictions if alpha_result is not None else pd.DataFrame()
        ),
        "alpha_summary": (
            alpha_result.summary if alpha_result is not None else pd.DataFrame()
        ),
        "alpha_coefficients": (
            alpha_result.coefficients if alpha_result is not None else pd.DataFrame()
        ),
    }
    unique_policy_ids = sorted({policy.policy_id for policy in applied_policies})
    first_policy = applied_policies[0]
    constant_policy = all(
        policy.as_dict() == first_policy.as_dict()
        for policy in applied_policies[1:]
    )
    metadata: dict[str, object] = {
        "decision": decision,
        "decision_reason": reason,
        "cohort_role": bundle.role,
        "event_count": len(bundle.events),
        "risk_aversion": parsed_aversion.value,
        "solver": solver,
        "numerical_scaling": "per_name",
        "verify_hard_constraints": True,
        "alpha_calibration": alpha_mode,
        "alpha_calibrated_event_count": (
            int(alpha_result.audit["status"].eq("calibrated").sum())
            if alpha_result is not None
            else 0
        ),
        "policy_source": (
            "fixed_risk_profile" if policy_coefficients is None else "supplied"
        ),
        "policy_ids": unique_policy_ids,
        "policy_aggressiveness": (
            first_policy.policy_aggressiveness if constant_policy else np.nan
        ),
        "risk_frontier_fraction": (
            first_policy.risk_frontier_fraction if constant_policy else np.nan
        ),
        "liquidity_quantile": (
            first_policy.liquidity_quantile if constant_policy else np.nan
        ),
        "liquidity_shape_fraction": (
            first_policy.liquidity_shape_fraction if constant_policy else np.nan
        ),
        "alpha_confidence": (
            first_policy.alpha_confidence if constant_policy else np.nan
        ),
        "factor_stress_fraction": (
            first_policy.factor_stress_fraction if constant_policy else np.nan
        ),
        "source_hashes": dict(bundle.source_hashes),
    }
    return outputs, metadata


def _event_policy_coefficients(
    event_id: str,
    risk_aversion: RiskAversion,
    supplied: (
        InvestmentPolicyCoefficients
        | Mapping[str, InvestmentPolicyCoefficients]
        | None
    ),
) -> InvestmentPolicyCoefficients:
    if isinstance(supplied, InvestmentPolicyCoefficients):
        return supplied
    if isinstance(supplied, Mapping):
        return supplied[event_id]
    preference = DEFAULT_RISK_PREFERENCES[risk_aversion]
    return InvestmentPolicyCoefficients(
        policy_id=f"fixed_{risk_aversion.value}",
        policy_aggressiveness=preference.risk_frontier_fraction,
        risk_frontier_fraction=preference.risk_frontier_fraction,
        liquidity_quantile=LIQUIDITY_QUANTILE_BY_RISK[risk_aversion.value],
        liquidity_shape_fraction=liquidity_shape_fraction_for_risk_profile(
            risk_aversion
        ),
        alpha_confidence=alpha_confidence_for_risk_profile(risk_aversion),
        factor_stress_fraction=factor_stress_fraction_for_risk_profile(
            risk_aversion
        ),
    )


def _historical_liquidity_frame(
    *,
    event_id: str,
    event,
    quantile_forecast_adv: np.ndarray,
    planning_adv: np.ndarray,
) -> pd.DataFrame:
    flat_adv = np.asarray(event.ctx.adv_shares, dtype=float)
    realized_adv = np.asarray(event.realized_adv_shares, dtype=float)
    rows = []
    for date_index, date in enumerate(event.ctx.dates):
        for symbol_index, symbol in enumerate(event.ctx.symbols):
            rows.append(
                {
                    "event_id": event_id,
                    "date": date,
                    "day_index": date_index + 1,
                    "symbol": symbol,
                    "flat_adv_shares": flat_adv[date_index, symbol_index],
                    "quantile_forecast_adv_shares": quantile_forecast_adv[
                        date_index, symbol_index
                    ],
                    "forecast_adv_shares": planning_adv[
                        date_index, symbol_index
                    ],
                    "realized_adv_shares": realized_adv[
                        date_index, symbol_index
                    ],
                    "forecast_adv_multiplier": planning_adv[
                        date_index, symbol_index
                    ]
                    / flat_adv[date_index, symbol_index],
                    "realized_adv_multiplier": realized_adv[
                        date_index, symbol_index
                    ]
                    / flat_adv[date_index, symbol_index],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--role",
        choices=("development", "holdout", "backtest"),
        required=True,
        help="Explicitly authorize which cohort role may be opened.",
    )
    parser.add_argument(
        "--risk-aversion",
        choices=("high", "medium", "low"),
        default="medium",
    )
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument(
        "--alpha-calibration",
        choices=("walk_forward", "none"),
        default="walk_forward",
        help=(
            "Chronologically calibrate holding alpha from earlier available "
            "events, or explicitly disable it."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("artifacts/historical_replay_development"),
    )
    args = parser.parse_args()
    bundle = load_historical_replay_bundle(
        args.bundle,
        expected_role=args.role,
    )
    outputs, metadata = run_historical_experiment(
        bundle,
        risk_aversion=args.risk_aversion,
        solver=args.solver,
        alpha_calibration=args.alpha_calibration,
    )
    prefix: Path = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(prefix.with_name(prefix.name + f"_{name}.csv"), index=False)
    chart = prefix.with_suffix(".png")
    plot_results(outputs, chart)
    print(outputs["summary"].round(4).to_string(index=False))
    print("\nAcceptance gates:")
    print(outputs["gates"].to_string(index=False))
    print(f"\ndecision: {metadata['decision']}")
    print(f"reason: {metadata['decision_reason']}")
    print(f"cohort role: {metadata['cohort_role']}")
    print(f"risk aversion: {metadata['risk_aversion']}")
    print(
        "alpha calibration: "
        f"{metadata['alpha_calibration']} "
        f"({metadata['alpha_calibrated_event_count']} calibrated events)"
    )
    print(
        "automatic liquidity quantile/shape: "
        f"P{100 * metadata['liquidity_quantile']:.0f} / "
        f"{100 * metadata['liquidity_shape_fraction']:.0f}%"
    )
    print(f"artifacts: {prefix.parent / (prefix.name + '*')}")


if __name__ == "__main__":
    main()
