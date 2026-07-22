"""Fresh validation of risk-profile-scaled event-liquidity shape."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.liquidity_forecast_walkforward import (
    LIQUIDITY_CALIBRATION_SEED,
    LIQUIDITY_EVENT_SEEDS,
    LIQUIDITY_SCENARIO_SEEDS,
    REALIZED_LIQUIDITY_SEEDS,
    plot_results,
    run_experiment,
)
from experiments.profit_floor_walkforward import (
    PROFIT_FLOOR_EVENT_SEEDS,
    PROFIT_FLOOR_LIQUIDITY_SEEDS,
    PROFIT_FLOOR_SCENARIO_SEEDS,
)


RISK_SCALED_EVENT_SEEDS = tuple(20291001 + offset for offset in range(24))
RISK_SCALED_SCENARIO_SEEDS = tuple(20291101 + offset for offset in range(24))
RISK_SCALED_LIQUIDITY_SEEDS = tuple(20291201 + offset for offset in range(24))
RISK_SCALED_EVENT_INDEX_OFFSET = 72
RISK_SCALED_DEVELOPMENT_EVENTS = 12

_prior_seeds = set(
    LIQUIDITY_EVENT_SEEDS
    + LIQUIDITY_SCENARIO_SEEDS
    + REALIZED_LIQUIDITY_SEEDS
    + PROFIT_FLOOR_EVENT_SEEDS
    + PROFIT_FLOOR_SCENARIO_SEEDS
    + PROFIT_FLOOR_LIQUIDITY_SEEDS
    + (LIQUIDITY_CALIBRATION_SEED,)
)
_new_seeds = (
    RISK_SCALED_EVENT_SEEDS
    + RISK_SCALED_SCENARIO_SEEDS
    + RISK_SCALED_LIQUIDITY_SEEDS
)
assert len(set(_new_seeds)) == len(_new_seeds)
assert _prior_seeds.isdisjoint(_new_seeds)


def run_risk_scaled_experiment(
    *,
    solver: str = "OSQP",
    n_events: int = 12,
    event_start: int = 0,
    risk_aversion: str = "medium",
):
    """Run the frozen risk-scaled policy on fresh events 73-96."""

    return run_experiment(
        solver=solver,
        n_events=n_events,
        event_start=event_start,
        risk_aversion=risk_aversion,
        coefficient_policy="baseline_locked",
        alpha_policy="capacity_slack_confidence",
        factor_policy="minimax_factor_stress",
        liquidity_shape_policy="risk_scaled",
        event_seeds=RISK_SCALED_EVENT_SEEDS,
        scenario_seeds=RISK_SCALED_SCENARIO_SEEDS,
        realized_liquidity_seeds=RISK_SCALED_LIQUIDITY_SEEDS,
        event_index_offset=RISK_SCALED_EVENT_INDEX_OFFSET,
        development_event_count=RISK_SCALED_DEVELOPMENT_EVENTS,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="OSQP")
    parser.add_argument("--n-events", type=int, default=12)
    parser.add_argument("--event-start", type=int, default=0)
    parser.add_argument(
        "--risk-aversion",
        choices=("high", "medium", "low"),
        default="medium",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("artifacts/risk_scaled_liquidity_fresh_dev"),
    )
    args = parser.parse_args()
    outputs, metadata = run_risk_scaled_experiment(
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
    plot_results(outputs, chart)
    print(outputs["summary"].round(4).to_string(index=False))
    print("\nAcceptance gates:")
    print(outputs["gates"].to_string(index=False))
    print(f"\ndecision: {metadata['decision']}")
    print(f"reason: {metadata['decision_reason']}")
    print(
        "liquidity shape fraction: "
        f"{100.0 * metadata['liquidity_shape_fraction']:.1f}%"
    )
    print(f"holdout untouched: {metadata['holdout_untouched']}")
    print(f"artifacts: {prefix.parent / (prefix.name + '*')}")


if __name__ == "__main__":
    main()
