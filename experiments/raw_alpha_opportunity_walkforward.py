"""Fresh replay of risk-scaled liquidity with raw alpha opportunity cost."""

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
from experiments.risk_scaled_liquidity_walkforward import (
    RISK_SCALED_EVENT_SEEDS,
    RISK_SCALED_LIQUIDITY_SEEDS,
    RISK_SCALED_SCENARIO_SEEDS,
)


RAW_ALPHA_EVENT_SEEDS = tuple(20301001 + offset for offset in range(24))
RAW_ALPHA_SCENARIO_SEEDS = tuple(20301101 + offset for offset in range(24))
RAW_ALPHA_LIQUIDITY_SEEDS = tuple(20301201 + offset for offset in range(24))
RAW_ALPHA_EVENT_INDEX_OFFSET = 96
RAW_ALPHA_DEVELOPMENT_EVENTS = 12

_prior_seeds = set(
    LIQUIDITY_EVENT_SEEDS
    + LIQUIDITY_SCENARIO_SEEDS
    + REALIZED_LIQUIDITY_SEEDS
    + PROFIT_FLOOR_EVENT_SEEDS
    + PROFIT_FLOOR_SCENARIO_SEEDS
    + PROFIT_FLOOR_LIQUIDITY_SEEDS
    + RISK_SCALED_EVENT_SEEDS
    + RISK_SCALED_SCENARIO_SEEDS
    + RISK_SCALED_LIQUIDITY_SEEDS
    + (LIQUIDITY_CALIBRATION_SEED,)
)
_new_seeds = (
    RAW_ALPHA_EVENT_SEEDS
    + RAW_ALPHA_SCENARIO_SEEDS
    + RAW_ALPHA_LIQUIDITY_SEEDS
)
assert len(set(_new_seeds)) == len(_new_seeds)
assert _prior_seeds.isdisjoint(_new_seeds)


def run_raw_alpha_experiment(
    *,
    solver: str = "OSQP",
    n_events: int = 12,
    event_start: int = 0,
    risk_aversion: str = "medium",
):
    """Run the frozen raw-alpha policy on fresh events 97-120."""

    return run_experiment(
        solver=solver,
        n_events=n_events,
        event_start=event_start,
        risk_aversion=risk_aversion,
        coefficient_policy="baseline_locked",
        alpha_policy="raw",
        factor_policy="minimax_factor_stress",
        liquidity_shape_policy="risk_scaled",
        event_seeds=RAW_ALPHA_EVENT_SEEDS,
        scenario_seeds=RAW_ALPHA_SCENARIO_SEEDS,
        realized_liquidity_seeds=RAW_ALPHA_LIQUIDITY_SEEDS,
        event_index_offset=RAW_ALPHA_EVENT_INDEX_OFFSET,
        development_event_count=RAW_ALPHA_DEVELOPMENT_EVENTS,
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
        default=Path("artifacts/raw_alpha_opportunity_fresh_dev"),
    )
    args = parser.parse_args()
    outputs, metadata = run_raw_alpha_experiment(
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
    print("alpha policy: raw point-in-time expected return")
    print(
        "liquidity shape fraction: "
        f"{100.0 * metadata['liquidity_shape_fraction']:.1f}%"
    )
    print(f"holdout untouched: {metadata['holdout_untouched']}")
    print(f"artifacts: {prefix.parent / (prefix.name + '*')}")


if __name__ == "__main__":
    main()
