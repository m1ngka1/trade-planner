"""Fresh-cohort replay of the minimax planner with a forecast-profit floor."""

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


PROFIT_FLOOR_EVENT_SEEDS = tuple(20281001 + offset for offset in range(24))
PROFIT_FLOOR_SCENARIO_SEEDS = tuple(20281101 + offset for offset in range(24))
PROFIT_FLOOR_LIQUIDITY_SEEDS = tuple(20281201 + offset for offset in range(24))
PROFIT_FLOOR_EVENT_INDEX_OFFSET = 48
PROFIT_FLOOR_DEVELOPMENT_EVENTS = 12

_old_seeds = set(
    LIQUIDITY_EVENT_SEEDS
    + LIQUIDITY_SCENARIO_SEEDS
    + REALIZED_LIQUIDITY_SEEDS
    + (LIQUIDITY_CALIBRATION_SEED,)
)
_new_seeds = (
    PROFIT_FLOOR_EVENT_SEEDS
    + PROFIT_FLOOR_SCENARIO_SEEDS
    + PROFIT_FLOOR_LIQUIDITY_SEEDS
)
assert len(set(_new_seeds)) == len(_new_seeds)
assert _old_seeds.isdisjoint(_new_seeds)


def run_profit_floor_experiment(
    *,
    solver: str = "CLARABEL",
    n_events: int = 12,
    event_start: int = 0,
    risk_aversion: str = "medium",
):
    """Run the fixed profit-floor candidate on fresh events 49-72."""

    return run_experiment(
        solver=solver,
        n_events=n_events,
        event_start=event_start,
        risk_aversion=risk_aversion,
        coefficient_policy="baseline_locked",
        alpha_policy="capacity_slack_confidence",
        factor_policy="minimax_factor_stress",
        profit_policy="baseline_materiality_floor",
        event_seeds=PROFIT_FLOOR_EVENT_SEEDS,
        scenario_seeds=PROFIT_FLOOR_SCENARIO_SEEDS,
        realized_liquidity_seeds=PROFIT_FLOOR_LIQUIDITY_SEEDS,
        event_index_offset=PROFIT_FLOOR_EVENT_INDEX_OFFSET,
        development_event_count=PROFIT_FLOOR_DEVELOPMENT_EVENTS,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="CLARABEL")
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
        default=Path("artifacts/profit_floor_dev"),
    )
    args = parser.parse_args()
    outputs, metadata = run_profit_floor_experiment(
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
        "forecast-profit materiality: "
        f"{metadata['forecast_profit_materiality_bps']:.1f} bp"
    )
    print(f"holdout untouched: {metadata['holdout_untouched']}")
    print(f"artifacts: {prefix.parent / (prefix.name + '*')}")


if __name__ == "__main__":
    main()
