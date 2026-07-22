"""Fixed spent-cohort mechanics screen for per-name decision scaling."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.liquidity_forecast_walkforward import plot_results, run_experiment


def run_numerical_scaling_screen():
    """Retry the exact failed net-P&L floor on spent events 25-26."""

    return run_experiment(
        solver="CLARABEL",
        n_events=2,
        event_start=0,
        risk_aversion="medium",
        coefficient_policy="baseline_locked",
        alpha_policy="capacity_slack_confidence",
        factor_policy="minimax_factor_stress",
        profit_policy="baseline_materiality_floor",
        numerical_scaling="per_name",
        verify_hard_constraints=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("artifacts/numerical_scaling_mechanics"),
    )
    args = parser.parse_args()
    outputs, metadata = run_numerical_scaling_screen()
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
    print(f"numerical scaling: {metadata['numerical_scaling']}")
    print(f"strict certificate: {metadata['verify_hard_constraints']}")
    print(f"artifacts: {prefix.parent / (prefix.name + '*')}")


if __name__ == "__main__":
    main()
