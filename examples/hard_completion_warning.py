"""Print and validate the hard-completion target-capping warning.

Run from the repository root with:

    python -m examples.hard_completion_warning
"""

from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np
import pandas as pd

from trade_planner import (
    CompositeCostModel,
    ParticipationCapModel,
    PlannerContext,
    StaticCovarianceRiskModel,
    TradePlanner,
    TradePlannerConfig,
)


EXPECTED_WARNING = """HardCompletionConstraint capped 2 target(s) to available horizon capacity; planning will continue.
AAA: original_target_shares=100, capped_target_shares=20, shortfall_shares=80
BBB: original_target_shares=-50, capped_target_shares=-20, shortfall_shares=30"""


def sample_context() -> PlannerContext:
    """Build two constrained names and one target that already fits."""
    symbols = ["AAA", "BBB", "CCC"]
    dates = pd.bdate_range("2026-07-01", periods=2)
    shape = (len(dates), len(symbols))
    orders = pd.DataFrame(
        {"target_shares": [100.0, -50.0, 10.0]},
        index=symbols,
    )
    return PlannerContext(
        symbols=symbols,
        dates=dates,
        orders=orders,
        panel=pd.DataFrame(
            index=pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        ),
        price=np.full(shape, 10.0),
        adv_shares=np.full(shape, 100.0),
        is_open=np.ones(shape, dtype=bool),
        base_participation=np.full(shape, 0.10),
        event_days=pd.DataFrame(np.inf, index=dates, columns=symbols),
    )


def installed_conic_solver() -> str:
    """Choose an installed solver that supports the planner's constraints."""
    for solver in ("CLARABEL", "SCS", "ECOS"):
        if solver in cp.installed_solvers():
            return solver
    raise RuntimeError("This example requires CLARABEL, SCS, or ECOS")


def run() -> None:
    """Solve the capped plan and fail if the warning format changes."""
    ctx = sample_context()
    original_orders = ctx.orders.copy()
    planner = TradePlanner(
        TradePlannerConfig(
            participation_model=ParticipationCapModel(),
            risk_model=StaticCovarianceRiskModel(),
            cost_model=CompositeCostModel(terms=()),
            solver=installed_conic_solver(),
        )
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = planner.solve(ctx)

    cap_warnings = [
        warning
        for warning in caught
        if str(warning.message).startswith("HardCompletionConstraint capped")
    ]
    if len(cap_warnings) != 1:
        raise AssertionError(f"Expected one target-capping warning, received {len(cap_warnings)}")

    actual_warning = str(cap_warnings[0].message)
    print("=== Captured warning ===")
    print(actual_warning)
    if actual_warning != EXPECTED_WARNING:
        raise AssertionError("Target-capping warning did not match EXPECTED_WARNING")

    traded = result.schedule.groupby("symbol")["trade_shares"].sum().reindex(ctx.symbols)
    expected_traded = pd.Series([20.0, -20.0, 10.0], index=ctx.symbols, name="trade_shares")
    np.testing.assert_allclose(traded, expected_traded, atol=1e-6)
    pd.testing.assert_frame_equal(ctx.orders, original_orders)

    print("\n=== Planned totals after capping ===")
    print(traded.rename("planned_trade_shares").round(6).to_string())
    print("\nWarning format, capped totals, and unchanged user orders all validated.")


if __name__ == "__main__":
    run()
