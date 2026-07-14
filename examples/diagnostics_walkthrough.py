"""Walk through a non-obvious infeasible trade plan without a MOSEK license.

The real CLARABEL solve proves that the model is infeasible.  Its mapped
constraint dual is useful but lower-confidence and incomplete, so the first
report labels it as directional fallback evidence.  The second report uses a
clearly labelled, analytically constructed certificate fixture to exercise the
repository's full certificate-to-business-label decoder.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from trade_planner import (
    CompositeCostModel,
    DailyGrossNotionalLimit,
    DirectionConstraint,
    HardCompletionConstraint,
    InfeasiblePlanError,
    MinCompletionByDate,
    ParticipationCapacityConstraint,
    ParticipationCapModel,
    PlannerContext,
    QuadraticParticipationImpact,
    StaticCovarianceRiskModel,
    TradePlanner,
    TradePlannerConfig,
    diagnose_problem,
    get_constraint_diagnostics,
)


DATA_DIR = Path(__file__).with_name("data")
MILESTONE_FRACTION = 0.60
BASE_PARTICIPATION = 0.10


def sample_context() -> tuple[PlannerContext, pd.Series]:
    """Load the small CSV fixture and build a three-day planner context."""
    order_input = pd.read_csv(DATA_DIR / "diagnostics_orders.csv").set_index("symbol")
    daily_input = pd.read_csv(
        DATA_DIR / "diagnostics_daily_limits.csv",
        parse_dates=["date"],
    ).set_index("date")
    daily_input.index = pd.DatetimeIndex(daily_input.index).normalize()

    symbols = order_input.index.astype(str).tolist()
    dates = daily_input.index
    shape = (len(dates), len(symbols))
    price = np.tile(order_input["price"].to_numpy(float), (len(dates), 1))
    adv = np.tile(order_input["adv_shares"].to_numpy(float), (len(dates), 1))
    context = PlannerContext(
        symbols=symbols,
        dates=dates,
        orders=order_input[["target_shares"]].copy(),
        panel=pd.DataFrame(index=pd.MultiIndex.from_product([dates, symbols])),
        price=price,
        adv_shares=adv,
        is_open=np.ones(shape, dtype=bool),
        base_participation=np.full(shape, BASE_PARTICIPATION),
        event_days=pd.DataFrame(np.inf, index=dates, columns=symbols),
    )
    return context, daily_input["gross_limit_dollars"]


def sample_planner(ctx: PlannerContext, gross_limits: pd.Series) -> TradePlanner:
    """Build the planner whose milestone and portfolio gross rules conflict."""
    return TradePlanner(
        TradePlannerConfig(
            participation_model=ParticipationCapModel(),
            risk_model=StaticCovarianceRiskModel(
                covariance=np.eye(len(ctx.symbols)) * 0.0001,
            ),
            cost_model=CompositeCostModel(
                terms=(QuadraticParticipationImpact(),),
            ),
            constraints=(
                ParticipationCapacityConstraint(),
                DirectionConstraint(),
                HardCompletionConstraint(),
                DailyGrossNotionalLimit(gross_limits),
                MinCompletionByDate(ctx.dates[0], MILESTONE_FRACTION),
            ),
            solver="CLARABEL",
        )
    )


def scenario_math(ctx: PlannerContext, gross_limits: pd.Series) -> dict[str, float]:
    """Return the aggregate arithmetic hidden across the two policy plugins."""
    target = np.abs(ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float))
    required = float(np.sum(MILESTONE_FRACTION * target * ctx.price[0]))
    allowed = float(gross_limits.iloc[0])
    horizon_capacity = np.sum(
        ctx.base_participation * ctx.adv_shares * ctx.is_open.astype(float),
        axis=0,
    )
    return {
        "required_first_day_gross": required,
        "allowed_first_day_gross": allowed,
        "gross_shortfall": max(required - allowed, 0.0),
        "minimum_name_capacity_ratio": float(np.min(horizon_capacity / target)),
    }


def attach_analytical_certificate_fixture(problem: Any, ctx: PlannerContext) -> None:
    """Attach teaching-only certificate data using the known contradiction.

    The milestone multipliers are the security prices: multiplying each
    minimum-share rule by price proves that day-one gross must be at least
    $2,460.  The gross-limit rule says it must be at most $2,000.  A production
    run must obtain mapped evidence from its solver; it must not manufacture it.
    """
    certificate: dict[int, Any] = {}
    for constraint in problem.constraints:
        metadata = get_constraint_diagnostics(constraint)
        name = metadata.name if metadata else ""
        if name == f"daily_gross_notional_limit[{ctx.dates[0].date()}]":
            certificate[constraint.id] = np.asarray(1.0)
        elif name == f"min_completion_by_date[{ctx.dates[0].date()}]":
            certificate[constraint.id] = ctx.price[0].copy()

    if len(certificate) != 2:
        raise RuntimeError("The example could not find both conflicting constraints")

    previous = getattr(problem, "solver_stats", None)
    problem._solver_stats = SimpleNamespace(
        solver_name="CLARABEL_WITH_ANALYTICAL_DEMO_CERTIFICATE",
        solve_time=getattr(previous, "solve_time", None),
        setup_time=getattr(previous, "setup_time", None),
        num_iters=getattr(previous, "num_iters", None),
        extra_stats={
            "IIS": certificate,
            "DEMO_ONLY": (
                "Analytical teaching fixture; not emitted by CLARABEL or MOSEK."
            ),
        },
    )


def run() -> dict[str, Any]:
    """Run the real failure, then decode the teaching certificate fixture."""
    ctx, gross_limits = sample_context()
    arithmetic = scenario_math(ctx, gross_limits)
    print("=== Sample orders ===")
    print(
        ctx.orders.assign(
            price=ctx.price[0],
            adv_shares=ctx.adv_shares[0],
        ).to_string()
    )
    print("\n=== Hidden aggregate conflict ===")
    print(
        f"Each name has at least {arithmetic['minimum_name_capacity_ratio']:.1f}x "
        "its target in horizon participation capacity."
    )
    print(
        f"The 60% day-one milestone requires ${arithmetic['required_first_day_gross']:,.0f}, "
        f"but the gross limit is ${arithmetic['allowed_first_day_gross']:,.0f}; "
        f"shortfall=${arithmetic['gross_shortfall']:,.0f}."
    )

    try:
        sample_planner(ctx, gross_limits).solve(ctx)
    except InfeasiblePlanError as error:
        if error.problem is None or error.diagnostics is None:
            raise RuntimeError("Expected the planner to preserve its failed problem") from error

        print("\n=== Real CLARABEL failure ===")
        print(str(error))
        print(error.diagnostics["text"])

        print("\n=== Same failed problem with teaching certificate fixture ===")
        print("WARNING: the mapped certificate below is analytical demo data, not solver output.")
        attach_analytical_certificate_fixture(error.problem, ctx)
        explained = diagnose_problem(error.problem)
        print(explained["text"])
        return {
            "arithmetic": arithmetic,
            "initial_report": error.diagnostics,
            "explained_report": explained,
        }

    raise RuntimeError("The deliberately conflicting sample unexpectedly solved")


if __name__ == "__main__":
    run()
