from __future__ import annotations

import unittest

import cvxpy as cp
import numpy as np
import pandas as pd

from trade_planner.config import TradePlannerConfig
from trade_planner.constraints import (
    ConstraintDiagnostics,
    DailyGrossNotionalLimit,
    DailyNetNotionalLimit,
    DirectionConstraint,
    FactorExposureLimit,
    HardCompletionConstraint,
    MinCompletionByDate,
    ParticipationCapacityConstraint,
    ZeroTargetConstraint,
    get_constraint_diagnostics,
    with_diagnostics,
)
from trade_planner.context import PlannerContext
from trade_planner.costs import CompositeCostModel
from trade_planner.diagnostics import diagnose_infeasible_problem, format_infeasibility_diagnosis
from trade_planner.participation import ParticipationCapModel
from trade_planner.planner import TradePlanner
from trade_planner.risk import StaticCovarianceRiskModel
from trade_planner.types import InfeasiblePlanError


def _solver() -> str:
    for candidate in ("CLARABEL", "SCS", "ECOS"):
        if candidate in cp.installed_solvers():
            return candidate
    raise unittest.SkipTest("No conic CVXPY solver installed")


def _context(targets: list[float] | None = None, base_participation: float = 0.1) -> PlannerContext:
    symbols = ["AAA", "BBB"]
    dates = pd.DatetimeIndex(pd.to_datetime(["2026-07-01", "2026-07-02"])).normalize()
    orders = pd.DataFrame({"target_shares": targets or [100.0, 0.0]}, index=symbols)
    price = np.full((len(dates), len(symbols)), 10.0)
    adv = np.full((len(dates), len(symbols)), 100.0)
    is_open = np.ones((len(dates), len(symbols)), dtype=bool)
    base = np.full((len(dates), len(symbols)), base_participation)
    event_days = pd.DataFrame(np.inf, index=dates, columns=symbols)
    panel = pd.DataFrame(index=pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"]))
    return PlannerContext(
        symbols=symbols,
        dates=dates,
        orders=orders,
        panel=panel,
        price=price,
        adv_shares=adv,
        is_open=is_open,
        base_participation=base,
        event_days=event_days,
    )


class ConstraintDiagnosticsTests(unittest.TestCase):
    def test_unsolved_problem_is_only_solved_when_explicitly_requested(self) -> None:
        x = cp.Variable(name="x")
        problem = cp.Problem(cp.Minimize(x), [x >= 1])

        report = diagnose_infeasible_problem(problem, solve_if_needed=False)
        self.assertEqual(report["summary"]["status_family"], "not_solved")
        self.assertIsNone(problem.status)

        report = diagnose_infeasible_problem(problem, solve_if_needed=True, solver=_solver())
        self.assertEqual(report["summary"]["status_family"], "solved")
        self.assertTrue(report["summary"]["was_solved_by_diagnostic"])

    def test_solved_problem_ranks_active_dual_bottleneck(self) -> None:
        x = cp.Variable(name="x")
        lower = with_diagnostics(
            x >= 1,
            ConstraintDiagnostics(name="minimum_position", suggested_relaxation="Lower the minimum."),
        )
        upper = with_diagnostics(
            x <= 10,
            ConstraintDiagnostics(name="maximum_position", suggested_relaxation="Raise the maximum."),
        )
        problem = cp.Problem(cp.Minimize(x), [lower, upper])
        problem.solve(solver=_solver())

        report = diagnose_infeasible_problem(problem)

        self.assertEqual(report["bottlenecks"][0]["diagnostics"]["name"], "minimum_position")
        self.assertEqual(report["bottlenecks"][0]["source"], "active_dual")
        self.assertIsNotNone(report["constraints"][0]["dual"])
        self.assertIsNotNone(report["constraints"][0]["violation"])
        self.assertIsNotNone(report["solver_stats"])

    def test_builtin_constraints_attach_diagnostics(self) -> None:
        ctx = _context()
        trades = cp.Variable((len(ctx.dates), len(ctx.symbols)))
        caps = ParticipationCapModel().caps(ctx)
        target = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
        state = TradePlanner._build_state(trades=trades, target=target, caps=caps, ctx=ctx)
        exposures = pd.DataFrame({"market": [1.0, -1.0]}, index=ctx.symbols)

        plugins = [
            ParticipationCapacityConstraint(),
            DirectionConstraint(),
            ZeroTargetConstraint(),
            HardCompletionConstraint(check_capacity=False),
            DailyGrossNotionalLimit(max_dollars=1000.0),
            DailyNetNotionalLimit(max_abs_dollars=1000.0),
            MinCompletionByDate(date=ctx.dates[0], fraction=0.5),
            FactorExposureLimit(exposures=exposures, max_abs_exposure=1000.0),
        ]

        constraints = []
        for plugin in plugins:
            constraints.extend(plugin.constraints(ctx, state))

        self.assertGreater(len(constraints), 0)
        for constraint in constraints:
            diagnostics = get_constraint_diagnostics(constraint)
            self.assertIsNotNone(diagnostics)
            self.assertTrue(diagnostics.name)
            self.assertTrue(diagnostics.potential_cause)
            self.assertTrue(diagnostics.suggested_relaxation)

    def test_scalar_infeasible_problem_reports_tagged_relaxations(self) -> None:
        x = cp.Variable(name="x")
        constraints = [
            with_diagnostics(
                x >= 10,
                ConstraintDiagnostics(
                    name="minimum_x",
                    group="lower_bound",
                    potential_cause="The lower bound is above the upper bound.",
                    suggested_relaxation="Decrease the lower bound.",
                    units="units",
                ),
            ),
            with_diagnostics(
                x <= 7,
                ConstraintDiagnostics(
                    name="maximum_x",
                    group="upper_bound",
                    potential_cause="The upper bound is below the lower bound.",
                    suggested_relaxation="Increase the upper bound.",
                    units="units",
                ),
            ),
        ]
        problem = cp.Problem(cp.Minimize(0), constraints)
        problem.solve(solver=_solver())

        report = diagnose_infeasible_problem(problem, run_elastic=True)
        text = format_infeasibility_diagnosis(report)

        self.assertIn("minimum_x", {row["diagnostics"]["name"] for row in report["constraints"]})
        self.assertIn("maximum_x", {row["diagnostics"]["name"] for row in report["constraints"]})
        self.assertGreater(len(report["elastic"]["violations"]), 0)
        self.assertIn("Required max relaxation", text)

    def test_planner_infeasible_error_carries_diagnostics(self) -> None:
        ctx = _context(targets=[100.0, 0.0], base_participation=0.01)
        config = TradePlannerConfig(
            participation_model=ParticipationCapModel(),
            risk_model=StaticCovarianceRiskModel(),
            cost_model=CompositeCostModel(terms=()),
            constraints=(
                ParticipationCapacityConstraint(),
                DirectionConstraint(),
                ZeroTargetConstraint(),
                HardCompletionConstraint(check_capacity=False),
            ),
            solver=_solver(),
        )

        with self.assertRaises(InfeasiblePlanError) as raised:
            TradePlanner(config).solve(ctx)

        diagnostics = raised.exception.diagnostics
        self.assertIsNotNone(diagnostics)
        self.assertIsNone(diagnostics["elastic"])
        text = diagnostics["text"].lower()
        self.assertIn("capacity", text)
        self.assertIn("completion", text)

    def test_factor_exposure_limit_is_identified_as_bottleneck(self) -> None:
        ctx = _context(targets=[100.0, 0.0], base_participation=0.5)
        exposures = pd.DataFrame({"market": [1.0, 0.0]}, index=ctx.symbols)
        config = TradePlannerConfig(
            participation_model=ParticipationCapModel(),
            risk_model=StaticCovarianceRiskModel(),
            cost_model=CompositeCostModel(terms=()),
            constraints=(
                ParticipationCapacityConstraint(),
                DirectionConstraint(),
                ZeroTargetConstraint(),
                HardCompletionConstraint(check_capacity=False),
                FactorExposureLimit(exposures=exposures, max_abs_exposure=100.0),
            ),
            solver=_solver(),
        )

        with self.assertRaises(InfeasiblePlanError) as raised:
            TradePlanner(config).solve(ctx)

        bottlenecks = raised.exception.diagnostics["bottlenecks"]
        groups = {row["diagnostics"]["group"] for row in bottlenecks}
        self.assertIn("factor_exposure", groups)
        self.assertTrue(all(row["source"] == "original_model_structure" for row in bottlenecks))

    def test_unbounded_problem_reports_bounds_not_slack(self) -> None:
        x = cp.Variable(name="x")
        problem = cp.Problem(
            cp.Minimize(-x),
            [
                with_diagnostics(
                    x >= 0,
                    ConstraintDiagnostics(
                        name="nonnegative_x",
                        group="bounds",
                        potential_cause="x has no upper bound.",
                        suggested_relaxation="Add an upper bound.",
                    ),
                )
            ],
        )
        problem.solve(solver=_solver())

        report = diagnose_infeasible_problem(problem)
        text = format_infeasibility_diagnosis(report).lower()

        self.assertIsNone(report["elastic"])
        self.assertIn("bounds", text)
        self.assertIn("objective", text)


if __name__ == "__main__":
    unittest.main()
