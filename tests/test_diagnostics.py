from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

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
    VariableDiagnostics,
    ZeroTargetConstraint,
    get_constraint_diagnostics,
    with_diagnostics,
    with_variable_diagnostics,
)
from trade_planner.context import PlannerContext
from trade_planner.costs import CompositeCostModel
from trade_planner.diagnostics import diagnose_problem, format_diagnosis
from trade_planner.mosek_diagnostics import DiagnosticMOSEK, _snapshot_task
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


def _mock_stats(extra_stats: dict, solver_name: str = "mock") -> SimpleNamespace:
    return SimpleNamespace(
        solver_name=solver_name,
        solve_time=0.0,
        setup_time=0.0,
        num_iters=1,
        extra_stats=extra_stats,
    )


class ConstraintDiagnosticsTests(unittest.TestCase):
    def test_unsolved_problem_is_only_solved_when_explicitly_requested(self) -> None:
        x = cp.Variable(name="x")
        problem = cp.Problem(cp.Minimize(x), [x >= 1])

        report = diagnose_problem(problem)
        self.assertEqual(report["summary"]["status_family"], "not_solved")
        self.assertIsNone(problem.status)

        report = diagnose_problem(problem, solve_if_needed=True, solver=_solver())
        self.assertEqual(report["summary"]["status_family"], "solved")
        self.assertTrue(report["summary"]["was_solved_by_diagnostic"])
        self.assertEqual(report["summary"]["additional_solves"], 1)

    def test_solved_problem_reports_shadow_price_as_sensitivity(self) -> None:
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

        report = diagnose_problem(problem)

        self.assertEqual(report["bottlenecks"][0]["diagnostics"]["name"], "minimum_position")
        self.assertEqual(report["bottlenecks"][0]["source"], "optimal_shadow_price")
        self.assertEqual(report["solver_evidence"]["kind"], "optimal_shadow_prices")
        self.assertIn("not infeasibility certificates", report["decision"]["evidence_limit"])

    def test_mapped_certificate_names_exact_setting_location_and_value(self) -> None:
        x = cp.Variable(2, name="positions")
        lower = with_diagnostics(
            x >= np.array([10.0, 0.0]),
            ConstraintDiagnostics(
                name="required_positions",
                group="completion",
                potential_cause="Required shares exceed an upper policy limit.",
                suggested_relaxation="Reduce the parent target.",
                units="shares",
                axis_labels={"symbol": ("AAA", "BBB")},
                setting_name="required shares",
                bound_values=np.array([10.0, 0.0]),
            ),
        )
        upper = with_diagnostics(
            x <= np.array([7.0, 100.0]),
            ConstraintDiagnostics(
                name="position_caps",
                group="capacity",
                potential_cause="The cap is below required shares.",
                suggested_relaxation="Raise the position cap.",
                units="shares",
                axis_labels={"symbol": ("AAA", "BBB")},
                setting_name="position cap",
                bound_values=np.array([7.0, 100.0]),
            ),
        )
        problem = cp.Problem(cp.Minimize(0), [lower, upper])
        problem.solve(solver=_solver())
        problem._solver_stats = _mock_stats(
            {"IIS": {lower.id: np.array([2.0, 0.0]), upper.id: np.array([2.0, 0.0])}},
            solver_name="MOSEK",
        )

        report = diagnose_problem(problem)
        cap = next(row for row in report["bottlenecks"] if row["constraint_id"] == upper.id)
        element = cap["affected_elements"][0]

        self.assertEqual(report["solver_evidence"]["kind"], "primal_infeasibility_certificate")
        self.assertEqual(cap["source"], "mosek_infeasibility_certificate")
        self.assertEqual(element["location"]["symbol"], "AAA")
        self.assertEqual(element["current_setting"], 7.0)
        self.assertEqual(report["decision"]["what_to_change"][1]["setting"], "position cap")
        self.assertIn("Raise the position cap", report["text"])
        self.assertIn("0 diagnostic re-solves", report["text"])
        json.dumps(report)

    def test_diagnosis_never_calls_solve_or_disables_a_constraint(self) -> None:
        x = cp.Variable(name="x")
        constraints = [x >= 1, x <= 0]
        problem = cp.Problem(cp.Minimize(0), constraints)
        problem.solve(solver=_solver())
        original_ids = [constraint.id for constraint in problem.constraints]
        original_duals = [np.array(constraint.dual_value, copy=True) for constraint in problem.constraints]
        solve_mock = Mock(side_effect=AssertionError("diagnostics must not solve"))
        problem.solve = solve_mock

        report = diagnose_problem(problem)

        solve_mock.assert_not_called()
        self.assertEqual(report["summary"]["additional_solves"], 0)
        self.assertEqual([constraint.id for constraint in problem.constraints], original_ids)
        for before, constraint in zip(original_duals, problem.constraints):
            np.testing.assert_allclose(constraint.dual_value, before)
        self.assertNotIn("verification", report)
        self.assertNotIn("single_constraint_recovery", json.dumps(report))

    def test_large_vector_certificate_is_sparse_in_report_and_does_not_solve(self) -> None:
        n_dates, n_symbols = 60, 200
        trades = cp.Variable((n_dates, n_symbols), name="trade_shares")
        caps = np.full((n_dates, n_symbols), 10.0)
        labels = {
            "date": tuple(f"D{index}" for index in range(n_dates)),
            "symbol": tuple(f"S{index}" for index in range(n_symbols)),
        }
        capacity = with_diagnostics(
            trades <= caps,
            ConstraintDiagnostics(
                name="participation_capacity",
                axis_labels=labels,
                setting_name="participation cap",
                bound_values=caps,
                suggested_relaxation="Increase the cap at the reported date and symbol.",
            ),
        )
        problem = cp.Problem(cp.Minimize(0), [capacity])
        certificate = np.zeros((n_dates, n_symbols))
        certificate[37, 151] = 4.0
        certificate[5, 2] = 1.0
        problem._status = cp.INFEASIBLE
        problem._value = np.inf
        problem._solver_stats = _mock_stats({"IIS": {capacity.id: certificate}}, solver_name="MOSEK")
        problem.solve = Mock(side_effect=AssertionError("diagnostics must not solve"))

        report = diagnose_problem(problem, top_elements=2)

        problem.solve.assert_not_called()
        top = report["bottlenecks"][0]["affected_elements"][0]
        self.assertEqual(top["location"], {"index": [37, 151], "date": "D37", "symbol": "S151"})
        self.assertEqual(top["current_setting"], 10.0)
        self.assertLess(len(json.dumps(report)), 200_000)

    def test_cvxpy_mosek_canonicalization_preserves_original_constraint_ids(self) -> None:
        x = cp.Variable(3, name="trade_shares")
        capacity = with_diagnostics(
            cp.abs(x) <= np.ones(3),
            ConstraintDiagnostics(name="participation_capacity"),
        )
        completion = with_diagnostics(cp.sum(x) == 10, ConstraintDiagnostics(name="hard_completion"))
        problem = cp.Problem(cp.Minimize(0), [capacity, completion])

        data, _, _ = problem.get_problem_data(DiagnosticMOSEK())
        canonical_ids = {constraint.id for constraint in data["param_prob"].constraints}

        self.assertTrue({capacity.id, completion.id}.issubset(canonical_ids))

    def test_builtin_constraints_own_business_metadata_and_current_settings(self) -> None:
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

        constraints = [constraint for plugin in plugins for constraint in plugin.constraints(ctx, state)]

        self.assertGreater(len(constraints), 0)
        for constraint in constraints:
            diagnostics = get_constraint_diagnostics(constraint)
            self.assertIsNotNone(diagnostics)
            self.assertTrue(diagnostics.name)
            self.assertTrue(diagnostics.potential_cause)
            self.assertTrue(diagnostics.suggested_relaxation)
            self.assertTrue(diagnostics.setting_name)
            self.assertIsNotNone(diagnostics.bound_values)
        capacity = next(
            get_constraint_diagnostics(constraint)
            for constraint in constraints
            if get_constraint_diagnostics(constraint).name == "participation_capacity"
        )
        context = capacity.element_context((0, 0))
        self.assertEqual(context["parent_target_abs_shares"], 100.0)
        self.assertEqual(context["total_horizon_capacity_shares"], 20.0)
        self.assertEqual(context["capacity_shortfall_shares"], 80.0)
        self.assertIn("add at least 80 shares", context["pm_action"])

    def test_multiple_conflicts_are_reported_as_certificate_members_not_single_fix(self) -> None:
        x = cp.Variable(name="x")
        y = cp.Variable(name="y")
        constraints = [x >= 1, x <= 0, y >= 1, y <= 0]
        problem = cp.Problem(cp.Minimize(0), constraints)
        problem.solve(solver=_solver())
        problem._solver_stats = _mock_stats(
            {"IIS": {constraint.id: np.array(1.0) for constraint in constraints}},
            solver_name="MOSEK",
        )

        report = diagnose_problem(problem)

        self.assertEqual(len(report["bottlenecks"]), 4)
        self.assertIn("choose one or more business levers", report["text"].lower())
        self.assertIn("does not prove that the top rule alone is sufficient", report["decision"]["evidence_limit"])

    def test_generic_constraint_types_are_all_inventoried(self) -> None:
        x = cp.Variable(2, name="x")
        t = cp.Variable(name="t")
        matrix = cp.Variable((2, 2), symmetric=True, name="matrix")
        exp_x = cp.Variable(name="exp_x")
        exp_y = cp.Variable(name="exp_y")
        exp_z = cp.Variable(name="exp_z")
        problem = cp.Problem(
            cp.Minimize(0),
            [cp.SOC(t, x), matrix >> 0, x[0] == 1, cp.ExpCone(exp_x, exp_y, exp_z)],
        )

        report = diagnose_problem(problem)

        self.assertEqual(report["coverage"]["inspected"], 4)
        self.assertEqual(
            {row["constraint_type"] for row in report["constraints"]},
            {"SOC", "PSD", "Equality", "ExpCone"},
        )
        self.assertTrue(all(row["state"] == "unavailable" for row in report["constraints"]))
        json.dumps(report)

    def test_planner_failure_carries_automatic_single_solve_report(self) -> None:
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

        report = raised.exception.diagnostics
        self.assertIsNotNone(report)
        self.assertIsNotNone(raised.exception.problem)
        self.assertEqual(report["summary"]["analysis_mode"], "single_solve")
        self.assertEqual(report["summary"]["additional_solves"], 0)
        self.assertIn("capacity", {row["diagnostics"]["group"] for row in report["constraints"]})
        self.assertIn("completion", {row["diagnostics"]["group"] for row in report["constraints"]})
        self.assertNotIn("verification", report)

    def test_factor_exposure_constraint_is_explained_from_its_own_metadata(self) -> None:
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

        factor_rows = [
            row for row in raised.exception.diagnostics["constraints"]
            if row["diagnostics"]["group"] == "factor_exposure"
        ]
        self.assertTrue(factor_rows)
        self.assertTrue(all(row["diagnostics"]["setting_name"] for row in factor_rows))

    def test_unbounded_ray_maps_joint_direction_to_business_labels(self) -> None:
        trades = with_variable_diagnostics(
            cp.Variable((2, 2), name="trade_shares"),
            VariableDiagnostics(
                name="trade shares",
                units="shares",
                axis_labels={"date": ("D1", "D2"), "symbol": ("AAA", "BBB")},
            ),
        )
        problem = cp.Problem(cp.Minimize(-cp.sum(trades)), [trades >= 0])
        problem.solve(solver=_solver())
        problem._solver_stats = _mock_stats(
            {
                "DUAL_RAY": {trades.id: np.array([[0.0, 2.0], [-1.0, 0.0]])},
                "MOSEK_DIAGNOSTICS": {
                    "canonical_problem_status": "prim_infeas",
                    "canonical_solution_status": "prim_infeas_cer",
                    "cvxpy_dualized": True,
                    "objective_slope": -3.0,
                },
            },
            solver_name="TRADE_PLANNER_MOSEK",
        )

        report = diagnose_problem(problem)
        elements = report["improving_direction"][0]["affected_elements"]

        self.assertEqual(report["solver_evidence"]["kind"], "dual_infeasibility_ray")
        self.assertEqual(elements[0]["location"]["symbol"], "BBB")
        self.assertEqual(elements[0]["direction"], "increase")
        self.assertEqual(elements[1]["location"]["symbol"], "AAA")
        self.assertEqual(elements[1]["direction"], "decrease")
        self.assertIn("joint direction", report["decision"]["evidence_limit"])

    def test_native_mosek_snapshot_recovers_original_unbounded_ray_without_license(self) -> None:
        try:
            import mosek
        except ModuleNotFoundError:
            self.skipTest("MOSEK Python package not installed")

        class FakeTask:
            def getnumintvar(self):
                return 0

            def getnumcone(self):
                return 0

            def getintparam(self, _):
                return mosek.optimizertype.intpnt

            def getprosta(self, _):
                return mosek.prosta.prim_infeas

            def getsolsta(self, _):
                return mosek.solsta.prim_infeas_cer

            def getsolutioninfo(self, _):
                return (0.0,) * 11

            def gety(self, _):
                return np.array([2.0, -1.0, 0.0])

        snapshot = _snapshot_task(
            FakeTask(),
            {},
            {
                "dualized": True,
                "objective": np.array([-1.0, 0.0, 0.0]),
                "variables": [
                    {"id": 11, "offset": 0, "size": 1, "shape": ()},
                    {"id": 12, "offset": 1, "size": 2, "shape": (2,)},
                ],
            },
        )

        self.assertEqual(snapshot["canonical_problem_status"], "prim_infeas")
        self.assertEqual(snapshot["canonical_solution_status"], "prim_infeas_cer")
        self.assertEqual(snapshot["objective_slope"], -2.0)
        self.assertTrue(snapshot["objective_improves"])
        self.assertEqual(snapshot["original_variable_ray"][11], 2.0)
        np.testing.assert_allclose(snapshot["original_variable_ray"][12], [-1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
