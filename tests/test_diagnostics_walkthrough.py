from __future__ import annotations

import unittest

from examples.diagnostics_walkthrough import (
    attach_analytical_certificate_fixture,
    sample_context,
    sample_planner,
    scenario_math,
)
from trade_planner import InfeasiblePlanError, diagnose_problem


class DiagnosticsWalkthroughTests(unittest.TestCase):
    def test_hidden_gross_milestone_conflict_is_reproducible(self) -> None:
        ctx, gross_limits = sample_context()
        arithmetic = scenario_math(ctx, gross_limits)

        self.assertEqual(arithmetic["required_first_day_gross"], 2460.0)
        self.assertEqual(arithmetic["allowed_first_day_gross"], 2000.0)
        self.assertEqual(arithmetic["gross_shortfall"], 460.0)
        self.assertEqual(arithmetic["minimum_name_capacity_ratio"], 3.0)

        with self.assertRaises(InfeasiblePlanError) as caught:
            sample_planner(ctx, gross_limits).solve(ctx)

        problem = caught.exception.problem
        self.assertIsNotNone(problem)
        attach_analytical_certificate_fixture(problem, ctx)
        report = diagnose_problem(problem)
        names = {
            row["diagnostics"]["name"]
            for row in report["bottlenecks"]
        }
        self.assertEqual(
            names,
            {
                "min_completion_by_date[2026-07-15]",
                "daily_gross_notional_limit[2026-07-15]",
            },
        )
        self.assertEqual(report["summary"]["additional_solves"], 0)


if __name__ == "__main__":
    unittest.main()
