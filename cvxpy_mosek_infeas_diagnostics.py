#!/usr/bin/env python3
"""Compatibility entry point for CVXPY infeasibility diagnostics.

The implementation lives in `trade_planner.diagnostics` so package code can use
the same helper that this standalone module exports.
"""

from __future__ import annotations

import argparse
from typing import Sequence

import cvxpy as cp

from trade_planner.constraints import ConstraintDiagnostics, with_diagnostics
from trade_planner.diagnostics import (
    diagnose_infeasible_problem,
    elastic_feasibility_report,
    format_infeasibility_diagnosis,
)

__all__ = [
    "diagnose_infeasible_problem",
    "elastic_feasibility_report",
    "format_infeasibility_diagnosis",
]


def build_toy_infeasible_problem() -> cp.Problem:
    """Build a tiny infeasible problem for manual smoke testing."""
    units = cp.Variable(name="ship_units")
    constraints = [
        with_diagnostics(
            units >= 10,
            ConstraintDiagnostics(
                name="demand_minimum",
                group="demand",
                description="Shipment must satisfy committed customer demand.",
                potential_cause="Demand is higher than available stock.",
                suggested_relaxation="Lower committed demand or add supply.",
                units="units",
            ),
        ),
        with_diagnostics(
            units <= 7,
            ConstraintDiagnostics(
                name="stock_maximum",
                group="inventory",
                description="Shipment cannot exceed available stock.",
                potential_cause="Stock is below committed demand.",
                suggested_relaxation="Increase available stock or split the shipment.",
                units="units",
            ),
        ),
    ]
    return cp.Problem(cp.Minimize(units), constraints)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a small CVXPY infeasibility diagnostic example.")
    parser.add_argument("--example", action="store_true", help="Run the built-in infeasible scalar example.")
    parser.add_argument("--top-k", type=int, default=10, help="Maximum report rows to print.")
    args = parser.parse_args(argv)

    if not args.example:
        parser.print_help()
        return 0

    problem = build_toy_infeasible_problem()
    problem.solve(solver=_installed_conic_solver())
    report = diagnose_infeasible_problem(problem, top_k=args.top_k)
    print(format_infeasibility_diagnosis(report, top_k=args.top_k))
    return 0


def _installed_conic_solver() -> str:
    for solver in ("CLARABEL", "ECOS", "SCS", "MOSEK"):
        if solver in cp.installed_solvers():
            return solver
    raise RuntimeError("No conic CVXPY solver is installed.")


if __name__ == "__main__":
    raise SystemExit(main())
