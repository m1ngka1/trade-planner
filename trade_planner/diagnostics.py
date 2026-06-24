"""CVXPY infeasibility diagnostics for trade-planner constraints."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Mapping

import cvxpy as cp
import numpy as np

from .constraints import ConstraintDiagnostics, get_constraint_diagnostics


def diagnose_infeasible_problem(
    problem: cp.Problem,
    *,
    run_elastic: bool = True,
    solver: Any | None = None,
    tol: float = 1e-7,
    top_k: int = 10,
) -> dict[str, Any]:
    """Inspect a solved CVXPY problem and return user-facing infeasibility diagnostics."""
    status = getattr(problem, "status", None)
    status_family = _status_family(status)
    constraints = [_constraint_row(i, c) for i, c in enumerate(problem.constraints)]
    solver_evidence = _solver_constraint_evidence(problem, tol=tol, top_k=top_k)

    elastic = None
    if run_elastic and status_family in {"infeasible", "infeasible_inaccurate"}:
        elastic = elastic_feasibility_report(problem, solver=solver, tol=tol, top_k=top_k)

    report: dict[str, Any] = {
        "summary": {
            "status": status,
            "status_family": status_family,
            "value": _finite_or_str(getattr(problem, "value", None)),
            "is_dcp": problem.is_dcp(),
            "num_constraints": len(problem.constraints),
            "num_variables": len(problem.variables()),
            "message": _summary_message(status_family, elastic),
        },
        "constraints": constraints,
        "solver_evidence": solver_evidence,
        "elastic": elastic,
        "recommendations": _recommendations(status_family, elastic, solver_evidence, top_k),
    }
    report["text"] = format_infeasibility_diagnosis(report, top_k=top_k)
    return report


def elastic_feasibility_report(
    problem: cp.Problem,
    *,
    solver: Any | None = None,
    tol: float = 1e-7,
    top_k: int = 10,
    penalize_by_size: bool = False,
) -> dict[str, Any]:
    """Solve a relaxed feasibility model and rank required constraint slacks."""
    relaxed_constraints: list[cp.Constraint] = []
    penalties: list[cp.Expression] = []
    slack_records: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    for index, constraint in enumerate(problem.constraints):
        expr = getattr(constraint, "expr", None)
        diagnostics = _diagnostics_or_default(index, constraint)
        constraint_type = type(constraint).__name__

        if expr is None:
            relaxed_constraints.append(constraint)
            unsupported.append(_unsupported_row(index, constraint, diagnostics, "constraint has no expression"))
            continue

        if constraint_type == "Inequality":
            slack = cp.Variable(expr.shape, nonneg=True, name=f"slack__{_safe_name(diagnostics.name)}")
            relaxed_constraints.append(expr <= slack)
            penalties.append(float(diagnostics.weight) * _penalty(slack, expr.shape, penalize_by_size))
            slack_records.append(
                {
                    "index": index,
                    "constraint": constraint,
                    "diagnostics": diagnostics,
                    "slack": slack,
                    "kind": "inequality",
                    "shape": tuple(expr.shape),
                }
            )
        elif constraint_type == "Equality":
            slack = cp.Variable(expr.shape, nonneg=True, name=f"slack__{_safe_name(diagnostics.name)}")
            relaxed_constraints.extend([expr <= slack, -expr <= slack])
            penalties.append(float(diagnostics.weight) * _penalty(slack, expr.shape, penalize_by_size))
            slack_records.append(
                {
                    "index": index,
                    "constraint": constraint,
                    "diagnostics": diagnostics,
                    "slack": slack,
                    "kind": "equality",
                    "shape": tuple(expr.shape),
                }
            )
        else:
            relaxed_constraints.append(constraint)
            unsupported.append(
                _unsupported_row(index, constraint, diagnostics, "only Equality and Inequality constraints are relaxed")
            )

    if not penalties:
        return {
            "status": "not_run",
            "solver": None,
            "objective": None,
            "violations": [],
            "unsupported_constraints": unsupported,
            "solve_error": "no relaxable equality or inequality constraints were found",
        }

    relaxed_problem = cp.Problem(cp.Minimize(sum(penalties)), relaxed_constraints)
    solver_used, solve_error = _solve_relaxed_problem(relaxed_problem, solver)

    violations: list[dict[str, Any]] = []
    if solve_error is None:
        for record in slack_records:
            slack_value = record["slack"].value
            if slack_value is None:
                continue

            slack_array = np.asarray(slack_value, dtype=float)
            finite = np.isfinite(slack_array)
            if not np.any(finite):
                continue

            max_relaxation = float(np.max(slack_array[finite]))
            if max_relaxation <= tol:
                continue

            total_relaxation = float(np.sum(slack_array[finite]))
            mean_relaxation = float(np.mean(slack_array[finite]))
            max_flat_index = int(np.nanargmax(slack_array))
            diagnostics: ConstraintDiagnostics = record["diagnostics"]

            row = {
                "cvxpy_index": record["index"],
                "constraint_type": type(record["constraint"]).__name__,
                "kind": record["kind"],
                "shape": record["shape"],
                "max_relaxation": max_relaxation,
                "total_relaxation": total_relaxation,
                "mean_relaxation": mean_relaxation,
                "max_location": _slack_location(slack_array.shape, max_flat_index, diagnostics),
                "slack_sample": slack_array.ravel()[:20].tolist(),
                "diagnostics": _diagnostics_dict(diagnostics),
                "constraint": _short_str(record["constraint"], 400),
            }
            row["recommendation"] = _relaxation_recommendation(row)
            violations.append(row)

    violations.sort(key=lambda r: (r["total_relaxation"], r["max_relaxation"]), reverse=True)

    return {
        "status": getattr(relaxed_problem, "status", None) if solve_error is None else "solve_error",
        "solver": solver_used,
        "objective": _finite_or_str(getattr(relaxed_problem, "value", None)) if solve_error is None else None,
        "violations": violations[:top_k],
        "unsupported_constraints": unsupported,
        "solve_error": solve_error,
    }


def format_infeasibility_diagnosis(report: Mapping[str, Any], *, top_k: int = 10) -> str:
    """Format a diagnostic report into a concise text explanation."""
    summary = dict(report.get("summary", {}) or {})
    elastic = report.get("elastic")
    recommendations = list(report.get("recommendations", []) or [])

    lines: list[str] = []
    lines.append("=== CVXPY infeasibility diagnosis ===")
    lines.append(f"Status: {summary.get('status')} ({summary.get('status_family')})")
    lines.append(str(summary.get("message") or "No status message was generated."))
    lines.append("")

    if isinstance(elastic, Mapping):
        if elastic.get("solve_error"):
            lines.append(f"Elastic relaxation failed: {elastic.get('solve_error')}")
            lines.append("")
        else:
            lines.append(
                "Elastic relaxation: "
                f"status={elastic.get('status')}, solver={elastic.get('solver')}, "
                f"minimum weighted slack={_fmt_float(elastic.get('objective'))}"
            )
            violations = list(elastic.get("violations", []) or [])
            if violations:
                lines.append("Likely infeasible constraints and relaxation suggestions:")
                for rank, row in enumerate(violations[:top_k], start=1):
                    diagnostics = dict(row.get("diagnostics", {}) or {})
                    units = f" {diagnostics.get('units')}" if diagnostics.get("units") else ""
                    location = _format_location(row.get("max_location"))
                    lines.append(
                        f"  {rank}. {diagnostics.get('name')} "
                        f"[{diagnostics.get('group') or 'ungrouped'}]: "
                        f"max slack {_fmt_float(row.get('max_relaxation'))}{units}{location}"
                    )
                    if diagnostics.get("potential_cause"):
                        lines.append(f"     Potential cause: {diagnostics.get('potential_cause')}")
                    if row.get("recommendation"):
                        lines.append(f"     Suggested relaxation: {row.get('recommendation')}")
            else:
                lines.append("No positive slack was required in relaxable constraints.")
            lines.append("")

    if recommendations:
        lines.append("Recommendations:")
        for item in recommendations[:top_k]:
            lines.append(f"  - {item.get('title')}: {item.get('detail')}")
            if item.get("action"):
                lines.append(f"    Action: {item.get('action')}")

    return "\n".join(lines)


def _constraint_row(index: int, constraint: cp.Constraint) -> dict[str, Any]:
    diagnostics = _diagnostics_or_default(index, constraint)
    return {
        "cvxpy_index": index,
        "constraint_id": getattr(constraint, "id", None),
        "constraint_type": type(constraint).__name__,
        "shape": tuple(getattr(constraint, "shape", ())),
        "size": int(getattr(constraint, "size", 0)),
        "diagnostics": _diagnostics_dict(diagnostics),
        "constraint": _short_str(constraint, 400),
    }


def _diagnostics_or_default(index: int, constraint: cp.Constraint) -> ConstraintDiagnostics:
    diagnostics = get_constraint_diagnostics(constraint)
    if diagnostics is not None:
        return diagnostics
    return ConstraintDiagnostics(
        name=f"constraint_{index}",
        group="unclassified",
        description="Unnamed CVXPY constraint.",
        potential_cause="This unnamed constraint may conflict with the rest of the model.",
        suggested_relaxation="Attach ConstraintDiagnostics to provide a domain-specific relaxation suggestion.",
    )


def _diagnostics_dict(diagnostics: ConstraintDiagnostics) -> dict[str, Any]:
    data = asdict(diagnostics)
    data["axis_labels"] = {str(k): list(v) for k, v in diagnostics.axis_labels.items()}
    data["details"] = dict(diagnostics.details)
    return data


def _penalty(slack: cp.Variable, shape: tuple[int, ...], penalize_by_size: bool) -> cp.Expression:
    penalty = cp.sum(slack)
    if not penalize_by_size:
        return penalty
    size = int(np.prod(shape or (1,)))
    return penalty / max(1, size)


def _solve_relaxed_problem(problem: cp.Problem, solver: Any | None) -> tuple[str | None, str | None]:
    if solver is not None:
        try:
            problem.solve(solver=solver)
            return str(solver), None
        except Exception as exc:
            return str(solver), repr(exc)

    errors: list[str] = []
    for candidate in _installed_solver_preference():
        try:
            problem.solve(solver=candidate)
            return str(candidate), None
        except Exception as exc:
            errors.append(f"{candidate}: {exc!r}")
    return None, "; ".join(errors) if errors else "no installed CVXPY solver was available"


def _installed_solver_preference() -> list[str]:
    installed = set(cp.installed_solvers())
    preferred = ["CLARABEL", "ECOS", "SCS", "MOSEK", "OSQP", "SCIPY"]
    return [solver for solver in preferred if solver in installed]


def _slack_location(shape: tuple[int, ...], flat_index: int, diagnostics: ConstraintDiagnostics) -> dict[str, Any]:
    if not shape:
        return {}

    multi_index = np.unravel_index(flat_index, shape)
    axis_labels = list(diagnostics.axis_labels.items())
    location: dict[str, Any] = {"index": [int(i) for i in multi_index]}
    if len(axis_labels) == len(shape):
        for axis, axis_index in zip(axis_labels, multi_index):
            name, labels = axis
            if int(axis_index) < len(labels):
                location[str(name)] = labels[int(axis_index)]
    elif len(axis_labels) == 1 and len(shape) == 1:
        name, labels = axis_labels[0]
        axis_index = int(multi_index[0])
        if axis_index < len(labels):
            location[str(name)] = labels[axis_index]
    if diagnostics.details:
        for key in ("date", "symbol", "factor"):
            if key in diagnostics.details and key not in location:
                location[key] = diagnostics.details[key]
    return location


def _format_location(location: Any) -> str:
    if not isinstance(location, Mapping) or not location:
        return ""
    parts = [f"{key}={value}" for key, value in location.items() if key != "index"]
    if not parts and location.get("index") is not None:
        parts = [f"index={location.get('index')}"]
    return " at " + ", ".join(parts) if parts else ""


def _relaxation_recommendation(row: Mapping[str, Any]) -> str:
    diagnostics = dict(row.get("diagnostics", {}) or {})
    amount = _fmt_float(row.get("max_relaxation"))
    units = f" {diagnostics.get('units')}" if diagnostics.get("units") else ""
    suggestion = diagnostics.get("suggested_relaxation") or "Loosen this constraint."
    return f"{suggestion} Required max relaxation is {amount}{units}."


def _solver_constraint_evidence(problem: cp.Problem, *, tol: float, top_k: int) -> dict[str, Any]:
    stats = getattr(problem, "solver_stats", None)
    extra = getattr(stats, "extra_stats", None) if stats is not None else None
    if not isinstance(extra, Mapping) or "IIS" not in extra:
        return {"available": False, "note": "No high-level IIS evidence was exposed by solver_stats.extra_stats."}

    iis = extra.get("IIS")
    if not isinstance(iis, Mapping):
        return {"available": False, "note": "IIS evidence was present but not mapping-shaped."}

    by_id = {getattr(c, "id", None): (i, c) for i, c in enumerate(problem.constraints)}
    rows: list[dict[str, Any]] = []
    for key, value in iis.items():
        arr = _numeric_array(value)
        if arr is None:
            continue
        nz = np.flatnonzero(np.isfinite(arr) & (np.abs(arr) > tol))
        if nz.size == 0:
            continue
        index, constraint = by_id.get(key, (None, None))
        if constraint is None:
            try:
                index, constraint = by_id.get(int(key), (None, None))
            except Exception:
                constraint = None
        diagnostics = _diagnostics_or_default(index if index is not None else -1, constraint) if constraint is not None else None
        rows.append(
            {
                "constraint_id": key,
                "cvxpy_index": index,
                "max_abs_certificate_entry": float(np.max(np.abs(arr[nz]))),
                "num_nonzero_certificate_entries": int(nz.size),
                "diagnostics": _diagnostics_dict(diagnostics) if diagnostics is not None else None,
            }
        )
    rows.sort(key=lambda r: r["max_abs_certificate_entry"], reverse=True)
    return {"available": bool(rows), "constraints": rows[:top_k]}


def _recommendations(
    status_family: str,
    elastic: Mapping[str, Any] | None,
    solver_evidence: Mapping[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []

    if status_family in {"infeasible", "infeasible_inaccurate"}:
        if isinstance(elastic, Mapping) and elastic.get("violations"):
            for row in list(elastic.get("violations", []) or [])[: min(top_k, 5)]:
                diagnostics = dict(row.get("diagnostics", {}) or {})
                recommendations.append(
                    {
                        "title": f"Relax {diagnostics.get('name')}",
                        "detail": row.get("recommendation"),
                        "action": diagnostics.get("suggested_relaxation"),
                    }
                )
        elif isinstance(elastic, Mapping) and elastic.get("solve_error"):
            recommendations.append(
                {
                    "title": "Elastic relaxation did not solve",
                    "detail": "The diagnostic could not compute required slack amounts.",
                    "action": "Inspect the solve error and try a conic solver such as CLARABEL or SCS.",
                }
            )
        else:
            recommendations.append(
                {
                    "title": "Run elastic relaxation",
                    "detail": "The solved status is infeasible but no relaxation amounts were computed.",
                    "action": "Call diagnose_infeasible_problem(..., run_elastic=True).",
                }
            )

    if status_family in {"unbounded", "unbounded_inaccurate", "infeasible_or_unbounded"}:
        recommendations.append(
            {
                "title": "Check bounds and objective direction",
                "detail": "The status is dual-infeasible or unbounded; slack relaxations are not the primary explanation.",
                "action": (
                    "Verify variable bounds, participation/capacity constraints, hard completion constraints, "
                    "objective signs, and any missing domain constraints."
                ),
            }
        )

    if solver_evidence.get("available"):
        names = [
            str((row.get("diagnostics") or {}).get("name"))
            for row in list(solver_evidence.get("constraints", []) or [])[:top_k]
            if row.get("diagnostics")
        ]
        recommendations.append(
            {
                "title": "Inspect solver IIS evidence",
                "detail": "Solver certificate evidence references: " + ", ".join(names),
                "action": "Look for contradictory requirements among these constraint groups.",
            }
        )

    return recommendations


def _unsupported_row(
    index: int,
    constraint: cp.Constraint,
    diagnostics: ConstraintDiagnostics,
    reason: str,
) -> dict[str, Any]:
    return {
        "cvxpy_index": index,
        "constraint_type": type(constraint).__name__,
        "reason": reason,
        "diagnostics": _diagnostics_dict(diagnostics),
    }


def _status_family(status: Any) -> str:
    text = str(status or "not_solved").lower()
    if "infeasible" in text and "unbounded" in text:
        return "infeasible_or_unbounded"
    if "infeasible" in text and "inaccurate" in text:
        return "infeasible_inaccurate"
    if "unbounded" in text and "inaccurate" in text:
        return "unbounded_inaccurate"
    if "infeasible" in text:
        return "infeasible"
    if "unbounded" in text:
        return "unbounded"
    if "optimal" in text or "solved" in text:
        return "solved"
    if "unknown" in text or "user_limit" in text or "inaccurate" in text:
        return "uncertain"
    return "not_solved"


def _summary_message(status_family: str, elastic: Mapping[str, Any] | None) -> str:
    if status_family in {"infeasible", "infeasible_inaccurate"}:
        if isinstance(elastic, Mapping) and elastic.get("violations"):
            return "The problem is primal infeasible; the elastic pass found concrete relaxation candidates."
        if isinstance(elastic, Mapping) and elastic.get("solve_error"):
            return "The problem is primal infeasible, but the elastic relaxation pass failed."
        return "The problem is primal infeasible; run elastic relaxation for concrete relaxation amounts."
    if status_family in {"unbounded", "unbounded_inaccurate"}:
        return "The problem appears dual infeasible or unbounded; inspect missing bounds and objective direction."
    if status_family == "infeasible_or_unbounded":
        return "The solver could not distinguish infeasible from unbounded; inspect bounds and re-solve with stricter settings."
    if status_family == "solved":
        return "The problem solved successfully; no infeasibility relaxation is indicated."
    return "The problem status is not a solved infeasible/unbounded terminal status."


def _numeric_array(value: Any) -> np.ndarray | None:
    try:
        return np.asarray(value, dtype=float).ravel()
    except Exception:
        return None


def _safe_name(name: str) -> str:
    chars = [ch if ch.isalnum() or ch == "_" else "_" for ch in str(name)]
    safe = "".join(chars).strip("_")
    return safe or "constraint"


def _short_str(value: Any, max_chars: int) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _fmt_float(value: Any) -> str:
    if value is None:
        return "None"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.6g}"


def _finite_or_str(value: Any) -> Any:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return str(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return number
