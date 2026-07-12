"""Generic diagnostics and recovery checks for CVXPY problem objects."""

from __future__ import annotations

import copy
from dataclasses import asdict
import math
from typing import Any, Mapping

import cvxpy as cp
import numpy as np

from .constraints import ConstraintDiagnostics, get_constraint_diagnostics


def diagnose_problem(
    problem: cp.Problem,
    *,
    solve_if_needed: bool = False,
    verify_bottlenecks: bool = True,
    max_verification_checks: int | None = None,
    solver: Any | None = None,
    solve_kwargs: Mapping[str, Any] | None = None,
    tol: float = 1e-7,
    top_k: int = 10,
) -> dict[str, Any]:
    """Inspect every original constraint and verify single-constraint recovery.

    The original object is never mutated. For an infeasible problem, isolated
    copies are solved with one constraint omitted at a time; no multi-constraint
    slack or synthetic relaxation model is constructed. Set
    ``verify_bottlenecks=False`` to skip these counterfactual checks.
    ``solve_if_needed=True`` permits solving an unsolved original problem once.
    """
    initial_status = problem.status
    solve_error = _solve_original(problem, solver, solve_kwargs) if initial_status is None and solve_if_needed else None
    status_family = _status_family(problem.status)
    constraints = [_inspect_constraint(i, constraint, tol) for i, constraint in enumerate(problem.constraints)]
    solver_evidence = _solver_evidence(problem, constraints, tol)
    candidates = _rank_evidenced_constraints(
        constraints,
        solver_evidence,
        status_family,
        tol,
        len(constraints),
    )
    verification = _empty_verification()
    bottlenecks = candidates[:top_k]
    if verify_bottlenecks and status_family in {"infeasible", "infeasible_inaccurate"}:
        verified, verification = _verify_single_constraint_recoveries(
            problem,
            constraints,
            candidates,
            solver,
            solve_kwargs,
            max_verification_checks,
        )
        if verified:
            bottlenecks = verified[:top_k]
        elif verification.get("complete"):
            bottlenecks = []
    coverage = _coverage(constraints, bottlenecks, verification)

    report: dict[str, Any] = {
        "summary": {
            "status": problem.status,
            "status_family": status_family,
            "value": _finite_or_str(problem.value),
            "is_dcp": problem.is_dcp(),
            "num_constraints": len(problem.constraints),
            "num_variables": len(problem.variables()),
            "was_solved_by_diagnostic": initial_status is None and problem.status is not None,
            "solve_error": solve_error,
            "message": _summary_message(status_family, bottlenecks, solver_evidence, verification),
        },
        "coverage": coverage,
        "constraints": constraints,
        "variables": [_inspect_variable(variable) for variable in problem.variables()],
        "bottlenecks": bottlenecks,
        "verification": verification,
        "solver_stats": _solver_stats(problem),
        "solver_evidence": solver_evidence,
        "recommendations": _recommendations(status_family, bottlenecks, solver_evidence, verification),
    }
    report["text"] = format_diagnosis(report, top_k=top_k)
    return report


# Backward-compatible name; the implementation is no longer infeasibility-only.
diagnose_infeasible_problem = diagnose_problem


def format_diagnosis(report: Mapping[str, Any], *, top_k: int = 10) -> str:
    """Render a concise, evidence-qualified model diagnosis."""
    summary = dict(report.get("summary", {}) or {})
    coverage = dict(report.get("coverage", {}) or {})
    bottlenecks = list(report.get("bottlenecks", []) or [])
    recommendations = list(report.get("recommendations", []) or [])

    lines = [
        "=== CVXPY model diagnosis ===",
        f"Status: {summary.get('status')} ({summary.get('status_family')})",
        str(summary.get("message") or "No status message was generated."),
        (
            f"Inspected {coverage.get('inspected', 0)}/{summary.get('num_constraints', 0)} constraints; "
            f"metadata={coverage.get('with_metadata', 0)}, primal metrics={coverage.get('with_primal_metrics', 0)}, "
            f"dual values={coverage.get('with_dual_values', 0)}."
        ),
    ]

    if bottlenecks:
        lines.extend(["", "Evidence-backed bottlenecks:"])
        for row in bottlenecks[:top_k]:
            diagnostics = dict(row.get("diagnostics", {}) or {})
            lines.append(
                f"  {row.get('rank')}. {diagnostics.get('name')} "
                f"[{diagnostics.get('group') or 'ungrouped'}]: "
                f"{row.get('source')}={_fmt_float(row.get('severity'))}"
            )
            if diagnostics.get("potential_cause"):
                lines.append(f"     Interpretation: {diagnostics.get('potential_cause')}")
            verification = dict(row.get("verification", {}) or {})
            if verification.get("restores_feasibility"):
                witness = (verification.get("witness_violation") or {}).get("max_abs")
                units = f" {diagnostics.get('units')}" if diagnostics.get("units") else ""
                location = _format_location(verification.get("witness_location"))
                lines.append(
                    f"     Verified: omitting only this constraint solved the remaining model "
                    f"({verification.get('status_without_constraint')}); "
                    f"feasible-witness violation={_fmt_float(witness)}{units}{location}."
                )

    if recommendations:
        lines.extend(["", "Recommendations:"])
        for item in recommendations[:top_k]:
            lines.append(f"  - {item.get('title')}: {item.get('detail')}")
            if item.get("action"):
                lines.append(f"    Action: {item.get('action')}")
    return "\n".join(lines)


format_infeasibility_diagnosis = format_diagnosis


def _inspect_constraint(index: int, constraint: cp.Constraint, tol: float) -> dict[str, Any]:
    attached = get_constraint_diagnostics(constraint)
    diagnostics = attached or _default_diagnostics(index, constraint)
    dual = _array_summary(getattr(constraint, "dual_value", None))
    violation = _constraint_violation(constraint)
    slack, active = _constraint_slack(constraint, tol)
    has_primal = violation is not None

    if violation and violation.get("max_abs", 0.0) > tol:
        state = "violated"
    elif active is True:
        state = "active"
    elif has_primal:
        state = "satisfied"
    else:
        state = "unavailable"

    return {
        "cvxpy_index": index,
        "constraint_id": getattr(constraint, "id", None),
        "constraint_type": type(constraint).__name__,
        "shape": tuple(getattr(constraint, "shape", ())),
        "size": int(getattr(constraint, "size", 0)),
        "is_dcp": _safe_bool_call(constraint, "is_dcp"),
        "state": state,
        "variables": [variable.name() for variable in constraint.variables()],
        "parameters": [parameter.name() for parameter in constraint.parameters()],
        "diagnostics_attached": attached is not None,
        "diagnostics": _diagnostics_dict(diagnostics),
        "dual": dual,
        "violation": violation,
        "slack": slack,
        "is_active": active,
        "constraint": _short_str(constraint, 400),
    }


def _inspect_variable(variable: cp.Variable) -> dict[str, Any]:
    attributes = {
        key: _json_safe(value)
        for key, value in variable.attributes.items()
        if _meaningful_attribute(value)
    }
    return {
        "variable_id": getattr(variable, "id", None),
        "name": variable.name(),
        "shape": tuple(variable.shape),
        "size": int(variable.size),
        "attributes": attributes,
        "has_value": variable.value is not None,
    }


def _constraint_violation(constraint: cp.Constraint) -> dict[str, Any] | None:
    try:
        return _array_summary(constraint.violation())
    except (ValueError, TypeError, NotImplementedError):
        return None


def _constraint_witness_violation(constraint: cp.Constraint) -> dict[str, Any] | None:
    """Return elementwise witness violations where the constraint API allows it."""
    constraint_type = type(constraint).__name__
    if constraint_type in {"Inequality", "Equality"}:
        try:
            value = np.asarray(constraint.expr.value, dtype=float)
        except (AttributeError, TypeError, ValueError):
            value = None
        if value is not None:
            residual = np.maximum(value, 0.0) if constraint_type == "Inequality" else np.abs(value)
            return _array_summary(residual)
    return _constraint_violation(constraint)


def _constraint_slack(constraint: cp.Constraint, tol: float) -> tuple[dict[str, Any] | None, bool | None]:
    constraint_type = type(constraint).__name__
    if constraint_type not in {"Inequality", "Equality"}:
        violation = _constraint_violation(constraint)
        return None, bool(violation and violation.get("max_abs", math.inf) <= tol) if violation else None
    try:
        expr = constraint.expr
    except (AttributeError, ValueError):
        return None, None
    value = getattr(expr, "value", None)
    if value is None:
        return None, None
    if constraint_type == "Inequality":
        slack_values = np.maximum(-np.asarray(value, dtype=float), 0.0)
        return _array_summary(slack_values), bool(np.min(slack_values) <= tol)
    return None, True


def _rank_evidenced_constraints(
    constraints: list[dict[str, Any]],
    solver_evidence: Mapping[str, Any],
    status_family: str,
    tol: float,
    top_k: int,
) -> list[dict[str, Any]]:
    ranked: dict[int, dict[str, Any]] = {}
    for evidence in solver_evidence.get("constraints", []) or []:
        index = evidence.get("cvxpy_index")
        if index is None:
            continue
        row = constraints[int(index)]
        ranked[int(index)] = _bottleneck_row(
            row,
            "solver_certificate",
            float(evidence.get("max_abs_certificate_entry", 0.0)),
        )

    for row in constraints:
        index = int(row["cvxpy_index"])
        violation = (row.get("violation") or {}).get("max_abs")
        dual = (row.get("dual") or {}).get("max_abs")
        if violation is not None and float(violation) > tol:
            ranked[index] = _bottleneck_row(row, "residual_violation", float(violation))
        elif status_family in {"infeasible", "infeasible_inaccurate"} and dual is not None and float(dual) > tol:
            ranked[index] = _bottleneck_row(row, "infeasibility_dual", float(dual))
        elif index not in ranked and row.get("is_active") and dual is not None and float(dual) > tol:
            ranked[index] = _bottleneck_row(row, "active_dual", float(dual))

    result = sorted(ranked.values(), key=lambda item: item["severity"], reverse=True)[:top_k]
    for rank, row in enumerate(result, start=1):
        row["rank"] = rank
    return result


def _bottleneck_row(row: Mapping[str, Any], source: str, severity: float) -> dict[str, Any]:
    return {
        "cvxpy_index": row.get("cvxpy_index"),
        "constraint_id": row.get("constraint_id"),
        "source": source,
        "severity": severity,
        "diagnostics": row.get("diagnostics"),
        "dual": row.get("dual"),
        "violation": row.get("violation"),
        "slack": row.get("slack"),
    }


def _empty_verification() -> dict[str, Any]:
    return {
        "performed": False,
        "complete": False,
        "checked": 0,
        "solve_errors": 0,
        "single_constraint_recoveries": 0,
    }


def _verify_single_constraint_recoveries(
    problem: cp.Problem,
    constraints: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    solver: Any | None,
    solve_kwargs: Mapping[str, Any] | None,
    max_checks: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prove whether omitting one original constraint restores an optimum."""
    candidate_by_index = {int(row["cvxpy_index"]): row for row in candidates}
    ordered_indices = list(candidate_by_index)
    ordered_indices.extend(index for index in range(len(constraints)) if index not in candidate_by_index)
    if max_checks is not None:
        ordered_indices = ordered_indices[: max(0, max_checks)]

    recovered: list[dict[str, Any]] = []
    solve_errors = 0
    solver_to_use = _verification_solver(problem, solver)
    for index in ordered_indices:
        try:
            cloned = _clone_problem(problem)
            removed = cloned.constraints[index]
            remaining = [constraint for i, constraint in enumerate(cloned.constraints) if i != index]
            counterfactual = cp.Problem(cloned.objective, remaining)
            _solve_counterfactual(counterfactual, solver_to_use, solve_kwargs)
        except Exception:
            solve_errors += 1
            continue

        if _status_family(counterfactual.status) not in {"solved", "solved_inaccurate"}:
            continue
        row = dict(
            candidate_by_index.get(index)
            or _bottleneck_row(constraints[index], "no_prior_numeric_evidence", 0.0)
        )
        row["evidence_source"] = row["source"]
        row["source"] = "single_constraint_recovery"
        row["verification"] = {
            "method": "single_constraint_omission",
            "restores_feasibility": True,
            "status_without_constraint": counterfactual.status,
            "solver": solver_to_use,
            "witness_violation": _constraint_witness_violation(removed),
        }
        row["verification"]["witness_location"] = _witness_location(
            row["verification"]["witness_violation"],
            row.get("diagnostics") or {},
        )
        recovered.append(row)

    recovered.sort(
        key=lambda row: (
            bool((row.get("diagnostics") or {}).get("hard", True)),
            -float(row.get("severity", 0.0)),
        )
    )
    for rank, row in enumerate(recovered, start=1):
        row["rank"] = rank
    return recovered, {
        "performed": True,
        "complete": len(ordered_indices) == len(constraints) and solve_errors == 0,
        "checked": len(ordered_indices),
        "solve_errors": solve_errors,
        "single_constraint_recoveries": len(recovered),
    }


def _clone_problem(problem: cp.Problem) -> cp.Problem:
    """Clone an expression tree with fresh leaves and no shared solve state."""
    replacements: dict[int, Any] = {}
    for variable in problem.variables():
        replacements[id(variable)] = cp.Variable(
            variable.shape,
            name=variable.name(),
            **_leaf_attributes(variable.attributes),
        )
    for parameter in problem.parameters():
        replacements[id(parameter)] = cp.Parameter(
            parameter.shape,
            name=parameter.name(),
            value=copy.deepcopy(parameter.value),
            **_leaf_attributes(parameter.attributes),
        )
    objective = problem.objective.tree_copy(replacements)
    constraints = [constraint.tree_copy(replacements) for constraint in problem.constraints]
    return cp.Problem(objective, constraints)


def _leaf_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in attributes.items()
        if _meaningful_attribute(value)
    }


def _verification_solver(problem: cp.Problem, solver: Any | None) -> Any | None:
    if solver is not None:
        return solver
    stats = getattr(problem, "solver_stats", None)
    original_solver = getattr(stats, "solver_name", None) if stats is not None else None
    return original_solver if original_solver in cp.installed_solvers() else None


def _solve_counterfactual(
    problem: cp.Problem,
    solver: Any | None,
    solve_kwargs: Mapping[str, Any] | None,
) -> None:
    kwargs = dict(solve_kwargs or {})
    kwargs.pop("solver", None)
    if solver is None:
        problem.solve(**kwargs)
    else:
        problem.solve(solver=solver, **kwargs)


def _solver_evidence(
    problem: cp.Problem,
    constraints: list[dict[str, Any]],
    tol: float,
) -> dict[str, Any]:
    stats = getattr(problem, "solver_stats", None)
    extra = getattr(stats, "extra_stats", None) if stats is not None else None
    if extra is not None and not isinstance(extra, Mapping) and hasattr(extra, "__dict__"):
        extra = vars(extra)
    if not isinstance(extra, Mapping):
        return {"available": False, "mapped": False, "keys": [], "constraints": [], "note": "No solver-specific evidence was exposed."}

    certificate = next(
        (extra[key] for key in ("IIS", "iis") if isinstance(extra.get(key), Mapping)),
        None,
    )
    by_id = {row["constraint_id"]: row for row in constraints}
    mapped: list[dict[str, Any]] = []
    if certificate is not None:
        for key, value in certificate.items():
            row = by_id.get(key)
            if row is None:
                try:
                    row = by_id.get(int(key))
                except (TypeError, ValueError):
                    row = None
            array = _numeric_array(value)
            if row is None or array is None:
                continue
            nonzero = array[np.isfinite(array) & (np.abs(array) > tol)]
            if nonzero.size:
                mapped.append(
                    {
                        "constraint_id": row["constraint_id"],
                        "cvxpy_index": row["cvxpy_index"],
                        "max_abs_certificate_entry": float(np.max(np.abs(nonzero))),
                        "num_nonzero_certificate_entries": int(nonzero.size),
                    }
                )
    mapped.sort(key=lambda item: item["max_abs_certificate_entry"], reverse=True)
    return {
        "available": bool(extra),
        "mapped": bool(mapped),
        "keys": sorted(str(key) for key in extra.keys()),
        "constraints": mapped,
        "note": (
            "Solver evidence was mapped to original CVXPY constraint ids."
            if mapped
            else "Solver data exists, but it does not expose a constraint-id mapping; it is not used to rank constraints."
        ),
    }


def _coverage(
    constraints: list[dict[str, Any]],
    bottlenecks: list[dict[str, Any]],
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "inspected": len(constraints),
        "with_metadata": sum(bool(row["diagnostics_attached"]) for row in constraints),
        "with_primal_metrics": sum(row["violation"] is not None for row in constraints),
        "with_dual_values": sum(row["dual"] is not None for row in constraints),
        "evidence_backed_bottlenecks": len(bottlenecks),
        "verified_single_constraint_recoveries": int(verification.get("single_constraint_recoveries", 0)),
        "unresolved": not bool(bottlenecks),
    }


def _recommendations(
    status_family: str,
    bottlenecks: list[dict[str, Any]],
    solver_evidence: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if bottlenecks:
        return [
            {
                "title": f"Inspect {(row.get('diagnostics') or {}).get('name')}",
                "detail": _recommendation_detail(row),
                "action": (row.get("diagnostics") or {}).get("suggested_relaxation"),
            }
            for row in bottlenecks
        ]
    if verification.get("performed") and verification.get("complete"):
        return [{
            "title": "No single-constraint recovery",
            "detail": "Removing each original constraint individually failed to restore an optimum.",
            "action": "Treat this as a multi-constraint conflict; do not promise that relaxing one constraint will solve it.",
        }]
    if status_family in {"infeasible", "infeasible_inaccurate"}:
        return [{
            "title": "No original-constraint mapping available",
            "detail": "The infeasible solve did not populate primal/dual values or a mapped certificate, so naming one constraint would be speculative.",
            "action": "Review the complete constraints list or enable a solver-native certificate that preserves original constraint ids.",
        }]
    if status_family in {"unbounded", "unbounded_inaccurate", "infeasible_or_unbounded"}:
        return [{
            "title": "Inspect objective direction and variable bounds",
            "detail": "Unboundedness is evidence about a feasible direction, not a violated constraint.",
            "action": "Review the variables section, objective signs, and missing explicit bounds.",
        }]
    if status_family == "solved_inaccurate":
        return [{
            "title": "Validate reduced-accuracy results",
            "detail": "Residual and dual rankings may be noisy.",
            "action": "Tighten solver tolerances and compare with another suitable solver.",
        }]
    if not solver_evidence.get("mapped") and status_family == "not_solved":
        return [{
            "title": "Solve the original problem first",
            "detail": "Structural inventory is available, but numeric evidence requires a solve.",
            "action": "Solve externally or pass solve_if_needed=True.",
        }]
    return []


def _summary_message(
    status_family: str,
    bottlenecks: list[dict[str, Any]],
    solver_evidence: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> str:
    recoveries = int(verification.get("single_constraint_recoveries", 0))
    if recoveries:
        return f"Verified {recoveries} constraint(s) whose individual removal restores an optimal solution."
    if verification.get("performed") and verification.get("complete"):
        return "No single original constraint can be removed to restore an optimal solution; the conflict requires multiple changes."
    if bottlenecks:
        return f"The original problem exposes evidence for {len(bottlenecks)} limiting constraint(s)."
    if status_family in {"infeasible", "infeasible_inaccurate"}:
        return "The original problem is infeasible, but it exposes no evidence that can be mapped to a specific constraint."
    if status_family in {"unbounded", "unbounded_inaccurate"}:
        return "The original problem is unbounded; inspect objective direction and variable bounds."
    if status_family == "infeasible_or_unbounded":
        return "The solver could not distinguish infeasible from unbounded."
    if status_family == "solved_inaccurate":
        return "The original problem solved to reduced accuracy."
    if status_family == "solved":
        return "The original problem solved; active constraints are ranked when dual evidence is available."
    if status_family == "not_solved":
        return "The original problem has not been solved; all constraints were inventoried without mutation."
    return str(solver_evidence.get("note") or "The problem status is uncertain.")


def _recommendation_detail(row: Mapping[str, Any]) -> str:
    verification = dict(row.get("verification", {}) or {})
    if verification.get("restores_feasibility"):
        witness = (verification.get("witness_violation") or {}).get("max_abs")
        diagnostics = dict(row.get("diagnostics", {}) or {})
        units = f" {diagnostics.get('units')}" if diagnostics.get("units") else ""
        location = _format_location(verification.get("witness_location"))
        return (
            f"Omitting only this constraint made every remaining original constraint solvable "
            f"with status {verification.get('status_without_constraint')}; "
            f"the feasible witness violates it by at most {_fmt_float(witness)}{units}{location}."
        )
    diagnostics = dict(row.get("diagnostics", {}) or {})
    return diagnostics.get("potential_cause") or f"Evidence source: {row.get('source')}."


def _solver_stats(problem: cp.Problem) -> dict[str, Any] | None:
    stats = getattr(problem, "solver_stats", None)
    if stats is None:
        return None
    return {
        "solver_name": getattr(stats, "solver_name", None),
        "solve_time": _finite_or_str(getattr(stats, "solve_time", None)),
        "setup_time": _finite_or_str(getattr(stats, "setup_time", None)),
        "num_iters": getattr(stats, "num_iters", None),
    }


def _solve_original(
    problem: cp.Problem,
    solver: Any | None,
    solve_kwargs: Mapping[str, Any] | None,
) -> str | None:
    try:
        kwargs = dict(solve_kwargs or {})
        problem.solve(**kwargs) if solver is None else problem.solve(solver=solver, **kwargs)
        return None
    except Exception as exc:
        return repr(exc)


def _default_diagnostics(index: int, constraint: cp.Constraint) -> ConstraintDiagnostics:
    return ConstraintDiagnostics(
        name=f"constraint_{index}",
        group="unclassified",
        description=f"Original {type(constraint).__name__} constraint without domain metadata.",
        potential_cause="No domain-specific interpretation is attached.",
        suggested_relaxation="Attach ConstraintDiagnostics if a domain-specific action is required.",
    )


def _diagnostics_dict(diagnostics: ConstraintDiagnostics) -> dict[str, Any]:
    return _json_safe(asdict(diagnostics))


def _array_summary(value: Any) -> dict[str, Any] | None:
    shaped = None
    if not isinstance(value, (list, tuple)):
        try:
            shaped = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            shaped = None
    array = _numeric_array(value)
    if array is None or not array.size:
        return None
    finite = array[np.isfinite(array)]
    if not finite.size:
        return None
    summary = {
        "max_abs": float(np.max(np.abs(finite))),
        "l2_norm": float(np.linalg.norm(finite)),
        "sample": _json_safe(array[:20]),
    }
    if shaped is not None and shaped.size and np.any(np.isfinite(shaped)):
        finite_abs = np.where(np.isfinite(shaped), np.abs(shaped), -np.inf)
        flat_index = int(np.argmax(finite_abs))
        summary["shape"] = list(shaped.shape)
        summary["max_index"] = [int(index) for index in np.unravel_index(flat_index, shaped.shape)] if shaped.shape else []
    return summary


def _witness_location(witness: Mapping[str, Any] | None, diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    if not witness or witness.get("max_index") is None:
        return {}
    indices = list(witness.get("max_index") or [])
    if not indices:
        return {}
    location: dict[str, Any] = {"index": indices}
    axes = list((diagnostics.get("axis_labels") or {}).items())
    if len(axes) == len(indices):
        for (axis, labels), index in zip(axes, indices):
            if index < len(labels):
                location[str(axis)] = labels[index]
    for key in ("date", "symbol", "factor"):
        details = diagnostics.get("details") or {}
        if key in details and key not in location:
            location[key] = details[key]
    return location


def _format_location(location: Any) -> str:
    if not isinstance(location, Mapping) or not location:
        return ""
    parts = [f"{key}={value}" for key, value in location.items() if key != "index"]
    if not parts and "index" in location:
        parts = [f"index={location['index']}"]
    return " at " + ", ".join(parts) if parts else ""


def _numeric_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = [_numeric_array(item) for item in value]
        valid = [part for part in parts if part is not None]
        return np.concatenate(valid) if valid else None
    try:
        return np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return None


def _status_family(status: Any) -> str:
    if status is None:
        return "not_solved"
    text = str(status).lower()
    if "infeasible" in text and "unbounded" in text:
        return "infeasible_or_unbounded"
    if "infeasible" in text:
        return "infeasible_inaccurate" if "inaccurate" in text else "infeasible"
    if "unbounded" in text:
        return "unbounded_inaccurate" if "inaccurate" in text else "unbounded"
    if "optimal" in text:
        return "solved_inaccurate" if "inaccurate" in text else "solved"
    return "uncertain"


def _safe_bool_call(value: Any, method: str) -> bool | None:
    try:
        return bool(getattr(value, method)())
    except (AttributeError, TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _meaningful_attribute(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (list, tuple)) and not value:
        return False
    return True


def _short_str(value: Any, max_chars: int) -> str:
    text = str(value)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _fmt_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.6g}" if math.isfinite(number) else str(number)


def _finite_or_str(value: Any) -> Any:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return number if math.isfinite(number) else str(number)
