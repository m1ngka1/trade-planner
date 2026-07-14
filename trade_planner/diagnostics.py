"""Single-solve diagnostics for solved CVXPY problem objects.

Reader's map: constraint plugins attach business meaning with
``with_diagnostics`` in :mod:`trade_planner.constraints`; the planner solves the
original model once; :func:`diagnose_problem` maps evidence from that same
problem back to those labels; and :func:`format_diagnosis` produces the concise
PM-facing explanation.  MOSEK-specific evidence capture is isolated in
``mosek_diagnostics.py`` so the public report shape remains solver-independent.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import cvxpy as cp
import numpy as np

from .constraints import (
    ConstraintDiagnostics,
    VariableDiagnostics,
    get_constraint_diagnostics,
    get_variable_diagnostics,
)
from .mosek_diagnostics import diagnostic_mosek_if_requested


def diagnose_problem(
    problem: cp.Problem,
    *,
    solve_if_needed: bool = False,
    solver: Any | None = None,
    solve_kwargs: Mapping[str, Any] | None = None,
    tol: float = 1e-7,
    certificate_relative_tol: float = 1e-7,
    top_k: int = 10,
    top_elements: int = 3,
) -> dict[str, Any]:
    """Explain one CVXPY solve without relaxing or re-solving the model.

    A solved problem is only inspected. ``solve_if_needed=True`` permits one
    solve of the same original object when it has no status. MOSEK's mapped IIS
    certificate is preferred for primal infeasibility; a captured MOSEK ray is
    used for original unboundedness. Certificate magnitudes are normalized
    because infeasibility certificates have arbitrary scale.
    """
    initial_status = problem.status
    solve_error = None
    if initial_status is None and solve_if_needed:
        solve_error = _solve_original(problem, solver, solve_kwargs)

    # 1. Snapshot the original CVXPY objects and their attached business
    # metadata.  This layer is useful with every solver, even without a native
    # infeasibility certificate.
    status_family = _status_family(problem.status)
    constraints = [
        _inspect_constraint(index, constraint, tol)
        for index, constraint in enumerate(problem.constraints)
    ]
    variables = [_inspect_variable(variable) for variable in problem.variables()]
    # 2. Decode evidence already produced by the solve.  Diagnostics do not
    # disable constraints, build a relaxed model, or run an ablation loop.
    solver_evidence = _solver_evidence(
        problem,
        constraints,
        status_family,
        certificate_relative_tol,
        top_elements,
    )
    bottlenecks = list(solver_evidence.get("constraint_members", ()))[:top_k]
    improving_direction = list(solver_evidence.get("variable_directions", ()))[:top_k]
    decision = _decision(status_family, bottlenecks, improving_direction, solver_evidence)
    recommendations = _recommendations(status_family, decision, solver_evidence)

    # 3. Keep both machine-readable rows and a short human-readable decision so
    # callers can log, render, or programmatically inspect the same report.
    report: dict[str, Any] = {
        "summary": {
            "status": problem.status,
            "status_family": status_family,
            "value": _finite_or_str(problem.value),
            "is_dcp": problem.is_dcp(),
            "num_constraints": len(problem.constraints),
            "num_variables": len(problem.variables()),
            "analysis_mode": "single_solve",
            "additional_solves": int(initial_status is None and problem.status is not None),
            "was_solved_by_diagnostic": initial_status is None and problem.status is not None,
            "solve_error": solve_error,
            "message": decision["plain_english"],
        },
        "decision": decision,
        "coverage": _coverage(constraints, bottlenecks, improving_direction, solver_evidence),
        "constraints": constraints,
        "variables": variables,
        "bottlenecks": bottlenecks,
        "improving_direction": improving_direction,
        "solver_evidence": solver_evidence,
        "solver_stats": _solver_stats(problem),
        "recommendations": recommendations,
    }
    report["text"] = format_diagnosis(report, top_k=top_k)
    return _json_safe(report)


diagnose_infeasible_problem = diagnose_problem


def format_diagnosis(report: Mapping[str, Any], *, top_k: int = 10) -> str:
    """Render the report for a PM who does not need solver terminology."""
    summary = dict(report.get("summary") or {})
    decision = dict(report.get("decision") or {})
    evidence = dict(report.get("solver_evidence") or {})
    bottlenecks = list(report.get("bottlenecks") or [])
    directions = list(report.get("improving_direction") or [])

    if summary.get("status") is None:
        analysis = "Analysis: model structure only; no solve was run; no constraints disabled or relaxed."
    else:
        analysis = "Analysis: existing original solve only; 0 diagnostic re-solves; no constraints disabled or relaxed."
    lines = [
        "=== Trade-plan solve diagnosis ===",
        f"Outcome: {decision.get('outcome', 'unknown').replace('_', ' ').upper()}",
        str(decision.get("plain_english") or summary.get("message") or "No explanation is available."),
        analysis,
    ]
    native = dict(evidence.get("native_mosek") or {})
    if native:
        lines.append(
            "MOSEK: "
            f"canonical status={native.get('canonical_problem_status')}, "
            f"solution={native.get('canonical_solution_status')}."
        )

    if bottlenecks:
        conflict = evidence.get("kind") == "primal_infeasibility_certificate"
        heading = (
            "Conflict members — choose one or more business levers from this set:"
            if conflict
            else "Active sensitivities — these are not infeasibility causes:"
        )
        weight_label = "certificate weight" if conflict else "absolute shadow-price weight"
        lines.extend(["", heading])
        for row in bottlenecks[:top_k]:
            diagnostics = dict(row.get("diagnostics") or {})
            share = 100.0 * float(row.get("certificate_share", 0.0))
            lines.append(
                f"  {row.get('rank')}. {diagnostics.get('name')} "
                f"[{diagnostics.get('group') or 'ungrouped'}] — {share:.1f}% of {weight_label}"
            )
            elements = list(row.get("affected_elements") or [])
            for element in elements:
                location = _format_location(element.get("location"))
                current = _current_setting_text(diagnostics, element.get("current_setting"))
                lines.append(f"     Where: {location or 'whole constraint'}{current}")
                context = dict(element.get("business_context") or {})
                if context:
                    rendered_context = _format_business_context(context)
                    if rendered_context:
                        lines.append(f"     Context: {rendered_context}")
            if diagnostics.get("potential_cause"):
                lines.append(f"     Why: {diagnostics['potential_cause']}")
            element_action = next(
                (
                    (element.get("business_context") or {}).get("pm_action")
                    for element in elements
                    if (element.get("business_context") or {}).get("pm_action")
                ),
                None,
            )
            if conflict and (element_action or diagnostics.get("suggested_relaxation")):
                lines.append(f"     PM action: {element_action or diagnostics['suggested_relaxation']}")

    if directions:
        lines.extend(["", "Unbounded improving direction — bound this joint move:"])
        for row in directions[:top_k]:
            diagnostics = dict(row.get("diagnostics") or {})
            lines.append(
                f"  {row.get('rank')}. {diagnostics.get('name') or row.get('name')} "
                f"({100.0 * float(row.get('direction_share', 0.0)):.1f}% of ray magnitude)"
            )
            for element in list(row.get("affected_elements") or []):
                location = _format_location(element.get("location"))
                lines.append(
                    f"     {element.get('direction')} {location or 'this variable'} "
                    f"(ray component {_fmt_float(element.get('value'))})"
                )
            lines.append("     PM action: add or tighten a finite bound that blocks this joint direction.")

    caveat = decision.get("evidence_limit")
    if caveat:
        lines.extend(["", f"Evidence limit: {caveat}"])
    if not bottlenecks and not directions and report.get("recommendations"):
        lines.extend(["", "Next action:"])
        for item in list(report["recommendations"])[:top_k]:
            lines.append(f"  - {item.get('action') or item.get('detail')}")
    return "\n".join(lines)


format_infeasibility_diagnosis = format_diagnosis


def _inspect_constraint(index: int, constraint: cp.Constraint, tol: float) -> dict[str, Any]:
    attached = get_constraint_diagnostics(constraint)
    diagnostics = attached or _default_diagnostics(index, constraint)
    dual = _array_summary(getattr(constraint, "dual_value", None))
    violation = _constraint_violation(constraint)
    slack, active = _constraint_slack(constraint, tol)
    if violation and violation.get("max_abs", 0.0) > tol:
        state = "violated"
    elif active is True:
        state = "active"
    elif violation is not None:
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
    attached = get_variable_diagnostics(variable)
    diagnostics = attached or VariableDiagnostics(name=variable.name())
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
        "diagnostics_attached": attached is not None,
        "diagnostics": _variable_diagnostics_dict(diagnostics),
    }


def _constraint_violation(constraint: cp.Constraint) -> dict[str, Any] | None:
    try:
        return _array_summary(constraint.violation())
    except (ValueError, TypeError, NotImplementedError):
        return None


def _constraint_slack(constraint: cp.Constraint, tol: float) -> tuple[dict[str, Any] | None, bool | None]:
    constraint_type = type(constraint).__name__
    if constraint_type not in {"Inequality", "Equality"}:
        violation = _constraint_violation(constraint)
        return None, bool(violation and violation.get("max_abs", math.inf) <= tol) if violation else None
    value = getattr(getattr(constraint, "expr", None), "value", None)
    if value is None:
        return None, None
    if constraint_type == "Inequality":
        slack_values = np.maximum(-np.asarray(value, dtype=float), 0.0)
        return _array_summary(slack_values), bool(np.min(slack_values) <= tol)
    return None, True


def _solver_evidence(
    problem: cp.Problem,
    constraints: list[dict[str, Any]],
    status_family: str,
    relative_tol: float,
    top_elements: int,
) -> dict[str, Any]:
    # ``extra_stats`` is the bridge from solver-native output to original
    # CVXPY constraint IDs.  Once mapped, constraint-owned metadata supplies
    # names, locations, current settings, causes, and practical actions.
    extra = _extra_stats(problem)
    native = extra.get("MOSEK_DIAGNOSTICS") if isinstance(extra.get("MOSEK_DIAGNOSTICS"), Mapping) else {}
    certificate = next(
        (extra[key] for key in ("IIS", "iis") if isinstance(extra.get(key), Mapping)),
        None,
    )
    solver_name = str(getattr(getattr(problem, "solver_stats", None), "solver_name", ""))
    source = (
        "mosek_infeasibility_certificate"
        if certificate is not None and (native or "MOSEK" in solver_name.upper())
        else "solver_infeasibility_certificate" if certificate is not None else ""
    )
    confidence = "solver_certificate" if certificate is not None else "none"

    if certificate is None and status_family in {"infeasible", "infeasible_inaccurate"}:
        # Some backends expose mapped values only as constraint duals.  Treat
        # this as weaker fallback evidence, not as a solver-certified IIS.
        fallback = {
            constraint.id: constraint.dual_value
            for constraint in problem.constraints
            if constraint.dual_value is not None
        }
        if fallback:
            certificate = fallback
            source = "constraint_infeasibility_dual"
            confidence = "solver_mapped_fallback"

    mapping_coverage = _certificate_mapping_coverage(certificate or {}, constraints)

    members = _map_constraint_certificate(
        certificate or {},
        constraints,
        problem.constraints,
        source,
        confidence,
        max(relative_tol, 0.05) if source == "constraint_infeasibility_dual" else relative_tol,
        top_elements,
    )
    ray = extra.get("DUAL_RAY") if isinstance(extra.get("DUAL_RAY"), Mapping) else {}
    directions = _map_variable_ray(ray, problem.variables(), relative_tol, top_elements)

    if status_family in {"solved", "solved_inaccurate"} and not members:
        # On a successful solve, active duals describe sensitivity.  They are
        # intentionally labelled as such and must not be reported as causes of
        # infeasibility.
        duals = {
            constraint.id: constraint.dual_value
            for constraint, row in zip(problem.constraints, constraints)
            if row.get("is_active") and constraint.dual_value is not None
        }
        members = _map_constraint_certificate(
            duals,
            constraints,
            problem.constraints,
            "optimal_shadow_price",
            "sensitivity_not_infeasibility",
            relative_tol,
            top_elements,
        )

    return {
        "available": bool(extra) or bool(members) or bool(directions),
        "kind": _evidence_kind(status_family, members, directions),
        "mapped": bool(members) or bool(directions),
        "source": source or (members[0]["source"] if members else ""),
        "confidence": confidence,
        "mapping_coverage": mapping_coverage,
        "keys": sorted(str(key) for key in extra),
        "native_mosek": dict(native),
        "constraint_members": members,
        "variable_directions": directions,
        "note": _evidence_note(status_family, members, directions, bool(extra)),
    }


def _map_constraint_certificate(
    certificate: Mapping[Any, Any],
    rows: list[dict[str, Any]],
    original_constraints: list[cp.Constraint],
    source: str,
    confidence: str,
    relative_tol: float,
    top_elements: int,
) -> list[dict[str, Any]]:
    # Certificate scale is arbitrary, so ranking uses relative magnitude and
    # reports conflict-set participation rather than claiming the top row alone
    # is the unique cause or fix.
    by_id = {row["constraint_id"]: (row, original_constraints[row["cvxpy_index"]]) for row in rows}
    prepared: list[tuple[dict[str, Any], cp.Constraint, np.ndarray]] = []
    global_max = 0.0
    for key, value in certificate.items():
        pair = by_id.get(key)
        if pair is None:
            try:
                pair = by_id.get(int(key))
            except (TypeError, ValueError):
                pair = None
        if pair is None:
            continue
        row, constraint = pair
        array = _shaped_numeric_array(value, tuple(row.get("shape") or ()))
        if array is None or not np.any(np.isfinite(array)):
            continue
        global_max = max(global_max, float(np.max(np.abs(array[np.isfinite(array)]))))
        prepared.append((row, constraint, array))
    if global_max <= 0.0:
        return []

    cutoff = global_max * max(float(relative_tol), np.finfo(float).eps * 10.0)
    total = sum(
        float(np.sum(np.abs(array[np.isfinite(array) & (np.abs(array) > cutoff)])))
        for _, _, array in prepared
    )
    result = []
    for row, constraint, array in prepared:
        mask = np.isfinite(array) & (np.abs(array) > cutoff)
        if not np.any(mask):
            continue
        absolute = np.abs(array[mask])
        diagnostics = dict(row.get("diagnostics") or {})
        result.append(
            {
                "cvxpy_index": row["cvxpy_index"],
                "constraint_id": row["constraint_id"],
                "source": source,
                "confidence": confidence,
                "severity": float(np.max(absolute) / global_max),
                "certificate_share": float(np.sum(absolute) / total) if total else 0.0,
                "max_abs_certificate_entry": float(np.max(absolute)),
                "num_nonzero_certificate_entries": int(np.count_nonzero(mask)),
                "diagnostics": diagnostics,
                "affected_elements": _ranked_elements(
                    array,
                    mask,
                    diagnostics.get("axis_labels") or {},
                    getattr(get_constraint_diagnostics(constraint), "bound_values", None),
                    getattr(get_constraint_diagnostics(constraint), "element_context", None),
                    top_elements,
                ),
            }
        )
    result.sort(key=lambda item: (item["certificate_share"], item["severity"]), reverse=True)
    for rank, row in enumerate(result, 1):
        row["rank"] = rank
    return result


def _certificate_mapping_coverage(
    certificate: Mapping[Any, Any],
    constraints: list[dict[str, Any]],
) -> dict[str, Any]:
    original_ids = {row["constraint_id"] for row in constraints}
    mapped = 0
    unmapped: list[str] = []
    for key in certificate:
        normalized = key
        if normalized not in original_ids:
            try:
                normalized = int(key)
            except (TypeError, ValueError):
                pass
        if normalized in original_ids:
            mapped += 1
        else:
            unmapped.append(str(key))
    return {
        "certificate_ids": len(certificate),
        "mapped_original_constraint_ids": mapped,
        "unmapped_canonical_constraint_ids": len(unmapped),
        "unmapped_id_sample": unmapped[:10],
    }


def _map_variable_ray(
    ray: Mapping[Any, Any],
    variables: list[cp.Variable],
    relative_tol: float,
    top_elements: int,
) -> list[dict[str, Any]]:
    by_id = {variable.id: variable for variable in variables}
    prepared: list[tuple[cp.Variable, np.ndarray]] = []
    global_max = 0.0
    for key, value in ray.items():
        variable = by_id.get(key)
        if variable is None:
            try:
                variable = by_id.get(int(key))
            except (TypeError, ValueError):
                variable = None
        if variable is None:
            continue
        array = _shaped_numeric_array(value, tuple(variable.shape))
        if array is None or not np.any(np.isfinite(array)):
            continue
        global_max = max(global_max, float(np.max(np.abs(array[np.isfinite(array)]))))
        prepared.append((variable, array))
    if global_max <= 0.0:
        return []
    cutoff = global_max * max(float(relative_tol), np.finfo(float).eps * 10.0)
    total = sum(
        float(np.sum(np.abs(array[np.isfinite(array) & (np.abs(array) > cutoff)])))
        for _, array in prepared
    )
    result = []
    for variable, array in prepared:
        mask = np.isfinite(array) & (np.abs(array) > cutoff)
        if not np.any(mask):
            continue
        diagnostics = get_variable_diagnostics(variable) or VariableDiagnostics(name=variable.name())
        elements = _ranked_elements(array, mask, diagnostics.axis_labels, None, None, top_elements)
        for element in elements:
            element["direction"] = "increase" if element["value"] > 0 else "decrease"
        magnitude = float(np.sum(np.abs(array[mask])))
        result.append(
            {
                "variable_id": variable.id,
                "name": variable.name(),
                "diagnostics": _variable_diagnostics_dict(diagnostics),
                "direction_share": magnitude / total if total else 0.0,
                "max_abs_ray_entry": float(np.max(np.abs(array[mask]))),
                "affected_elements": elements,
            }
        )
    result.sort(key=lambda item: item["direction_share"], reverse=True)
    for rank, row in enumerate(result, 1):
        row["rank"] = rank
    return result


def _ranked_elements(
    array: np.ndarray,
    mask: np.ndarray,
    axis_labels: Mapping[str, Any],
    bound_values: Any,
    element_context: Any,
    limit: int,
) -> list[dict[str, Any]]:
    flat_indices = np.flatnonzero(mask.ravel())
    ordered = sorted(flat_indices, key=lambda index: abs(float(array.ravel()[index])), reverse=True)
    result = []
    for flat_index in ordered[: max(1, limit)]:
        index = tuple(int(item) for item in np.unravel_index(flat_index, array.shape)) if array.shape else ()
        context = {}
        if callable(element_context):
            try:
                context = dict(element_context(index) or {})
            except Exception:
                context = {}
        result.append({
            "index": list(index),
            "location": _location(index, array.shape, axis_labels),
            "value": float(array.ravel()[flat_index]),
            "current_setting": _bound_value_at(bound_values, index, array.shape),
            "business_context": context,
        })
    return result


def _location(index: tuple[int, ...], shape: tuple[int, ...], axis_labels: Mapping[str, Any]) -> dict[str, Any]:
    if not index:
        return {}
    location: dict[str, Any] = {"index": list(index)}
    axes = list(axis_labels.items())
    if len(axes) == len(shape):
        for (axis, labels), item in zip(axes, index):
            if item < len(labels):
                location[str(axis)] = labels[item]
    return location


def _bound_value_at(bound_values: Any, index: tuple[int, ...], shape: tuple[int, ...]) -> Any:
    if bound_values is None:
        return None
    try:
        array = np.asarray(bound_values, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim == 0:
        return float(array)
    if array.shape == shape and index:
        return float(array[index])
    if array.size == int(np.prod(shape, dtype=int)) and index:
        return float(array.reshape(shape)[index])
    return None


def _decision(
    status_family: str,
    bottlenecks: list[dict[str, Any]],
    directions: list[dict[str, Any]],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if status_family in {"infeasible", "infeasible_inaccurate"}:
        if bottlenecks:
            actions = [_action_row(row) for row in bottlenecks]
            fallback = evidence.get("confidence") == "solver_mapped_fallback"
            unmapped = int((evidence.get("mapping_coverage") or {}).get("unmapped_canonical_constraint_ids", 0))
            return {
                "outcome": "no_feasible_schedule",
                "plain_english": (
                    f"No trade schedule satisfies all hard rules. "
                    + ("Solver-returned infeasibility duals highlight " if fallback else "The solver certificate identifies ")
                    + f"{len(bottlenecks)} policy constraint(s) in the conflict."
                ),
                "what_to_change": actions,
                "evidence_limit": (
                    ("This fallback is not MOSEK's mapped IIS; treat its ranking as directional. " if fallback else "")
                    + "Certificate weight ranks participation in one mathematical conflict and depends on model scaling; "
                    "it does not prove that "
                    "the top rule alone is sufficient or calculate the smallest safe limit change."
                    + (f" {unmapped} canonical certificate row(s) were internal to CVXPY and could not be named." if unmapped else "")
                ),
            }
        return {
            "outcome": "no_feasible_schedule_unmapped",
            "plain_english": "No trade schedule satisfies all hard rules, but this solve exposed no mappable certificate.",
            "what_to_change": [],
            "evidence_limit": "Naming one rule without mapped solver evidence would be speculation.",
        }
    if status_family in {"unbounded", "unbounded_inaccurate"}:
        return {
            "outcome": "objective_unbounded",
            "plain_english": (
                "The objective can keep improving without a finite limit. "
                + ("The solver ray identifies the joint variable direction to block." if directions else "No variable ray was preserved by this solver path.")
            ),
            "what_to_change": [_direction_action(row) for row in directions],
            "evidence_limit": "An unbounded ray is a joint direction; one component should not be blamed in isolation.",
        }
    if status_family == "infeasible_or_unbounded":
        return {
            "outcome": "solver_status_ambiguous",
            "plain_english": "The solver could not distinguish no feasible schedule from an unbounded objective.",
            "what_to_change": [],
            "evidence_limit": "A definitive PM action requires a solver status that separates the two cases.",
        }
    if status_family in {"solved", "solved_inaccurate"}:
        return {
            "outcome": "schedule_found" if status_family == "solved" else "schedule_found_inaccurately",
            "plain_english": "A feasible trade schedule was found." if status_family == "solved" else "A schedule was found, but solver accuracy is reduced.",
            "what_to_change": [],
            "evidence_limit": "Shadow prices are sensitivities at the solution, not infeasibility certificates.",
        }
    if status_family == "not_solved":
        return {
            "outcome": "not_solved",
            "plain_english": "The model has not been solved; only its structure can be inspected.",
            "what_to_change": [],
            "evidence_limit": "Numeric causes require one solver result.",
        }
    return {
        "outcome": "solver_status_uncertain",
        "plain_english": str(evidence.get("note") or "The solver outcome is uncertain."),
        "what_to_change": [],
        "evidence_limit": "No definitive certificate interpretation is available.",
    }


def _action_row(row: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = dict(row.get("diagnostics") or {})
    element = next(iter(row.get("affected_elements") or []), {})
    return {
        "priority": row.get("rank"),
        "constraint": diagnostics.get("name"),
        "setting": diagnostics.get("setting_name") or diagnostics.get("name"),
        "where": element.get("location") or {},
        "current_setting": element.get("current_setting"),
        "business_context": element.get("business_context") or {},
        "action": (element.get("business_context") or {}).get("pm_action") or diagnostics.get("suggested_relaxation"),
        "certificate_share": row.get("certificate_share"),
    }


def _direction_action(row: Mapping[str, Any]) -> dict[str, Any]:
    element = next(iter(row.get("affected_elements") or []), {})
    return {
        "priority": row.get("rank"),
        "variable": (row.get("diagnostics") or {}).get("name") or row.get("name"),
        "where": element.get("location") or {},
        "direction": element.get("direction"),
        "action": "Add or tighten a finite bound that blocks this joint improving direction.",
    }


def _recommendations(
    status_family: str,
    decision: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    actions = list(decision.get("what_to_change") or [])
    if actions:
        return [
            {
                "title": f"Priority {item.get('priority')}: {item.get('setting') or item.get('variable')}",
                "detail": f"Location: {_format_location(item.get('where')) or 'whole rule'}.",
                "action": item.get("action"),
            }
            for item in actions
        ]
    if status_family in {"infeasible", "infeasible_inaccurate"}:
        return [{
            "title": "Preserve a mapped certificate",
            "detail": str(evidence.get("note")),
            "action": "Solve with MOSEK and inspect its CVXPY IIS mapping; do not disable constraints one by one.",
        }]
    if status_family in {"unbounded", "unbounded_inaccurate"}:
        return [{
            "title": "Preserve the MOSEK improving ray",
            "detail": str(evidence.get("note")),
            "action": "Use DiagnosticMOSEK, then add finite business bounds around the reported joint direction.",
        }]
    if status_family == "not_solved":
        return [{"title": "Solve once", "detail": "No numeric evidence exists yet.", "action": "Solve the original model once."}]
    return []


def _coverage(
    constraints: list[dict[str, Any]],
    bottlenecks: list[dict[str, Any]],
    directions: list[dict[str, Any]],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "inspected": len(constraints),
        "with_metadata": sum(bool(row["diagnostics_attached"]) for row in constraints),
        "with_primal_metrics": sum(row["violation"] is not None for row in constraints),
        "with_dual_values": sum(row["dual"] is not None for row in constraints),
        "mapped_certificate_constraints": len(bottlenecks),
        "mapped_unbounded_variables": len(directions),
        "solver_evidence_available": bool(evidence.get("available")),
        "unresolved": not bool(bottlenecks or directions),
    }


def _extra_stats(problem: cp.Problem) -> dict[str, Any]:
    stats = getattr(problem, "solver_stats", None)
    extra = getattr(stats, "extra_stats", None) if stats is not None else None
    if isinstance(extra, Mapping):
        return dict(extra)
    if extra is not None and hasattr(extra, "__dict__"):
        return dict(vars(extra))
    return {}


def _evidence_kind(status_family: str, members: list[Any], directions: list[Any]) -> str:
    if status_family in {"infeasible", "infeasible_inaccurate"} and members:
        return "primal_infeasibility_certificate"
    if status_family in {"unbounded", "unbounded_inaccurate"} and directions:
        return "dual_infeasibility_ray"
    if status_family in {"solved", "solved_inaccurate"} and members:
        return "optimal_shadow_prices"
    return "none"


def _evidence_note(status_family: str, members: list[Any], directions: list[Any], has_extra: bool) -> str:
    if members and status_family in {"infeasible", "infeasible_inaccurate"}:
        return "Mapped a scale-normalized infeasibility certificate to original CVXPY constraint ids."
    if directions:
        return "Mapped MOSEK's improving ray to original CVXPY variable ids."
    if has_extra:
        return "Solver-specific data exists, but it contains no certificate mapped to the original model."
    return "This solver exposed no structured certificate data."


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


def _solve_original(problem: cp.Problem, solver: Any, solve_kwargs: Mapping[str, Any] | None) -> str | None:
    try:
        kwargs = dict(solve_kwargs or {})
        kwargs.pop("solver", None)
        selected = diagnostic_mosek_if_requested(solver)
        if selected is None:
            problem.solve(**kwargs)
        else:
            problem.solve(solver=selected, **kwargs)
        return None
    except Exception as exc:
        return repr(exc)


def _default_diagnostics(index: int, constraint: cp.Constraint) -> ConstraintDiagnostics:
    return ConstraintDiagnostics(
        name=f"constraint_{index}",
        group="unclassified",
        description=f"Original {type(constraint).__name__} constraint without domain metadata.",
        potential_cause="No business interpretation is attached.",
        suggested_relaxation="Attach ConstraintDiagnostics to name the setting and PM action.",
    )


def _diagnostics_dict(diagnostics: ConstraintDiagnostics) -> dict[str, Any]:
    return _json_safe({
        "name": diagnostics.name,
        "group": diagnostics.group,
        "description": diagnostics.description,
        "potential_cause": diagnostics.potential_cause,
        "suggested_relaxation": diagnostics.suggested_relaxation,
        "units": diagnostics.units,
        "weight": diagnostics.weight,
        "hard": diagnostics.hard,
        "axis_labels": diagnostics.axis_labels,
        "details": diagnostics.details,
        "setting_name": diagnostics.setting_name,
    })


def _variable_diagnostics_dict(diagnostics: VariableDiagnostics) -> dict[str, Any]:
    return _json_safe({
        "name": diagnostics.name,
        "description": diagnostics.description,
        "units": diagnostics.units,
        "axis_labels": diagnostics.axis_labels,
    })


def _array_summary(value: Any) -> dict[str, Any] | None:
    array = _shaped_numeric_array(value)
    if array is None or not array.size:
        return None
    finite = array[np.isfinite(array)]
    if not finite.size:
        return None
    finite_abs = np.where(np.isfinite(array), np.abs(array), -np.inf)
    flat_index = int(np.argmax(finite_abs))
    return {
        "max_abs": float(np.max(np.abs(finite))),
        "l2_norm": float(np.linalg.norm(finite)),
        "sample": array.ravel()[:20].tolist(),
        "shape": list(array.shape),
        "max_index": list(np.unravel_index(flat_index, array.shape)) if array.shape else [],
    }


def _shaped_numeric_array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        if isinstance(value, (list, tuple)):
            parts = [_shaped_numeric_array(item) for item in value]
            valid = [part.ravel() for part in parts if part is not None]
            array = np.concatenate(valid) if valid else None
        else:
            array = None
    if array is None:
        return None
    if shape is not None and array.size == int(np.prod(shape, dtype=int)):
        array = array.reshape(shape)
    return array


def _format_location(location: Any) -> str:
    if not isinstance(location, Mapping) or not location:
        return ""
    parts = [f"{key}={value}" for key, value in location.items() if key != "index"]
    if not parts and "index" in location:
        parts = [f"index={location['index']}"]
    return ", ".join(parts)


def _current_setting_text(diagnostics: Mapping[str, Any], current: Any) -> str:
    if current is None:
        return ""
    units = f" {diagnostics.get('units')}" if diagnostics.get("units") else ""
    setting = diagnostics.get("setting_name") or "current setting"
    return f"; {setting}={_fmt_float(current)}{units}"


def _format_business_context(context: Mapping[str, Any]) -> str:
    return "; ".join(
        f"{str(key).replace('_', ' ')}={_fmt_float(value)}"
        for key, value in context.items()
        if key != "pm_action"
    )


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
