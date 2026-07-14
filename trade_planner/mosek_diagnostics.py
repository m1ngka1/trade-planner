"""Preserve MOSEK certificate evidence that CVXPY normally discards.

The control flow is intentionally split in two:

1. :class:`DiagnosticMOSEK` runs as the solver and stores evidence on the
   returned CVXPY ``Solution.extra_stats``.
2. ``diagnostics.py`` runs *after* the solve and converts that evidence into
   constraint names, date/symbol labels, current settings, and PM actions.

Keeping the native ``Task`` handling here is important: CVXPY closes the Task
inside its MOSEK ``invert`` step, so the native arrays must be copied before
calling ``super().invert(...)``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from cvxpy import settings as s
from cvxpy.error import SolverError
from cvxpy.reductions.solvers.conic_solvers.mosek_conif import MOSEK


class DiagnosticMOSEK(MOSEK):
    """MOSEK with structured status and unbounded-ray data in ``extra_stats``.

    CVXPY already maps MOSEK's certificate for an infeasible continuous model
    to ``extra_stats["IIS"]``. It normally discards the opposite certificate
    (the improving direction when the original problem is unbounded). This
    adapter snapshots that direction before CVXPY closes the native Task.
    """

    def name(self) -> str:
        return "TRADE_PLANNER_MOSEK"

    def solve_via_data(
        self,
        data: Mapping[str, Any],
        warm_start: bool,
        verbose: bool,
        solver_opts: Mapping[str, Any],
        solver_cache: Any = None,
    ) -> Any:
        # Let CVXPY build and optimize its normal MOSEK Task. We do not build a
        # relaxed model and we do not run an additional diagnostic solve.
        try:
            output = super().solve_via_data(data, warm_start, verbose, solver_opts, solver_cache)
        except Exception as exc:
            missing_package = isinstance(exc, ModuleNotFoundError) and exc.name == "mosek"
            if missing_package or type(exc).__module__.split(".", 1)[0] == "mosek":
                raise SolverError(f"MOSEK could not run: {exc}") from exc
            raise
        if isinstance(output, dict) and "task" in output:
            # Save the canonical variable layout next to the live Task. It is
            # needed later to split a flat MOSEK ray back into CVXPY variables.
            output["trade_planner_context"] = _canonical_context(data)
        return output

    def invert(self, solver_output: Any, inverse_data: Any) -> Any:
        # Snapshot first. MOSEK.invert calls Task.__exit__, after which gety,
        # getslc, getsuc, getslx, etc. can no longer be queried.
        native: dict[str, Any] = {}
        if isinstance(solver_output, Mapping) and "task" in solver_output:
            native = _snapshot_task(
                solver_output["task"],
                solver_output.get("solver_options") or {},
                solver_output.get("trade_planner_context") or {},
            )

        # CVXPY's standard inversion performs the important original-space
        # mapping for primal infeasibility and writes it to extra_stats["IIS"].
        solution = super().invert(solver_output, inverse_data)
        if not native:
            return solution

        extra = dict(solution.attr.get(s.EXTRA_STATS) or {})
        ray = native.pop("original_variable_ray", None)
        # diagnostics.py consumes only these structured extra_stats fields; it
        # never needs the native Task itself and therefore never solves again.
        extra["MOSEK_DIAGNOSTICS"] = native
        if ray:
            extra["DUAL_RAY"] = ray
        solution.attr[s.EXTRA_STATS] = extra
        return solution


def diagnostic_mosek_if_requested(solver: Any) -> Any:
    """Use the certificate-preserving adapter for a requested MOSEK solve."""
    if isinstance(solver, str) and solver.upper() == "MOSEK":
        return DiagnosticMOSEK()
    return solver


def _canonical_context(data: Mapping[str, Any]) -> dict[str, Any]:
    """Remember how CVXPY packed named variables into its canonical vector."""
    param_problem = data.get(s.PARAM_PROB)
    variables: list[dict[str, Any]] = []
    if param_problem is not None:
        offsets = getattr(param_problem, "var_id_to_col", {})
        for variable in getattr(param_problem, "variables", ()):
            offset = offsets.get(variable.id)
            if offset is None:
                continue
            variables.append(
                {
                    "id": int(variable.id),
                    "name": variable.name(),
                    "offset": int(offset),
                    "size": int(variable.size),
                    "shape": tuple(variable.shape),
                }
            )
    # For CVXPY's continuous MOSEK reduction, B is the objective vector of the
    # original canonical minimization problem. Its dot product with the ray
    # must be negative before we call that ray an improving direction.
    objective = np.asarray(data.get(s.B, ()), dtype=float).ravel()
    return {
        "dualized": bool(data.get("dualized")),
        "variables": variables,
        "objective": objective,
    }


def _snapshot_task(task: Any, solver_options: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    """Copy statuses and certificates from the still-live native MOSEK Task."""
    import mosek

    solution_type = _solution_type(task, solver_options, mosek)
    problem_status = task.getprosta(solution_type)
    try:
        solution_status = task.getsolsta(solution_type)
    except Exception:
        solution_status = None

    result: dict[str, Any] = {
        "canonical_problem_status": _enum_name(problem_status),
        "canonical_solution_status": _enum_name(solution_status),
        "solution_type": _enum_name(solution_type),
        "cvxpy_dualized": bool(context.get("dualized")),
        "interpretation": _interpretation(problem_status, context, mosek),
        "solution_quality": _solution_quality(task, solution_type),
        "certificate_activity": _certificate_activity(
            task,
            solution_type,
            problem_status,
            mosek,
        ),
    }

    # CVXPY dualizes continuous models. Therefore a primal-infeasible *MOSEK
    # canonical task* means the original CVXPY problem is unbounded. MOSEK's y
    # vector is the corresponding direction in the original canonical x-space.
    if bool(context.get("dualized")) and problem_status == mosek.prosta.prim_infeas:
        ray_vector = _task_vector(task, "gety", solution_type)
        if ray_vector is not None:
            objective = np.asarray(context.get("objective", ()), dtype=float).ravel()
            if objective.size == ray_vector.size:
                slope = float(objective @ ray_vector)
                denominator = float(np.linalg.norm(objective) * np.linalg.norm(ray_vector))
                normalized = slope / denominator if denominator > np.finfo(float).tiny else 0.0
                result["objective_slope"] = slope
                result["objective_slope_normalized"] = normalized
                # Reject numerically dubious rays instead of presenting a PM
                # action unless the direction genuinely improves minimization.
                result["objective_improves"] = normalized < -1e-8
            if result.get("objective_improves"):
                result["original_variable_ray"] = _split_variable_vector(
                    ray_vector,
                    context.get("variables", ()),
                )
            else:
                result["ray_warning"] = "The captured direction did not pass the objective-improvement sign check."
    return result


def _solution_type(task: Any, solver_options: Mapping[str, Any], mosek: Any) -> Any:
    """Select the same MOSEK solution slot that CVXPY will subsequently read."""
    if task.getnumintvar() > 0:
        return mosek.soltype.itg
    simplex = {mosek.optimizertype.primal_simplex, mosek.optimizertype.dual_simplex}
    optimizer = task.getintparam(mosek.iparam.optimizer)
    bfs = bool(solver_options.get("bfs")) and task.getnumcone() == 0
    return mosek.soltype.bas if optimizer in simplex or bfs else mosek.soltype.itr


def _solution_quality(task: Any, solution_type: Any) -> dict[str, float]:
    try:
        values = task.getsolutioninfo(solution_type)
    except Exception:
        return {}
    names = (
        "primal_objective",
        "primal_constraint_violation",
        "primal_variable_violation",
        "primal_bar_variable_violation",
        "primal_cone_violation",
        "integer_violation",
        "dual_objective",
        "dual_constraint_violation",
        "dual_variable_violation",
        "dual_bar_variable_violation",
        "dual_cone_violation",
    )
    return {
        name: float(value)
        for name, value in zip(names, values)
        if np.isfinite(value)
    }


def _task_vector(task: Any, method: str, solution_type: Any) -> np.ndarray | None:
    """Read an optional Task vector without making diagnostics fail the solve."""
    try:
        value = getattr(task, method)(solution_type)
    except Exception:
        return None
    try:
        array = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    return array if array.size else None


def _certificate_activity(
    task: Any,
    solution_type: Any,
    problem_status: Any,
    mosek: Any,
) -> dict[str, Any]:
    """Summarize the same native certificate arrays used by MOSEK's report.

    ``slc/suc`` are lower/upper multipliers for constraint bounds and
    ``slx/sux`` are lower/upper multipliers for variable bounds. These indices
    describe CVXPY's canonical MOSEK task, so they are retained as native audit
    evidence. The PM-facing original-constraint mapping comes from CVXPY's
    ``extra_stats["IIS"]`` instead of pretending these canonical indices are
    original constraint IDs.
    """
    methods = (
        ("constraint_lower", "getslc"),
        ("constraint_upper", "getsuc"),
        ("variable_lower", "getslx"),
        ("variable_upper", "getsux"),
    ) if problem_status == mosek.prosta.prim_infeas else (
        ("primal_ray", "getxx"),
    ) if problem_status == mosek.prosta.dual_infeas else ()
    vectors = {
        label: vector
        for label, method in methods
        if (vector := _task_vector(task, method, solution_type)) is not None
    }
    # Certificates can be multiplied by any positive constant. Use a relative
    # threshold so the summary does not depend on their arbitrary scale.
    scale = max(
        (float(np.max(np.abs(vector[np.isfinite(vector)]))) for vector in vectors.values()),
        default=0.0,
    )
    if scale <= 0.0:
        return {}
    cutoff = scale * 1e-7
    return {
        label: {
            "nonzero_count": int(np.count_nonzero(np.isfinite(vector) & (np.abs(vector) > cutoff))),
            "max_abs": float(np.max(np.abs(vector[np.isfinite(vector)]))),
            "top_indices": [
                int(index)
                for index in np.argsort(np.abs(vector))[-5:][::-1]
                if np.isfinite(vector[index]) and abs(vector[index]) > cutoff
            ],
        }
        for label, vector in vectors.items()
    }


def _split_variable_vector(vector: np.ndarray, variables: Any) -> dict[int, Any]:
    """Split a flat canonical ray into arrays keyed by CVXPY variable ID."""
    result: dict[int, Any] = {}
    for variable in variables:
        start = int(variable["offset"])
        stop = start + int(variable["size"])
        if stop > vector.size:
            continue
        value = vector[start:stop]
        shape = tuple(variable.get("shape") or ())
        if shape:
            # CVXPY vectorizes matrix variables in Fortran/column-major order.
            value = value.reshape(shape, order="F")
        elif value.size == 1:
            value = value.item()
        result[int(variable["id"])] = value
    return result


def _interpretation(problem_status: Any, context: Mapping[str, Any], mosek: Any) -> str:
    """Explain the status flip caused by CVXPY's continuous dualization."""
    if not context.get("dualized"):
        return "The MOSEK status describes the original task directly."
    if problem_status == mosek.prosta.dual_infeas:
        return "CVXPY dualized the model: canonical dual infeasibility is original primal infeasibility."
    if problem_status == mosek.prosta.prim_infeas:
        return "CVXPY dualized the model: canonical primal infeasibility is original unboundedness."
    return "CVXPY dualized the continuous model before the MOSEK solve."


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    text = str(name) if name is not None else str(value)
    return text.rsplit(".", 1)[-1]
