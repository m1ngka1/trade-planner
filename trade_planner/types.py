"""Shared types and exceptions."""

from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np


Array = np.ndarray


class InfeasiblePlanError(ValueError):
    """Raised when hard constraints make full completion impossible."""

    def __init__(
        self,
        message: str,
        diagnostics: dict[str, Any] | None = None,
        problem: cp.Problem | None = None,
    ):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.problem = problem
