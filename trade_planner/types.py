"""Shared types and exceptions."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


class InfeasiblePlanError(ValueError):
    """Raised when hard constraints make full completion impossible."""
