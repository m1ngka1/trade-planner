"""Internal numeric helpers."""

from __future__ import annotations

import numpy as np

from .types import Array


def as_psd(matrix: Array, jitter: float = 1e-10) -> Array:
    """Return a symmetric positive semidefinite matrix with small diagonal repair."""
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    min_eig = float(np.linalg.eigvalsh(matrix).min())
    if min_eig < jitter:
        matrix = matrix + (jitter - min_eig) * np.eye(matrix.shape[0])
    return matrix


def safe_numeric(values: Array, floor: float = 1.0) -> Array:
    values = np.asarray(values, dtype=float)
    return np.maximum(values, floor)
