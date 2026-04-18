"""
SHAP-style feature attribution
===============================

A lightweight, dependency-free implementation of **Kernel SHAP**
(Lundberg & Lee, 2017) for tabular, text, or any feature-indexable
input.

Why re-implement?
-----------------
The reference `shap` library is fantastic but pulls in a large
dependency tree.  For research scaffolding — especially teaching
or evaluation notebooks — a short pure-numpy implementation is
often more useful.  You can swap in the real `shap` package later
with almost identical call sites.

What this module provides
-------------------------
* ``kernel_shap_values(...)`` — a functional API, good for scripts.
* ``ShapExplainer`` — a class that caches the background dataset.
* ``ShapExplanation`` — a plotting-friendly dataclass with the
  attributions, the ranked feature order, and a pretty-printer.

The implementation follows the Kernel SHAP optimisation of the
Shapley-value game: a weighted linear regression over random
coalitions of "features present vs. absent", where absence is
realised by substituting each missing feature with a draw from the
background distribution.

Author: Dimitri Romanov
Project: humanising-ai
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------
@dataclass
class ShapExplanation:
    """Per-feature attributions for one explained instance."""

    feature_names: List[str]
    values: np.ndarray                           # shape (d,)
    base_value: float
    prediction: float

    # ------------------------------------------------------------------
    def ranked(self) -> List[Tuple[str, float]]:
        """Features sorted by |attribution|, descending."""
        order = np.argsort(-np.abs(self.values))
        return [(self.feature_names[i], float(self.values[i]))
                for i in order]

    def top(self, k: int = 5) -> List[Tuple[str, float]]:
        return self.ranked()[:k]

    def __repr__(self) -> str:
        lines = [
            f"ShapExplanation  base={self.base_value:+.4f}  "
            f"prediction={self.prediction:+.4f}  "
            f"Σφ={self.values.sum():+.4f}"
        ]
        for name, phi in self.top(min(8, len(self.values))):
            bar = "█" * max(1, int(round(abs(phi) * 20)))
            sign = "+" if phi >= 0 else "-"
            lines.append(f"  {name:<24} {sign}{abs(phi):.4f}  {bar}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Kernel-SHAP weighting function
# ---------------------------------------------------------------------------
def _shap_kernel_weight(M: int, s: int) -> float:
    """
    Shapley kernel weight for a coalition that includes `s` of the
    `M` features.  Coalitions of size 0 and M are degenerate and
    are handled separately in the fitting routine.
    """
    if s == 0 or s == M:
        return 0.0
    return (M - 1) / (math.comb(M, s) * s * (M - s))


def _sample_coalitions(
    M: int, n_samples: int, rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample binary coalitions.  We always include the all-off and
    all-on coalitions (they pin the base value and the prediction),
    then sample the rest uniformly over sizes and features.
    """
    n_samples = max(n_samples, 4)
    rows = [np.zeros(M, dtype=int), np.ones(M, dtype=int)]
    while len(rows) < n_samples:
        s = rng.integers(1, M)  # exclude degenerate 0 and M
        row = np.zeros(M, dtype=int)
        idx = rng.choice(M, size=s, replace=False)
        row[idx] = 1
        rows.append(row)
    return np.asarray(rows, dtype=int)


# ---------------------------------------------------------------------------
# Core Kernel SHAP routine
# ---------------------------------------------------------------------------
def kernel_shap_values(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    background: np.ndarray,
    n_samples: int = 128,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Estimate Shapley attributions for a single instance.

    Parameters
    ----------
    predict_fn : callable
        Takes a (B, d) array, returns a length-B vector of scalar
        predictions (e.g. class-probabilities or regression outputs).
    x : (d,) array
        The instance to explain.
    background : (N, d) array
        Samples from the background distribution used to marginalise
        "absent" features.  Any reasonable reference set works.
    n_samples : int
        Number of random coalitions to sample.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    values : (d,) array of Shapley attributions
    base_value : float
        Expected prediction under the background distribution.
    prediction : float
        The actual prediction at ``x``.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float).reshape(-1)
    background = np.asarray(background, dtype=float)
    M = x.shape[0]
