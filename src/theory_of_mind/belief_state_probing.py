"""
Belief-State Probing
====================

Lightweight utilities for testing whether a model's **internal** or
**output** representations encode the belief state required to pass
a false-belief scenario.

Probing methodology
-------------------
Follow-up to a long interpretability tradition (Alain & Bengio,
2016; Hewitt & Manning, 2019; Li et al., 2021 — "implicit world
models"):

1. Run the model over a batch of `BeliefScenario`s.
2. For each scenario, obtain a vector representation
   ``h ∈ R^d``.  This can be:
       * a hidden-state vector extracted from the model, or
       * any *observable* feature vector (e.g. probabilities over
         candidate answers, n-gram statistics of the narrative, etc.).
3. Train a small linear (logistic) classifier ``p(belief | h)`` on
   half the scenarios and evaluate on the other half.
4. High test accuracy ⇒ the representation *linearly encodes*
   the belief state — a necessary, though not sufficient, condition
   for genuine mentalising.

This module deliberately avoids hard dependencies on any particular
model library.  You supply the representation extractor; we train
and score the probe with plain numpy.

Author: Dimitri Romanov
Project: humanising-ai
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .tom_benchmark import BeliefScenario


FeatureExtractor = Callable[[BeliefScenario], Sequence[float]]


# ---------------------------------------------------------------------------
# Small logistic-regression solver (batch gradient descent, numpy only)
# ---------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def logistic_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int = 500,
    lr: float = 0.1,
    l2: float = 1e-3,
    verbose: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Fit a binary logistic regression via batch gradient descent.

    Parameters
    ----------
    X : (n, d) float array
    y : (n,)  int   array (0 / 1)
    n_iter : int
    lr : float
    l2 : float
        Ridge penalty.
    verbose : bool
        Print the cross-entropy loss every 100 iterations.

    Returns
    -------
    w : (d,) array, b : float
    """
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for t in range(n_iter):
        logits = X @ w + b
        p = _sigmoid(logits)
        # Cross-entropy gradient + L2 shrinkage
        grad_w = X.T @ (p - y) / n + l2 * w
        grad_b = (p - y).mean()
        w -= lr * grad_w
        b -= lr * grad_b

        if verbose and t % 100 == 0:
            loss = -np.mean(
                y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)
            )
            print(f"  iter {t:4d}   loss {loss:.4f}")
    return w, float(b)


def logistic_predict(
    X: np.ndarray, w: np.ndarray, b: float
) -> np.ndarray:
    return _sigmoid(X @ w + b)


# ---------------------------------------------------------------------------
# Linear probe over arbitrary feature vectors
# ---------------------------------------------------------------------------
@dataclass
class LinearProbe:
    """A fitted linear probe with its own accuracy score."""

    w: np.ndarray
    b: float
    train_accuracy: float
    test_accuracy: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (logistic_predict(X, self.w, self.b) >= 0.5).astype(int)


class BeliefStateProbe:
    """
    End-to-end probe: take `BeliefScenario`s + a feature extractor,
    build a (X, y) dataset, train a linear classifier, and report
    whether the belief state is linearly decodable.

    Labels
    ------
    For each scenario, the probe tries to predict whether the
    **correct belief-location** is the *initial* or the *moved*
    location.  Both are valid "distractors" in the same sense — a
    linear probe that can recover this bit from a representation
    demonstrates that the belief-state information is present.

    Parameters
    ----------
    feature_extractor : callable
        Maps a `BeliefScenario` to a fixed-length numeric vector.
    seed : int
        RNG seed for the train/test split.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        seed: int = 0,
    ):
        self.extract = feature_extractor
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def featurise(
        self, scenarios: Iterable[BeliefScenario]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build (X, y) from a list of scenarios."""
        X_rows: List[Sequence[float]] = []
        y_rows: List[int] = []
        for s in scenarios:
            feats = list(self.extract(s))
            X_rows.append(feats)
            # Label = 1 if the correct belief-location happens to be
            # lexicographically greater than the actual-location, 0
            # otherwise.  This yields a balanced binary label that a
            # representation must *know about both locations* to predict.
            y_rows.append(int(s.correct_location > s.actual_location))
        return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=int)

    # ------------------------------------------------------------------
    def fit_eval(
        self,
        scenarios: Sequence[BeliefScenario],
        test_frac: float = 0.3,
        n_iter: int = 500,
        lr: float = 0.1,
        l2: float = 1e-3,
    ) -> LinearProbe:
        """Split the scenarios into train/test, fit, and score."""
        X, y = self.featurise(scenarios)
        n = X.shape[0]
        if n < 4:
            raise ValueError(
                "Need at least 4 scenarios to train + evaluate."
            )

        idx = np.arange(n)
        self.rng.shuffle(idx)
        cut = max(1, int(n * (1.0 - test_frac)))
        train_idx, test_idx = idx[:cut], idx[cut:]

        # Standardise using train-only statistics
        mu = X[train_idx].mean(axis=0)
        sigma = X[train_idx].std(axis=0) + 1e-6
        Xn = (X - mu) / sigma

        w, b = logistic_fit(
            Xn[train_idx], y[train_idx],
            n_iter=n_iter, lr=lr, l2=l2, verbose=False,
        )

        train_acc = float(
            ((logistic_predict(Xn[train_idx], w, b) >= 0.5).astype(int)
             == y[train_idx]).mean()
        )
        test_acc = float(
            ((logistic_predict(Xn[test_idx], w, b) >= 0.5).astype(int)
             == y[test_idx]).mean()
        )

        return LinearProbe(
            w=w, b=b,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
        )


# ---------------------------------------------------------------------------
# Reference feature extractors
# ---------------------------------------------------------------------------
def bag_of_location_features(locations: Sequence[str]) -> FeatureExtractor:
    """
    A trivial *observable* feature extractor: one-hot over a fixed
    set of locations indicating whether each one appears in the
    scenario prompt.

    Useful as a sanity baseline — if *this* beats chance, the scenario
    distribution itself is leaking the label and your probe needs
    better controls.
    """
    locs = list(locations)

    def fn(scenario: BeliefScenario) -> List[float]:
        text = scenario.prompt.lower()
        return [float(l.lower() in text) for l in locs]

    return fn


def output_probability_features(
    candidate_locations: Sequence[str],
    model_logprobs_fn: Callable[[str, Sequence[str]], Sequence[float]],
) -> FeatureExtractor:
    """
    Represent each scenario by the **model's log-probabilities**
    over a fixed set of candidate locations — a clean output-side
    feature vector that doesn't require model internals.

    Parameters
    ----------
    candidate_locations : sequence of str
        The locations to query the model about.
    model_logprobs_fn : callable
        Takes (prompt, candidates) → sequence of log-probabilities
        of each candidate answer under the model.  Users provide
        this wrapper around their own model; it's left abstract so
        the probe stays dependency-light.
    """
    def fn(scenario: BeliefScenario) -> Sequence[float]:
        return list(
            model_logprobs_fn(scenario.prompt, candidate_locations)
        )
    return fn


if __name__ == "__main__":
    # Self-contained smoke test: the bag-of-locations baseline
    # should NOT beat chance, because the label is defined by the
    # relative lexicographic order of two location names — a bit of
    # information that's genuinely independent of which locations
    # simply appear in the prompt.
    from .tom_benchmark import generate_sally_anne_scenarios

    scenarios = (
        generate_sally_anne_scenarios(n=60, order=1, seed=0)
        + generate_sally_anne_scenarios(n=60, order=2, seed=1)
    )

    extractor = bag_of_location_features(
        ["the basket", "the box", "the drawer", "the cupboard",
         "the shelf", "the backpack"]
    )
    probe = BeliefStateProbe(extractor, seed=42).fit_eval(scenarios)

    print("=" * 60)
    print("Humanising AI: Belief-state probe — leakage sanity check")
    print("=" * 60)
    print(f"Train accuracy : {probe.train_accuracy:.2%}")
    print(f"Test  accuracy : {probe.test_accuracy:.2%}")
    print(
        "\nWe expect test accuracy near 50% with this baseline feature "
        "set — confirming the benchmark itself doesn't leak the label."
    )
