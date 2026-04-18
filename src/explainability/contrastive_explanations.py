"""
Contrastive explanations
========================

"Why X and not Y?" — contrastive explanations are the form of
explanation humans ask for and accept most naturally (Miller, 2019;
Lipton, 1990).  Rather than "why is this prediction what it is?",
we answer "why *this* outcome rather than *that* one?".

Two complementary approaches are implemented here:

1. **Feature-contrast (white-box inputs)**
   For a prediction function ``f : R^d -> R^C`` and an input ``x``,
   find the smallest set of feature changes that would flip the
   prediction from class `fact` to class `foil`.  Useful for tabular
   data and structured features.

2. **Token-contrast (black-box text)**
   For a text scorer ``f : str -> R^C``, find which individual
   tokens most contribute to the *difference* ``f(x)[fact] -
   f(x)[foil]`` via leave-one-out perturbation.  Works with any
   text classifier without needing gradients or internals.

Both return a `ContrastiveExplanation` carrying the ranked
contributions plus a human-readable verdict.

Author: Dimitri Romanov
Project: humanising-ai
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class ContrastiveExplanation:
    """
    Output of a contrastive explainer.

    Attributes
    ----------
    fact : str
        The class actually predicted (or the class being defended).
    foil : str
        The class the user wanted to compare against.
    score_gap : float
        ``f(x)[fact] - f(x)[foil]`` — positive means the fact wins.
    contributions : list of (name, value)
        Per-feature/token contributions to *the gap*, sorted by
        absolute magnitude, descending.  Positive values support
        the fact over the foil; negative values support the foil.
    flipped_at : int, optional
        For feature-contrast: how many features had to change to
        flip the prediction.  None if no flip was found.
    """

    fact: str
    foil: str
    score_gap: float
    contributions: List[Tuple[str, float]]
    flipped_at: Optional[int] = None

    def top(self, k: int = 5) -> List[Tuple[str, float]]:
        return self.contributions[:k]

    def verdict(self, k: int = 3) -> str:
        """A short natural-language summary of the top drivers."""
        if not self.contributions:
            return (f"{self.fact!r} and {self.foil!r} score identically "
                    f"on this input.")

        winning = [n for n, v in self.contributions[:k] if v > 0]
        losing = [n for n, v in self.contributions[:k] if v < 0]
        parts = [f"{self.fact!r} wins over {self.foil!r} by "
                 f"{self.score_gap:+.3f}."]
        if winning:
            parts.append(
                "Main drivers for " + repr(self.fact) + ": "
                + ", ".join(winning) + "."
            )
        if losing:
            parts.append(
                "Main pull toward " + repr(self.foil) + ": "
                + ", ".join(losing) + "."
            )
        if self.flipped_at is not None:
            parts.append(
                f"Flipping to {self.foil!r} requires changing "
                f"{self.flipped_at} feature(s)."
            )
        return " ".join(parts)

    def __repr__(self) -> str:
        head = (f"ContrastiveExplanation  fact={self.fact!r}  "
                f"foil={self.foil!r}  gap={self.score_gap:+.4f}")
        body = "\n".join(
            f"  {name:<24} {val:+.4f}"
            for name, val in self.top(8)
        )
        return head + "\n" + body


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TokenScorer = Callable[[str], np.ndarray]          # str -> (C,) scores
VectorScorer = Callable[[np.ndarray], np.ndarray]  # (B,d) -> (B,C) scores


_TOKEN_RE = re.compile(r"\w+|[^\s\w]", re.UNICODE)


def _tokenise(text: str) -> List[str]:
    """Simple, deterministic whitespace+punctuation tokenizer."""
    return _TOKEN_RE.findall(text)


def _detokenise(tokens: Sequence[str]) -> str:
    """Reassemble tokens with reasonable spacing."""
    out: List[str] = []
    for i, tok in enumerate(tokens):
        if i == 0:
            out.append(tok)
            continue
        if re.match(r"[^\w]", tok):
            out.append(tok)
        else:
            out.append(" " + tok)
    return "".join(out)


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------
class ContrastiveExplainer:
    """
    Build "why X and not Y?" explanations for black-box predictors.

    Two modes:

    * **Feature mode** — call `explain_features(x, fact, foil, ...)`
      on a numeric vector with a `VectorScorer`.
    * **Token mode** — call `explain_tokens(text, fact, foil, ...)`
      on a string with a `TokenScorer`.

    Parameters
    ----------
    class_names : sequence of str
        Names for the score columns returned by the scorer.
    """

    def __init__(self, class_names: Sequence[str]):
        self.class_names = list(class_names)
        self._index = {name: i for i, name in enumerate(self.class_names)}

    # ------------------------------------------------------------------
    # Feature-space contrastive explanations
    # ------------------------------------------------------------------
    def explain_features(
        self,
        x: np.ndarray,
        fact: str,
        foil: str,
        scorer: VectorScorer,
        baseline: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
        greedy_flip: bool = True,
    ) -> ContrastiveExplanation:
        """
        Explain a numeric prediction contrastively.

        Parameters
        ----------
        x : (d,) array
            The instance to explain.
        fact, foil : str
            Two class names from `class_names` to contrast.
        scorer : VectorScorer
            Model callable returning per-class scores for a batch.
        baseline : (d,) array
            Reference / "neutral" values to substitute when asking
            "what if this feature were absent?".
        feature_names : list[str], optional
            Names for the per-feature columns.
        greedy_flip : bool
            If True, also report how many features have to change to
            flip the prediction to `foil`, via greedy replacement.
        """
        fact_i, foil_i = self._pair(fact, foil)
        x = np.asarray(x, dtype=float).reshape(-1)
        d = x.shape[0]
        names = list(feature_names or [f"f{i}" for i in range(d)])
        if len(names) != d:
            raise ValueError("feature_names must match x length.")

        baseline = np.asarray(baseline, dtype=float).reshape(-1)
        if baseline.shape[0] != d:
            raise ValueError("baseline must have same length as x.")

        # Per-feature attribution via leave-one-in
        scores_x = scorer(x[None, :])[0]
        gap_x = float(scores_x[fact_i] - scores_x[foil_i])

        contribs: List[Tuple[str, float]] = []
        batch = np.tile(x, (d, 1))
        for i in range(d):
            batch[i, i] = baseline[i]
        scores_masked = scorer(batch)
        for i in range(d):
            gap_masked = float(
                scores_masked[i, fact_i] - scores_masked[i, foil_i]
            )
            contribs.append((names[i], gap_x - gap_masked))

        contribs.sort(key=lambda kv: -abs(kv[1]))

        flipped_at: Optional[int] = None
        if greedy_flip:
            flipped_at = self._greedy_flip_count(
                x, baseline, fact_i, foil_i, scorer,
            )

        return ContrastiveExplanation(
            fact=fact,
            foil=foil,
            score_gap=gap_x,
            contributions=contribs,
            flipped_at=flipped_at,
        )

    # ------------------------------------------------------------------
    # Text contrastive explanations
    # ------------------------------------------------------------------
    def explain_tokens(
        self,
        text: str,
        fact: str,
        foil: str,
        scorer: TokenScorer,
        replacement: str = "___",
    ) -> ContrastiveExplanation:
        """
        Explain a text prediction contrastively via leave-one-out.

        Parameters
        ----------
        text : str
            The input text.
        fact, foil : str
            Class names to contrast.
        scorer : TokenScorer
            Model callable: str -> (C,) per-class scores.
        replacement : str
            Token used to stand in for a "removed" input token.
        """
        fact_i, foil_i = self._pair(fact, foil)
        tokens = _tokenise(text)
        if not tokens:
            return ContrastiveExplanation(
                fact=fact, foil=foil, score_gap=0.0, contributions=[],
            )

        base_scores = scorer(text)
        gap_base = float(base_scores[fact_i] - base_scores[foil_i])

        contribs: List[Tuple[str, float]] = []
        for i, tok in enumerate(tokens):
            perturbed = tokens.copy()
            perturbed[i] = replacement
            new_scores = scorer(_detokenise(perturbed))
            gap_new = float(new_scores[fact_i] - new_scores[foil_i])
            # Positive contribution ⇒ removing this token *reduced* the
            # gap, i.e. the token was helping the fact over the foil.
            contribs.append((tok, gap_base - gap_new))

        contribs.sort(key=lambda kv: -abs(kv[1]))
        return ContrastiveExplanation(
            fact=fact,
            foil=foil,
            score_gap=gap_base,
            contributions=contribs,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _pair(self, fact: str, foil: str) -> Tuple[int, int]:
        if fact not in self._index:
            raise ValueError(f"Unknown fact class {fact!r}")
        if foil not in self._index:
            raise ValueError(f"Unknown foil class {foil!r}")
        if fact == foil:
            raise ValueError("fact and foil must differ.")
        return self._index[fact], self._index[foil]

    @staticmethod
    def _greedy_flip_count(
        x: np.ndarray,
        baseline: np.ndarray,
        fact_i: int,
        foil_i: int,
        scorer: VectorScorer,
    ) -> Optional[int]:
        """
        Greedily replace whichever single feature most reduces the
        fact-vs-foil gap at each step, until it flips sign.
        Returns the count, or None if no flip is reachable.
        """
        x_cur = x.copy()
        d = x.shape[0]
        replaced: set = set()
        for step in range(1, d + 1):
            best_i = -1
            best_gap = math.inf
            for i in range(d):
                if i in replaced:
                    continue
                cand = x_cur.copy()
                cand[i] = baseline[i]
                s = scorer(cand[None, :])[0]
                gap = float(s[fact_i] - s[foil_i])
                if gap < best_gap:
                    best_gap = gap
                    best_i = i
            if best_i < 0:
                return None
            x_cur[best_i] = baseline[best_i]
            replaced.add(best_i)
            if best_gap <= 0:
                return step
        return None


if __name__ == "__main__":
    # Feature-mode demo: a linear scorer with two classes.
    rng = np.random.default_rng(0)
    d = 4
