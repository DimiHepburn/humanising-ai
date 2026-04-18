"""
Sentiment / Emotion pipeline
=============================

A unified interface for single-utterance emotion classification with
two interchangeable backends:

1. **LexiconBackend** (default) — a small, transparent, regex-based
   classifier over the emotion vocabulary of
   [GoEmotions](https://arxiv.org/abs/2005.00547).  Runs instantly
   with zero model downloads.

2. **TransformerBackend** — a thin wrapper around a HuggingFace
   ``text-classification`` pipeline.  Only imported lazily; if
   ``transformers`` isn't installed a clear error is raised *at
   use* so importing this file is always cheap.

Both backends return the same data structure:

    { "joy": 0.62, "sadness": 0.11, ... }     (dict[str, float], sums to 1.0)

so downstream code (trackers, dialogue systems, evaluation) doesn't
care which one is in use.

Author: Dimitri Romanov
Project: humanising-ai
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Protocol


# ---------------------------------------------------------------------------
# The GoEmotions label set — we use this as a stable, widely recognised
# taxonomy even when the lexicon backend is active.
# ---------------------------------------------------------------------------
GOEMOTIONS_LABELS: List[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]


# ---------------------------------------------------------------------------
# Lexicon for the lightweight backend.  Intentionally short — this is a
# transparent default, not a state-of-the-art classifier.
# ---------------------------------------------------------------------------
LEXICON: Dict[str, List[str]] = {
    "admiration":    ["admire", "amazing", "incredible", "impressive"],
    "amusement":     ["funny", "lol", "hilarious", "haha"],
    "anger":         ["angry", "furious", "mad", "hate", "pissed"],
    "annoyance":     ["annoying", "irritated", "bothered", "frustrat"],
    "approval":      ["agree", "support", "yes", "absolutely"],
    "caring":        ["care", "love you", "worried about you"],
    "confusion":     ["confused", "don't understand", "unsure", "lost"],
    "curiosity":     ["curious", "wonder", "interested", "intrigued"],
    "desire":        ["want", "wish", "need", "crave"],
    "disappointment":["disappointed", "let down", "expected more"],
    "disapproval":   ["disapprove", "against", "no way"],
    "disgust":       ["disgusted", "gross", "revolting", "yuck"],
    "embarrassment": ["embarrassed", "awkward", "ashamed"],
    "excitement":    ["excited", "can't wait", "thrilled"],
    "fear":          ["scared", "afraid", "terrified", "worried"],
    "gratitude":     ["thank", "grateful", "appreciate"],
    "grief":         ["grief", "bereaved", "mourning"],
    "joy":           ["happy", "joyful", "delighted", "glad"],
    "love":          ["love", "adore", "cherish"],
    "nervousness":   ["nervous", "anxious", "uneasy", "jittery"],
    "optimism":      ["hopeful", "optimistic", "things will"],
    "pride":         ["proud", "accomplished"],
    "realization":   ["realise", "realize", "figured out", "now i see"],
    "relief":        ["relieved", "phew", "thank god"],
    "remorse":       ["regret", "sorry", "my fault"],
    "sadness":       ["sad", "down", "depressed", "miserable", "heartbroken"],
    "surprise":      ["surprised", "shocked", "didn't expect", "wow"],
}


def _softmax_from_counts(counts: Dict[str, float]) -> Dict[str, float]:
    """Normalise a count dictionary into a proper distribution (adds
    a small 'neutral' floor so an empty document still returns a
    valid distribution)."""
    if not counts or sum(counts.values()) == 0.0:
        return {"neutral": 1.0}
    total = sum(counts.values())
    dist = {k: v / total for k, v in counts.items() if v > 0}
    # Add a light 'neutral' floor so subsequent smoothing is stable
    dist.setdefault("neutral", 0.0)
    return dist


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------
class EmotionBackend(Protocol):
    """A callable that maps a string to a distribution over emotions."""

    def __call__(self, text: str) -> Dict[str, float]: ...


class LexiconBackend:
    """
    Regex-based emotion classifier.

    Deliberately simple: for each emotion in `LEXICON`, count how many
    of its trigger words/phrases appear in the text, then normalise.

    This is **not** a state-of-the-art classifier — it's a transparent
    default so every module in the repo is runnable with no heavy
    dependencies.  Swap in the TransformerBackend for anything serious.
    """

    def __init__(self, lexicon: Optional[Dict[str, List[str]]] = None):
        self.lexicon = lexicon or LEXICON
        self._patterns: Dict[str, List[re.Pattern]] = {
            label: [re.compile(re.escape(w), re.IGNORECASE) for w in words]
            for label, words in self.lexicon.items()
        }

    def __call__(self, text: str) -> Dict[str, float]:
        counts: Dict[str, float] = {}
        for label, patterns in self._patterns.items():
            c = sum(1 for p in patterns if p.search(text))
            if c > 0:
                counts[label] = float(c)
        return _softmax_from_counts(counts)


class TransformerBackend:
    """
    Lazy-loading wrapper around a HuggingFace ``text-classification``
    pipeline with ``top_k=None`` so we get full per-label scores.

    Example
    -------
    >>> backend = TransformerBackend("SamLowe/roberta-base-go_emotions")
    >>> clf = EmotionClassifier(backend=backend)

    If ``transformers`` is not installed, *constructing* this object
    raises a helpful ImportError — but the module itself still imports
    fine, so the lexicon default remains usable.
    """

    def __init__(
        self,
        model: str = "SamLowe/roberta-base-go_emotions",
        device: Optional[Any] = None,
    ):
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as e:
            raise ImportError(
                "TransformerBackend needs the `transformers` package.\n"
                "Install it with:  pip install transformers torch\n"
                "Or use the default LexiconBackend for a no-dependency run."
            ) from e

        self._pipeline = pipeline(
            "text-classification",
            model=model,
            top_k=None,
            device=device,
        )

    def __call__(self, text: str) -> Dict[str, float]:
        raw = self._pipeline(text)
        # Pipelines return either [[{...}]] or [{...}] depending on version
        rows = raw[0] if isinstance(raw[0], list) else raw
        scores = {r["label"]: float(r["score"]) for r in rows}
        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in scores.items()}


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------
class EmotionClassifier:
    """
    Backend-agnostic emotion classifier.

    Parameters
    ----------
    backend : EmotionBackend, optional
        Any callable mapping str -> dict[str, float].  Defaults to
        `LexiconBackend()`.

    Examples
    --------
    >>> clf = EmotionClassifier()                     # lexicon default
    >>> clf("I'm so excited about tomorrow!")
    {'excitement': 1.0, 'neutral': 0.0}

    >>> clf = EmotionClassifier(backend=TransformerBackend())
    >>> clf("I'm so excited about tomorrow!")
    {'excitement': 0.87, 'joy': 0.06, ...}
    """

    def __init__(self, backend: Optional[EmotionBackend] = None):
        self.backend: EmotionBackend = backend or LexiconBackend()

    def __call__(self, text: str) -> Dict[str, float]:
        return self.backend(text)

    # Convenience helpers ---------------------------------------------------
    def top(self, text: str, k: int = 3) -> List[tuple[str, float]]:
        """Return the top-k (label, probability) pairs."""
        scores = self(text)
        return sorted(scores.items(), key=lambda kv: -kv[1])[:k]

    def dominant(self, text: str) -> str:
        """Return the single most likely label."""
        scores = self(text)
        return max(scores.items(), key=lambda kv: kv[1])[0]


if __name__ == "__main__":
    clf = EmotionClassifier()      # lexicon backend
    samples = [
        "I'm so grateful — you really helped me get through this.",
        "I can't believe he said that to me, I'm furious.",
        "I don't know what to do, I feel lost and anxious.",
        "That was hilarious, I haven't laughed this hard in ages.",
    ]
    print("=" * 60)
    print("Humanising AI: EmotionClassifier demo (lexicon backend)")
    print("=" * 60)
    for s in samples:
        print(f"\n> {s}")
        for label, p in clf.top(s, k=3):
            print(f"    {label:<14} {p:.2f}")
