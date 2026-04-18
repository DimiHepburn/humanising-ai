"""
Emotional Context Tracker
==========================

Conversations aren't single utterances — they're **arcs**.  A good
empathetic system needs to track how emotion accumulates, shifts
and recovers across many turns.

This module implements a simple but principled tracker:

1. Each new utterance produces a distribution over emotions
   (from any `EmotionBackend`).
2. The distribution is fused into a running **emotional state**
   via exponential smoothing.
3. The tracker exposes useful derived quantities: the dominant
   emotion, volatility (how fast the state is changing), and
   valence / arousal proxies.

Parameters like the decay constant and the valence map are
user-configurable so the same tracker can model short-horizon
emotional "weather" or long-horizon emotional "climate".

Author: Dimitri Romanov
Project: humanising-ai
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from .sentiment_pipeline import EmotionClassifier, EmotionBackend


# ---------------------------------------------------------------------------
# Russell's circumplex (Russell, 1980): approximate valence / arousal
# coordinates for the GoEmotions labels used elsewhere in this package.
# Values are in [-1, 1] for both axes.
# ---------------------------------------------------------------------------
VALENCE_AROUSAL: Dict[str, Tuple[float, float]] = {
    # label        : (valence, arousal)
    "admiration":    (+0.6, +0.3),
    "amusement":     (+0.7, +0.5),
    "anger":         (-0.6, +0.8),
    "annoyance":     (-0.4, +0.4),
    "approval":      (+0.4, +0.1),
    "caring":        (+0.5, +0.1),
    "confusion":     (-0.1, +0.2),
    "curiosity":     (+0.2, +0.4),
    "desire":        (+0.3, +0.5),
    "disappointment":(-0.6, -0.2),
    "disapproval":   (-0.4, +0.1),
    "disgust":       (-0.7, +0.4),
    "embarrassment": (-0.4, +0.3),
    "excitement":    (+0.7, +0.8),
    "fear":          (-0.6, +0.7),
    "gratitude":     (+0.7, +0.1),
    "grief":         (-0.8, -0.3),
    "joy":           (+0.8, +0.4),
    "love":          (+0.8, +0.3),
    "nervousness":   (-0.3, +0.5),
    "optimism":      (+0.6, +0.2),
    "pride":         (+0.5, +0.4),
    "realization":   (+0.1, +0.2),
    "relief":        (+0.6, -0.2),
    "remorse":       (-0.5, +0.0),
    "sadness":       (-0.7, -0.4),
    "surprise":      (+0.1, +0.7),
    "neutral":       (+0.0, +0.0),
}


@dataclass
class EmotionalSnapshot:
    """Point-in-time view of the tracker's state."""
    turn: int
    utterance: str
    instant: Dict[str, float]        # distribution for this utterance
    smoothed: Dict[str, float]       # post-smoothing cumulative state
    dominant: str                    # argmax of smoothed
    valence: float                   # running valence, in [-1, 1]
    arousal: float                   # running arousal, in [-1, 1]
    volatility: float                # L1 distance since previous snapshot


class EmotionalContextTracker:
    """
    Tracks emotional state across conversation turns.

    Parameters
    ----------
    classifier : EmotionClassifier, optional
        Classifier used to score each incoming utterance. Defaults
        to a fresh `EmotionClassifier()` (lexicon backend).
    decay : float in (0, 1)
        Smoothing factor.  `state = decay * state + (1 - decay) * new`.
        Higher = more memory, slower adaptation.
    window : int
        Size of the recent-history window retained for inspection.
    valence_arousal : dict, optional
        Custom label -> (valence, arousal) map.  Defaults to a
        circumplex based on Russell (1980).
    """

    def __init__(
        self,
        classifier: Optional[EmotionClassifier] = None,
        decay: float = 0.7,
        window: int = 16,
        valence_arousal: Optional[Dict[str, Tuple[float, float]]] = None,
        backend: Optional[EmotionBackend] = None,
    ):
        if classifier is None:
            classifier = EmotionClassifier(backend=backend)
        self.classifier = classifier
        self.decay = float(decay)
        self.state: Dict[str, float] = {}
        self.history: Deque[EmotionalSnapshot] = deque(maxlen=int(window))
        self.va_map = valence_arousal or VALENCE_AROUSAL
        self._turn = 0

    # ------------------------------------------------------------------ API
    def update(self, utterance: str) -> EmotionalSnapshot:
        """Score a new utterance and fold it into the running state."""
        self._turn += 1
        instant = self.classifier(utterance)

        # Exponential smoothing
        new_state: Dict[str, float] = dict(self.state)
        for label, score in instant.items():
            prev = new_state.get(label, 0.0)
            new_state[label] = self.decay * prev + (1 - self.decay) * score

        # Renormalise so the state remains a proper distribution
        total = sum(new_state.values()) or 1.0
        new_state = {k: v / total for k, v in new_state.items()}

        # Derived signals
        valence, arousal = self._va(new_state)
        volatility = self._l1_distance(self.state, new_state)
        dominant = max(new_state.items(), key=lambda kv: kv[1])[0]

        self.state = new_state
        snap = EmotionalSnapshot(
            turn=self._turn,
            utterance=utterance,
            instant=instant,
            smoothed=dict(new_state),
            dominant=dominant,
            valence=valence,
            arousal=arousal,
            volatility=volatility,
        )
        self.history.append(snap)
        return snap

    # ----------------------------------------------------------- inspection
    def top(self, k: int = 3) -> List[Tuple[str, float]]:
        """Top-k labels in the current smoothed state."""
        return sorted(self.state.items(), key=lambda kv: -kv[1])[:k]

    def valence(self) -> float:
        """Running valence (−1 negative, +1 positive)."""
        return self._va(self.state)[0]

    def arousal(self) -> float:
        """Running arousal (−1 calm, +1 activated)."""
        return self._va(self.state)[1]

    def snapshots(self) -> List[EmotionalSnapshot]:
        """List of snapshots in the current window."""
        return list(self.history)

    def reset(self) -> None:
        self.state.clear()
        self.history.clear()
        self._turn = 0

    # ---------------------------------------------------------- internals
    def _va(self, dist: Dict[str, float]) -> Tuple[float, float]:
        v = a = 0.0
        for label, p in dist.items():
            vv, aa = self.va_map.get(label, (0.0, 0.0))
            v += p * vv
            a += p * aa
        return v, a

    @staticmethod
    def _l1_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
        labels = set(a) | set(b)
        return sum(abs(a.get(l, 0.0) - b.get(l, 0.0)) for l in labels)


if __name__ == "__main__":
    tracker = EmotionalContextTracker(decay=0.6)

    conversation = [
        "I've been really struggling with anxiety lately.",
        "Work has been overwhelming and I can't sleep.",
        "Thanks for listening, it helps to talk about it.",
        "Actually — I feel a bit lighter now.",
        "Maybe I'll try that meditation thing tomorrow.",
    ]

    print("=" * 60)
    print("Humanising AI: Emotional context tracker demo")
    print("=" * 60)
    print(f"{'turn':>4} {'dominant':<14} {'val':>+5} {'aro':>+5} {'∥Δ∥1':>6}"
          f"  utterance")
    for text in conversation:
        s = tracker.update(text)
        print(f"{s.turn:>4} {s.dominant:<14} "
              f"{s.valence:+.2f} {s.arousal:+.2f} {s.volatility:>6.3f}"
              f"  {text}")
