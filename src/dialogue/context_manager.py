"""
Conversation Context Manager
=============================

Long-horizon memory for a conversation.  Unlike a flat chat history,
a `ConversationContext` tracks:

* **Turn history** — the raw utterances, with timestamps and roles.
* **Emotional arc** — per-turn valence / arousal / dominant emotion
  (delegated to `EmotionalContextTracker`).
* **Salience-weighted summary** — a compressed view of what the
  conversation has been *about*, so downstream response generators
  can stay grounded even as the window grows long.

Everything is in-memory, JSON-serialisable, and dependency-light
so it can be embedded in agents, notebooks, or lightweight services
without pulling in heavy ML stacks.

Author: Dimitri Romanov
Project: humanising-ai
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from ..affective.emotion_tracker import (
    EmotionalContextTracker,
    EmotionalSnapshot,
)


Role = Literal["user", "assistant", "system"]


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------
@dataclass
class ConversationTurn:
    """A single message, with optional attached emotional snapshot."""
    role: Role
    text: str
    timestamp: float = field(default_factory=time.time)
    emotion: Optional[EmotionalSnapshot] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "role": self.role,
            "text": self.text,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }
        if self.emotion is not None:
            d["emotion"] = {
                "turn": self.emotion.turn,
                "dominant": self.emotion.dominant,
                "valence": self.emotion.valence,
                "arousal": self.emotion.arousal,
                "volatility": self.emotion.volatility,
                "top": sorted(
                    self.emotion.smoothed.items(),
                    key=lambda kv: -kv[1],
                )[:3],
            }
        return d


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class ConversationContext:
    """
    In-memory conversation state with emotional tracking.

    Parameters
    ----------
    window : int
        Maximum number of turns to retain in full fidelity.  Older
        turns are retained only through the summary.
    tracker : EmotionalContextTracker, optional
        Injected tracker; a fresh one is created if not provided.
    summary_every : int
        Recompute the salience-weighted summary every N turns.
    stopwords : iterable of str, optional
        Custom stop-word list for summarisation.
    """

    _DEFAULT_STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "if", "then", "so",
        "is", "are", "was", "were", "be", "been", "being",
        "i", "you", "he", "she", "it", "we", "they", "me", "him",
        "her", "us", "them", "my", "your", "his", "its", "our",
        "their", "this", "that", "these", "those",
        "of", "to", "in", "on", "at", "for", "with", "from",
        "have", "has", "had", "do", "does", "did",
        "not", "no", "yes", "okay", "ok", "just", "really", "like",
        "can", "could", "would", "should", "will", "may", "might",
        "about", "as", "by", "because", "what", "when", "where",
        "how", "why", "which", "who",
    }

    def __init__(
        self,
        window: int = 32,
        tracker: Optional[EmotionalContextTracker] = None,
        summary_every: int = 4,
        stopwords: Optional[Iterable[str]] = None,
    ):
        self.window = int(window)
        self.tracker = tracker or EmotionalContextTracker()
        self.summary_every = int(summary_every)
        self.stopwords = set(stopwords) if stopwords else set(
            self._DEFAULT_STOPWORDS
        )

        self._turns: List[ConversationTurn] = []
        self._summary: str = ""

    # ------------------------------------------------------------------ API
    def add(
        self,
        role: Role,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationTurn:
        """Append a new turn and (for user turns) update the emotion
        tracker."""
        emo: Optional[EmotionalSnapshot] = None
        if role == "user":
            emo = self.tracker.update(text)

        turn = ConversationTurn(
            role=role,
            text=text,
            emotion=emo,
            metadata=dict(metadata or {}),
        )
        self._turns.append(turn)

        # Maintain the retention window — older turns are folded into
        # the summary rather than dropped silently.
        if len(self._turns) > self.window:
            self._turns = self._turns[-self.window :]

        if len(self._turns) % self.summary_every == 0:
            self._summary = self._build_summary()

        return turn

    # ----------------------------------------------------------------- views
    def recent(self, k: int = 6) -> List[ConversationTurn]:
        """Most recent k turns (oldest first)."""
        return list(self._turns[-k:])

    def as_messages(self) -> List[Dict[str, str]]:
        """Turns in the ``{role, content}`` shape used by chat LLMs."""
        return [{"role": t.role, "content": t.text} for t in self._turns]

    def emotional_arc(self) -> List[Tuple[int, str, float, float]]:
        """(turn, dominant, valence, arousal) per user turn."""
        arc = []
        for t in self._turns:
            if t.emotion is not None:
                arc.append((
                    t.emotion.turn,
                    t.emotion.dominant,
                    t.emotion.valence,
                    t.emotion.arousal,
                ))
        return arc

    def dominant_user_emotion(self) -> Optional[str]:
        """Current dominant emotion in the user's running state."""
        if not self.tracker.state:
            return None
        return max(self.tracker.state.items(), key=lambda kv: kv[1])[0]

    def summary(self, rebuild: bool = False) -> str:
        """Return (or recompute) a short salience-weighted summary."""
        if rebuild or not self._summary:
            self._summary = self._build_summary()
        return self._summary

    # ------------------------------------------------------------- serialise
    def to_json(self) -> str:
        """Serialise the whole context for persistence / debugging."""
        data = {
            "window": self.window,
            "summary": self.summary(),
            "turns": [t.to_dict() for t in self._turns],
            "emotional_state": dict(self.tracker.state),
            "valence": self.tracker.valence(),
            "arousal": self.tracker.arousal(),
        }
        return json.dumps(data, indent=2, default=str)

    def clear(self) -> None:
        self._turns.clear()
        self._summary = ""
        self.tracker.reset()

    # -------------------------------------------------------- internals
    def _tokens(self, text: str) -> List[str]:
        return [
            t for t in re.findall(r"[a-zA-Z]{3,}", text.lower())
            if t not in self.stopwords
        ]

    def _build_summary(self) -> str:
        """
        Produce a short, human-readable summary of the conversation.

        Approach: most-frequent content words (minus stop-words)
        plus the current dominant emotion.  Deliberately shallow so
        it runs in microseconds on any machine — swap in an LLM
        summariser via `metadata["summary"]` if you want something
        fancier.
        """
        if not self._turns:
            return ""
        token_counts: Counter = Counter()
        for t in self._turns:
            token_counts.update(self._tokens(t.text))

        top_terms = [w for w, _ in token_counts.most_common(8)]
        emo = self.dominant_user_emotion() or "neutral"
        val = self.tracker.valence()
        arc = "lightening" if val > 0.15 else (
            "heavy" if val < -0.15 else "level"
        )

        terms_str = ", ".join(top_terms) if top_terms else "—"
        return (
            f"{len(self._turns)} turns; emotional tone is {arc} "
            f"(dominant: {emo}, valence={val:+.2f}). "
            f"Salient terms: {terms_str}."
        )

    # -------------------------------------------------------------- magic
    def __len__(self) -> int:
        return len(self._turns)

    def __iter__(self):
        return iter(list(self._turns))


if __name__ == "__main__":
    ctx = ConversationContext(window=16, summary_every=2)

    exchanges = [
        ("user",      "I've been really overwhelmed with work this week."),
        ("assistant", "That sounds exhausting. What's been weighing on you "
                      "the most?"),
        ("user",      "Mostly deadlines and feeling like I can't keep up."),
        ("assistant", "Falling behind on things that matter is stressful. "
                      "Have you been getting any rest?"),
        ("user",      "Not really. But talking about it helps a bit, thanks."),
    ]

    print("=" * 60)
    print("Humanising AI: Conversation context demo")
    print("=" * 60)
    for role, text in exchanges:
        ctx.add(role, text)

    print("\nEmotional arc (user turns only):")
    for turn, dom, v, a in ctx.emotional_arc():
        print(f"  turn {turn}: {dom:<10} v={v:+.2f} a={a:+.2f}")

    print("\nSummary:")
    print(f"  {ctx.summary(rebuild=True)}")

    print(f"\nFinal dominant user emotion: {ctx.dominant_user_emotion()}")
