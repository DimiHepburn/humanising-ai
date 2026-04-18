"""
Empathetic Responder
=====================

End-to-end pipeline for empathy-aware dialogue.

Pipeline
--------
    user utterance
          │
          ▼
    ┌─────────────────────────┐
    │ EmotionalContextTracker │  ← per-turn emotion inference
    └─────────────────────────┘
          │                      (delegates to affective sub-package)
          ▼
    ┌─────────────────────────┐
    │   ConversationContext   │  ← long-horizon memory & arc
    └─────────────────────────┘
          │
          ▼
    ┌─────────────────────────┐
    │    ResponseGenerator    │  ← template / LLM / custom callable
    └─────────────────────────┘
          │
          ▼
        reply

`EmpatheticResponder` is a deliberately thin orchestration layer
so you can swap any stage out without rewriting the others:

* swap the emotion classifier by passing a different tracker
* swap the generator (TemplateGenerator, LLMGenerator, or any
  callable that matches the interface)
* swap the empathy evaluator by injecting one into `respond()`

Author: Dimitri Romanov
Project: humanising-ai
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

from ..affective.emotion_tracker import EmotionalContextTracker
from .context_manager import ConversationContext


# ---------------------------------------------------------------------------
# Generator interface
# ---------------------------------------------------------------------------
class ResponseGenerator(Protocol):
    """Anything that maps (user_text, context) → assistant_text."""

    def __call__(
        self,
        user_text: str,
        context: ConversationContext,
    ) -> str: ...


# ---------------------------------------------------------------------------
# Template-based generator (no deps, no network, always available)
# ---------------------------------------------------------------------------
@dataclass
class _TemplateBank:
    """Small banks of empathy-marked response templates, keyed by
    emotion family.  Each template contains the literal placeholder
    ``{echo}`` which is filled with a short, grounded echo of the
    user's own words."""

    openers: Dict[str, List[str]]
    followups: Dict[str, List[str]]


_DEFAULT_BANK = _TemplateBank(
    openers={
        "sadness": [
            "That sounds really heavy. {echo}",
            "I'm sorry you're going through that. {echo}",
            "That makes sense — what you're describing is painful. {echo}",
        ],
        "grief": [
            "I'm so sorry for what you're carrying right now. {echo}",
            "Loss like that takes its own time. {echo}",
        ],
        "fear": [
            "That uncertainty sounds exhausting. {echo}",
            "It's understandable to feel on edge about this. {echo}",
        ],
        "anger": [
            "That would frustrate me too. {echo}",
            "That does sound unfair. {echo}",
        ],
        "annoyance": [
            "That's genuinely irritating. {echo}",
            "No wonder you're fed up. {echo}",
        ],
        "joy": [
            "That's wonderful to hear. {echo}",
            "I'm really glad for you. {echo}",
        ],
        "gratitude": [
            "I appreciate you saying that. {echo}",
        ],
        "confusion": [
            "It's okay to not have it all mapped out. {echo}",
        ],
        "neutral": [
            "Thanks for sharing that with me. {echo}",
        ],
    },
    followups={
        "sadness": [
            "What feels hardest about it right now?",
            "Is there something specific that's been weighing on you?",
        ],
        "grief": [
            "Would it help to talk a little more about them?",
        ],
        "fear": [
            "Is there a part of this that feels most unknown?",
        ],
        "anger": [
            "What would feel like a fair outcome to you?",
        ],
        "annoyance": [
            "What's the part that's bugging you the most?",
        ],
        "joy": [
            "What's been the best bit so far?",
        ],
        "gratitude": [
            "Anything else on your mind?",
        ],
        "confusion": [
            "Want to think it through out loud together?",
        ],
        "neutral": [
            "Tell me more?",
        ],
    },
)


class TemplateGenerator:
    """
    Deterministic, dependency-free response generator.

    Chooses an opener template based on the dominant user emotion,
    inserts a short echo of the user's own phrasing, then adds a
    gentle, open-ended follow-up to invite elaboration.

    Parameters
    ----------
    bank : _TemplateBank, optional
        Override the default opener/follow-up templates.
    rng : random.Random, optional
        For reproducible outputs in tests / demos.
    """

    def __init__(
        self,
        bank: Optional[_TemplateBank] = None,
        rng: Optional[random.Random] = None,
    ):
        self.bank = bank or _DEFAULT_BANK
        self.rng = rng or random.Random()

    def __call__(
        self,
        user_text: str,
        context: ConversationContext,
    ) -> str:
        emotion = context.dominant_user_emotion() or "neutral"
        key = self._map_emotion(emotion)

        opener_template = self.rng.choice(
            self.bank.openers.get(key, self.bank.openers["neutral"])
        )
        followup = self.rng.choice(
            self.bank.followups.get(key, self.bank.followups["neutral"])
        )

        echo = self._build_echo(user_text)
        opener = opener_template.format(echo=echo).strip()
        if not opener.endswith((".", "!", "?")):
            opener += "."
        return f"{opener} {followup}"

    # ---------------------------------------------------------- helpers
    @staticmethod
    def _map_emotion(label: str) -> str:
        """Collapse the GoEmotions label set to the template families
        we cover."""
        low_valence = {"sadness", "disappointment", "remorse", "embarrassment"}
        fearlike = {"fear", "nervousness"}
        angerlike = {"anger", "disapproval", "disgust"}
        joylike = {"joy", "excitement", "love", "amusement", "pride",
                   "relief", "optimism"}
        if label in low_valence:
            return "sadness"
        if label == "grief":
            return "grief"
        if label in fearlike:
            return "fear"
        if label in angerlike:
            return "anger"
        if label == "annoyance":
            return "annoyance"
        if label == "gratitude":
            return "gratitude"
        if label == "confusion":
            return "confusion"
        if label in joylike:
            return "joy"
        return "neutral"

    @staticmethod
    def _build_echo(user_text: str) -> str:
        """Paraphrase-style short echo of the user's utterance —
        deliberately simple: take the first clause up to ~12 words
        and rephrase into second person, so the response feels
        grounded without quoting verbatim."""
        text = user_text.strip().rstrip(".!?")
        first_clause = text.split(",")[0]
        words = first_clause.split()
        snippet = " ".join(words[:12])
        if not snippet:
            return ""
        # Lightweight first-person → second-person flip
        replacements = [
            (" i ", " you "),
            ("I ", "You "),
            (" my ", " your "),
            ("My ", "Your "),
            (" me ", " you "),
            (" we ", " you both "),
        ]
        padded = f" {snippet} "
        for src, dst in replacements:
            padded = padded.replace(src, dst)
        return f"It sounds like{padded.rstrip()}."


# ---------------------------------------------------------------------------
# LLM-backed generator (optional — only used if the user asks for it)
# ---------------------------------------------------------------------------
class LLMGenerator:
    """
    Thin wrapper around a user-provided LLM `chat` function.

    We don't depend on any specific SDK — instead, the caller
    supplies a callable with the signature

        chat(messages: list[dict[str, str]], **kwargs) -> str

    which matches the shape used by OpenAI / Anthropic / local
    llama.cpp servers alike.  This keeps the responder dependency-
    free while still being directly usable with real models.

    Parameters
    ----------
    chat_fn : callable
        The LLM callable described above.
    system_prompt : str, optional
        Overrides the default empathy-aware system prompt.
    extra_params : dict, optional
        Extra keyword arguments passed through to ``chat_fn``.
    """

    DEFAULT_SYSTEM = (
        "You are a calm, emotionally attuned conversation partner. "
        "Before responding, silently consider: what is the speaker "
        "feeling, what might they need from me right now, and how "
        "can I acknowledge that without rushing to fix it? Respond "
        "briefly (2-4 sentences), validating their experience before "
        "offering anything else."
    )

    def __init__(
        self,
        chat_fn: Callable[..., str],
        system_prompt: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ):
        self.chat_fn = chat_fn
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM
        self.extra_params = dict(extra_params or {})

    def __call__(
        self,
        user_text: str,
        context: ConversationContext,
    ) -> str:
        msgs: List[Dict[str, str]] = [{"role": "system",
                                       "content": self._augmented_system(context)}]
        msgs.extend(context.as_messages())
        # The latest user turn is already the last item, but we
        # re-assert it to be safe if someone calls us out-of-order.
        if not msgs or msgs[-1].get("role") != "user" \
                or msgs[-1].get("content") != user_text:
            msgs.append({"role": "user", "content": user_text})
        return self.chat_fn(msgs, **self.extra_params)

    def _augmented_system(self, context: ConversationContext) -> str:
        emo = context.dominant_user_emotion() or "neutral"
        val = context.tracker.valence()
        summary = context.summary(rebuild=False)
        hint = (
            f"\n\nConversation signal:\n"
            f"- dominant user emotion: {emo}\n"
            f"- running valence: {val:+.2f}  (−1 low, +1 positive)\n"
            f"- summary: {summary or '(new conversation)'}"
        )
        return self.system_prompt + hint


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------
class EmpatheticResponder:
    """
    A single entry-point for emotion-aware dialogue.

    Parameters
    ----------
    generator : ResponseGenerator, optional
        Defaults to `TemplateGenerator()` so it runs out of the box.
    context : ConversationContext, optional
        Defaults to a fresh context with a fresh emotion tracker.

    Example
    -------
    >>> bot = EmpatheticResponder()
    >>> bot.respond("I've been really struggling lately.")
    "That sounds really heavy. It sounds like you've been really
    struggling lately. What feels hardest about it right now?"
    """

    def __init__(
        self,
        generator: Optional[ResponseGenerator] = None,
        context: Optional[ConversationContext] = None,
    ):
        self.generator: ResponseGenerator = generator or TemplateGenerator()
        self.context = context or ConversationContext()

    # ------------------------------------------------------------------ API
    def respond(
        self,
        user_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Ingest a user message, generate and log an assistant reply."""
        self.context.add("user", user_text, metadata=metadata)
        reply = self.generator(user_text, self.context)
        self.context.add("assistant", reply)
        return reply

    # Convenience accessors ----------------------------------------------
    @property
    def tracker(self) -> EmotionalContextTracker:
        return self.context.tracker

    def history(self) -> List[Dict[str, str]]:
        return self.context.as_messages()

    def reset(self) -> None:
        self.context.clear()


if __name__ == "__main__":
    rng = random.Random(0)
    bot = EmpatheticResponder(generator=TemplateGenerator(rng=rng))

    turns = [
        "I've been really overwhelmed with work this week.",
        "I just feel like no matter how hard I try, I'm behind.",
        "Talking about it helps a bit though, thanks.",
        "I think I'm going to try to rest this weekend.",
    ]

    print("=" * 60)
    print("Humanising AI: Empathetic responder demo")
    print("=" * 60)
    for t in turns:
        print(f"\nUser      : {t}")
        print(f"Assistant : {bot.respond(t)}")
        snap = bot.tracker.history[-1]
        print(f"  [signal]: {snap.dominant:<10} "
              f"v={snap.valence:+.2f} a={snap.arousal:+.2f}")

    print("\n--- Final context summary ---")
    print(bot.context.summary(rebuild=True))
