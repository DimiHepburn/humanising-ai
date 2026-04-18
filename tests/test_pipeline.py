"""
Pipeline tests for humanising-ai
=================================

Where `test_smoke.py` checks that each sub-package *runs*, and
`test_invariants.py` checks that the values it produces satisfy
their invariants, this file checks that the sub-packages still
**compose**.

Concretely: we drive a short conversation through the four stages
of the Humanising Loop — perceive → attribute → attune /
respond → account — and assert that every stage hands off
something the next stage can actually use.

This is the cheapest early-warning signal we have for "modules
silently diverged while everyone's own tests kept passing."

Run with:
    pytest tests/test_pipeline.py

Author: Dimitri Romanov
Project: humanising-ai
"""

from __future__ import annotations

import json
import random
from typing import List

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared fixture: a small, emotionally varied conversation
# ---------------------------------------------------------------------------

CONVERSATION: List[str] = [
    "I've been really overwhelmed with work this week.",
    "I just feel like no matter how hard I try, I'm behind.",
    "Talking about it helps a bit though, thanks.",
    "I think I'm going to try to rest this weekend.",
]


@pytest.fixture()
def rng() -> random.Random:
    return random.Random(0)


# ---------------------------------------------------------------------------
# Stage 1 → Stage 3: Perceive feeds the dialogue context
# ---------------------------------------------------------------------------

def test_affective_signal_flows_into_dialogue_context(rng):
    """Each user turn should update the tracker inside the
    conversation context, producing an emotional arc of the
    expected length and shape."""
    from src.dialogue import EmpatheticResponder, TemplateGenerator

    bot = EmpatheticResponder(generator=TemplateGenerator(rng=rng))
    for turn in CONVERSATION:
        bot.respond(turn)

    arc = bot.context.emotional_arc()
    # One snapshot per user turn.
    assert len(arc) == len(CONVERSATION)

    for turn_idx, dominant, val, aro in arc:
        assert isinstance(dominant, str) and dominant
        assert -1.0 - 1e-9 <= val <= 1.0 + 1e-9
        assert -1.0 - 1e-9 <= aro <= 1.0 + 1e-9

    # A non-degenerate arc should show *some* movement in valence
    # across the conversation — if every turn has identical
    # valence the tracker isn't actually perceiving anything.
    valences = [v for _, _, v, _ in arc]
    assert max(valences) - min(valences) > 1e-6


# ---------------------------------------------------------------------------
# Stage 3 → Stage 4: Attune shapes Respond
# ---------------------------------------------------------------------------

def test_register_matches_emotion_family(rng):
    """A visibly distressed opener should produce a reply in the
    sadness/grief register; a visibly positive opener should not."""
    from src.dialogue import EmpatheticResponder, TemplateGenerator

    sad_bot = EmpatheticResponder(generator=TemplateGenerator(rng=rng))
    sad_reply = sad_bot.respond(
        "I've been crying on and off all day and I can't explain why."
    )

    happy_bot = EmpatheticResponder(
        generator=TemplateGenerator(rng=random.Random(1))
    )
    happy_reply = happy_bot.respond(
        "I got the job I wanted — I'm honestly over the moon."
    )

    sad_markers = {"heavy", "sorry", "painful"}
    happy_markers = {"wonderful", "glad", "appreciate"}

    assert any(m in sad_reply.lower() for m in sad_markers), sad_reply
    assert any(m in happy_reply.lower() for m in happy_markers), happy_reply
    # And crucially, the two registers must not be interchangeable:
    assert not any(m in happy_reply.lower() for m in sad_markers), happy_reply


# ---------------------------------------------------------------------------
# Full loop: Perceive → Attune → Respond → Account
# ---------------------------------------------------------------------------

def test_full_loop_produces_auditable_trace(rng):
    """After a short conversation, the system should expose every
    signal the Attunement Audit needs to score the interaction."""
    from src.dialogue import EmpatheticResponder, TemplateGenerator

    bot = EmpatheticResponder(generator=TemplateGenerator(rng=rng))
    for turn in CONVERSATION:
        bot.respond(turn)

    trace = json.loads(bot.context.to_json())

    # Perceive evidence
    assert "valence" in trace and "arousal" in trace
    assert -1.0 <= trace["valence"] <= 1.0
    assert -1.0 <= trace["arousal"] <= 1.0

    # Respond evidence — one assistant turn per user turn
    user_turns = [t for t in trace["turns"] if t["role"] == "user"]
    assistant_turns = [t for t in trace["turns"] if t["role"] == "assistant"]
    assert len(user_turns) == len(CONVERSATION)
    assert len(assistant_turns) == len(CONVERSATION)

    # Attune evidence — every assistant reply ends in an invitation
    for t in assistant_turns:
        assert t["text"].strip().endswith("?"), t["text"]

    # Account evidence — summary is non-empty and reflects the
    # actual content of the conversation, not a static string.
    summary = bot.context.summary(rebuild=True)
    assert summary and any(
        word in summary.lower() for word in
        ["overwhelmed", "behind", "rest", "weekend", "work"]
    ), summary


# ---------------------------------------------------------------------------
# Stage 5: Account — explainability plugs cleanly onto a scorer
# ---------------------------------------------------------------------------

def _toy_classifier():
    """A transparent two-class scorer we can reason about."""
    _POS = {"wonderful", "love", "great", "kind", "thank", "helpful", "calm"}
    _NEG = {"terrible", "hate", "awful", "cruel", "rude", "useless", "angry"}

    def scorer(text: str) -> np.ndarray:
        toks = [t.lower() for t in text.replace(".", " ").split()]
        pos = sum(t in _POS for t in toks)
        neg = sum(t in _NEG for t in toks)
        z = np.array([pos - 0.5 * neg, neg - 0.5 * pos], dtype=float)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    return scorer


def test_explainability_can_justify_a_classifier_verdict():
    """A classifier's top-1 verdict on an unambiguous input should
    be justified by at least one strong pro-verdict token."""
    from src.explainability import ContrastiveExplainer

    scorer = _toy_classifier()
    text = "The support agent was kind and helpful."
    probs = scorer(text)
    fact_idx = int(np.argmax(probs))
    fact, foil = ["positive", "negative"][fact_idx], \
                 ["positive", "negative"][1 - fact_idx]

    expl = ContrastiveExplainer(
        class_names=["positive", "negative"],
    ).explain_tokens(
        text=text,
        fact=fact,
        foil=foil,
        scorer=scorer,
    )

    # Gap should point the same way as the verdict
    assert expl.score_gap > 0

    # Top contribution should be a pro-fact token
    top_name, top_val = expl.contributions[0]
    assert top_val > 0, expl.contributions
    assert top_name.lower() in {
        "wonderful", "love", "great", "kind", "thank", "helpful", "calm",
    }, top_name


# ---------------------------------------------------------------------------
# Cross-stage negative control: the dialogue loop does not hide
# from the explainer
# ---------------------------------------------------------------------------

def test_dialogue_history_is_inspectable_for_audit():
    """The Attunement Audit requires the trace to be inspectable.
    That means replaying `as_messages()` should yield exactly what
    was said, in order, with no silent rewriting."""
    from src.dialogue import EmpatheticResponder, TemplateGenerator

    bot = EmpatheticResponder(
        generator=TemplateGenerator(rng=random.Random(0))
    )
    for turn in CONVERSATION:
        bot.respond(turn)

    messages = bot.context.as_messages()
    # Interleaved user/assistant, starting with user
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant"] * len(CONVERSATION)

    # Original user utterances are preserved verbatim
    user_texts = [m["content"] for m in messages if m["role"] == "user"]
    assert user_texts == CONVERSATION
