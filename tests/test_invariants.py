"""
Invariant tests for humanising-ai
==================================

Where `test_smoke.py` checks that the public API *runs*, this
file checks that the values it produces *satisfy the invariants
downstream code depends on*.

The four invariants documented in `src/README.md` are each
exercised here, plus a couple of "negative-control must stay
broken" checks, which are the most common source of silent
benchmark regressions.

Run with:
    pytest tests/test_invariants.py

Author: Dimitri Romanov
Project: humanising-ai
"""

from __future__ import annotations

import json
import math
import random

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. Valence / arousal bounds on every emotional snapshot
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "I'm absolutely thrilled about this — it's the best news all year.",
    "I feel completely empty and exhausted.",
    "Honestly I don't really feel anything about it either way.",
    "I'm furious and I don't know what to do with it.",
    "I'm a bit nervous but also kind of excited, if that makes sense.",
])
def test_emotional_snapshot_is_bounded(text):
    from src.affective import EmotionalContextTracker

    tracker = EmotionalContextTracker(decay=0.5)
    snap = tracker.update(text)
    assert -1.0 - 1e-9 <= snap.valence <= 1.0 + 1e-9
    assert -1.0 - 1e-9 <= snap.arousal <= 1.0 + 1e-9
    assert isinstance(snap.dominant, str) and snap.dominant


def test_tracker_state_is_a_probability_like_dict():
    from src.affective import EmotionalContextTracker

    tracker = EmotionalContextTracker(decay=0.7)
    tracker.update("I'm genuinely grateful for everything you've done.")
    state = tracker.state
    assert isinstance(state, dict)
    for label, score in state.items():
        assert isinstance(label, str) and label
        assert 0.0 - 1e-9 <= score <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 2. SHAP efficiency axiom
# ---------------------------------------------------------------------------

def test_kernel_shap_satisfies_efficiency_axiom():
    """Σ φ_i must recover f(x) − E[f(X_background)] to sampling tolerance."""
    from src.explainability import kernel_shap_values

    rng = np.random.default_rng(0)
    d = 5
    true_weights = rng.normal(size=d)

    def predict_fn(X):
        X = np.atleast_2d(X)
        return X @ true_weights

    background = rng.normal(size=(200, d))
    x = rng.normal(size=d)

    values, base, pred = kernel_shap_values(
        predict_fn=predict_fn,
        x=x,
        background=background,
        n_samples=512,
        seed=0,
    )

    gap = pred - base
    # Sampling noise on 512 coalitions is small for a linear rule;
    # a tolerance of 1e-2 is generous but still informative.
    assert abs(values.sum() - gap) < 1e-2, (
        f"Efficiency violated: Σφ={values.sum():+.4f}, "
        f"f(x)−E[f]={gap:+.4f}"
    )


# ---------------------------------------------------------------------------
# 3. ToM benchmark structural invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [1, 2])
def test_sally_anne_scenarios_are_well_formed(order):
    from src.theory_of_mind import generate_sally_anne_scenarios

    scenarios = generate_sally_anne_scenarios(n=20, order=order, seed=0)
    assert len(scenarios) == 20
    for s in scenarios:
        # A scenario that has the correct answer *equal* to the
        # distractor is trivially passable — the benchmark relies
        # on this never being the case.
        assert s.correct_location != s.actual_location
        assert s.order == order
        assert s.prompt and isinstance(s.prompt, str)


def test_recency_baseline_remains_broken():
    """The recency baseline is a *designed* negative control.

    If it ever starts scoring near perfectly on first-order
    Sally-Anne, that is a benchmark bug, not a capability win.
    """
    from src.theory_of_mind import (
        generate_sally_anne_scenarios,
        ToMBenchmark,
    )
    from src.theory_of_mind.tom_benchmark import recency_baseline

    scenarios = generate_sally_anne_scenarios(n=40, order=1, seed=0)
    result = ToMBenchmark(scenarios).evaluate(recency_baseline)
    acc_by_order = result.accuracy_by_order()
    # First-order accuracy for a pure recency picker should be
    # well below human-level. If this ever exceeds 0.7, the
    # benchmark is probably leaking the answer.
    assert acc_by_order[1] < 0.7


# ---------------------------------------------------------------------------
# 4. Belief-state probe: bag-of-locations should NOT solve the task
# ---------------------------------------------------------------------------

def test_bag_of_locations_probe_stays_near_chance():
    """Another negative control: if a pure location-presence feature
    set starts *solving* the probe, the label is leaking through."""
    from src.theory_of_mind import BeliefStateProbe, generate_sally_anne_scenarios
    from src.theory_of_mind.belief_state_probing import (
        bag_of_location_features,
    )

    scenarios = (
        generate_sally_anne_scenarios(n=60, order=1, seed=0)
        + generate_sally_anne_scenarios(n=60, order=2, seed=1)
    )
    extractor = bag_of_location_features([
        "the basket", "the box", "the drawer",
        "the cupboard", "the shelf", "the backpack",
    ])
    probe = BeliefStateProbe(extractor, seed=42).fit_eval(scenarios)

    # Chance is 1/|locations| = ~0.17; generous tolerance of 0.45.
    assert probe.test_accuracy < 0.45, (
        f"Location-presence features appear to be leaking the "
        f"answer: test_acc={probe.test_accuracy:.2%}"
    )


# ---------------------------------------------------------------------------
# 5. Dialogue reply shape: acknowledge + invite
# ---------------------------------------------------------------------------

ACK_MARKERS = (
    "sounds", "makes sense", "i'm sorry", "i hear", "it sounds like",
    "no wonder", "that would", "i appreciate", "i can see",
)

@pytest.mark.parametrize("prompt", [
    "I've been really overwhelmed this week.",
    "I just feel like no matter how hard I try, I'm behind.",
    "I lost my dad last month and I can't seem to get back to normal.",
])
def test_template_reply_has_both_acknowledgement_and_invitation(prompt):
    from src.dialogue import EmpatheticResponder, TemplateGenerator

    bot = EmpatheticResponder(
        generator=TemplateGenerator(rng=random.Random(0))
    )
    reply = bot.respond(prompt)
    low = reply.lower()

    assert any(m in low for m in ACK_MARKERS), (
        f"No acknowledgement marker found in reply: {reply!r}"
    )
    assert reply.strip().endswith("?"), (
        f"Reply does not end in an invitation: {reply!r}"
    )


# ---------------------------------------------------------------------------
# 6. JSON round-trip: conversation state stays replayable
# ---------------------------------------------------------------------------

def test_conversation_context_round_trips_through_json():
    from src.dialogue import ConversationContext

    ctx = ConversationContext(window=16, summary_every=2)
    for role, text in [
        ("user", "I've been feeling quite anxious about the move."),
        ("assistant", "That sounds stressful. What's weighing on you most?"),
        ("user", "Mostly leaving my friends, I think."),
    ]:
        ctx.add(role, text)

    blob = ctx.to_json()
    # Must be valid JSON
    data = json.loads(blob)

    # Key pieces expected by downstream audit tooling
    assert "turns" in data and len(data["turns"]) == 3
    assert "summary" in data
    assert "valence" in data and -1.0 <= data["valence"] <= 1.0
    assert "arousal" in data and -1.0 <= data["arousal"] <= 1.0
    # Every turn carries a role and text
    for turn in data["turns"]:
        assert turn["role"] in {"user", "assistant", "system"}
        assert isinstance(turn["text"], str) and turn["text"]


# ---------------------------------------------------------------------------
# 7. Contrastive explanation: gap sign matches contribution sum sign
# ---------------------------------------------------------------------------

def test_contrastive_text_contributions_are_consistent():
    """If the fact wins (gap > 0), the top contributions supporting
    the fact should collectively outweigh those supporting the foil."""
    from src.explainability import ContrastiveExplainer

    positive = {"great", "kind", "helpful", "thank", "love"}
    negative = {"terrible", "awful", "rude", "hate"}

    def scorer(text: str) -> np.ndarray:
        toks = [t.lower() for t in text.replace(".", " ").split()]
        pos = sum(t in positive for t in toks)
        neg = sum(t in negative for t in toks)
        z = np.array([pos - 0.5 * neg, neg - 0.5 * pos], dtype=float)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    text = "The agent was kind and helpful, though the wait was terrible."
    expl = ContrastiveExplainer(
        class_names=["positive", "negative"],
    ).explain_tokens(
        text=text,
        fact="positive",
        foil="negative",
        scorer=scorer,
    )

    assert expl.fact == "positive"
    assert expl.foil == "negative"

    pos_contrib = sum(v for _, v in expl.contributions if v > 0)
    neg_contrib = sum(-v for _, v in expl.contributions if v < 0)

    if expl.score_gap > 0:
        assert pos_contrib >= neg_contrib, (
            "fact wins overall but pro-fact contributions do not "
            "outweigh pro-foil contributions — check sign convention."
        )
