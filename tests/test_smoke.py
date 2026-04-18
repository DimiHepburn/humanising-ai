"""
Smoke tests for humanising-ai
==============================

Quick end-to-end exercises for every sub-package.  They verify that
imports work, the default (no-heavy-deps) code paths run, and
outputs have the shapes / types downstream code expects.

Run with:

    pytest tests/              # if pytest is installed
    python -m tests.test_smoke # plain fallback

Author: Dimitri Romanov
Project: humanising-ai
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Affective
# ---------------------------------------------------------------------------
def test_emotion_classifier_lexicon():
    from src.affective import EmotionClassifier

    clf = EmotionClassifier()
    dist = clf("I am so excited and grateful right now!")
    assert isinstance(dist, dict)
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    # Some recognisable positive emotion should be on top
    dominant = max(dist.items(), key=lambda kv: kv[1])[0]
    assert dominant in {"excitement", "gratitude", "joy", "neutral"}


def test_emotional_context_tracker_updates():
    from src.affective import EmotionalContextTracker

    tracker = EmotionalContextTracker(decay=0.5)
    snaps = []
    for text in [
        "I feel sad and alone.",
        "Talking to you helps, I feel a bit lighter.",
        "Actually, I feel hopeful about tomorrow.",
    ]:
        snaps.append(tracker.update(text))

    # Every snapshot exposes the expected fields
    for s in snaps:
        assert s.dominant
        assert -1.0 <= s.valence <= 1.0
        assert -1.0 <= s.arousal <= 1.0

    # Valence should trend upward across the conversation
    assert snaps[-1].valence >= snaps[0].valence - 0.2


def test_multimodal_fuser_handles_missing_modality():
    from src.affective import (
        EmotionClassifier, MultimodalEmotionFuser,
    )
    from src.affective.multimodal_fusion import default_text_modality

    labels = ["joy", "sadness", "anger", "fear", "neutral"]

    def fake_audio(cue):
        return {"sadness": 0.7, "neutral": 0.3}

    fuser = MultimodalEmotionFuser(
        modalities={"text": default_text_modality(), "audio": fake_audio},
        weights={"text": 0.5, "audio": 0.5},
    )
    # Only text supplied — should still fuse fine.
    res = fuser.fuse({"text": "I'm really struggling today."})
    assert isinstance(res.fused, dict)
    assert abs(sum(res.fused.values()) - 1.0) < 1e-6
    assert "text" in res.per_modality
    assert "audio" not in res.per_modality


# ---------------------------------------------------------------------------
# Dialogue
# ---------------------------------------------------------------------------
def test_context_manager_tracks_emotional_arc():
    from src.dialogue import ConversationContext

    ctx = ConversationContext(window=8, summary_every=2)
    ctx.add("user", "I feel anxious and overwhelmed.")
    ctx.add("assistant", "That sounds really hard.")
    ctx.add("user", "Thanks, actually I feel a bit better already.")

    arc = ctx.emotional_arc()
    assert len(arc) == 2                # only user turns produce snapshots
    assert all(-1.0 <= v <= 1.0 for _, _, v, _ in arc)

    # Summary is non-empty after `summary_every` turns
    assert ctx.summary(rebuild=True)


def test_empathetic_responder_produces_reasonable_reply():
    import random
    from src.dialogue import EmpatheticResponder, TemplateGenerator

    bot = EmpatheticResponder(
        generator=TemplateGenerator(rng=random.Random(0))
    )
    reply = bot.respond("I'm really scared about the results tomorrow.")

    assert isinstance(reply, str) and len(reply) > 0
    # Should contain *some* empathetic acknowledgement, not just a
    # bare follow-up question.
    assert "?" in reply
    assert reply.count(".") + reply.count("!") + reply.count("?") >= 2


# ---------------------------------------------------------------------------
# Theory of Mind
# ---------------------------------------------------------------------------
def test_tom_benchmark_generator_and_evaluation():
    from src.theory_of_mind import (
        generate_sally_anne_scenarios, ToMBenchmark,
    )
    from src.theory_of_mind.tom_benchmark import recency_baseline

    scenarios = (
        generate_sally_anne_scenarios(n=8, order=1, seed=0)
        + generate_sally_anne_scenarios(n=8, order=2, seed=1)
    )
    assert len(scenarios) == 16
    assert all(s.correct_location != s.actual_location for s in scenarios)

    bench = ToMBenchmark(scenarios)
    result = bench.evaluate(recency_baseline)
    assert result.total == 16
    # The recency baseline is *designed* to pick the actual location,
    # so distractor rate should be substantial — certainly > 0.
    assert result.distractor > 0


def test_belief_state_probe_runs_end_to_end():
    from src.theory_of_mind import BeliefStateProbe
    from src.theory_of_mind import generate_sally_anne_scenarios
    from src.theory_of_mind.belief_state_probing import (
        bag_of_location_features,
    )

    scenarios = (
        generate_sally_anne_scenarios(n=40, order=1, seed=0)
        + generate_sally_anne_scenarios(n=40, order=2, seed=1)
    )
    extractor = bag_of_location_features(
        ["the basket", "the box", "the drawer", "the cupboard",
         "the shelf", "the backpack"]
    )
    probe = BeliefStateProbe(extractor, seed=42).fit_eval(scenarios)

    assert 0.0 <= probe.train_accuracy <= 1.0
    assert 0.0 <= probe.test_accuracy <= 1.0


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
def test_shap_explainer_recovers_linear_rule():
    from src.explainability import ShapExplainer

    rng = np.random.default_rng(0)
