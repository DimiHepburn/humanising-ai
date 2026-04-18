# Notebooks

**Five walkthroughs — four that build one stage of the [Humanising Loop](../frameworks/humanising_loop.md) each, and one capstone that runs the evaluative frameworks on a conversation trace.**

These notebooks are the didactic companion to the `src/`
codebase. The first four read top-to-bottom as a single arc —
perception → mentalising → interaction → accountability — with
each one targeting exactly one module from `src/`. The fifth
steps back and *uses* the frameworks rather than building
toward them: it scores a logged conversation against the
[Attunement Audit](../frameworks/attunement_audit.md) and the
[Handoff Threshold](../frameworks/handoff_threshold.md).

Files are stored as **Jupytext-style `.py`** so they review cleanly
in pull requests. Open them as notebooks with `jupytext` or run
them directly as scripts; both work.

---

## The arc

| # | Notebook | `src/` module | Loop stage | Audit dimension |
|---|----------|---------------|------------|-----------------|
| 01 | [`01_emotion_detection.py`](./01_emotion_detection.py) | [`affective`](../src/affective) | Perceive | Perceptual Fidelity |
| 02 | [`02_theory_of_mind_evals.py`](./02_theory_of_mind_evals.py) | [`theory_of_mind`](../src/theory_of_mind) | Attribute | Attributive Honesty |
| 03 | [`03_dialogue_grounding.py`](./03_dialogue_grounding.py) | [`dialogue`](../src/dialogue) | Attune / Respond | Registerial Attunement / Generative Restraint |
| 04 | [`04_explainability.py`](./04_explainability.py) | [`explainability`](../src/explainability) | Account | Interrogable Accounting |
| 05 | [`05_audit_and_handoff.py`](./05_audit_and_handoff.py) | *(consumes all four)* | *(evaluates the loop)* | *(all five + Handoff Threshold)* |

The mapping for notebooks 01–04 is not decorative. Each is an
attempt to make the corresponding stage of the
[Humanising Loop](../frameworks/humanising_loop.md) empirically
tractable, and each produces at least one signal that can feed
directly into the Attunement Audit. Notebook 05 then closes the
loop by *being* the audit pipeline.

---

## 01 — Emotion detection

*Reading the person as they actually presented.*

Builds an `EmotionalContextTracker` on top of a GoEmotions-style
classifier, with exponential smoothing so the tracker has
**momentum**: a single sarcastic turn doesn't erase the running
emotional state. Shows how to plot a per-turn valence/arousal
trace, surface the dominant emotion, and snapshot the full state
for downstream use.

Outputs consumed by the Audit: `EmotionalSnapshot` fields
(dominant, valence, arousal, volatility).

---

## 02 — Theory of Mind evaluations

*Distinguishing what the user believes from what the system knows.*

Generates **Sally-Anne false-belief scenarios** programmatically
at first and second order, evaluates any `str -> str` model
callable against them, and runs a leakage sanity check via a
bag-of-locations probe. Includes a deliberately failing recency
baseline so we can see the distractor-rate fingerprint of a
non-mentalising system.

Outputs consumed by the Audit: `ToMBenchmark` accuracy-by-order,
belief-probe test accuracy.

---

## 03 — Dialogue grounding

*Matching register to the moment before matching words to content.*

Wires the affective tracker, the conversation context and the
response generator into a single `EmpatheticResponder`. Plots the
emotional arc across turns, scores each reply against two
heuristic empathy markers (acknowledgement + invitation), and
runs an "advice-trap" stress test. Shows how to swap the template
backend for an OpenAI, Anthropic or local-LLM backend without
changing any of the evaluation code.

Outputs consumed by the Audit: acknowledgement score, invitation
score, register trace, conversation summary.

---

## 04 — Explainability

*Showing the working, contrastively.*

A dependency-free **Kernel SHAP** walkthrough on a transparent
toy model (so ground truth is visible), followed by contrastive
"why X and not Y?" explanations in both feature and token space.
Ends with a `flipped_at` diagnostic — *how far was this decision
from flipping?* — which is often more humane than the verdict
itself.

Outputs consumed by the Audit: per-feature attributions,
contrastive verdicts, flip thresholds.

---

## 05 — Running the Attunement Audit and Handoff Threshold

*From prose framework to pipeline.*

The capstone. Takes a logged conversation — the same JSON trace
`ConversationContext.to_json()` produces in notebook 03 — and
runs the full evaluative stack against it:

- Scores the interaction on all five dimensions of the
  [Attunement Audit](../frameworks/attunement_audit.md),
  with each score derived from concrete signals in the trace
  rather than from human judgement. Plots the five-tuple as a
  radar chart, because the *shape* of the score matters more
  than its mean.
- Evaluates each turn against the five criteria of the
  [Handoff Threshold](../frameworks/handoff_threshold.md),
  using traffic-light levels and the framework's disjunctive
  decision rule (any Red, or two or more Ambers, triggers
  handoff).
- Emits a humane handoff reply when the threshold trips —
  *acknowledge*, *limit*, *route* — so deployments have a
  concrete fallback to use.
- Produces a single JSON report combining both framework
  outputs, shaped so it can be logged, reviewed, or replayed.

Outputs consumed by anyone: an auditor-ready JSON artefact with
per-turn handoff verdicts and a per-dimension audit summary.

The risk and consent detectors in this notebook are **toys by
design** — a production deployment must swap in an externally
audited risk classifier. The rest of the pipeline does not
change when you do.

---

## How to run

```bash
pip install -r requirements.txt
python notebooks/01_emotion_detection.py
```

Or open any notebook as a Jupyter notebook via Jupytext:

```bash
pip install jupytext
jupytext --to notebook notebooks/01_emotion_detection.py
jupyter lab notebooks/01_emotion_detection.ipynb
```

All notebooks are CPU-only and make **no network calls by
default**. The LLM hooks (OpenAI / Anthropic / local) are shown
inline in each notebook but never required to run the core
experiments.

---

## Reading order

If you want the conceptual story, read the four frameworks in
[`../frameworks`](../frameworks) first —
[Friction Protocol](../frameworks/friction_protocol.md),
[Humanising Loop](../frameworks/humanising_loop.md),
[Attunement Audit](../frameworks/attunement_audit.md),
[Handoff Threshold](../frameworks/handoff_threshold.md) — and
then the notebooks in numerical order. Notebook 05 will then read
as the place where the prose gets operationalised.

If you want the empirical story first, run the notebooks
top-to-bottom and read the
[frameworks README](../frameworks/README.md) afterwards to see
what the pipeline was collectively trying to demonstrate.

Either path lands in the same place: a small, auditable pipeline
for AI interactions that take the person on the other end
seriously — and a principled rule for when to hand off to a
human instead.
