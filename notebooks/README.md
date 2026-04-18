# Notebooks

**Four walkthroughs — one per stage of the [Humanising Loop](../frameworks/humanising_loop.md).**

These notebooks are the didactic companion to the `src/`
codebase. They read top-to-bottom as a single arc —
perception → mentalising → interaction → accountability — and
each one targets exactly one module from `src/`.

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

The mapping is not decorative. Each notebook is an attempt to
make the corresponding stage of the [Humanising Loop](../frameworks/humanising_loop.md)
empirically tractable, and each produces at least one signal
that can feed directly into the [Attunement Audit](../frameworks/attunement_audit.md).

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

If you want the conceptual story, read the
[Humanising Loop](../frameworks/humanising_loop.md) first, then
the notebooks in numerical order. If you want the empirical
story first, run the notebooks top-to-bottom and then read the
[frameworks README](../frameworks/README.md) to see what they
were collectively trying to demonstrate.

Either path lands in the same place: a small, auditable pipeline
for AI interactions that take the person on the other end
seriously.
