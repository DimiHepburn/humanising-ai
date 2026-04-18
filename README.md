# 🤝 humanising-ai

> *What would it mean for an AI to truly understand you — not just your words, but your context, your history, your emotional state?*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen?style=flat-square)]()

---

## Overview

**Humanising AI** is a research project examining what it would genuinely take to make artificial intelligence systems more human — not in the superficial sense of giving chatbots friendly names and avatars, but in the deep sense of building systems that are:

- Genuinely **emotionally aware** — able to recognise, respond to, and reason about human affective states
- **Contextually grounded** — understanding not just what is said, but what is meant, implied, and felt
- **Ethically aligned** — behaving in ways that respect human values, dignity, and autonomy
- **Transparent and trustworthy** — able to explain their reasoning in ways humans can actually understand

This is not just an engineering challenge. It sits at the intersection of psychology, neuroscience, ethics, linguistics, and machine learning.

The repository pairs a lightweight reference **codebase** (`src/`, `notebooks/`) with four original **frameworks** (`frameworks/`) that describe what the code is *for*: how a human should engage with AI, how an AI should engage with a human, how a third party can audit whether either actually happened, and when the AI should step out of the interaction entirely.

---

## 🧩 The Problem Space

### Why Current AI Falls Short

Most modern AI systems — even the most capable large language models — fail to be genuinely human in several key ways:

| Dimension | What Humans Do | What Current AI Does |
|-----------|---------------|----------------------|
| **Emotion** | Recognise, feel, and appropriately respond to emotions | Pattern-match emotional cues without genuine understanding |
| **Memory** | Maintain rich episodic memories across long timescales | Lose context after a few thousand tokens |
| **Theory of Mind** | Model others' mental states, beliefs, and intentions | Approximate this statistically, often incorrectly |
| **Values** | Have deeply held, contextually applied values | Follow static rules or RLHF preferences |
| **Embodiment** | Understand the world through a physical body | Have no sensorimotor grounding at all |

---

## 📐 The Four Frameworks

Four original frameworks sit in [`frameworks/`](./frameworks) and form a deliberately complete set — three for how to interact well, and one for when not to interact at all:

| Framework | Perspective | Mode | Asks |
|-----------|-------------|------|------|
| [Friction Protocol](./frameworks/friction_protocol.md) | Human → AI | Prescriptive | *How should I engage critically with AI output?* |
| [Humanising Loop](./frameworks/humanising_loop.md) | AI → Human | Prescriptive | *How should the system engage carefully with the person in front of it?* |
| [Attunement Audit](./frameworks/attunement_audit.md) | Third-party → Exchange | Evaluative | *Did this interaction, in fact, humanise anyone?* |
| [Handoff Threshold](./frameworks/handoff_threshold.md) | AI → (Out of interaction) | Safety-critical | *When should the system step out of the conversation entirely?* |

The [frameworks README](./frameworks/README.md) explains why four are needed and how to read them in order. The rest of the repository is an implementation of what those frameworks prescribe.

---

## 🔬 Research Modules

### 1. Affective Computing & Emotion Recognition

Building AI systems that can accurately detect and respond to human emotional states:

- **Lexicon-based classifier** that runs on any laptop with no ML dependencies, plus a clean `EmotionBackend` interface that drops in a HuggingFace transformer (e.g. GoEmotions fine-tuned RoBERTa) with a single argument
- **Contextual emotion modelling** — tracking emotional arcs across conversation turns, not just single utterances, with exponential smoothing for emotional *momentum*
- **Multimodal fusion** — a lightweight `MultimodalEmotionFuser` that combines text + tone + any custom modality and gracefully handles missing channels

Implemented in [`src/affective/`](./src/affective). Walkthrough: [`notebooks/01_emotion_detection.py`](./notebooks/01_emotion_detection.py).

```python
from src.affective import EmotionalContextTracker

tracker = EmotionalContextTracker(decay=0.7)
snap = tracker.update("I've been really overwhelmed with work this week.")
print(snap.dominant, snap.valence, snap.arousal)
```

---

### 2. Theory of Mind in Language Models

Theory of Mind (ToM) — the ability to attribute mental states (beliefs, desires, intentions) to others — is one of the hallmarks of human social cognition. Do LLMs have it? The evidence is mixed and fascinating. This module:

- Generates **Sally-Anne false-belief scenarios** programmatically at configurable order (1st- and 2nd-order)
- Provides a backend-agnostic `ToMBenchmark` that evaluates any `str -> str` model callable
- Ships a **belief-state probe** for testing whether a feature set linearly encodes the correct belief
- Includes a **recency baseline** designed to fail, as a validity check on the benchmark itself

Implemented in [`src/theory_of_mind/`](./src/theory_of_mind). Walkthrough: [`notebooks/02_theory_of_mind_evals.py`](./notebooks/02_theory_of_mind_evals.py).

Key finding from our experiments: GPT-4-class models pass first-order ToM tasks at near-human rates, but fail meaningfully on second-order ("what does Alice think Bob thinks?") scenarios — suggesting statistical mimicry rather than genuine mentalising.

---

### 3. Empathetic Dialogue Systems

What does empathetic conversation actually require? We decompose it into a four-stage skeleton — acknowledge → mirror → invite → (optionally) offer — and implement it as a thin orchestration layer so the generator can be swapped freely between a template backend, an OpenAI/Anthropic API, or a local LLM.

Implemented in [`src/dialogue/`](./src/dialogue). Walkthrough: [`notebooks/03_dialogue_grounding.py`](./notebooks/03_dialogue_grounding.py).

```python
from src.dialogue import EmpatheticResponder, TemplateGenerator

bot = EmpatheticResponder(generator=TemplateGenerator())
print(bot.respond("I've been really overwhelmed with work this week."))
```

---

### 4. Explainability as a Human Value

For AI to be genuinely human-centred, it must be **explainable** — not just technically interpretable, but communicable to non-expert users in meaningful ways. This module provides:

- A dependency-free **Kernel SHAP** implementation for per-feature attribution
- **Contrastive explanations** — "why *X* and not *Y*?" — for both feature-space and token-space inputs
- A `flipped_at` diagnostic that reports **how close a decision was to flipping**, which is often more humane than the verdict itself

Implemented in [`src/explainability/`](./src/explainability). Walkthrough: [`notebooks/04_explainability.py`](./notebooks/04_explainability.py).

---

### 5. Safety: The Handoff Threshold

The four sub-packages above describe how to interact *well* with a user. They do not, on their own, describe when an AI system should stop trying to help and defer to a human. That is the job of the [Handoff Threshold](./frameworks/handoff_threshold.md): five criteria — risk, epistemic limit, attunement failure, emotional load, consent — evaluated per turn, encoded as a disjunction rather than a weighted sum.

This repository deliberately does **not** ship a production-grade risk classifier. Any deployment using the Handoff Threshold is expected to supply one, audited separately. The framework document specifies what signals from the existing sub-packages feed each criterion.

---

## 🧠 Theoretical Grounding

This project draws on several interdisciplinary fields:

- **Affective neuroscience** (Panksepp, Damasio) — understanding how emotions are generated and represented in biological brains
- **Social psychology** — empathy models, attribution theory, in-group/out-group dynamics
- **Phenomenology** (Husserl, Merleau-Ponty) — what subjective experience actually is and what it would mean for a machine to have it
- **Philosophy of mind** — the hard problem of consciousness, functionalism vs. biological naturalism
- **HCI research** — what actually makes humans trust and connect with AI systems
- **Clinical & pedagogical theory** (Rogers, Stern, Schön, Vygotsky, Polanyi, Gawande, Perrow) — underpinning the four frameworks in [`frameworks/`](./frameworks)

---

## 📂 Repository Structure

```
humanising-ai/
├── frameworks/
│   ├── README.md
│   ├── friction_protocol.md         # Human → AI
│   ├── humanising_loop.md           # AI → Human
│   ├── attunement_audit.md          # Third-party → Exchange
│   └── handoff_threshold.md         # AI → (Out of interaction)
├── notebooks/
│   ├── README.md
│   ├── 01_emotion_detection.py
│   ├── 02_theory_of_mind_evals.py
│   ├── 03_dialogue_grounding.py
│   └── 04_explainability.py
├── src/
│   ├── README.md
│   ├── affective/
│   │   ├── __init__.py
│   │   ├── emotion_tracker.py       # EmotionalContextTracker
│   │   ├── sentiment_pipeline.py    # EmotionClassifier, Lexicon/Transformer backends
│   │   └── multimodal_fusion.py     # MultimodalEmotionFuser
│   ├── theory_of_mind/
│   │   ├── __init__.py
│   │   ├── tom_benchmark.py         # Sally-Anne generator, ToMBenchmark
│   │   └── belief_state_probing.py  # BeliefStateProbe
│   ├── dialogue/
│   │   ├── __init__.py
│   │   ├── context_manager.py       # ConversationContext, ConversationTurn
│   │   └── empathetic_responder.py  # EmpatheticResponder, Template/LLM generators
│   └── explainability/
│       ├── __init__.py
│       ├── shap_explainer.py        # Kernel SHAP (pure NumPy)
│       └── contrastive_explanations.py  # "Why X and not Y?"
├── tests/
│   ├── README.md
│   ├── test_smoke.py                # end-to-end exercises per sub-package
│   ├── test_invariants.py           # bounds, efficiency axiom, negative controls
│   └── test_pipeline.py             # Humanising Loop integration
├── LICENSE
├── requirements.txt
└── README.md
```

---

## 🗺️ How the Frameworks Map onto the Code

| Humanising Loop stage | Attunement Audit dimension | Code module | Notebook |
|------------------------|-----------------------------|-------------|----------|
| Perceive | Perceptual Fidelity | [`src/affective/`](./src/affective) | `01_emotion_detection.py` |
| Attribute | Attributive Honesty | [`src/theory_of_mind/`](./src/theory_of_mind) | `02_theory_of_mind_evals.py` |
| Attune / Respond | Registerial Attunement / Generative Restraint | [`src/dialogue/`](./src/dialogue) | `03_dialogue_grounding.py` |
| Account | Interrogable Accounting | [`src/explainability/`](./src/explainability) | `04_explainability.py` |

The **Friction Protocol** sits *outside* this table on purpose — it is the framework the reader is invited to apply to everything else in the repository, including these frameworks themselves. The **Handoff Threshold** also sits outside, for a different reason: it composes signals from every sub-package, plus at least one (risk classification) the repository deliberately does not ship.

---

## 📚 Key References

- Rashkin, H. et al. (2019). Towards Empathetic Open-domain Conversation Models
- Le, M. et al. (2019). Revisiting the Evaluation of Theory of Mind through Question Answering
- Lundberg, S.M. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions
- Miller, T. (2019). Explanation in Artificial Intelligence: Insights from the Social Sciences
- Baron-Cohen, S., Leslie, A.M. & Frith, U. (1985). Does the autistic child have a "theory of mind"?
- Gawande, A. (2009). *The Checklist Manifesto*
- Perrow, C. (1984). *Normal Accidents: Living with High-Risk Technologies*
- Damasio, A. (1994). *Descartes' Error: Emotion, Reason, and the Human Brain*
- Panksepp, J. (1998). *Affective Neuroscience*
- Rogers, C.R. (1957). The necessary and sufficient conditions of therapeutic personality change

---

## 🚀 Getting Started

```bash
git clone https://github.com/DimiHepburn/humanising-ai.git
cd humanising-ai
pip install -r requirements.txt

# Open any notebook as a Jupytext-paired .py file, or run directly:
python notebooks/01_emotion_detection.py

# Run the test suite
pytest tests/
```

All notebooks run with CPU-only defaults and no network calls. The LLM hooks (OpenAI / Anthropic / local) are illustrated inline in each notebook but never required.

---

*Part of a broader research programme on neuroscience-inspired AI. See also: [neuro-ai-bridge](https://github.com/DimiHepburn/neuro-ai-bridge)*
