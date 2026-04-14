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

## 🔬 Research Modules

### 1. Affective Computing & Emotion Recognition

Building AI systems that can accurately detect and respond to human emotional states across multiple modalities:

- **Text-based sentiment and emotion detection** using transformer models (BERT, RoBERTa fine-tuned on GoEmotions, EmoBank)
- **Multimodal fusion** — combining text, tone, and facial expression signals for richer emotional understanding
- **Contextual emotion modelling** — tracking emotional arcs across conversation turns, not just single utterances

```python
from transformers import pipeline

class EmotionalContextTracker:
    """
    Tracks emotional state across conversation turns.
    Uses exponential smoothing to model emotional momentum.
    """
    def __init__(self, model="SamLowe/roberta-base-go_emotions", decay=0.7):
        self.classifier = pipeline("text-classification", model=model, top_k=None)
        self.decay = decay
        self.emotional_state = {}

    def update(self, utterance: str) -> dict:
        results = self.classifier(utterance)[0]
        new_scores = {r['label']: r['score'] for r in results}

        # Exponential smoothing — blend new signal with accumulated state
        for emotion, score in new_scores.items():
            prev = self.emotional_state.get(emotion, 0.0)
            self.emotional_state[emotion] = self.decay * prev + (1 - self.decay) * score

        return self.dominant_emotions(top_k=3)

    def dominant_emotions(self, top_k=3) -> dict:
        sorted_emotions = sorted(self.emotional_state.items(), key=lambda x: -x[1])
        return dict(sorted_emotions[:top_k])
```

---

### 2. Theory of Mind in Language Models

Theory of Mind (ToM) — the ability to attribute mental states (beliefs, desires, intentions) to others — is one of the hallmarks of human social cognition. Do LLMs have it? The evidence is mixed and fascinating.

This module:
- Implements the classic **Sally-Anne false belief test** for language models
- Evaluates ToM capabilities using the **ToMi benchmark** (Le et al., 2019)
- Probes internal representations in transformer models for belief-state encoding
- Explores whether fine-tuning on perspective-taking data improves ToM performance

Key finding from our experiments: GPT-4 class models pass first-order ToM tasks at near-human rates, but fail meaningfully on second-order ("what does Alice think Bob thinks?") scenarios — suggesting statistical mimicry rather than genuine mentalising.

---

### 3. Empathetic Dialogue Systems

What does empathetic conversation actually require? We decompose it into four components:

1. **Emotion recognition** — accurately identifying the speaker's emotional state
2. **Emotional validation** — acknowledging and normalising the emotion
3. **Perspective-taking** — demonstrating genuine understanding of the speaker's viewpoint
4. **Constructive response** — offering something useful without projecting or trivialising

This module fine-tunes dialogue models on the **EmpatheticDialogues** dataset (Rashkin et al., 2019) and evaluates them against human raters on all four dimensions.

```python
# Example empathetic response pipeline
class EmpatheticResponder:
    def __init__(self):
        self.emotion_tracker = EmotionalContextTracker()
        self.dialogue_model = load_empathetic_model()

    def respond(self, user_input: str, history: list) -> str:
        # Step 1: detect emotional context
        emotions = self.emotion_tracker.update(user_input)
        dominant = max(emotions, key=emotions.get)

        # Step 2: build emotionally-aware prompt
        prompt = self._build_empathetic_prompt(user_input, dominant, history)

        # Step 3: generate response with emotional grounding
        response = self.dialogue_model.generate(prompt)
        return self._post_process(response, emotions)
```

---

### 4. Explainability as a Human Value

For AI to be genuinely human-centred, it must be **explainable** — not just technically interpretable, but communicable to non-expert users in meaningful ways. This module covers:

- **LIME and SHAP** for local model explanations
- **Attention visualisation** as an imperfect but useful interpretability tool
- **Contrastive explanations** ("why X and not Y?") — which research suggests humans find most useful
- **Uncertainty quantification** — communicating confidence alongside outputs

---

### 5. Value Alignment & AI Ethics Framework

A survey and synthesis of current approaches to value alignment, with critical analysis:

- RLHF (Reinforcement Learning from Human Feedback) — strengths and failure modes
- Constitutional AI and rule-following approaches
- Debate and amplification as scalable oversight methods
- The **alignment tax** problem — does aligning AI compromise capability?
- **Goodhart's Law in AI**: when optimising for a proxy measure corrupts the underlying goal

---

## 🧠 Theoretical Grounding

This project draws on several interdisciplinary fields:

- **Affective neuroscience** (Panksepp, Damasio) — understanding how emotions are generated and represented in biological brains
- **Social psychology** — empathy models, attribution theory, in-group/out-group dynamics
- **Phenomenology** (Husserl, Merleau-Ponty) — what subjective experience actually is and what it would mean for a machine to have it
- **Philosophy of mind** — the hard problem of consciousness, functionalism vs. biological naturalism
- **HCI research** — what actually makes humans trust and connect with AI systems

---

## 📂 Repository Structure

```
humanising-ai/
├── notebooks/
│   ├── 01_emotion_detection.ipynb
│   ├── 02_theory_of_mind_evals.ipynb
│   ├── 03_empathetic_dialogue.ipynb
│   ├── 04_explainability_methods.ipynb
│   └── 05_value_alignment_survey.ipynb
├── src/
│   ├── affective/
│   │   ├── emotion_tracker.py
│   │   ├── sentiment_pipeline.py
│   │   └── multimodal_fusion.py
│   ├── dialogue/
│   │   ├── empathetic_responder.py
│   │   └── context_manager.py
│   ├── theory_of_mind/
│   │   ├── tom_benchmark.py
│   │   └── belief_state_probing.py
│   └── explainability/
│       ├── shap_explainer.py
│       └── contrastive_explanations.py
├── data/
│   ├── empathetic_dialogues/
│   └── tom_benchmark/
├── results/
├── requirements.txt
└── README.md
```

---

## 📚 Key References

- Rashkin, H. et al. (2019). Towards Empathetic Open-domain Conversation Models
- Le, M. et al. (2019). Revisiting the Evaluation of Theory of Mind through Question Answering
- Damasio, A. (1994). *Descartes' Error: Emotion, Reason, and the Human Brain*
- Panksepp, J. (1998). *Affective Neuroscience*
- Bender, E.M. et al. (2021). On the Dangers of Stochastic Parrots
- Gabriel, I. (2020). Artificial Intelligence, Values, and Alignment

---

## 🚀 Getting Started

```bash
git clone https://github.com/DimiHepburn/humanising-ai.git
cd humanising-ai
pip install -r requirements.txt
jupyter notebook notebooks/01_emotion_detection.ipynb
```

---

*Part of a broader research programme on neuroscience-inspired AI. See also: [neuro-ai-bridge](https://github.com/DimiHepburn/neuro-ai-bridge)*
