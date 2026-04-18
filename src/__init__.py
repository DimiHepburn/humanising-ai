"""
humanising-ai
=============

Research scaffolding and frameworks for building emotionally
intelligent, context-aware, and genuinely human-centred AI systems.

Top-level package structure:

    src/
    ├── affective/         — emotion classifiers, context tracker,
    │                        multimodal fusion
    ├── dialogue/          — long-horizon context, empathetic responder
    ├── theory_of_mind/    — ToM benchmark generators + probing
    ├── explainability/    — SHAP-style + contrastive explanations
    └── empathy_score.py   — legacy multi-dimensional empathy scorer

Each sub-package ships with a **lightweight, dependency-free
default backend** so the full repo is runnable on any machine,
and well-defined plug-in points for HuggingFace / LLM / local
model integrations when you want them.
"""

from . import affective
from . import dialogue
from . import theory_of_mind
from . import explainability
from .empathy_score import EmpathyScore, score_empathy

__all__ = [
    "affective",
    "dialogue",
    "theory_of_mind",
    "explainability",
    "EmpathyScore",
    "score_empathy",
]
