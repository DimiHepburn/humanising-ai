"""
Explainability sub-package
===========================

Tools for turning opaque model predictions into something a human
can actually reason about.

Contents
--------
- ``shap_explainer.py``            : model-agnostic feature-attribution
                                     based on a SHAP-like sampling
                                     approximation; runs with no heavy
                                     dependencies
- ``contrastive_explanations.py``  : "why X and not Y?" explanations,
                                     which research suggests humans find
                                     most useful (Miller, 2019)

Every explainer in this package only needs a black-box prediction
function ``f : input -> score`` — so it works equally well with
classical ML, neural networks, or LLM-based scorers.
"""

from .shap_explainer import (
    ShapExplainer,
    ShapExplanation,
    kernel_shap_values,
)
from .contrastive_explanations import (
    ContrastiveExplainer,
    ContrastiveExplanation,
)

__all__ = [
    "ShapExplainer",
    "ShapExplanation",
    "kernel_shap_values",
    "ContrastiveExplainer",
    "ContrastiveExplanation",
]
