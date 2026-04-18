"""
Multimodal Emotion Fusion
==========================

Real emotional understanding rarely relies on text alone.  Humans
integrate **language**, **prosody** (tone of voice) and **facial
expression** into a single percept, and deep-learning systems that
approach human-level affective competence do the same
(Poria et al., 2017; Baltrušaitis et al., 2019).

This module provides a **modality-agnostic** fusion layer.
Each modality is treated as a black box that emits a probability
distribution over a shared emotion vocabulary; the fuser combines
them using well-understood rules:

* **weighted mean** — simple, interpretable, robust to missing modalities
* **product of experts** — sharpens consensus, penalises disagreement
* **confidence-gated** — modalities with high self-confidence dominate

Like the rest of the affective package, this module runs with
**no heavy dependencies**.  If you have a vision or speech model,
just wrap it in any callable that maps input → dict[label, float]
and hand it to the fuser.

Author: Dimitri Romanov
Project: humanising-ai
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .sentiment_pipeline import EmotionClassifier


ModalityInput = Tuple[str, Any]           # (modality_name, raw_input)
Distribution = Dict[str, float]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _normalise(dist: Distribution) -> Distribution:
    total = sum(dist.values())
    if total <= 0:
        return {"neutral": 1.0}
    return {k: v / total for k, v in dist.items()}


def _align_labels(dists: Iterable[Distribution]) -> List[Distribution]:
    """Pad each distribution to the union of labels with 0.0 entries."""
    labels = set()
    for d in dists:
        labels.update(d.keys())
    return [{l: d.get(l, 0.0) for l in labels} for d in dists]


def _entropy(dist: Distribution) -> float:
    """Shannon entropy in nats — used as a (negative) confidence proxy."""
    return -sum(p * math.log(p + 1e-12) for p in dist.values() if p > 0)


# ---------------------------------------------------------------------------
# Default modality stubs
# ---------------------------------------------------------------------------
def default_text_modality() -> Callable[[str], Distribution]:
    """Wrap the lexicon EmotionClassifier as a modality callable."""
    clf = EmotionClassifier()
    return lambda text: clf(text)


def uniform_modality(labels: List[str]) -> Callable[[Any], Distribution]:
    """Fallback modality that always returns a uniform distribution.
    Useful as a placeholder when a real prosody/vision model isn't
    available — the fuser will still run end-to-end."""
    p = 1.0 / len(labels)
    return lambda _x: {l: p for l in labels}


# ---------------------------------------------------------------------------
# The fuser itself
# ---------------------------------------------------------------------------
@dataclass
class FusionResult:
    fused: Distribution
    per_modality: Dict[str, Distribution]
    weights_used: Dict[str, float]
    dominant: str

    def __repr__(self) -> str:
        top = sorted(self.fused.items(), key=lambda kv: -kv[1])[:3]
        body = ", ".join(f"{k}={v:.2f}" for k, v in top)
        return (f"FusionResult(dominant={self.dominant!r}, "
                f"top3={{{body}}})")


class MultimodalEmotionFuser:
    """
    Combine per-modality emotion distributions into a single fused one.

    Parameters
    ----------
    modalities : dict[str, callable]
        Mapping from modality name to a callable that turns a raw
        input into a dict[str, float] distribution.  The names
        passed to `fuse()` must match keys here.
    weights : dict[str, float], optional
        Prior weights per modality.  Defaults to equal weights.
    strategy : {"weighted_mean", "product_of_experts", "confidence_gated"}
        How distributions are combined.
    """

    def __init__(
        self,
        modalities: Dict[str, Callable[[Any], Distribution]],
        weights: Optional[Dict[str, float]] = None,
        strategy: str = "weighted_mean",
    ):
        if not modalities:
            raise ValueError("Provide at least one modality.")
        self.modalities = dict(modalities)
        self.weights = self._init_weights(weights)
        self.strategy = strategy

    def _init_weights(self, w: Optional[Dict[str, float]]) -> Dict[str, float]:
        if w is None:
            n = len(self.modalities)
            return {name: 1.0 / n for name in self.modalities}
        # Normalise user-supplied weights
        total = sum(w.get(name, 0.0) for name in self.modalities) or 1.0
        return {name: w.get(name, 0.0) / total for name in self.modalities}

    # ------------------------------------------------------------------
    def fuse(self, inputs: Dict[str, Any]) -> FusionResult:
        """
        Parameters
        ----------
        inputs : dict[str, Any]
            Keyed by modality name, same keys as the `modalities`
            dict passed at construction time.  Missing keys are
            allowed: only present modalities contribute to the fuse.
        """
        per_modality: Dict[str, Distribution] = {}
        present_weights: Dict[str, float] = {}
        for name, fn in self.modalities.items():
            if name in inputs and inputs[name] is not None:
                per_modality[name] = _normalise(fn(inputs[name]))
                present_weights[name] = self.weights[name]

        if not per_modality:
            raise ValueError("No recognised modalities supplied.")

        # Renormalise over *present* modalities
        s = sum(present_weights.values()) or 1.0
        present_weights = {k: v / s for k, v in present_weights.items()}

        fused = self._combine(per_modality, present_weights)
        dominant = max(fused.items(), key=lambda kv: kv[1])[0]
        return FusionResult(
            fused=fused,
            per_modality=per_modality,
            weights_used=present_weights,
            dominant=dominant,
        )

    # ------------------------------------------------------------- strategies
    def _combine(
        self,
        dists: Dict[str, Distribution],
        weights: Dict[str, float],
    ) -> Distribution:
        if self.strategy == "weighted_mean":
            return self._weighted_mean(dists, weights)
        if self.strategy == "product_of_experts":
            return self._product_of_experts(dists, weights)
        if self.strategy == "confidence_gated":
            return self._confidence_gated(dists)
        raise ValueError(f"Unknown fusion strategy: {self.strategy}")

    @staticmethod
    def _weighted_mean(
        dists: Dict[str, Distribution],
        weights: Dict[str, float],
    ) -> Distribution:
        aligned = _align_labels(dists.values())
        fused: Distribution = {}
        for name, dist in zip(dists.keys(), aligned):
            w = weights[name]
            for label, p in dist.items():
                fused[label] = fused.get(label, 0.0) + w * p
        return _normalise(fused)

    @staticmethod
    def _product_of_experts(
        dists: Dict[str, Distribution],
        weights: Dict[str, float],
        eps: float = 1e-6,
    ) -> Distribution:
        aligned = _align_labels(dists.values())
        log_fused: Dict[str, float] = {}
        for name, dist in zip(dists.keys(), aligned):
            w = weights[name]
            for label, p in dist.items():
                log_fused[label] = (
                    log_fused.get(label, 0.0) + w * math.log(p + eps)
                )
        # Back to probability space
        m = max(log_fused.values())
        fused = {l: math.exp(v - m) for l, v in log_fused.items()}
        return _normalise(fused)

    @staticmethod
    def _confidence_gated(dists: Dict[str, Distribution]) -> Distribution:
        """Weight each modality by (1 - normalised entropy)."""
        # Confidence = 1 - H / H_max
        max_H = math.log(max(len(d) for d in dists.values()) or 1)
        conf = {
            name: max(1.0 - (_entropy(d) / max_H), 1e-6)
            for name, d in dists.items()
        }
        total = sum(conf.values()) or 1.0
        conf = {k: v / total for k, v in conf.items()}
        return MultimodalEmotionFuser._weighted_mean(dists, conf)


if __name__ == "__main__":
    # No prosody / vision model? Use stubs so the demo runs anywhere.
    text_mod = default_text_modality()
    labels = ["joy", "sadness", "anger", "fear", "neutral"]
    # Fake "audio" model that leans sad if the text mentions tears
    def fake_audio(audio_cue: str) -> Distribution:
        if audio_cue == "shaky_voice":
            return {"sadness": 0.6, "fear": 0.3, "neutral": 0.1}
        if audio_cue == "laughing":
            return {"joy": 0.8, "neutral": 0.2}
        return {l: 0.2 for l in labels}

    fuser = MultimodalEmotionFuser(
        modalities={"text": text_mod, "audio": fake_audio},
        weights={"text": 0.5, "audio": 0.5},
        strategy="weighted_mean",
    )

    print("=" * 60)
    print("Humanising AI: Multimodal fusion demo")
    print("=" * 60)

    examples = [
        {"text": "I'm fine, really, nothing's wrong.", "audio": "shaky_voice"},
        {"text": "That was hilarious!", "audio": "laughing"},
        {"text": "I don't know, I just feel tired.", "audio": None},
    ]

    for ex in examples:
        res = fuser.fuse(ex)
        print(f"\n> text: {ex['text']}   audio: {ex['audio']}")
        for name, d in res.per_modality.items():
            top = max(d.items(), key=lambda kv: kv[1])
            print(f"    {name:<6} → {top[0]} ({top[1]:.2f})")
        print(f"    fused  → {res}")
