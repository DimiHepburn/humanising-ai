"""
Affective computing sub-package
================================

Emotion recognition and emotional-context tracking for text
(and, optionally, multimodal) input.

Design goals
------------
* **Runs on any laptop by default.**  A lexicon-based classifier
  is shipped so every module is usable without installing any
  heavy ML libraries.
* **Transformer-ready.**  A clean `EmotionBackend` interface lets
  you swap in a HuggingFace model (e.g. GoEmotions fine-tuned
  RoBERTa) with a single argument when you want real performance.
"""

from .sentiment_pipeline import (
    EmotionBackend,
    LexiconBackend,
    TransformerBackend,
    EmotionClassifier,
)
from .emotion_tracker import EmotionalContextTracker
from .multimodal_fusion import MultimodalEmotionFuser

__all__ = [
    "EmotionBackend",
    "LexiconBackend",
    "TransformerBackend",
    "EmotionClassifier",
    "EmotionalContextTracker",
    "MultimodalEmotionFuser",
]
