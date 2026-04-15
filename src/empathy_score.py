"""
Empathy Scoring Module
======================
A computational approach to measuring empathic resonance in AI-generated text.

This module implements a multi-dimensional empathy scoring framework based on
established psychological models of empathy (Davis, 1983; Decety & Jackson, 2004),
adapted for evaluating AI conversational output.

Dimensions scored:
    1. Affective resonance — Does the response mirror emotional tone?
    2. Perspective-taking — Does the response demonstrate understanding of the other's viewpoint?
    3. Validation — Does the response acknowledge the other's experience as legitimate?
    4. Warmth signals — Presence of linguistic markers associated with interpersonal warmth

Author: Dimitri Romanov
Project: humanising-ai

References
----------
Davis, M.H. (1983). Measuring individual differences in empathy: Evidence
    for a multidimensional approach. Journal of Personality and Social
    Psychology, 44(1), 113-126.
Decety, J. & Jackson, P.L. (2004). The functional architecture of human
    empathy. Behavioral and Cognitive Neuroscience Reviews, 3(2), 71-100.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class EmpathyScore:
    """Multi-dimensional empathy assessment result."""
    affective_resonance: float  # 0-1
    perspective_taking: float   # 0-1
    validation: float           # 0-1
    warmth: float              # 0-1
    
    @property
    def composite(self) -> float:
        """Weighted composite score."""
        weights = {
            'affective_resonance': 0.30,
            'perspective_taking': 0.30,
            'validation': 0.25,
            'warmth': 0.15
        }
        return sum(
            getattr(self, dim) * w
            for dim, w in weights.items()
        )
    
    @property
    def level(self) -> str:
        """Human-readable empathy level."""
        c = self.composite
        if c >= 0.8: return "Deeply empathic"
        if c >= 0.6: return "Empathic"
        if c >= 0.4: return "Moderately empathic"
        if c >= 0.2: return "Low empathy"
        return "Non-empathic"
    
    def __repr__(self) -> str:
        return (
            f"EmpathyScore(\n"
            f"  affective_resonance={self.affective_resonance:.2f},\n"
            f"  perspective_taking={self.perspective_taking:.2f},\n"
            f"  validation={self.validation:.2f},\n"
            f"  warmth={self.warmth:.2f},\n"
            f"  composite={self.composite:.2f} [{self.level}]\n"
            f")"
        )


# Linguistic markers for each dimension
WARMTH_MARKERS = [
    r"\bi understand\b", r"\bthat makes sense\b", r"\bi hear you\b",
    r"\bthat sounds\b", r"\bit's okay\b", r"\bno wonder\b",
    r"\bof course\b", r"\bi can see\b", r"\bthat must\b",
    r"\bi appreciate\b", r"\bthank you for sharing\b",
    r"\bthat takes courage\b", r"\byou're not alone\b",
]

VALIDATION_MARKERS = [
    r"\byour feelings are valid\b", r"\bit makes sense that\b",
    r"\banyone would feel\b", r"\bthat's understandable\b",
    r"\byou have every right\b", r"\bthat's a natural\b",
    r"\bit's completely normal\b", r"\bgiven what you've\b",
]

PERSPECTIVE_MARKERS = [
    r"\bfrom your perspective\b", r"\bin your position\b",
    r"\bif i were in your\b", r"\bseeing it through\b",
    r"\bfor you\b", r"\bin your shoes\b",
    r"\byour experience\b", r"\bwhat you're going through\b",
]

EMOTION_WORDS = {
    'positive': [
        'happy', 'glad', 'excited', 'hopeful', 'proud', 'grateful',
        'relieved', 'peaceful', 'joyful', 'content', 'optimistic'
    ],
    'negative': [
        'sad', 'angry', 'frustrated', 'anxious', 'worried', 'scared',
        'overwhelmed', 'lonely', 'hurt', 'disappointed', 'stressed',
        'exhausted', 'confused', 'helpless'
    ]
}


def _count_markers(text: str, patterns: list) -> int:
    """Count occurrences of linguistic marker patterns."""
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower))


def _detect_emotion_tone(text: str) -> Tuple[float, float]:
    """Detect positive and negative emotional tone."""
    text_lower = text.lower()
    pos = sum(1 for w in EMOTION_WORDS['positive'] if w in text_lower)
    neg = sum(1 for w in EMOTION_WORDS['negative'] if w in text_lower)
    total = max(pos + neg, 1)
    return pos / total, neg / total


def score_empathy(
    user_message: str,
    ai_response: str,
    context: Optional[List[Dict[str, str]]] = None
) -> EmpathyScore:
    """
    Score an AI response for empathic quality.
    
    Parameters
    ----------
    user_message : str
        The human's message (provides emotional context)
    ai_response : str
        The AI's response to evaluate
    context : list of dict, optional
        Previous conversation turns for context-aware scoring
    
    Returns
    -------
    EmpathyScore
        Multi-dimensional empathy assessment
    """
    # Affective resonance: does the AI mirror the emotional valence?
    user_pos, user_neg = _detect_emotion_tone(user_message)
    ai_pos, ai_neg = _detect_emotion_tone(ai_response)
    
    # Resonance = how well emotional tones align
    if user_neg > user_pos:
        # User is expressing negative emotions — AI should acknowledge them
        resonance = min(1.0, ai_neg * 2 + _count_markers(ai_response, WARMTH_MARKERS) * 0.15)
    else:
        resonance = min(1.0, ai_pos * 1.5 + 0.3)
    
    # Perspective-taking
    perspective_count = _count_markers(ai_response, PERSPECTIVE_MARKERS)
    perspective = min(1.0, perspective_count * 0.25 + 0.1)
    
    # Validation
    validation_count = _count_markers(ai_response, VALIDATION_MARKERS)
    validation = min(1.0, validation_count * 0.3 + 0.1)
    
    # Warmth
    warmth_count = _count_markers(ai_response, WARMTH_MARKERS)
    warmth = min(1.0, warmth_count * 0.2 + 0.1)
    
    # Bonus: response length indicates engagement (up to a point)
    length_bonus = min(0.15, len(ai_response.split()) / 500)
    
    return EmpathyScore(
        affective_resonance=min(1.0, resonance + length_bonus),
        perspective_taking=perspective,
        validation=validation,
        warmth=warmth
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Humanising AI: Empathy Score Demo")
    print("=" * 60)
    
    # Example 1: High empathy response
    user = "I've been really struggling with anxiety lately and I don't know what to do."
    response_good = (
        "That sounds really difficult, and I appreciate you sharing that with me. "
        "Anxiety can feel overwhelming, especially when you're not sure what's causing it. "
        "What you're going through is completely understandable — anyone in your position "
        "would find it hard. From your perspective, what feels like the biggest source of worry?"
    )
    
    print("\n--- High Empathy Example ---")
    print(f"User: {user}")
    print(f"AI: {response_good}")
    print(score_empathy(user, response_good))
    
    # Example 2: Low empathy response
    response_low = "You should try meditation. Here are five steps to reduce anxiety."
    
    print("\n--- Low Empathy Example ---")
    print(f"User: {user}")
    print(f"AI: {response_low}")
    print(score_empathy(user, response_low))
