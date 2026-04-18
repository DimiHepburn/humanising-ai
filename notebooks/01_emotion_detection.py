# %% [markdown]
# # 01 — Emotion Detection
#
# The starting point for any "humanising" layer on top of AI is
# being able to recognise emotion in text. This notebook walks
# through three things:
#
# 1. **Single-utterance classification** — the `EmotionClassifier`
#    and its two interchangeable backends (lexicon / transformer).
# 2. **Conversation-level tracking** — how the
#    `EmotionalContextTracker` turns one-shot labels into a
#    running affective state across many turns.
# 3. **Multimodal fusion** — combining text with other modalities
#    (prosody, vision) via `MultimodalEmotionFuser`.
#
# Every cell runs with *zero* heavy dependencies — the lexicon
# backend makes the whole pipeline usable on any laptop.

# %%
from __future__ import annotations
import sys
import pathlib

ROOT = pathlib.Path().resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.affective import (
    EmotionClassifier,
    EmotionalContextTracker,
    MultimodalEmotionFuser,
)
from src.affective.multimodal_fusion import default_text_modality

# %% [markdown]
# ## 1. Single-utterance classification
#
# The default `EmotionClassifier()` uses the shipped
# lexicon backend — fast, transparent, and dependency-free.
# The exact same interface works with a HuggingFace transformer
# if you pass `backend=TransformerBackend(...)`.

# %%
clf = EmotionClassifier()

examples = [
    "I'm so grateful — you really helped me through this.",
    "I can't believe he said that to me, I'm furious.",
    "I don't know what to do, I feel lost and anxious.",
    "That was hilarious, I haven't laughed this hard in ages.",
    "Everything is fine I guess.",
]

print(f"{'text':<60} → top 3")
print("-" * 80)
for text in examples:
    top = clf.top(text, k=3)
    labels = ", ".join(f"{lab}={p:.2f}" for lab, p in top)
    print(f"{text[:58]:<60} → {labels}")

# %% [markdown]
# The lexicon backend is a toy for teaching and lightweight use —
# swap to a fine-tuned RoBERTa with one line when you want real
# performance:
#
# ```python
# from src.affective import TransformerBackend, EmotionClassifier
# clf = EmotionClassifier(backend=TransformerBackend())   # downloads the model
# ```

# %% [markdown]
# ## 2. Tracking an emotional arc across a conversation
#
# One-shot emotion labels are noisy. The tracker smooths them
# exponentially and exposes two summary signals drawn from
# Russell's circumplex model: **valence** (pleasant ↔ unpleasant)
# and **arousal** (calm ↔ activated).

# %%
tracker = EmotionalContextTracker(decay=0.55)

conversation = [
    "I've been really overwhelmed with work this week.",
    "I can't sleep, I just lie there thinking about deadlines.",
    "My manager finally said I can take a day off.",
    "Actually, just saying that out loud helped a bit.",
    "Maybe I'll go for a walk tomorrow.",
    "I might even be looking forward to the weekend.",
]

snapshots = [tracker.update(u) for u in conversation]

# %%
# Plot valence and arousal over time
v = [s.valence for s in snapshots]
a = [s.arousal for s in snapshots]

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(v, marker="o", label="valence")
ax.plot(a, marker="s", label="arousal")
ax.axhline(0, color="gray", lw=0.5)
ax.set_xticks(range(len(conversation)))
ax.set_xticklabels([f"T{i+1}" for i in range(len(conversation))])
ax.set_ylim(-1, 1)
ax.set_ylabel("signal (−1 … +1)")
ax.set_title("Emotional arc across the conversation")
ax.legend()
plt.tight_layout()
plt.show()

# %%
# Peek at the running top emotions
print("Turn-by-turn dominant emotions:")
for i, s in enumerate(snapshots):
    print(f"  T{i+1}: {s.dominant:<12}  v={s.valence:+.2f}  "
          f"a={s.arousal:+.2f}  Δ={s.volatility:.3f}")

# %% [markdown]
# Notice how the tracker smooths over noise — a single positive
# utterance doesn't immediately flip the running state, which
# models the **momentum** of emotional experience. The `decay`
# parameter controls how much history influences the next step.

# %% [markdown]
# ## 3. Multimodal fusion
#
# Text alone is a thin signal. A real affective system would
# fuse text with prosody, facial expression, and sometimes
# physiological signals.
#
# The fuser is **modality-agnostic** — you hand it any callable
# that returns a distribution over emotions, and it handles the
# combination strategy. Below we simulate a prosody model with a
# simple rule-based stub so everything runs without extra
# dependencies.

# %%
labels = ["joy", "sadness", "anger", "fear", "neutral"]

def fake_prosody(cue: str):
    table = {
        "shaky_voice":  {"sadness": 0.6, "fear": 0.3, "neutral": 0.1},
        "laughing":     {"joy": 0.8, "neutral": 0.2},
        "flat":         {"sadness": 0.4, "neutral": 0.6},
        "raised_pitch": {"anger": 0.5, "fear": 0.3, "neutral": 0.2},
    }
    return table.get(cue, {l: 1.0 / len(labels) for l in labels})


fuser = MultimodalEmotionFuser(
    modalities={
        "text":    default_text_modality(),
        "prosody": fake_prosody,
    },
    weights={"text": 0.5, "prosody": 0.5},
    strategy="weighted_mean",
)

scenarios = [
    # Text contradicts prosody — a classic "I'm fine" misclassification
    {"text": "I'm fine, really, nothing's wrong.", "prosody": "shaky_voice"},
    # Text and prosody agree
    {"text": "That was hilarious!", "prosody": "laughing"},
    # Only one modality available
    {"text": "I don't know what to do.", "prosody": None},
]

for s in scenarios:
    res = fuser.fuse(s)
    print(f"\n> text={s['text']!r}  prosody={s['prosody']!r}")
    for name, dist in res.per_modality.items():
        top = max(dist.items(), key=lambda kv: kv[1])
        print(f"    {name:<8} → {top[0]} ({top[1]:.2f})")
    print(f"    fused    → {res.dominant}")

# %% [markdown]
# In the first scenario notice how the fuser correctly leans
# **sad** despite the textually "fine" utterance — because the
# prosody contradicts the words. This is exactly the kind of
# case purely text-based emotion systems fail on, and it's the
# main argument for multimodal affect.

# %% [markdown]
# ## 4. Take-aways
#
# * Lexicon-based emotion detection is a useful starting point
#   but brittle on sarcasm, negation, and mixed emotion.
# * Tracking across turns matters: valence and arousal are more
#   informative than raw labels.
# * Real affective understanding likely requires fusion — words
#   alone can't disambiguate "I'm fine".
#
# ### Suggested next steps
#
# * Swap the lexicon backend for `TransformerBackend` and compare
#   per-utterance confidence.
# * Build your own `prosody_model` wrapper (e.g. over
#   [wav2vec2-emotion](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim))
#   and plug it into the fuser.
# * Try a longer conversation (30+ turns) and plot the valence
#   trajectory — is there an emotional "attractor" the tracker
#   converges to?
