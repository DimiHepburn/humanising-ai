# %% [markdown]
# # 04 — Explainability: Shapley attributions and contrastive "why X not Y?"
#
# Notebooks 01–03 built a small stack for *reading* the user
# (emotion), *modelling* the user (theory of mind), and
# *responding* to the user (empathetic dialogue). None of that
# is humanising if the user can't interrogate **why** the system
# did what it did.
#
# This notebook tackles explainability on two fronts:
#
# 1. **Feature attribution with Kernel SHAP.** For any black-box
#    scorer `f : R^d -> R`, we decompose a single prediction into
#    per-feature contributions whose sum recovers the score gap
#    from the baseline (efficiency axiom).
# 2. **Contrastive explanations ("why X and not Y?").** Research
#    (Miller, 2019; Lipton, 1990) consistently shows that humans
#    ask for, and accept, *contrastive* explanations — not
#    monolithic ones. We cover both feature-space and text-token
#    modes.
#
# Everything runs with numpy only; the same call sites work
# against sklearn, torch or LLM scorers by just swapping the
# `predict_fn`.

# %%
from __future__ import annotations

import sys
import pathlib

ROOT = pathlib.Path().resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.explainability import (
    ShapExplainer,
    ShapExplanation,
    kernel_shap_values,
    ContrastiveExplainer,
    ContrastiveExplanation,
)

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. A toy model we can reason about by hand
#
# To keep the ground truth visible, we use a transparent linear
# scorer plus a non-linear interaction term. We know exactly
# which features *should* get credit, so we can check that SHAP
# recovers the right story.

# %%
FEATURES = ["age", "income", "debt_ratio", "late_payments", "savings"]
TRUE_WEIGHTS = np.array([0.05, 0.004, -1.2, -0.4, 0.002])

def predict_fn(X: np.ndarray) -> np.ndarray:
    """
    Mock credit-risk scorer: higher = safer.
    Linear in the weights, plus a mild non-linear bonus for
    (savings AND low debt_ratio) — realistic "compensation"
    behaviour SHAP should attribute fairly.
    """
    X = np.atleast_2d(X)
    linear = X @ TRUE_WEIGHTS
    interaction = 0.3 * (X[:, 4] > 5000).astype(float) * (X[:, 2] < 0.3).astype(float)
    return linear + interaction

# A plausible background distribution (the "population")
background = np.column_stack([
    rng.normal(42, 12, size=200).clip(18, 80),        # age
    rng.normal(55_000, 20_000, size=200).clip(10_000, 200_000),  # income
    rng.beta(2, 5, size=200),                         # debt_ratio ∈ [0, 1]
    rng.poisson(1.2, size=200),                       # late_payments
    rng.gamma(2.0, 2500, size=200),                   # savings
])

# The applicant we want to explain
x = np.array([34, 72_000, 0.18, 0, 9_500], dtype=float)
print(f"Applicant score : {predict_fn(x)[0]:+.4f}")
print(f"Population mean : {predict_fn(background).mean():+.4f}")

# %% [markdown]
# ## 2. Kernel SHAP on a single decision
#
# `ShapExplainer` caches the background distribution; the
# returned `ShapExplanation` has a nice `ranked()` / `top()` API
# and a printable repr with proportional bars.

# %%
explainer = ShapExplainer(
    predict_fn=predict_fn,
    background=background,
    feature_names=FEATURES,
    n_samples=256,
    seed=0,
)
explanation: ShapExplanation = explainer.explain(x)

print(explanation)

# Efficiency check: Σ φ_i ≈ f(x) − E[f(background)]
gap = float(predict_fn(x)[0] - predict_fn(background).mean())
print(f"\nΣ φ_i          = {explanation.values.sum():+.4f}")
print(f"f(x) − E[f(X)] = {gap:+.4f}")

# %% [markdown]
# ## 3. Visualising the attribution
#
# A horizontal bar chart, sorted by |φ|, is the single most
# useful SHAP plot for most audiences — it reads like a
# pros/cons list with magnitudes attached.

# %%
ranked = explanation.ranked()
names = [n for n, _ in ranked][::-1]
vals  = [v for _, v in ranked][::-1]
colors = ["#4caf50" if v > 0 else "#e53935" for v in vals]

fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.barh(names, vals, color=colors)
ax.axvline(0, color="grey", lw=0.8)
ax.set_xlabel("φ  (signed contribution to score above/below baseline)")
ax.set_title(f"SHAP attribution for one applicant  "
             f"(f(x) = {explanation.prediction:+.3f}, "
             f"E[f] = {explanation.base_value:+.3f})")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Functional API: the same thing in one line
#
# For scripts or quick checks, the functional form is often
# cleaner than instantiating a class.

# %%
values, base, pred = kernel_shap_values(
    predict_fn=predict_fn,
    x=x,
    background=background,
    n_samples=256,
    seed=0,
)
print(f"Base value : {base:+.4f}")
print(f"Prediction : {pred:+.4f}")
for name, v in sorted(zip(FEATURES, values), key=lambda kv: -abs(kv[1])):
    print(f"  {name:<14} {v:+.4f}")

# %% [markdown]
# ## 5. Contrastive explanations — "why approved, not denied?"
#
# Monolithic attributions answer "why this score?" — but users
# usually ask a **different** question: "why *this* outcome
# rather than *that* one?". `ContrastiveExplainer` answers that
# directly, in feature space or token space.
#
# For a two-class version of the same decision, we turn the
# scorer into a calibrated pair `(approve, deny)` via a logistic:

# %%
def class_scorer(X: np.ndarray) -> np.ndarray:
    """Returns (B, 2) scores for classes ['approve', 'deny']."""
    raw = predict_fn(X)
    p_approve = 1.0 / (1.0 + np.exp(-raw))
    return np.column_stack([p_approve, 1.0 - p_approve])

contra = ContrastiveExplainer(class_names=["approve", "deny"])

baseline_row = background.mean(axis=0)  # the "average applicant"

feat_expl: ContrastiveExplanation = contra.explain_features(
    x=x,
    fact="approve",
    foil="deny",
    scorer=class_scorer,
    baseline=baseline_row,
    feature_names=FEATURES,
)

print(feat_expl)
print("\nVerdict:")
print(" ", feat_expl.verdict(k=3))

# %% [markdown]
# The `flipped_at` field tells the applicant *how close to the
# other side of the fence they are* — a remarkably humane piece
# of information. "Your application would have flipped if any
# two features had moved toward the population average" is far
# more actionable than any monolithic score.

# %%
if feat_expl.flipped_at is not None:
    print(f"Greedy flip threshold: {feat_expl.flipped_at} feature(s) "
          f"out of {len(FEATURES)}.")
else:
    print("No realistic feature swap flips the decision — the "
          "applicant is comfortably on the approve side.")

# %% [markdown]
# ## 6. Contrastive explanations on text
#
# The same idea works token-by-token for any black-box text
# classifier. Here we fake a sentiment scorer so the notebook
# stays dependency-free; swap in a HuggingFace pipeline and the
# call site is identical.

# %%
_POS = {"wonderful", "love", "great", "kind", "thank", "helpful", "calm"}
_NEG = {"terrible", "hate", "awful", "cruel", "rude", "useless", "angry"}

def fake_sentiment_scorer(text: str) -> np.ndarray:
    """Returns (2,) scores for classes ['positive', 'negative']."""
    toks = [t.lower() for t in text.replace(".", " ").split()]
    pos = sum(t in _POS for t in toks)
    neg = sum(t in _NEG for t in toks)
    z = np.array([pos - 0.5 * neg, neg - 0.5 * pos], dtype=float)
    # softmax
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

text = "The support agent was kind and helpful, though the wait was terrible."

text_contra = ContrastiveExplainer(class_names=["positive", "negative"])
tok_expl = text_contra.explain_tokens(
    text=text,
    fact="positive",
    foil="negative",
    scorer=fake_sentiment_scorer,
)

print("Text:", text)
print(tok_expl)
print("\nVerdict:")
print(" ", tok_expl.verdict(k=3))

# %% [markdown]
# Positive contributions are tokens whose removal *narrowed*
# the positive-over-negative margin — i.e. words that were
# doing the work for the fact. Negative contributions are
# tokens pulling toward the foil. This decomposition is
# exactly what a user needs in order to **disagree with the
# model on evidence, not on vibes**.

# %% [markdown]
# ## 7. Hooking up a real model
#
# All the explainers above need only a `predict_fn` or a text
# `scorer`. Real-world wrappers are one-liners:
#
# ### sklearn classifier
# ```python
# from sklearn.ensemble import GradientBoostingClassifier
# clf = GradientBoostingClassifier().fit(X_train, y_train)
#
# def scorer(X): return clf.predict_proba(X)  # returns (B, C)
# ```
#
# ### HuggingFace text classifier
# ```python
# from transformers import pipeline
# pipe = pipeline("text-classification",
#                 model="distilbert-base-uncased-finetuned-sst-2-english",
#                 return_all_scores=True)
#
# def scorer(text: str) -> np.ndarray:
#     out = pipe(text, truncation=True)[0]
#     label2score = {d["label"].lower(): d["score"] for d in out}
#     return np.array([label2score["positive"], label2score["negative"]])
# ```
#
# ### LLM as scorer
# ```python
# def scorer(text: str) -> np.ndarray:
#     logprobs = llm_score(text, candidates=["approve", "deny"])
#     p = np.exp(logprobs - logprobs.max()); p /= p.sum()
#     return p
# ```
#
# The SHAP and contrastive code paths are completely unchanged.

# %% [markdown]
# ## 8. Take-aways
#
# * Monolithic explanations ("why this score?") are useful for
#   engineers; contrastive explanations ("why this and not that?")
#   are what users, regulators and auditors actually ask for.
# * Kernel SHAP's efficiency axiom — Σ φ_i = f(x) − E[f] — gives
#   you a free sanity check every time: if the attributions don't
#   sum to the gap, something is off.
# * The same API covers tabular features and free-text tokens,
#   so explainability doesn't have to be rebuilt per-modality.
# * `flipped_at` is a humane, actionable metric: *how far are
#   you from the other side of the decision?* It converts an
#   opaque verdict into a conversation about evidence.
#
# ### Suggested next steps
#
# * Add a **stability** diagnostic: re-run SHAP with different
#   `n_samples` and report attribution variance — a lot of real
#   disputes between explanations are just Monte-Carlo noise.
# * Compare `kernel_shap_values` output against the reference
#   `shap` library on the same background + instance, as a
#   correctness regression test.
# * For dialogue (notebook 03), pipe `explain_tokens` over the
#   user's final utterance against a fact/foil pair of
#   *responses* the system considered — this gives you **model
#   self-explanation** of its own reply choices.
