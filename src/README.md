# `src/` — Reference implementation

**The four sub-packages that implement the [Humanising Loop](../frameworks/humanising_loop.md).**

This folder is deliberately small and dependency-light. Each
sub-package corresponds to exactly one stage of the Humanising
Loop and produces at least one signal consumable by the
[Attunement Audit](../frameworks/attunement_audit.md).
The [notebooks](../notebooks) walk through each sub-package in
the same order.

---

## Layout

| Sub-package | Loop stage | Audit dimension | Notebook |
|-------------|------------|-----------------|----------|
| [`affective/`](./affective) | Perceive | Perceptual Fidelity | [`01_emotion_detection.py`](../notebooks/01_emotion_detection.py) |
| [`theory_of_mind/`](./theory_of_mind) | Attribute | Attributive Honesty | [`02_theory_of_mind_evals.py`](../notebooks/02_theory_of_mind_evals.py) |
| [`dialogue/`](./dialogue) | Attune / Respond | Registerial Attunement / Generative Restraint | [`03_dialogue_grounding.py`](../notebooks/03_dialogue_grounding.py) |
| [`explainability/`](./explainability) | Account | Interrogable Accounting | [`04_explainability.py`](../notebooks/04_explainability.py) |

Everything is pure-Python with NumPy as the only hard dependency.
Model backends (HuggingFace, OpenAI, Anthropic, local llama.cpp)
are *optional plug-ins* wired in through thin callable interfaces
— the core logic never imports them.

---

## `affective/` — reading the person

The `EmotionalContextTracker` runs a GoEmotions-style classifier
turn-by-turn and smooths its outputs with a decay factor, giving
each utterance **emotional momentum** rather than amnesia. An
`EmotionalSnapshot` carries the per-turn dominant emotion,
valence, arousal and volatility for downstream use.

```python
from src.affective.emotion_tracker import EmotionalContextTracker

tracker = EmotionalContextTracker(decay=0.7)
snap = tracker.update("I've been really overwhelmed this week.")
print(snap.dominant, snap.valence, snap.arousal)
```

Drop-in replacements: swap the classifier via the `model`
argument; swap the smoothing rule by subclassing the tracker.
The rest of the stack does not change.

---

## `theory_of_mind/` — modelling what the user believes

Three pieces sit here:

- `tom_benchmark.py` — programmatic **Sally-Anne false-belief**
  generator at configurable order, plus a `ToMBenchmark` that
  will score any `str -> str` model callable. A deliberately
  failing `recency_baseline` is shipped as a negative control.
- `belief_state_probing.py` — a `BeliefStateProbe` for asking
  whether a feature set **linearly encodes** the correct belief
  state. Bag-of-locations features are included as a leakage
  sanity check.

```python
from src.theory_of_mind import (
    generate_sally_anne_scenarios,
    ToMBenchmark,
)
from src.theory_of_mind.tom_benchmark import recency_baseline

scenarios = generate_sally_anne_scenarios(n=30, order=1, seed=0)
bench = ToMBenchmark(scenarios)
print(bench.evaluate(recency_baseline).summary())
```

Designed so the benchmark's validity can be checked *before* any
model weights are paid for.

---

## `dialogue/` — attuning and responding

Two modules, one protocol:

- `context_manager.py` — `ConversationContext` + `ConversationTurn`.
  Long-horizon memory with an emotional arc and a
  salience-weighted summary, all JSON-serialisable.
- `empathetic_responder.py` — `EmpatheticResponder` orchestrates
  emotion tracking, context memory and response generation. Two
  built-in generators: `TemplateGenerator` (dependency-free,
  deterministic) and `LLMGenerator` (backend-agnostic wrapper
  around any chat callable).

```python
from src.dialogue import EmpatheticResponder, TemplateGenerator

bot = EmpatheticResponder(generator=TemplateGenerator())
print(bot.respond("I've been really overwhelmed this week."))
```

The `ResponseGenerator` protocol is the one extension point:
anything matching `(user_text, context) -> reply` slots in. That
keeps the attunement logic and the generation logic
orthogonally swappable.

---

## `explainability/` — showing the working

Two complementary explainers, both black-box:

- `shap_explainer.py` — a pure-NumPy **Kernel SHAP** for any
  `predict_fn : R^d -> R`. Returns a `ShapExplanation` with per-
  feature attributions that satisfy the efficiency axiom
  (Σφᵢ = f(x) − E[f(X)]).
- `contrastive_explanations.py` — `ContrastiveExplainer` for
  "why *X* and not *Y*?" questions. Works on numeric features
  *and* raw text via leave-one-out token perturbation, and
  reports a `flipped_at` threshold showing how many feature
  changes would flip the decision.

```python
from src.explainability import ShapExplainer, ContrastiveExplainer
```

Contrastive explanations are favoured because they match the
form of the "why" questions humans actually ask (Miller, 2019).
Monolithic attributions are useful for engineers; contrastive
attributions are useful for everyone else.

---

## Design invariants

These hold across the whole package. If a change violates them,
it belongs somewhere else — not in `src/`.

1. **NumPy-only core.** Every sub-package must run without any
   ML framework installed. Model backends are plug-ins, not
   dependencies.
2. **Callable interfaces, not concrete classes.** Benchmarks,
   explainers and responders accept plain callables so they can
   wrap sklearn, PyTorch, HuggingFace, OpenAI, Anthropic or
   local-LLM endpoints identically.
3. **JSON-serialisable state.** Conversation state, explanations
   and probe results should round-trip through `json.dumps` so
   they are replayable and auditable.
4. **Failure modes first.** Every evaluator ships with a
   deliberately-failing baseline (recency baseline for ToM,
   bag-of-locations probe for belief leakage, etc.) so the
   benchmark's own validity can be checked.

---

## Testing

Unit tests live in [`../tests`](../tests) and follow the same
sub-package layout. The notebooks double as integration tests:
if they run top-to-bottom with the current `src/`, the public
API is intact.

---

## Extending the package

To add a new humanising capability:

1. Decide which loop stage it belongs to. If it doesn't fit
   cleanly into *perceive / attribute / attune / respond /
   account*, that is useful information — the [Humanising Loop](../frameworks/humanising_loop.md)
   may need revising before the code does.
2. Add a sub-package under `src/` exporting its public surface
   via `__init__.py`, matching the existing pattern.
3. Provide a deliberately-failing baseline in the same module
   (see `recency_baseline` and `bag_of_location_features` for
   examples).
4. Add a notebook under `notebooks/` walking through it, ending
   with signals the [Attunement Audit](../frameworks/attunement_audit.md)
   can score against.

The goal is not to grow the package. The goal is to keep each
piece small, honest and interrogable.
