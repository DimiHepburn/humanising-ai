# Tests

**Smoke tests for the reference implementation in [`../src`](../src).**

These tests are deliberately *smoke* tests, not exhaustive unit
tests. Their job is to catch the two failure modes that matter
most when the repository is used as a teaching and framework
reference:

1. **Import / API drift.** Something in `src/` was renamed, moved
   or removed in a way that would break the notebooks or the
   public surface documented in [`../src/README.md`](../src/README.md).
2. **Shape / invariant drift.** The outputs still exist but no
   longer satisfy the invariants downstream code (and the
   [Attunement Audit](../frameworks/attunement_audit.md)) relies
   on — e.g. valence outside `[-1, 1]`, probability distributions
   that don't sum to 1, SHAP attributions that violate the
   efficiency axiom.

Anything deeper — statistical power, model-level benchmarking,
calibration curves — lives in the notebooks, which double as
integration tests.

---

## Layout

| File | Covers |
|------|--------|
| [`test_smoke.py`](./test_smoke.py) | End-to-end exercises for every sub-package in `src/` |

There is intentionally **one** file at this stage. New test
modules should be added only when a sub-package grows beyond
what a single smoke file can clearly cover.

---

## What's tested

The smoke file has one or more tests per sub-package, each
checking at least one behavioural invariant rather than an
implementation detail.

### Affective
- `EmotionClassifier` returns a proper probability distribution
  and picks a plausible dominant label.
- `EmotionalContextTracker` emits snapshots with valence and
  arousal in `[-1, 1]` and shows a non-degenerate arc when the
  conversation actually shifts.
- `MultimodalEmotionFuser` handles a missing modality without
  crashing and still produces a normalised fused distribution.

### Dialogue
- `ConversationContext` tracks an emotional arc across turns and
  emits a non-empty summary once `summary_every` is reached.
- `EmpatheticResponder` + `TemplateGenerator` produces a reply
  that contains *both* acknowledgement structure and an invitation
  (i.e. not just a bare question).

### Theory of mind
- `generate_sally_anne_scenarios` produces the requested count
  and — critically — always places the correct answer away from
  the distractor (otherwise the benchmark is trivially passed).
- `ToMBenchmark(recency_baseline)` actually picks the distractor
  a non-trivial proportion of the time, which is the baseline's
  *designed* failure mode. If that ever hit zero we would have a
  silent benchmark bug.
- `BeliefStateProbe` with `bag_of_location_features` fits and
  evaluates cleanly end-to-end, with accuracies in `[0, 1]`.

### Explainability
- `ShapExplainer` recovers the known linear rule on a transparent
  toy model, with per-feature attributions obeying the efficiency
  axiom: Σφᵢ ≈ f(x) − E[f(X)].
- Contrastive explanations on a two-class scorer produce
  non-empty contributions and a well-defined `score_gap`.

---

## How to run

With pytest (preferred):

```bash
pip install -r requirements.txt
pytest tests/
```

Without pytest (plain Python fallback, useful in minimal envs):

```bash
python -m tests.test_smoke
```

Every test is CPU-only and makes no network calls. If a test
starts requiring either, it belongs in a notebook under
[`../notebooks`](../notebooks), not here.

---

## Design principles for tests in this repo

These mirror the design invariants documented in
[`../src/README.md`](../src/README.md):

1. **Invariants over examples.** Prefer asserting that a
   probability sums to 1, or that valence is in `[-1, 1]`, over
   asserting a specific label for a specific sentence.
2. **Baselines stay failing.** The deliberately-failing
   baselines shipped alongside each benchmark (recency baseline
   for ToM, bag-of-locations probe for belief leakage) must keep
   failing — they are our signal that the benchmark is still
   measuring what it claims to measure.
3. **No optional deps in the test path.** No HuggingFace, no
   OpenAI, no Anthropic, no torch. The tests exercise the default
   dependency-light code paths so they can be run in a CI
   container without credentials.
4. **JSON-serialisable outputs.** Whenever an object is meant to
   round-trip (conversation state, SHAP explanation, probe
   result), at least one test should confirm it does.

---

## Extending

To add coverage for a new sub-package or a new bit of `src/`:

1. If a single smoke test still fits, add a function to
   [`test_smoke.py`](./test_smoke.py) with a clear
   `test_<sub_package>_<behaviour>` name.
2. If the new area is big enough to deserve its own file, mirror
   the `src/` layout: `tests/test_<sub_package>.py`.
3. Include at least one assertion about an **invariant**, not
   just a type or shape.
4. If the code you're testing has a deliberately-failing
   baseline, add a test that confirms it still fails — silent
   improvement in a negative control is itself a bug.

The goal is not to maximise coverage. The goal is to make it
impossible for the public API documented in the `README.md`s to
drift away from the code without CI noticing.
