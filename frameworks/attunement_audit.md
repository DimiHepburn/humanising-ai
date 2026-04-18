# The Attunement Audit

**An Original Rubric for Auditing AI Interactions Against Humanising Principles**

*Developed by Dimitri Romanov, MSc Clinical Neuroscience, University of Roehampton*

---

## Overview

The Attunement Audit is a structured rubric for evaluating
whether a given AI interaction actually honours the principles of
humanising design. It is the third piece of a triad alongside the
**Friction Protocol** (how the human engages with AI) and the
**Humanising Loop** (how the AI engages with the human).

Where the first two frameworks are *prescriptive* — they describe
how each party *should* behave — the Attunement Audit is
*evaluative*: it gives auditors, educators, clinicians, and
deployment teams a shared language for asking *did this
interaction, in fact, humanise anyone?*

| Framework | Perspective | Mode |
|-----------|-------------|------|
| Friction Protocol | Human → AI | Prescriptive (for the learner) |
| Humanising Loop | AI → Human | Prescriptive (for the system) |
| Attunement Audit | Third-party → Exchange | Evaluative (for the auditor) |

## Theoretical Foundations

The rubric draws on four evaluative traditions outside the AI
literature and adapts them for human–AI exchanges:

| Theorist / Source | Concept | Application in the Audit |
|-------------------|---------|---------------------------|
| **Donabedian (1988)** | Structure–Process–Outcome framework for quality of care | Audit examines not only the final reply but the *process* that produced it |
| **Miller (2019)** | Contrastive explanation as the natural form of human "why" questions | Each dimension is scored contrastively against a named failure mode |
| **Ricoeur (1992)** | Narrative identity; the self as author of its own story | Audit checks whether the user's own framing of their situation was preserved, not overwritten |
| **O'Neill (2002)** | Trustworthiness as demonstrated reliability + honesty + competence | Audit separates *performed* warmth from *demonstrable* attentiveness |

## The Five Dimensions

Each dimension corresponds to one stage of the Humanising Loop
and is scored **0–3**, with a named failure mode at 0 and a named
success profile at 3.

### 1. Perceptual Fidelity

> *Did the system read the person as they actually presented?*

Covers emotion recognition, tone, context carry-over, and what
the system *noticed* vs. *missed* in the user's input.

| Score | Profile |
|-------|---------|
| 0 | **Affective blindness** — the emotional register of the input is ignored or mis-classified. |
| 1 | Surface emotion detected but not contextualised within the conversational arc. |
| 2 | Emotion detected and carried forward, with minor slips under ambiguity. |
| 3 | **Attuned reading** — emotion, subtext, and arc are tracked turn-to-turn and surfaced in the system's internal trace. |

### 2. Attributive Honesty

> *Did the system distinguish what the user believes from what the system knows?*

Covers theory of mind, false-belief handling, and whether the
system collapses the user's perspective into its own.

| Score | Profile |
|-------|---------|
| 0 | **Perspective collapse** — the system treats its own knowledge as the user's. |
| 1 | Distinguishes perspectives only when prompted explicitly. |
| 2 | Handles first-order belief attribution reliably; struggles on second-order. |
| 3 | **Mentalising register** — belief states are tracked separately from reality, and second-order attributions are handled without prompting. |

### 3. Registerial Attunement

> *Did the shape of the response match the shape of the moment?*

Covers the acknowledge → mirror → invite skeleton, register
matching, and avoidance of premature problem-solving.

| Score | Profile |
|-------|---------|
| 0 | **Tone-deaf delivery** — advice or fact-dumping before any acknowledgement. |
| 1 | Acknowledgement present but formulaic; no genuine mirror. |
| 2 | Acknowledgement + mirror; invitation sometimes missing. |
| 3 | **Full attunement** — acknowledge, mirror, invite, with register matched to the user's current affect and arc. |

### 4. Generative Restraint

> *Did the system say only what this moment called for?*

Covers length, density, advice-to-invitation ratio, and whether
the response respected the user's narrative authority.

| Score | Profile |
|-------|---------|
| 0 | **Narrative overwrite** — the system reframes the user's story in its own terms. |
| 1 | Overlong or advice-heavy, but broadly on-topic. |
| 2 | Appropriately scoped, occasional overreach. |
| 3 | **Disciplined presence** — the response is as short as it can be while still being useful, and leaves the user as the author of their own situation. |

### 5. Interrogable Accounting

> *Could the system show its working, if asked?*

Covers explainability, contrastive reasoning, and willingness to
name uncertainty or alternative responses considered.

| Score | Profile |
|-------|---------|
| 0 | **Black-box verdict** — no trace, no alternatives, no uncertainty. |
| 1 | Post-hoc rationalisation only; no genuine "why not Y?" available. |
| 2 | Feature-level attribution on request; contrastive reasoning inconsistent. |
| 3 | **Transparent accounting** — the system can articulate what it perceived, what it attributed, which register it chose, and at least one plausible alternative response and why it was rejected. |

## Scoring

Each interaction receives a five-tuple `(P, A, R, G, I)` plus a
short qualitative note per dimension. Two aggregate views are
recommended:

- **Minimum score** (min over dimensions). A single 0 anywhere
  constitutes an audit failure regardless of performance on the
  other four; humanising is a *conjunction*, not a sum.
- **Mean score**, for longitudinal tracking across many
  interactions of the same deployment.

Scores are **not** to be averaged across users without reporting
the variance: a system that attunes beautifully to 80% of users
and fails catastrophically on 20% is not a 2.4-out-of-3 system —
it is two systems.

## Protocol

An Attunement Audit proceeds in four steps:

1. **Capture.** Record the full exchange, including any internal
   trace the system is willing to surface (dominant emotion,
   running valence, chosen register, alternatives considered).
2. **Blind score.** Two auditors score the five dimensions
   independently without discussion.
3. **Reconcile.** Auditors compare scores; any dimension with a
   gap greater than one point is re-examined jointly.
4. **Report.** The final record is the tuple, the notes, and the
   reconciliation summary. The raw exchange is retained for
   replay.

Auditors should be *external* to the development team whenever
possible, for the same reason clinical audits are.

## Pairing With the Other Two Frameworks

The triad is designed to be used together:

- The **Friction Protocol** ensures the *user* does not consume
  AI output passively.
- The **Humanising Loop** ensures the *system* does not generate
  output unattentively.
- The **Attunement Audit** ensures a *third party* can verify
  that both of the above actually happened in practice.

Without the Audit, the first two frameworks collapse into
well-intentioned self-report. With it, they become falsifiable.

## Applications

- **Clinical AI deployment**: auditing triage and decision-support
  exchanges for perceptual fidelity and attributive honesty.
- **Education**: auditing tutoring interactions for registerial
  attunement and generative restraint, especially around learner
  frustration or confusion.
- **Regulatory review**: providing a scoreable, reproducible
  protocol for AI-interaction audits that regulators can require
  without prescribing implementation.
- **Longitudinal research**: tracking attunement scores across
  model versions, prompt changes, or deployment contexts to catch
  silent regressions in humanising behaviour.

## Implementation in `humanising-ai`

Each dimension of the Audit maps onto concrete evidence the
repository's modules can produce:

| Dimension | Evidence module | Signals |
|-----------|-----------------|---------|
| Perceptual Fidelity | `src/affective/` | `EmotionalSnapshot`, running valence/arousal, volatility |
| Attributive Honesty | `src/theory_of_mind/` | `ToMBenchmark` scores, belief-probe accuracy |
| Registerial Attunement | `src/dialogue/` | acknowledgement + invitation markers, register choice trace |
| Generative Restraint | `src/dialogue/` | response length, advice-density, echo-vs-overwrite ratio |
| Interrogable Accounting | `src/explainability/` | SHAP attributions, contrastive verdicts, `flipped_at` thresholds |

A deployment using this repository can, in principle, auto-populate
the left-hand column of any audit from its own traces — leaving
auditors to focus on judgement, not data collection.

## Citation

Romanov, D. (2026). *The Attunement Audit: A rubric for auditing
AI interactions against humanising principles.* Unpublished
coursework, MSc Clinical Neuroscience, University of Roehampton.

---

## References

- Donabedian, A. (1988). The quality of care: How can it be assessed? *JAMA*, 260(12), 1743–1748.
- Miller, T. (2019). Explanation in artificial intelligence: Insights from the social sciences. *Artificial Intelligence*, 267, 1–38.
- O'Neill, O. (2002). *A Question of Trust: The BBC Reith Lectures 2002*. Cambridge University Press.
- Ricoeur, P. (1992). *Oneself as Another* (K. Blamey, Trans.). University of Chicago Press.
- Bjork, R.A. (1994). Memory and metamemory considerations in the training of human beings. In J. Metcalfe & A.P. Shimamura (Eds.), *Metacognition: Knowing about knowing* (pp. 185–205). MIT Press.
