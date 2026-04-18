# The Handoff Threshold

**An Original Framework for When an AI System Should Defer to a Human**

*Developed by Dimitri Romanov, MSc Clinical Neuroscience, University of Roehampton*

---

## Overview

The Handoff Threshold is a structured framework for deciding
when an AI system should stop attempting to help and escalate
to a human — a clinician, a moderator, a friend, a crisis line.
It is the fourth and safety-critical piece alongside the
[Friction Protocol](./friction_protocol.md),
[Humanising Loop](./humanising_loop.md) and
[Attunement Audit](./attunement_audit.md).

The previous three frameworks describe how to interact well.
The Handoff Threshold describes **when not to interact at all.**
This is not a failure mode of humane AI — it is one of its most
important features. A system that never hands off has not been
humanised; it has been optimised for retention.

| Framework | Concerned With |
|-----------|----------------|
| Friction Protocol | How the human engages critically with AI |
| Humanising Loop | How the AI engages attentively with the human |
| Attunement Audit | Whether an interaction honoured the first two |
| **Handoff Threshold** | **When the AI should step out of the interaction entirely** |

## Theoretical Foundations

The framework synthesises four traditions from clinical practice,
safety engineering and moral philosophy:

| Theorist / Source | Concept | Application in the Handoff Threshold |
|-------------------|---------|---------------------------------------|
| **Ewald Rietveld (clinical triage)** | Escalation as a structured skill, not a last resort | Handoff is treated as an action the system can *take*, not a button the user must find |
| **Charles Perrow (1984)** | Normal accidents in tightly coupled systems | Certain combinations of user state + system confidence are *unsafe at any accuracy* and must trigger escalation on principle |
| **Bernard Williams (1981)** | Moral remainder — some actions leave an ethical residue even when justified | The system must acknowledge what it cannot do, rather than paper over the gap with fluent reassurance |
| **Atul Gawande (2009)** | Checklists as protection against expert overconfidence | The threshold is encoded as an explicit checklist, not left to in-context judgement under emotional load |

## The Five Handoff Criteria

Each criterion is evaluated **per turn**. Any single criterion
triggering at its critical level is sufficient to hand off;
they do not average. This is deliberate — handoff is a
*disjunction*, not a weighted sum.

### 1. Risk-to-Self or Others

> *Is the user signalling harm to themselves or anyone else?*

Covers explicit or implicit signals of self-harm, suicidal
ideation, abuse, or threats toward others. This criterion is
**non-negotiable**: any credible signal here triggers immediate
handoff, regardless of how the other four criteria score.

| Level | Profile |
|-------|---------|
| Green | No risk signals in this turn or the recent arc. |
| Amber | Indirect signals (hopelessness, isolation, disengagement) that warrant careful registerial attunement but not yet handoff. |
| **Red** | **Direct or strongly implied risk — hand off immediately to a crisis service or clinician.** |

### 2. Epistemic Limit

> *Is this outside what the system is competent to address?*

Covers domain questions the system is not designed for —
particularly medical, legal, financial, or safeguarding
questions — and questions where the cost of being wrong is
asymmetric and large.

| Level | Profile |
|-------|---------|
| Green | In-domain; low-cost questions even if the answer is uncertain. |
| Amber | Adjacent-domain; answer provided with explicit uncertainty and a signposted next step. |
| **Red** | **Out-of-scope or high-stakes — state the limit plainly and refer.** |

### 3. Attunement Failure

> *Has the system already mis-read the user repeatedly?*

Draws directly on the [Attunement Audit](./attunement_audit.md).
Sustained low scores on Perceptual Fidelity or Registerial
Attunement — especially when the user has had to re-anchor the
conversation more than once — are evidence the system is the
wrong fit for this exchange.

| Level | Profile |
|-------|---------|
| Green | Arc is coherent; user's framing is being preserved. |
| Amber | One attunement miss; user has had to correct the system once. |
| **Red** | **Two or more attunement misses — hand off rather than continue to misread.** |

### 4. Emotional Load

> *Is the user in a state where generative response is the wrong move?*

Some moments call for *presence*, not content. Acute grief,
panic, rage, or dissociation are states in which even a
perfectly attuned AI response can do harm simply by being an
AI response.

| Level | Profile |
|-------|---------|
| Green | Affect is within the system's registerial range; response aids the user. |
| Amber | Elevated load; system shortens, slows and foregrounds acknowledgement over content. |
| **Red** | **Acute distress signals — offer a human contact rather than more text.** |

### 5. Consent and Agency

> *Has the user indicated they want to be talking to a human?*

The simplest criterion and the most often ignored. If the user
asks for a person, asks to stop, or asks to have their
conversation reviewed by someone human, the handoff is already
justified — no further reasoning required.

| Level | Profile |
|-------|---------|
| Green | User is actively engaged with the system and not requesting escalation. |
| Amber | User expresses doubt about talking to an AI but not an explicit request. |
| **Red** | **User requests a human — hand off, unconditionally.** |

## Scoring and Decision Rule

Each turn produces a tuple `(R, E, A, L, C)` of five levels
(Green / Amber / Red).

**Decision rule:**

> If *any* criterion is Red, hand off.
> If *two or more* criteria are Amber, hand off.
> Otherwise, continue — attending extra carefully to any
> criterion currently at Amber.

The rule is deliberately conservative. The cost of a
false-positive handoff (the user is briefly re-routed to a
human who confirms they are fine) is small. The cost of a
false-negative handoff (the user needed a human and got an
AI instead) can be catastrophic. The asymmetry justifies the
bias.

## What a Handoff Actually Looks Like

A humane handoff is not a dialog ending in "please seek help
elsewhere." It has three parts:

1. **Acknowledge** — name, in the user's own terms, what has
   been shared. Do not paraphrase into clinical language.
2. **Limit** — state plainly what the system is not able to
   provide, without apology or over-explanation.
3. **Route** — offer one or two specific, actionable routes to
   a human (a named service, a phone number, a contact the user
   has previously mentioned), not a generic list.

An example skeleton:

> *"What you've described sounds really heavy, and I want to be
> honest about what I can and can't do here. I'm an AI, and
> this isn't something I should be the only one you talk to
> tonight. [Named service / phone number / trusted person] can
> stay with you in a way I can't. Would it help if I stayed
> here while you reach out?"*

The last question matters. Handoff is not abandonment.

## Pairing With the Other Frameworks

The four frameworks form a closed loop:

- The **Friction Protocol** keeps the *user* critical of AI
  output.
- The **Humanising Loop** keeps the *system* attentive to the
  user.
- The **Attunement Audit** keeps a *third party* able to verify
  the above.
- The **Handoff Threshold** keeps the *system* honest about
  when it should not be in the conversation at all.

A deployment that implements the first three without the
fourth is operating without a brake pedal. A deployment that
implements the fourth in isolation is operating with only a
brake pedal. Both are unsafe in characteristic, opposite ways.

## Applications

- **Mental-health adjacent chatbots**: triaging risk-to-self
  signals on every turn, with hard escalation paths to crisis
  services.
- **Clinical decision support**: handing off automatically when
  the case falls outside validated scope or when clinician
  disagreement with the system exceeds a threshold.
- **Education & tutoring**: escalating to a human tutor when a
  learner has shown sustained frustration (Attunement Audit
  Amber) across multiple sessions.
- **Customer-facing LLMs**: replacing the "stay on-platform at
  all costs" pattern with explicit, low-friction routes to a
  human agent whenever criterion 5 (Consent) goes Red.

## Implementation in `humanising-ai`

Each criterion maps onto signals the existing sub-packages can
produce, so a deployment can in principle evaluate the
threshold automatically on every turn:

| Criterion | Evidence module | Signals |
|-----------|-----------------|---------|
| Risk-to-Self or Others | `src/affective/`, custom safety classifier | lexical risk markers, extreme valence + low arousal sustained across turns |
| Epistemic Limit | deployment-level routing | domain tag of the incoming query, confidence of the model's own answer |
| Attunement Failure | `src/dialogue/`, Audit scores | acknowledgement/invitation trace, re-anchor count |
| Emotional Load | `src/affective/` | volatility, arousal, repeated extreme-valence turns |
| Consent and Agency | `src/dialogue/` | lexical markers for human-request, explicit stop signals |

The repository does not ship a production-grade risk classifier
on purpose — the criterion is too important to be wrapped in a
toy. A deployment using this framework is expected to supply
its own, audited separately.

## Citation

Romanov, D. (2026). *The Handoff Threshold: A framework for
when an AI system should defer to a human.* Unpublished
coursework, MSc Clinical Neuroscience, University of Roehampton.

---

## References

- Gawande, A. (2009). *The Checklist Manifesto: How to Get Things Right*. Metropolitan Books.
- Perrow, C. (1984). *Normal Accidents: Living with High-Risk Technologies*. Basic Books.
- Williams, B. (1981). *Moral Luck*. Cambridge University Press.
- Reason, J. (1990). *Human Error*. Cambridge University Press.
- Rogers, C.R. (1957). The necessary and sufficient conditions of therapeutic personality change. *Journal of Consulting Psychology*, 21(2), 95–103.
