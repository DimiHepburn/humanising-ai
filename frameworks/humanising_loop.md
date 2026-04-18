# The Humanising Loop

**An Original Design Framework for Emotionally and Cognitively Attuned AI Response**

*Developed by Dimitri Romanov, MSc Clinical Neuroscience, University of Roehampton*

---

## Overview

The Humanising Loop is a structured framework for designing AI
systems that respond to people, not just prompts. Where the
**Friction Protocol** tells the *human* how to engage critically
with AI, the Humanising Loop tells the *AI* how to engage
carefully with humans — by treating each interaction as a
five-stage loop rather than a one-shot generation.

The framework is deliberately the mirror image of the Friction
Protocol. Together they form a pair:

| Friction Protocol | Humanising Loop |
|-------------------|-----------------|
| How the human engages with AI output | How the AI engages with the human |
| Adds cognitive resistance to passive consumption | Adds relational attentiveness to automated generation |
| Protects the learner's epistemic agency | Protects the user's emotional agency |

## Theoretical Foundations

The Humanising Loop draws on four clinical and philosophical
traditions:

| Theorist | Concept | Application in the Humanising Loop |
|----------|---------|-------------------------------------|
| **Carl Rogers** | Unconditional positive regard; reflective listening | The AI's first move is always to *acknowledge*, not to solve |
| **Daniel Stern** | Affect attunement | The AI matches the emotional *contour* of the user before adjusting content |
| **Martin Buber** | I–Thou vs. I–It encounter | The user is addressed as a subject with their own world, not as an input to be processed |
| **Antonio Damasio** | Somatic markers; feeling as cognition | Affect is treated as information about the user's state, not noise to be filtered out |

## The Five Stages of the Loop

### 1. Perceive

> *"What is this person actually bringing to me?"*

Before generating anything, the system reads the **surface and
subtext** of the input: what is said, what is avoided, what
emotional tone carries it, and what the running conversational
arc looks like. In the `humanising-ai` codebase this stage is
handled by the `affective` sub-package.

### 2. Attribute

> *"What might they believe, want, or fear — that I cannot see?"*

The system explicitly models the user's **mental state** rather
than collapsing them into the literal prompt. This is a
theory-of-mind step: distinguishing belief from reality, and
checking whether an apparently simple question carries a harder
question beneath it. Covered in the `theory_of_mind` sub-package.

### 3. Attune

> *"What shape of response does this moment call for?"*

Before choosing **words**, the system chooses **register**:
acknowledgement vs. problem-solving, invitation vs. closure,
brevity vs. elaboration. The `dialogue` sub-package enforces this
by generating an *acknowledgement → mirror → invitation* skeleton
before any content is attached to it.

### 4. Respond

> *"Now — and only now — generate."*

Only after perceive, attribute and attune is content produced.
The response is constrained to respect the earlier stages:
register first, content second. This is the inverse of the usual
LLM failure mode, in which fluent content arrives before the
system has done any relational work.

### 5. Account

> *"If the person asks, can I show my working?"*

Every response is accompanied by an *interrogable trace*: what
emotional signal was read, what belief was attributed, which
register was chosen, and which alternative responses were
considered. The `explainability` sub-package turns this trace
into contrastive, feature-level explanations on demand.

## Why "Loop"?

A single forward pass cannot humanise anything. The Humanising
Loop insists that interaction is a **cycle**: each response
re-enters the Perceive stage, updating the system's model of the
person as the conversation evolves. The loop closes when the user
either ends the exchange or explicitly signals that their state
has shifted — at which point the system re-attunes rather than
continuing on its previous trajectory.

This framing resists three common pathologies of deployed AI:

- **Premature resolution** — jumping to advice before acknowledgement.
- **Tone collapse** — delivering every response in the same register.
- **Black-box verdicts** — emitting outputs with no interrogable trace.

## Pairing With the Friction Protocol

The two frameworks are designed to be used together:

- The **Friction Protocol** operates on the *human side* of the
  exchange, introducing five points of deliberate cognitive
  resistance to AI output.
- The **Humanising Loop** operates on the *AI side*, introducing
  five stages of deliberate relational attentiveness before and
  during generation.

An AI system that implements the Humanising Loop gives the
Friction Protocol something worth frictioning *with*: outputs
that are honest about their assumptions, traceable in their
reasoning, and attuned in their register. An educator or
clinician using the Friction Protocol, in turn, provides the
Humanising Loop with the kind of critical counterpart that keeps
it from drifting into sycophancy.

## Applications

- **Mental-health adjacent chatbots**: enforcing acknowledgement
  and invitation before any coping-strategy content.
- **Clinical decision support**: surfacing the belief state
  attributed to the clinician, and the contrastive "why this
  differential and not that one?" trace.
- **Education & tutoring**: matching register to the learner's
  affective state (frustration vs. curiosity vs. fatigue) before
  selecting pedagogical strategy.
- **Customer-facing LLMs**: replacing the default "fluent first,
  feeling later" pipeline with a perceive → attune → respond
  sequence that reduces tone-deaf failures.

## Implementation in `humanising-ai`

Each stage of the loop maps onto a concrete module in the
accompanying codebase:

| Stage | Module | Key Interfaces |
|-------|--------|----------------|
| Perceive | `src/affective/` | `EmotionalContextTracker`, `EmotionalSnapshot` |
| Attribute | `src/theory_of_mind/` | `generate_sally_anne_scenarios`, `ToMBenchmark`, `BeliefStateProbe` |
| Attune | `src/dialogue/` | `ConversationContext`, `TemplateGenerator` |
| Respond | `src/dialogue/` | `EmpatheticResponder`, `LLMGenerator` |
| Account | `src/explainability/` | `ShapExplainer`, `ContrastiveExplainer` |

The notebooks `01`–`04` walk through the stages in order, so the
Humanising Loop can be read either as a design document or as a
reading guide for the repository.

## Citation

Romanov, D. (2026). *The Humanising Loop: A design framework for
emotionally and cognitively attuned AI response.* Unpublished
coursework, MSc Clinical Neuroscience, University of Roehampton.

---

## References

- Buber, M. (1923/1970). *I and Thou* (W. Kaufmann, Trans.). Charles Scribner's Sons.
- Damasio, A.R. (1994). *Descartes' Error: Emotion, Reason, and the Human Brain*. Putnam.
- Rogers, C.R. (1957). The necessary and sufficient conditions of therapeutic personality change. *Journal of Consulting Psychology*, 21(2), 95–103.
- Stern, D.N. (1985). *The Interpersonal World of the Infant: A View from Psychoanalysis and Developmental Psychology*. Basic Books.
- Miller, T. (2019). Explanation in artificial intelligence: Insights from the social sciences. *Artificial Intelligence*, 267, 1–38.
