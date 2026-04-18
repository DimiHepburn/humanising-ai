# Frameworks

**Four original frameworks for humanising AI — one for the user, one for the system, one for the auditor, one for the safety edge.**

*Developed by Dimitri Romanov, MSc Clinical Neuroscience, University of Roehampton.*

---

## What this folder is

The `humanising-ai` repository is built around a simple claim:
making AI *humane* is not a feature of the model weights but of
the interaction that surrounds them. This folder contains the
four original frameworks that express that claim. Together they
form a deliberately complete set — each perspective covers
what the others cannot see.

| File | Perspective | Mode | Asks |
|------|-------------|------|------|
| [`friction_protocol.md`](./friction_protocol.md) | Human → AI | Prescriptive | *How should I engage critically with what the AI gives me?* |
| [`humanising_loop.md`](./humanising_loop.md) | AI → Human | Prescriptive | *How should the system engage carefully with the person in front of it?* |
| [`attunement_audit.md`](./attunement_audit.md) | Third-party → Exchange | Evaluative | *Did this interaction, in fact, humanise anyone?* |
| [`handoff_threshold.md`](./handoff_threshold.md) | AI → (Out of interaction) | Safety-critical | *When should the system step out of the conversation entirely?* |

## Why four frameworks and not three

A three-framework triad covers a well-functioning interaction,
but it presumes the interaction *should be happening* in the
first place. In real deployments — especially clinical,
educational, or mental-health adjacent ones — that assumption is
the single most dangerous thing to leave implicit.

The fourth framework therefore takes a different shape. Where
the first three are prescriptive or evaluative *within* an
exchange, the **Handoff Threshold** is a disjunctive safety
rule that can end the exchange entirely:

- **The user** brings an epistemic and emotional situation — and
  can consume AI output passively or critically. The **Friction
  Protocol** is for them.
- **The system** produces outputs — and can generate fluently but
  unattentively, or loop back through perceive → attribute →
  attune → respond → account. The **Humanising Loop** is for it.
- **Everyone else** — the educator, clinician, regulator,
  deployer — needs a way to tell whether the first two actually
  happened. The **Attunement Audit** is for them.
- **Safety** requires that none of the above is allowed to
  substitute for a human when a human is what the user needs.
  The **Handoff Threshold** is the brake pedal.

Remove any one of the first three and the other two collapse
into self-report. Remove the fourth and the whole system is
operating without a brake pedal.

## How to read the folder

1. Start with [`friction_protocol.md`](./friction_protocol.md) if
   you are a **learner or practitioner** meeting AI-generated
   content. Five friction points to convert passive consumption
   into genuine understanding.
2. Read [`humanising_loop.md`](./humanising_loop.md) next if you
   are a **designer or engineer** building AI that interacts with
   people. Five stages — perceive, attribute, attune, respond,
   account — that structure every exchange.
3. Read [`attunement_audit.md`](./attunement_audit.md) if you are
   an **auditor, educator, or deployment lead**. A five-dimension
   rubric for scoring any real interaction against the principles
   above.
4. Finish with [`handoff_threshold.md`](./handoff_threshold.md)
   if you are **responsible for user safety**. Five criteria,
   evaluated per turn, for when the system should defer to a
   human — encoded as a disjunction, not a weighted sum.

## How the set maps onto the codebase

Each framework is deliberately implementable. The `src/` modules
and the notebooks in `notebooks/` provide the machinery; the
frameworks describe what that machinery is *for*.

| Loop stage / Audit dimension | Code module | Notebook |
|------------------------------|-------------|----------|
| Perceive / Perceptual Fidelity | `src/affective/` | `01_emotion_detection.py` |
| Attribute / Attributive Honesty | `src/theory_of_mind/` | `02_theory_of_mind_evals.py` |
| Attune & Respond / Registerial Attunement & Generative Restraint | `src/dialogue/` | `03_dialogue_grounding.py` |
| Account / Interrogable Accounting | `src/explainability/` | `04_explainability.py` |

The **Friction Protocol** sits *outside* this table on purpose:
it is the framework the *reader* is invited to apply to
everything else in the repository, including these frameworks
themselves.

The **Handoff Threshold** also sits outside this table, for a
different reason: it is evaluated against signals the four
sub-packages produce (valence, arousal, acknowledgement scores,
re-anchor counts) plus at least one signal — risk classification —
that the repository deliberately does *not* ship. See the
"Implementation in `humanising-ai`" section of the Handoff
Threshold document for the expected evidence mapping.

## Citation

If you use any of these frameworks, please cite them
individually using the citation block at the bottom of each
file. A combined citation for the set:

> Romanov, D. (2026). *Humanising AI: Four frameworks for user
> engagement, system design, third-party audit, and safe
> handoff in human–AI interaction.* Unpublished coursework,
> MSc Clinical Neuroscience, University of Roehampton.

---

## Status

Each framework is a first published version (April 2026). The
accompanying code is a lightweight reference implementation —
dependency-light on purpose, so the frameworks can be examined
without committing to a particular ML stack. The Handoff
Threshold deliberately omits a production-grade risk
classifier; any deployment using it must supply one, audited
separately. Suggestions, critiques and replication attempts
are welcome via the repository's issue tracker.
