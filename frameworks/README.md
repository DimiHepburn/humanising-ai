# Frameworks

**Three original frameworks for humanising AI — one for the user, one for the system, one for the auditor.**

*Developed by Dimitri Romanov, MSc Clinical Neuroscience, University of Roehampton.*

---

## What this folder is

The `humanising-ai` repository is built around a simple claim:
making AI *humane* is not a feature of the model weights but of
the interaction that surrounds them. This folder contains the
three original frameworks that express that claim. Together they
form a deliberately complete triad — each perspective covers
what the other two cannot see.

| File | Perspective | Mode | Asks |
|------|-------------|------|------|
| [`friction_protocol.md`](./friction_protocol.md) | Human → AI | Prescriptive | *How should I engage critically with what the AI gives me?* |
| [`humanising_loop.md`](./humanising_loop.md) | AI → Human | Prescriptive | *How should the system engage carefully with the person in front of it?* |
| [`attunement_audit.md`](./attunement_audit.md) | Third-party → Exchange | Evaluative | *Did this interaction, in fact, humanise anyone?* |

## Why three frameworks and not one

A single framework cannot tell the whole story of a humane
interaction, because three distinct agents participate in it:

- **The user** brings an epistemic and emotional situation — and
  can consume AI output passively or critically. The **Friction
  Protocol** is for them.
- **The system** produces outputs — and can generate fluently but
  unattentively, or loop back through perceive → attribute →
  attune → respond → account. The **Humanising Loop** is for it.
- **Everyone else** — the educator, clinician, regulator,
  deployer — needs a way to tell whether the first two actually
  happened. The **Attunement Audit** is for them.

Remove any one of these and the other two collapse into
self-report.

## How to read the folder

1. Start with [`friction_protocol.md`](./friction_protocol.md) if
   you are a **learner or practitioner** meeting AI-generated
   content. Five friction points to convert passive consumption
   into genuine understanding.
2. Read [`humanising_loop.md`](./humanising_loop.md) next if you
   are a **designer or engineer** building AI that interacts with
   people. Five stages — perceive, attribute, attune, respond,
   account — that structure every exchange.
3. Finish with [`attunement_audit.md`](./attunement_audit.md) if
   you are an **auditor, educator, or deployment lead**. A five-
   dimension rubric for scoring any real interaction against the
   principles above.

## How the triad maps onto the codebase

Each framework is deliberately implementable. The `src/` modules
and the notebooks in `notebooks/` provide the machinery; the
frameworks describe what that machinery is *for*.

| Loop stage / Audit dimension | Code module | Notebook |
|------------------------------|-------------|----------|
| Perceive / Perceptual Fidelity | `src/affective/` | `01_emotion_detection.py` |
| Attribute / Attributive Honesty | `src/theory_of_mind/` | `02_theory_of_mind_evals.py` |
| Attune & Respond / Registerial Attunement & Generative Restraint | `src/dialogue/` | `03_dialogue_grounding.py` |
| Account / Interrogable Accounting | `src/explainability/` | `04_explainability.py` |

The Friction Protocol sits *outside* this table on purpose: it
is the framework the *reader* is invited to apply to everything
else in the repository, including these frameworks themselves.

## Citation

If you use any of these frameworks, please cite them
individually using the citation block at the bottom of each
file. A combined citation for the triad:

> Romanov, D. (2026). *Humanising AI: A triad of frameworks for
> user engagement, system design, and third-party audit of
> human–AI interaction.* Unpublished coursework, MSc Clinical
> Neuroscience, University of Roehampton.

---

## Status

Each framework is a first published version (April 2026). The
accompanying code is a lightweight reference implementation —
dependency-light on purpose, so the frameworks can be examined
without committing to a particular ML stack. Suggestions,
critiques and replication attempts are welcome via the
repository's issue tracker.
