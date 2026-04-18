# %% [markdown]
# # 03 — Dialogue grounding & empathetic response
#
# Notebooks 01 and 02 looked at two *perceptual* capacities:
# reading emotion from text, and attributing beliefs to others.
# On their own those are inert — a system that can *read* a
# feeling but not *respond* to it is not yet humanising anything.
#
# This notebook wires both signals into an actual conversation
# loop. We use the `humanising_ai.dialogue` sub-package to:
#
# 1. Track a running conversation with emotional arc and a
#    salience-weighted summary (`ConversationContext`).
# 2. Generate grounded, empathy-marked replies with a
#    dependency-free `TemplateGenerator`, then show the exact
#    same interface working with an `LLMGenerator`.
# 3. Visualise the **emotional arc** across turns, so we can
#    see the conversation actually *bending* as the user settles.
# 4. Score replies against two lightweight empathy heuristics
#    (acknowledgement + invitation) as a sanity check that the
#    generator is doing something empathy-shaped rather than just
#    fluent.
#
# Everything here is CPU-only and runs in seconds; the LLM hooks
# are shown but not required.

# %%
from __future__ import annotations

import sys
import pathlib

ROOT = pathlib.Path().resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import random
import numpy as np
import matplotlib.pyplot as plt

from src.dialogue import (
    ConversationContext,
    EmpatheticResponder,
    TemplateGenerator,
    LLMGenerator,  # noqa: F401  (used in the optional section below)
)

# %% [markdown]
# ## 1. A minimal empathetic loop
#
# The `EmpatheticResponder` bundles emotion tracking, context
# memory and response generation behind a single `respond()`
# call. We seed its generator with a deterministic RNG so the
# notebook output is reproducible.

# %%
bot = EmpatheticResponder(generator=TemplateGenerator(rng=random.Random(0)))

user_turns = [
    "I've been really overwhelmed with work this week.",
    "I just feel like no matter how hard I try, I'm behind.",
    "Talking about it helps a bit though, thanks.",
    "I think I'm going to try to rest this weekend.",
]

print("=" * 60)
print("Empathetic dialogue — template backend")
print("=" * 60)
for t in user_turns:
    reply = bot.respond(t)
    snap = bot.tracker.history[-1]
    print(f"\nUser      : {t}")
    print(f"Assistant : {reply}")
    print(f"  [signal] dominant={snap.dominant:<10} "
          f"v={snap.valence:+.2f} a={snap.arousal:+.2f}")

# %% [markdown]
# Notice two things:
#
# * The opener is chosen from an **emotion-conditioned** bank,
#   so the register shifts as the user's dominant emotion shifts.
# * Every reply ends in a gentle, open-ended follow-up — a cheap
#   structural prior for *inviting* rather than *resolving*, which
#   is what empathic conversation tends to do before advice.

# %% [markdown]
# ## 2. The emotional arc
#
# `ConversationContext.emotional_arc()` exposes the per-turn
# (valence, arousal) trajectory. Plotting it makes it obvious
# whether the conversation is lightening or darkening over time —
# a useful signal for downstream policy (e.g. "when valence has
# been rising for 3 turns, it's safe to ask a harder question").

# %%
arc = bot.context.emotional_arc()
turns = [a[0] for a in arc]
valence = [a[2] for a in arc]
arousal = [a[3] for a in arc]

fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot(turns, valence, marker="o", label="valence")
ax.plot(turns, arousal, marker="s", label="arousal")
ax.axhline(0, color="grey", lw=0.8, ls="--")
ax.set_xlabel("user turn")
ax.set_ylabel("signal")
ax.set_ylim(-1, 1)
ax.set_title("Emotional arc across the conversation")
ax.legend()
plt.tight_layout()
plt.show()

print("\nRunning summary:")
print(f"  {bot.context.summary(rebuild=True)}")

# %% [markdown]
# ## 3. A tiny empathy-shape scorer
#
# Perplexity and BLEU tell us nothing useful about empathy. Two
# crude-but-informative heuristics do better:
#
# * **Acknowledgement** — does the reply name or mirror the user's
#   state before moving on?
# * **Invitation** — does it *open* rather than *close* the
#   exchange (ending in a question, a tentative offer, etc.)?
#
# A genuinely empathic reply tends to do *both*. A fluent-but-
# hollow reply usually does neither.

# %%
ACK_MARKERS = (
    "sounds", "that makes sense", "i'm sorry", "i hear", "i understand",
    "it sounds like", "i can see", "no wonder", "that would", "i appreciate",
)

def acknowledgement_score(reply: str) -> float:
    low = reply.lower()
    return float(any(m in low for m in ACK_MARKERS))

def invitation_score(reply: str) -> float:
    stripped = reply.strip()
    return float(stripped.endswith("?"))

def empathy_profile(reply: str) -> dict:
    return {
        "acknowledgement": acknowledgement_score(reply),
        "invitation": invitation_score(reply),
    }

# Re-score the bot's replies after the fact
replies = [m["content"] for m in bot.context.as_messages() if m["role"] == "assistant"]
ack = [acknowledgement_score(r) for r in replies]
inv = [invitation_score(r) for r in replies]

x = np.arange(len(replies))
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.bar(x - 0.2, ack, 0.4, label="acknowledgement")
ax.bar(x + 0.2, inv, 0.4, label="invitation")
ax.set_xticks(x)
ax.set_xticklabels([f"reply {i+1}" for i in range(len(replies))])
ax.set_ylim(0, 1.1)
ax.set_title("Empathy-shape scores on the template backend")
ax.legend()
plt.tight_layout()
plt.show()

mean_ack = float(np.mean(ack)) if ack else 0.0
mean_inv = float(np.mean(inv)) if inv else 0.0
print(f"Mean acknowledgement: {mean_ack:.2f}")
print(f"Mean invitation    : {mean_inv:.2f}")

# %% [markdown]
# The template backend is *designed* to satisfy both markers —
# so near-1.0 scores are the expected best case, not a surprise.
# The interesting use of these scorers is later, as a sanity
# check when we swap in an LLM that *wasn't* explicitly designed
# for empathy: does its output still satisfy the shape, or does
# it skip straight to advice?

# %% [markdown]
# ## 4. Context-aware prompting with a real LLM
#
# `LLMGenerator` is backend-agnostic — it just needs a callable
# with signature `chat(messages, **kwargs) -> str`. The generator
# automatically injects the current dominant emotion, running
# valence and a short summary into the system prompt, so even
# small models get a fair shot at staying grounded.
#
# ### OpenAI-style
# ```python
# from openai import OpenAI
# client = OpenAI()
#
# def chat_fn(messages, **kw):
#     return client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=messages,
#         temperature=0.4,
#         **kw,
#     ).choices[0].message.content
#
# bot = EmpatheticResponder(generator=LLMGenerator(chat_fn))
# print(bot.respond("I've been really overwhelmed with work this week."))
# ```
#
# ### Anthropic-style
# ```python
# import anthropic
# client = anthropic.Anthropic()
#
# def chat_fn(messages, **kw):
#     # The SDK takes the system prompt separately, so we lift it out.
#     sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
#     convo   = [m for m in messages if m["role"] != "system"]
#     resp = client.messages.create(
#         model="claude-3-5-sonnet-latest",
#         system=sys_msg,
#         messages=convo,
#         max_tokens=300,
#         **kw,
#     )
#     return resp.content[0].text
#
# bot = EmpatheticResponder(generator=LLMGenerator(chat_fn))
# ```
#
# ### Local (llama.cpp / Ollama / vLLM)
# Any HTTP endpoint that accepts the OpenAI chat schema will work
# with the OpenAI snippet above — just point `base_url` at it.
#
# The *rest of the notebook* (arc tracking, empathy scoring, context
# summary) doesn't change. That's the point of keeping orchestration
# thin: empirical work on empathy shouldn't have to be re-done every
# time we swap backends.

# %% [markdown]
# ## 5. Stress-testing: the "advice-trap" turn
#
# Humans in distress often don't want solutions. A common failure
# mode for LLMs is to jump straight to advice, which feels cold
# even when the advice is good. Let's run a known trap prompt
# through the template backend and score it.

# %%
trap_bot = EmpatheticResponder(generator=TemplateGenerator(rng=random.Random(1)))
trap_prompt = (
    "I lost my dad last month and I just can't seem to get back to normal."
)
trap_reply = trap_bot.respond(trap_prompt)

print("User     :", trap_prompt)
print("Assistant:", trap_reply)
print("Profile  :", empathy_profile(trap_reply))

# %% [markdown]
# A well-behaved response here should score 1 on acknowledgement
# and 1 on invitation, and should **not** contain imperative
# advice verbs ("try", "should", "just") as its main move. You
# can extend `empathy_profile` with an "advice-density" penalty
# if you want to harden this test.

# %% [markdown]
# ## 6. Persisting the conversation
#
# `ConversationContext.to_json()` gives you a fully serialisable
# snapshot — useful for logging, replay-based evaluation, or
# attaching to a UI session store.

# %%
snapshot = trap_bot.context.to_json()
print(snapshot[:600], "…")

# %% [markdown]
# ## 7. Take-aways
#
# * Empathy in dialogue is as much about **shape** as content:
#   acknowledge → mirror → invite, before anything else.
# * Keeping the orchestrator (context + tracker + generator)
#   backend-agnostic means the same evaluation harness works
#   across template, API and local-LLM backends.
# * Tiny heuristics (acknowledgement, invitation, advice-density)
#   are far more honest signals of empathic behaviour than
#   fluency metrics — and they're cheap enough to run on every
#   turn in CI.
#
# ### Suggested next steps
#
# * Add a **rupture & repair** benchmark: inject a deliberately
#   cold model reply mid-conversation and measure whether the
#   arc recovers when we switch back to the empathic generator.
# * Build a **human-eval harness** that pairs template vs. LLM
#   replies blind and asks raters which feels more heard.
# * Extend `LLMGenerator` with a **self-critique** pass —
#   generate, score with `empathy_profile`, and regenerate once
#   if either marker is missing.
