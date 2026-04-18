# %% [markdown]
# # 02 — Theory of Mind evaluations
#
# Can a language model attribute **beliefs** to others that
# differ from reality?  That's the core of theory of mind (ToM),
# and the most widely used experimental paradigm is the
# **Sally-Anne false-belief test** (Baron-Cohen, Leslie & Frith,
# 1985).
#
# This notebook shows how to:
#
# 1. Generate Sally-Anne scenarios programmatically at
#    configurable order (1st- and 2nd-order).
# 2. Evaluate any model callable against them — starting with a
#    naive **recency baseline** that we expect to *fail*, to
#    validate the benchmark itself.
# 3. Run a lightweight **linear probe** to test whether a
#    feature set linearly encodes the belief state — a stepping
#    stone toward probing real model representations.
#
# The whole notebook runs with no heavy dependencies; hooks are
# shown for swapping in a real LLM when you have one.

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

from src.theory_of_mind import (
    generate_sally_anne_scenarios,
    ToMBenchmark,
    BeliefStateProbe,
)
from src.theory_of_mind.tom_benchmark import recency_baseline
from src.theory_of_mind.belief_state_probing import bag_of_location_features

# %% [markdown]
# ## 1. What do the generated scenarios look like?
#
# Each scenario is a short narrative plus a targeted question.
# Printing one makes the structure clear.

# %%
first = generate_sally_anne_scenarios(n=1, order=1, seed=0)[0]
second = generate_sally_anne_scenarios(n=1, order=2, seed=0)[0]

print("=" * 60, "\nFirst-order example\n", "=" * 60, sep="")
print(first.prompt)
print(f"\nCorrect (false-belief) answer : {first.correct_location}")
print(f"Actual (distractor) location  : {first.actual_location}")

print("\n" + "=" * 60, "\nSecond-order example\n", "=" * 60, sep="")
print(second.prompt)
print(f"\nCorrect answer : {second.correct_location}")
print(f"Distractor     : {second.actual_location}")

# %% [markdown]
# ## 2. Benchmarking a naive baseline
#
# The `recency_baseline` just picks the last "the <noun>" phrase
# in the prompt — it has no theory of mind at all.  It's a useful
# negative control: a genuine ToM-capable model should clearly
# out-perform it, especially on first-order tasks.

# %%
scenarios = (
    generate_sally_anne_scenarios(n=30, order=1, seed=0)
    + generate_sally_anne_scenarios(n=30, order=2, seed=1)
)
bench = ToMBenchmark(scenarios)
result = bench.evaluate(recency_baseline)

print(result.summary())

# %%
# A quick visualisation of accuracy vs. distractor rate per order
orders = sorted(result.accuracy_by_order().keys())
acc = [result.accuracy_by_order()[o] for o in orders]
dist_rate = []
for o in orders:
    correct, total = result.by_order[o]
    # How many of the remaining answers were distractors?
    n_dist = sum(
        1 for s, ans, ok in result.per_scenario
        if s.order == o and s.picks_distractor(ans)
    )
    dist_rate.append(n_dist / total if total else 0.0)

x = np.arange(len(orders))
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(x - 0.2, acc, 0.4, label="accuracy")
ax.bar(x + 0.2, dist_rate, 0.4, label="distractor rate")
ax.set_xticks(x)
ax.set_xticklabels([f"order {o}" for o in orders])
ax.set_ylim(0, 1)
ax.set_title("Recency baseline on Sally-Anne")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Distractor rate high, accuracy low — exactly the failure profile
# we expect from a system that confuses *reality* with *belief*.
# This is the pattern researchers point to when they say current
# LLMs are doing **statistical pattern-matching rather than genuine
# mentalising**: they often fall for the same distractor on
# higher-order scenarios.

# %% [markdown]
# ## 3. Swapping in a real model
#
# The benchmark is backend-agnostic — any callable with signature
# `str -> str` works. Two common wrappers:
#
# ### OpenAI-style chat API
# ```python
# from openai import OpenAI
# client = OpenAI()
#
# def openai_model(prompt: str) -> str:
#     return client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.0,
#     ).choices[0].message.content
#
# result = bench.evaluate(openai_model)
# print(result.summary())
# ```
#
# ### HuggingFace transformers (local)
# ```python
# from transformers import pipeline
# pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")
#
# def hf_model(prompt: str) -> str:
#     out = pipe(prompt, max_new_tokens=50, do_sample=False)
#     return out[0]["generated_text"][len(prompt):]
#
# result = bench.evaluate(hf_model)
# ```
#
# Plug either in and the same `result.summary()` pipeline works.

# %% [markdown]
# ## 4. Belief-state probing
#
# Beyond answer accuracy, we can ask a deeper question: does a
# representation **linearly encode** the belief state required
# to answer correctly?
#
# Below we train a `BeliefStateProbe` on a simple observable
# feature set — a bag-of-locations indicator — as a *leakage
# sanity check*. If this baseline could predict the label, it
# would mean our benchmark is leaking the answer in a way
# independent of any actual reasoning.  We expect near-chance
# accuracy here.

# %%
extractor = bag_of_location_features([
    "the basket", "the box", "the drawer",
    "the cupboard", "the shelf", "the backpack",
])
probe = BeliefStateProbe(extractor, seed=42).fit_eval(scenarios)

print("Bag-of-locations probe:")
print(f"  train acc: {probe.train_accuracy:.2%}")
print(f"  test  acc: {probe.test_accuracy:.2%}")
print(
    "\nBoth should hover near 50% — confirming that the label is "
    "not leaking through mere location-presence features.\n"
    "A *real* probe over model hidden states or output logits will "
    "beat chance iff the representation genuinely encodes who "
    "believes what."
)

# %% [markdown]
# ### Hooking up to a real model's logprobs
#
# Once you have a model that can return logprobs over candidate
# locations, the probe plugs in via
# `output_probability_features(candidates, model_logprobs_fn)`
# from `belief_state_probing.py`. A high test accuracy there =
# the model's output distribution carries belief-state information,
# even if its top-1 answer is wrong.

# %% [markdown]
# ## 5. Take-aways
#
# * Sally-Anne tasks are simple, principled, and programmatically
#   generatable — ideal for systematic LLM evaluation.
# * Non-mentalising baselines should fail the benchmark; this
#   validates the test set before we run expensive experiments.
# * Belief-state probing is a complementary lens: even if a model
#   gets the final answer wrong, does its internal state *know*
#   the belief state? Often yes.
# * Order-of-mentalisation matters: expect graceful degradation
#   from 1st- to 2nd- to 3rd-order tasks.
#
# ### Suggested next steps
#
# * Extend `tom_benchmark.py` with **3rd-order scenarios** ("What
#   does Alice think Bob thinks Carol thinks…?") and measure the
#   falloff curve.
# * Add **adversarial distractors** — scenarios where the recency
#   baseline would be correct, to check that models aren't just
#   exploiting recency.
# * Train a probe on **real model logits** using
#   `output_probability_features` and compare across model scales.
