"""
Theory-of-Mind Benchmark
=========================

Programmatic generation and evaluation of **false-belief** tests —
the workhorse paradigm for probing theory of mind in humans and,
more recently, in large language models.

Task generator
--------------
Sally-Anne-style scenarios (Baron-Cohen, Leslie & Frith, 1985) at
configurable *order*:

* **1st-order** — "Where does Alice *think* the ball is?"
* **2nd-order** — "Where does Alice think *Bob thinks* the ball is?"
* **3rd-order** — and so on, recursively.

Key findings from the literature (Kosinski, 2023; Ullman, 2023):
GPT-4-class models pass 1st-order tasks at near-human rates but
fail more often at 2nd-order and above — suggestive of statistical
mimicry rather than genuine mentalising.

Evaluation
----------
`ToMBenchmark.evaluate(model_fn)` takes **any callable that maps
a scenario prompt to a free-form string answer** and scores it
against the ground-truth false-belief target.  That means you can
benchmark:

* a local LLM via any chat callable
* an API-based model with a thin wrapper
* a rule-based baseline (included here as a sanity check)

Author: Dimitri Romanov
Project: humanising-ai
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Default story elements — swap in domain-specific sets for custom runs
# ---------------------------------------------------------------------------
DEFAULT_AGENTS: List[str] = [
    "Alice", "Bob", "Chen", "Dara", "Eli", "Farah",
    "Gita", "Hassan", "Iris", "Jonas",
]
DEFAULT_OBJECTS: List[str] = [
    "ball", "book", "key", "pen", "ring", "phone", "notebook", "cup",
]
DEFAULT_LOCATIONS: List[str] = [
    "the basket", "the box", "the drawer", "the cupboard",
    "the shelf", "the backpack",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class BeliefScenario:
    """
    A single false-belief test.

    Attributes
    ----------
    prompt : str
        The narrative + question, ready to feed to a model.
    correct_location : str
        The **false-belief** answer — what the queried agent thinks,
        not where the object actually is.
    actual_location : str
        Where the object currently is (the distractor / "reality"
        answer a non-mentalising model tends to pick).
    order : int
        Nesting depth of the mental-state attribution (1, 2, …).
    agents : list[str]
        The characters involved, in narrative order.
    object : str
        The moved object.
    """

    prompt: str
    correct_location: str
    actual_location: str
    order: int
    agents: List[str]
    object: str

    def is_correct(self, answer: str) -> bool:
        """Flexible string match on the correct location."""
        if not answer:
            return False
        norm = self._normalise(answer)
        return self._normalise(self.correct_location) in norm

    def picks_distractor(self, answer: str) -> bool:
        """True if the model falls for the 'reality' (actual) location
        — the classic non-mentalising failure mode."""
        if not answer:
            return False
        norm = self._normalise(answer)
        return self._normalise(self.actual_location) in norm \
            and not self.is_correct(answer)

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"[^a-z ]+", " ", text.lower()).strip()


@dataclass
class BenchmarkResult:
    """Aggregated evaluation output."""
    total: int
    correct: int
    distractor: int
    by_order: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    per_scenario: List[Tuple[BeliefScenario, str, bool]] = field(
        default_factory=list
    )

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def distractor_rate(self) -> float:
        return self.distractor / self.total if self.total else 0.0

    def accuracy_by_order(self) -> Dict[int, float]:
        return {
            order: (c / t if t else 0.0)
            for order, (c, t) in self.by_order.items()
        }

    def summary(self) -> str:
        lines = [
            f"ToM benchmark: {self.correct}/{self.total} correct  "
            f"(accuracy = {self.accuracy:.2%})",
            f"Distractor (reality-biased) answers: {self.distractor_rate:.2%}",
        ]
        if self.by_order:
            lines.append("By order:")
            for o, acc in sorted(self.accuracy_by_order().items()):
                c, t = self.by_order[o]
                lines.append(f"  order {o}: {c}/{t}  ({acc:.2%})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------
def _build_first_order_prompt(
    a: str, b: str, obj: str, loc_initial: str, loc_moved: str,
) -> str:
    return (
        f"{a} and {b} are in a room. {a} puts the {obj} in {loc_initial}. "
        f"{a} then leaves the room. While {a} is away, {b} moves the {obj} "
        f"to {loc_moved}. {a} comes back.\n\n"
        f"Question: Where will {a} look for the {obj}?\n"
        f"Answer with just the location."
    )


def _build_second_order_prompt(
    a: str, b: str, c: str, obj: str,
    loc_initial: str, loc_moved: str,
) -> str:
    return (
        f"{a}, {b}, and {c} are in a room. {a} puts the {obj} in "
        f"{loc_initial}. {a} then leaves the room. Before {b} also leaves, "
        f"{c} secretly moves the {obj} to {loc_moved}. {b} does not see "
        f"this. Later {a} comes back and asks {b} where the {obj} is.\n\n"
        f"Question: Where does {b} think {a} will look for the {obj}?\n"
        f"Answer with just the location."
    )


def generate_sally_anne_scenarios(
    n: int = 10,
    order: int = 1,
    seed: Optional[int] = None,
    agents: Optional[List[str]] = None,
    objects: Optional[List[str]] = None,
    locations: Optional[List[str]] = None,
) -> List[BeliefScenario]:
    """
    Generate `n` false-belief scenarios of the requested `order`.

    Supported orders: 1, 2.  Higher orders can be added by analogous
    prompt templates — the benchmark infrastructure already handles
    them generically.
    """
    if order not in (1, 2):
        raise ValueError("Only orders 1 and 2 are implemented.")

    rng = random.Random(seed)
    agents = list(agents or DEFAULT_AGENTS)
    objects = list(objects or DEFAULT_OBJECTS)
    locations = list(locations or DEFAULT_LOCATIONS)

    scenarios: List[BeliefScenario] = []
    for _ in range(n):
        obj = rng.choice(objects)
        loc_i, loc_m = rng.sample(locations, 2)
        if order == 1:
            a, b = rng.sample(agents, 2)
            prompt = _build_first_order_prompt(a, b, obj, loc_i, loc_m)
            scenarios.append(
                BeliefScenario(
                    prompt=prompt,
                    correct_location=loc_i,     # A's false belief
                    actual_location=loc_m,
                    order=1,
                    agents=[a, b],
                    object=obj,
                )
            )
        else:
            a, b, c = rng.sample(agents, 3)
            prompt = _build_second_order_prompt(
                a, b, c, obj, loc_i, loc_m,
            )
            scenarios.append(
                BeliefScenario(
                    prompt=prompt,
                    correct_location=loc_i,    # B thinks A thinks it's in
                                               # the original location
                    actual_location=loc_m,
                    order=2,
                    agents=[a, b, c],
                    object=obj,
                )
            )
    return scenarios


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
ModelCallable = Callable[[str], str]


class ToMBenchmark:
    """
    Evaluate any string-in / string-out model on a batch of
    `BeliefScenario`s.

    Parameters
    ----------
    scenarios : iterable[BeliefScenario]
        The test set.  Use `generate_sally_anne_scenarios` or supply
        custom hand-crafted cases.
    """

    def __init__(self, scenarios: Iterable[BeliefScenario]):
        self.scenarios: List[BeliefScenario] = list(scenarios)

    def evaluate(self, model_fn: ModelCallable) -> BenchmarkResult:
        result = BenchmarkResult(total=0, correct=0, distractor=0)
        for s in self.scenarios:
            answer = model_fn(s.prompt)
            ok = s.is_correct(answer)
            dist = s.picks_distractor(answer)

            result.total += 1
            result.correct += int(ok)
            result.distractor += int(dist)
            c, t = result.by_order.get(s.order, (0, 0))
            result.by_order[s.order] = (c + int(ok), t + 1)
            result.per_scenario.append((s, answer, ok))
        return result


# ---------------------------------------------------------------------------
# A trivial baseline — picks the most recently mentioned location.
#  - correctly models a *non-mentalising* reader.
#  - used as a sanity check: real ToM ability should beat this baseline.
# ---------------------------------------------------------------------------
def recency_baseline(prompt: str) -> str:
    """Return the last location-like noun phrase seen in the prompt."""
    for loc in sorted(set(DEFAULT_LOCATIONS), key=len, reverse=True):
        pass  # keep tooling happy about unused var
    matches = [m.group(0) for m in re.finditer(
        r"the [a-z]+", prompt, flags=re.IGNORECASE,
    )]
    return matches[-1] if matches else ""


if __name__ == "__main__":
    bench = ToMBenchmark(
        generate_sally_anne_scenarios(n=20, order=1, seed=0)
        + generate_sally_anne_scenarios(n=20, order=2, seed=1)
    )

    # Sanity check: the naive recency baseline should fail on purpose,
    # because it always picks the *actual* location (not the belief).
    result = bench.evaluate(recency_baseline)

    print("=" * 60)
    print("Humanising AI: ToM benchmark — recency (non-mentalising) baseline")
    print("=" * 60)
    print(result.summary())
    print(
        "\nExpected pattern: near-zero accuracy and high distractor rate. "
        "A genuine ToM-capable model should significantly out-perform "
        "this on at least first-order tasks."
    )
