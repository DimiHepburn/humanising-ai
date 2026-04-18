"""
Theory of Mind sub-package
===========================

Tools for **probing and evaluating** theory-of-mind (ToM) abilities
in language models.

Contents
--------
- ``tom_benchmark.py``         : generates Sally-Anne-style false-belief
                                 tests at configurable order (1st, 2nd, 3rd)
                                 and evaluates any callable model on them.
- ``belief_state_probing.py``  : lightweight probing utilities for
                                 analysing whether a model's internal
                                 (or output) representations reflect the
                                 belief state required by a scenario.

Both modules are dependency-light by default — they work with any
callable that returns free-form text or a list of token/label
probabilities, so they're usable with local models, API-based
models, or purely rule-based baselines.
"""

from .tom_benchmark import (
    BeliefScenario,
    ToMBenchmark,
    BenchmarkResult,
    generate_sally_anne_scenarios,
)
from .belief_state_probing import (
    BeliefStateProbe,
    LinearProbe,
    logistic_fit,
)

__all__ = [
    "BeliefScenario",
    "ToMBenchmark",
    "BenchmarkResult",
    "generate_sally_anne_scenarios",
    "BeliefStateProbe",
    "LinearProbe",
    "logistic_fit",
]
