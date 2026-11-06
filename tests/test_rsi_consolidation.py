"""Tests for experience-to-skill consolidation."""

from __future__ import annotations

from cetra.rsi.consolidation import (
    apply_library,
    consolidate,
    policy_c0,
    policy_c1,
    transfer_metrics,
)
from cetra.rsi.harness import make_seed_harness
from cetra.rsi.types import CaseRecord, Representation, Rule


def _cases() -> list[CaseRecord]:
    rows: list[CaseRecord] = []
    for index, (x, y) in enumerate(
        [("0", "a"), ("1", "a"), ("2", "a"), ("2", "b"), ("3", "a"), ("3", "b")]
    ):
        rows.append(
            CaseRecord(
                case_id=f"c{index}",
                split="test",
                narrative=f"x={x}, y={y}",
                factors={"x": x, "y": y},
                action="high" if x in {"2", "3"} and y == "a" else "low",
            )
        )
    return rows


def test_consolidation_promotes_stable_rules_and_transfers() -> None:
    cases = _cases()
    rep = Representation(
        rules=[
            Rule(
                conditions=(("y", "a"),),
                thresholds=(("x", 2),),
                action="high",
                support=2,
                precision=1.0,
            ),
            Rule(conditions=(("y", "b"),), action="low", support=2, precision=1.0),
            # Spurious on the source split: fails cross-validation on x=2,b.
            Rule(conditions=(("x", "2"),), action="high", support=2, precision=1.0),
        ]
    )
    loose = consolidate(rep, policy_c0(), source_family="src")
    strict = consolidate(rep, policy_c1(), source_family="src", validation_cases=cases)

    assert 0 < len(strict.skills) <= len(loose.skills)
    library_rep = apply_library(strict)
    assert len(library_rep.rules) == len(strict.skills)

    harness = make_seed_harness(concept_pool=("x", "y"))
    metrics = transfer_metrics(harness, strict, cases, source_rule_count=len(rep.rules))
    assert metrics["n"] == len(cases)
    assert metrics["coverage"] > 0.0
    assert metrics["transfer_gain"] >= 0.0
