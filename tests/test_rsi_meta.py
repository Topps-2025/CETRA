"""Tests for the meta loop and meta-edits."""

from __future__ import annotations

from cetra.rsi.benchmarks import build_suite
from cetra.rsi.meta import MetaEdit, MetaRSILoop, apply_meta_edit
from cetra.rsi.policy import seed_policy


def test_meta_edit_enables_synthesis() -> None:
    policy = seed_policy()
    child = apply_meta_edit(
        policy,
        MetaEdit("ENABLE_SYNTHESIS", args={"max_arity": 3, "beam": 16}),
    )
    assert child.synthesis_enabled
    assert child.synthesis_max_arity == 3
    assert child.synthesis_beam == 16
    assert not policy.synthesis_enabled


def test_meta_loop_improves_mmp_and_learns_a_meta_rule() -> None:
    suite = build_suite(families=("threshold",), seed=20261102)
    loop = MetaRSILoop(suite, generations=1, meta_generations=1, max_candidates=2)
    report = loop.run()

    assert report.final.mmp > report.initial.mmp
    assert report.final_policy.synthesis_enabled
    assert any(
        rule.when == "synthesis_missing" and rule.then == "ENABLE_SYNTHESIS"
        for rule in report.final_policy.meta_rules
    )
    assert report.records and report.records[0].decision["accepted"]
