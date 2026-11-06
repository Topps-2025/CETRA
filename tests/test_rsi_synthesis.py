"""Tests for open operator synthesis and program operators."""

from __future__ import annotations

from cetra.rsi.benchmarks import build_suite
from cetra.rsi.harness import make_seed_harness
from cetra.rsi.loop import CETRARSILoop
from cetra.rsi.policy import seed_policy
from cetra.rsi.synthesis import Condition, ProgramOperator, RuleProgram
from cetra.rsi.types import CaseRecord, Edit


def _cases() -> list[CaseRecord]:
    rows: list[CaseRecord] = []
    actions = {
        ("0", "a"): "low",
        ("0", "b"): "low",
        ("1", "a"): "low",
        ("1", "b"): "low",
        ("2", "a"): "high",
        ("2", "b"): "low",
        ("3", "a"): "high",
        ("3", "b"): "low",
    }
    for index, ((x, y), action) in enumerate(actions.items()):
        rows.append(
            CaseRecord(
                case_id=f"c{index}",
                split="train",
                narrative=f"x={x}, y={y}",
                factors={"x": x, "y": y},
                action=action,
            )
        )
    return rows


def test_program_operator_adds_threshold_rule() -> None:
    cases = _cases()
    program = RuleProgram(
        conditions=(
            Condition("x", "ge", 2),
            Condition("y", "eq", "a"),
        ),
        action="high",
    )
    operator = ProgramOperator(programs=(program,))
    state = operator.run(type("S", (), {"rules": [], "traces": {}})(), cases, {})
    assert len(state.rules) == 1
    rule = state.rules[0]
    assert rule.thresholds == (("x", 2),)
    assert rule.conditions == (("y", "a"),)
    assert rule.action == "high"


def test_synthesis_accepts_a_new_operator_on_a_threshold_gap() -> None:
    ds = build_suite(families=("threshold",), seed=20261102)["threshold"].meta_train
    harness = make_seed_harness(concept_pool=("authority", "retry_count", "retry_safety", "resource_state", "commit_state"))

    policy = seed_policy()
    policy.synthesis_enabled = True
    policy.synthesis_max_arity = 3
    policy.synthesis_beam = 12
    policy.max_proposals = 6
    loop = CETRARSILoop(ds, seed_harness=harness, policy=policy)
    report = loop.run(generations=1)

    accepted = [record for record in report.edit_log if record.accepted and record.synthesis]
    assert accepted, "synthesis should find a program for the threshold family"
    program = accepted[0].synthesis["program"]
    assert any(
        condition["op"] == "ge"
        for rule in program["programs"]
        for condition in rule["conditions"]
    )
    assert report.best_metrics["test"].risk < report.seed_metrics["test"].risk


def test_add_operator_accepts_program_payload() -> None:
    cases = _cases()
    program = RuleProgram(
        conditions=(Condition("x", "ge", 2), Condition("y", "eq", "a")),
        action="high",
    )
    payload = ProgramOperator(programs=(program,)).to_dict()
    harness = make_seed_harness(concept_pool=("x", "y"))
    from cetra.rsi.edits import apply_edit_bundle

    child = apply_edit_bundle(
        harness,
        [
            Edit("ADD_OPERATOR", "synth_test", {"program": payload}),
            Edit("REWIRE", "synth_test", {"before": "predict"}),
        ],
        "h_synth",
    )
    assert "synth_test" in child.operators
    rep = child.build(cases)
    assert any(rule.action == "high" for rule in rep.rules)
    assert child.predict(rep, cases[4]) == "high"
