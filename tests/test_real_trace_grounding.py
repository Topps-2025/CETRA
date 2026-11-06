"""Tests for the real-trace landing points: parser, tree learner, skills, gate."""

from __future__ import annotations

from pathlib import Path

from cetra.rsi.gate import accept_edit
from cetra.rsi.induction import build_chains, induce_decision_tree
from cetra.rsi.ontology import StructuredEntityDecomposer, flatten_entities
from cetra.rsi.skill import SkillRegistry, SkillRuntime, artifact_from_rule
from cetra.rsi.trace import StructuredTraceParser, TraceEvent
from cetra.rsi.types import CaseRecord, Metrics, Rule


def case(case_id: str, factors: dict[str, str], action: str) -> CaseRecord:
    return CaseRecord(
        case_id=case_id,
        split="train",
        narrative=f"case {case_id}",
        factors=factors,
        action=action,
        family="test",
    )


def test_structured_trace_parser_keeps_factors_and_surface():
    events = [
        TraceEvent("ctx", "observation", "env", "user asks for a speech; length missing"),
        TraceEvent(
            "act",
            "agent_action",
            "perception",
            "ask clarification",
            {
                "action": "ask_clarification",
                "missing_slots": "yes",
                "user_feedback": "none",
                "thread": "t-1",
                "request_id": "r-1",
            },
        ),
        TraceEvent("out", "observation", "env", "DST questions generated"),
    ]
    points = StructuredTraceParser().parse(events)
    assert len(points) == 1
    point = points[0]
    assert point.action == "ask_clarification"
    assert point.factors == {"missing_slots": "yes", "user_feedback": "none"}
    assert point.surface == {"thread": "t-1", "request_id": "r-1"}
    assert point.outcome == "DST questions generated"
    assert point.evidence == ("ctx", "act", "out")


def test_entity_decomposition_separates_components_and_attributes():
    payload = {
        "components": {
            "shell": {"texture": "crisp", "color": "pink"},
            "filling": {"flavor": "rose"},
        },
        "attributes": {"price": "8", "origin": "Paris"},
        "evidence": ["menu:item-42"],
    }
    node = StructuredEntityDecomposer().decompose("m1", "macaron", payload)
    flat = node.flatten_attributes()
    assert flat["shell.texture"] == "crisp"
    assert flat["filling.flavor"] == "rose"
    assert flat["price"] == "8"
    assert node.components[0].name == "shell"
    assert flatten_entities([node])["m1.price"] == "8"


def test_decision_tree_learns_conjunctions():
    cases = [
        case("1", {"missing_slots": "yes", "user_feedback": "none"}, "ask_clarification"),
        case("2", {"missing_slots": "yes", "user_feedback": "none"}, "ask_clarification"),
        case("3", {"missing_slots": "yes", "user_feedback": "skip"}, "proceed"),
        case("4", {"missing_slots": "yes", "user_feedback": "skip"}, "proceed"),
        case("5", {"missing_slots": "no", "has_enough_context": "no"}, "research"),
        case("6", {"missing_slots": "no", "has_enough_context": "no"}, "research"),
        case("7", {"missing_slots": "no", "has_enough_context": "yes"}, "write_report"),
        case("8", {"missing_slots": "no", "has_enough_context": "yes"}, "write_report"),
    ]
    rules = induce_decision_tree(cases, max_depth=3, min_support=2, min_precision=0.8)
    assert rules
    assert any(len(rule.conditions) >= 2 for rule in rules)
    assert any(rule.action == "ask_clarification" for rule in rules)


def test_chain_builder_connects_effect_to_precondition():
    rules = [
        Rule(conditions=(("missing_slots", "yes"),), action="ask_clarification", support=2, precision=1.0),
        Rule(conditions=(("missing_slots", "no"), ("has_enough_context", "no")), action="research", support=2, precision=1.0),
    ]
    chains = build_chains(rules)
    assert any(chain.actions == ("ask_clarification", "research") for chain in chains)


def test_skill_artifact_has_four_pieces_and_retrieval(tmp_path: Path):
    rule = Rule(
        conditions=(("missing_slots", "yes"), ("user_feedback", "none")),
        action="ask_clarification",
        support=2,
        precision=1.0,
    )
    validation = [
        case("v1", {"missing_slots": "yes", "user_feedback": "none"}, "ask_clarification"),
        case("v2", {"missing_slots": "yes", "user_feedback": "none"}, "ask_clarification"),
    ]
    artifact = artifact_from_rule(
        rule,
        source_family="trace",
        validation_cases=validation,
    )
    assert artifact.contract.preconditions == rule.conditions
    assert "ask_clarification" in artifact.handler.source
    assert artifact.evidence.validation_cases == ("v1", "v2")

    registry = SkillRegistry()
    registry.add(artifact)
    assert registry.retrieve({"missing_slots": "yes", "user_feedback": "none"})
    assert not registry.retrieve({"missing_slots": "no", "user_feedback": "none"})

    runtime = SkillRuntime(registry)
    action, skill_id = runtime.predict(
        case("q1", {"missing_slots": "yes", "user_feedback": "none"}, "ask_clarification")
    )
    assert action == "ask_clarification"
    assert skill_id == artifact.skill_id

    directory = artifact.write_dir(tmp_path)
    assert (directory / "SKILL.md").exists()
    assert (directory / "handler.py").exists()
    assert (directory / "evidence.json").exists()
    assert (directory / "provenance.json").exists()

    assert registry.retire(artifact.skill_id)
    assert not registry.retrieve({"missing_slots": "yes", "user_feedback": "none"})
    action, skill_id = runtime.predict(
        case("q2", {"missing_slots": "yes", "user_feedback": "none"}, "ask_clarification")
    )
    assert action is None and skill_id is None


def test_strict_gate_rejects_zero_metaproductivity():
    parent = {
        "iid": Metrics(n=2, correct=1, risk=0.5, cost=1.0, mp=0.0),
        "test": Metrics(n=2, correct=1, risk=0.5, cost=1.0, mp=0.0),
    }
    child = {
        "iid": Metrics(n=2, correct=1, risk=0.5, cost=1.0, mp=0.0),
        "test": Metrics(n=2, correct=1, risk=0.5, cost=1.0, mp=0.0),
    }
    decision = accept_edit(
        parent,
        child,
        [],
        [],
        gate_split="iid",
        mp_split="iid",
        require_mp=True,
    )
    assert not decision.accepted
    assert "no metaproductivity improvement" in decision.reason
