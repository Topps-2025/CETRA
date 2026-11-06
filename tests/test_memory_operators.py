"""Tests for the schema-driven operator workflow."""

from __future__ import annotations

import json
from pathlib import Path

from cetra.memory.decomp_program import compile_program
from cetra.memory.fields import FieldAtom, FieldValue
from cetra.memory.need import NeedEstimator
from cetra.memory.operators_generic import (
    ComposeOperator,
    ConsolidateOperator,
    DecomposeOperator,
    ProgramDecomposeOperator,
    ProjectOperator,
    Workflow,
    default_workflow,
    run_events,
    segment_to_field_atoms,
)
from cetra.memory.sources import SWEAgentSource
from cetra.memory.types import Event


def _atom(atom_id: str, action: str, **fields) -> FieldAtom:
    data = {"action": FieldValue("action", action)}
    for key, value in fields.items():
        data[key] = FieldValue(key, value)
    return FieldAtom(atom_id=atom_id, schema="decision", fields=data, scope="demo")


def test_need_estimator_accepts_informative_split() -> None:
    estimator = NeedEstimator(max_depth=3)
    score = estimator.score(
        "metadata",
        ["phase=plan", "phase=plan", "phase=execute", "phase=execute"],
        {
            "phase": ["plan", "plan", "execute", "execute"],
            "tool": ["plan", "plan", "edit", "python"],
        },
        action_values=["plan", "plan", "edit", "python"],
    )
    assert score.accepted
    assert score.gain > 0


def test_recursive_decompose_adds_child_fields() -> None:
    atoms = [
        _atom("a1", "plan", metadata="phase=plan;tool=plan"),
        _atom("a2", "plan", metadata="phase=plan;tool=plan"),
        _atom("a3", "edit", metadata="phase=execute;tool=edit"),
        _atom("a4", "python", metadata="phase=execute;tool=python"),
    ]
    result = DecomposeOperator(fields={"metadata"}, max_depth=2).apply(atoms)
    assert result.metrics["decompose_accepted"] >= 1
    assert result.metrics["fields_added"] >= 4
    assert "phase" in atoms[0].fields
    assert "tool" in atoms[0].fields


def test_project_and_compose_and_consolidate() -> None:
    atoms = [
        _atom("a1", "keep", has_context="no", needs_search="yes"),
        _atom("a2", "keep", has_context="no", needs_search="no"),
        _atom("a3", "archive", has_context="yes", needs_search="yes"),
        _atom("a4", "archive", has_context="yes", needs_search="no"),
        _atom("a5", "keep", has_context="no", needs_search="yes"),
        _atom("a6", "archive", has_context="yes", needs_search="yes"),
    ]
    workflow = Workflow(
        "test",
        [
            ProjectOperator(top_k=4),
            ComposeOperator(),
        ],
    )
    result = workflow.run(atoms)
    assert any(atom.schema == "chain" for atom in result.atoms)
    assert result.metrics["operators"][0]["name"] == "project"
    rule_atom = FieldAtom(
        atom_id="rule:test",
        schema="rule",
        fields={
            "conditions": FieldValue("conditions", [["has_context", "no"]]),
            "action": FieldValue("action", "keep"),
            "scope": FieldValue("scope", []),
            "support": FieldValue("support", 3),
            "precision": FieldValue("precision", 1.0),
        },
    )
    consolidated = ConsolidateOperator(min_support=1, min_precision=0.5).apply([rule_atom])
    assert any(atom.schema == "skill" for atom in consolidated.atoms)


def test_segment_to_field_atoms_keeps_decision_points() -> None:
    events = [
        Event("e1", "t1", "user", "user_message", "run the tests"),
        Event(
            "e2",
            "t2",
            "agent",
            "agent_action",
            "running pytest",
            metadata={"action": "run", "command": "pytest -q"},
        ),
        Event("e3", "t3", "tool", "tool_result", "12 passed"),
    ]
    atoms = segment_to_field_atoms(events)
    assert atoms
    assert any(atom.schema == "decision" for atom in atoms)
    assert sum(len(atom.evidence) for atom in atoms) == len(events)
    assert len(atoms) <= len(events)
    assert any(atom.get("command") == "pytest -q" for atom in atoms)


def test_program_decompose_operator_adds_fields() -> None:
    program = compile_program(
        {
            "factors": [
                {
                    "name": "has_enough_context",
                    "extractor": "key_value",
                    "config": {"key": "has_enough_context"},
                }
            ]
        },
        default_source="raw_context",
    )
    atom = FieldAtom(
        atom_id="a1",
        schema="decision",
        fields={
            "action": FieldValue("action", "answer"),
            "raw_context": FieldValue(
                "raw_context", "task_type=qa;has_enough_context=yes"
            ),
        },
    )
    result = ProgramDecomposeOperator(program).apply([atom])
    assert atom.get("has_enough_context") == "yes"
    assert result.metrics["fields_added"] == 1
    assert result.metrics["coverage"] == 1.0


def test_sweagent_source_fixture(tmp_path: Path) -> None:
    path = tmp_path / "demo.traj"
    payload = {
        "instance_id": "demo-1",
        "history": [
            {"role": "system", "message_type": "system_prompt", "content": "system"},
            {
                "role": "user",
                "message_type": "observation",
                "content": "We're currently solving the following issue. ISSUE: x",
            },
            {
                "role": "assistant",
                "message_type": "action",
                "content": "Let's run it.\n```\npython test.py\n```",
            },
            {
                "role": "user",
                "message_type": "observation",
                "content": "[File: test.py (3 lines total)]\n1:a\n2:b\n3:c",
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    events = SWEAgentSource(path).events()
    assert len(events) == 4
    assert events[1].kind == "user_message"
    assert events[2].kind == "agent_action"
    assert events[2].metadata["action"] == "python"
