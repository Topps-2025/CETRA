"""Tests for DecompIR, the LLM decomposer, and joint decomposition/SE."""

from __future__ import annotations

import itertools
import random
from dataclasses import replace

import pytest

from cetra.memory.decomp_evolve import run_joint_se
from cetra.memory.decomp_program import (
    CompileError,
    DecompositionProgram,
    apply_program,
    apply_programs,
    compile_program,
)
from cetra.memory.llm_decomposer import HeuristicDecomposer, LLMDecomposer, propose_candidates
from cetra.rsi.types import CaseRecord


def _case(case_id: str, raw: str, action: str, split: str = "fit") -> CaseRecord:
    return CaseRecord(
        case_id=case_id,
        split=split,
        narrative=raw,
        factors={"raw_context": raw},
        action=action,
    )


class _FakeClient:
    available = True

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def chat_json(self, system, user, **kwargs):
        self.calls += 1
        return self.payload


def test_compile_and_apply_key_value_program() -> None:
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
    cases = [_case("a", "task_type=qa;has_enough_context=yes", "answer")]
    rebuilt, report = apply_program(program, cases)
    assert rebuilt[0].factors["has_enough_context"] == "yes"
    assert report.coverage["has_enough_context"] == 1.0
    assert report.mean_coverage() == 1.0


def test_staged_programs_read_earlier_factors() -> None:
    stage1 = compile_program(
        {"factors": [{"name": "command", "extractor": "identity"}]},
        default_source="raw_context",
    )
    stage2 = compile_program(
        {
            "factors": [
                {
                    "name": "tool",
                    "source": "command",
                    "extractor": "token",
                    "config": {"index": 0},
                },
                {
                    "name": "arguments",
                    "source": "command",
                    "extractor": "regex",
                    "config": {"pattern": r"^\S+\s+(.*)$", "group": 1},
                },
            ]
        },
        default_source="command",
    )
    cases = [_case("a", "python test.py", "run")]
    rebuilt, reports = apply_programs([stage1, stage2], cases)
    assert rebuilt[0].factors["command"] == "python test.py"
    assert rebuilt[0].factors["tool"] == "python"
    assert rebuilt[0].factors["arguments"] == "test.py"
    assert len(reports) == 2


def test_compile_rejects_reserved_and_bad_regex() -> None:
    with pytest.raises(CompileError):
        compile_program(
            {"factors": [{"name": "action", "extractor": "identity"}]},
            default_source="raw",
        )
    with pytest.raises(CompileError):
        compile_program(
            {
                "factors": [
                    {
                        "name": "bad",
                        "extractor": "regex",
                        "config": {"pattern": "(", "group": 1},
                    }
                ]
            },
            default_source="raw",
        )


def test_program_restrict_is_versioned() -> None:
    program = compile_program(
        {
            "factors": [
                {"name": "a", "extractor": "key_value", "config": {"key": "a"}},
                {"name": "b", "extractor": "key_value", "config": {"key": "b"}},
            ]
        },
        default_source="raw",
    )
    child = program.restrict(["b"], rationale="drop a")
    assert child.factor_names() == ("b",)
    assert child.version == program.version + 1
    assert child.parent == program.program_id
    assert child.metadata["edit"] == "SELECT_FACTORS"


def test_heuristic_proposes_packed_factors() -> None:
    cases = [
        _case("a", "task_type=qa;has_enough_context=yes", "answer"),
        _case("b", "task_type=qa;has_enough_context=no", "research"),
        _case("c", "task_type=report;has_enough_context=yes", "write_report"),
    ]
    program = HeuristicDecomposer().propose(cases, source_field="raw_context")
    assert program is not None
    assert {"task_type", "has_enough_context"} <= set(program.factor_names())


def test_llm_extractor_uses_typed_backend() -> None:
    program = compile_program(
        {
            "factors": [
                {
                    "name": "user_feedback",
                    "extractor": "llm",
                    "config": {
                        "instruction": "none, skip, or revise",
                        "allowed_values": ["none", "skip", "revise"],
                    },
                }
            ]
        },
        default_source="narrative",
    )
    cases = [
        CaseRecord("a", "fit", "user asks to revise", {}, "replan"),
        CaseRecord("b", "fit", "fresh request", {}, "answer"),
    ]
    client = _FakeClient({"values": {"a": {"user_feedback": "revise"}}})
    rebuilt, report = apply_program(program, cases, client=client)
    assert rebuilt[0].factors["user_feedback"] == "revise"
    assert rebuilt[1].factors["user_feedback"] == "__missing__"
    assert report.coverage["user_feedback"] == 0.5
    assert report.llm_calls == 1


def test_llm_revision_compiles_with_parent() -> None:
    current = compile_program(
        {"factors": [{"name": "service", "extractor": "identity"}]},
        default_source="narrative",
    )
    client = _FakeClient(
        {
            "rationale": "revision requests were missing",
            "factors": [
                {
                    "name": "revision_request",
                    "extractor": "llm",
                    "config": {
                        "instruction": "whether the user asks to revise",
                        "allowed_values": ["true", "false"],
                    },
                }
            ],
        }
    )
    revised = LLMDecomposer(client).revise(
        current,
        [],
        [{"case_id": "x", "raw": "please revise", "action": "replan", "predicted": "answer"}],
        source_field="narrative",
    )
    assert revised is not None
    assert revised.parent == current.program_id
    assert revised.version == current.version + 1
    assert revised.factor_names() == ("revision_request",)


def test_joint_se_accepts_informative_decomposition() -> None:
    def action(factors: dict[str, str]) -> str:
        if factors["user_feedback"] == "revise" and factors["has_enough_context"] == "yes":
            return "replan"
        if factors["has_enough_context"] == "no":
            return "research"
        if factors["task_type"] in {"speech", "report"}:
            return "write_report"
        return "answer"

    rows: list[CaseRecord] = []
    for index, (task, enough, feedback, search) in enumerate(
        itertools.product(
            ["speech", "qa", "report"],
            ["yes", "no"],
            ["none", "skip", "revise"],
            ["yes", "no"],
        )
    ):
        factors = {
            "task_type": task,
            "has_enough_context": enough,
            "user_feedback": feedback,
            "needs_search": search,
        }
        raw = ";".join(f"{key}={value}" for key, value in factors.items())
        rows.append(_case(f"S{index}", raw, action(factors)))
    random.Random(0).shuffle(rows)
    rows = [
        replace(case, split="fit" if index < 20 else ("dev" if index < 28 else "test"))
        for index, case in enumerate(rows)
    ]
    fit = [case for case in rows if case.split == "fit"]
    dev = [case for case in rows if case.split == "dev"]
    test = [case for case in rows if case.split == "test"]

    candidates = propose_candidates(fit, source_field="raw_context", client=None)
    result = run_joint_se(
        fit=fit,
        dev=dev,
        test=test,
        candidates=candidates,
        source_field="raw_context",
        operator="induce_tree",
        max_fields=3,
        beam=6,
    )
    assert result["accepted"]
    assert result["final"]["test"]["risk"] < result["baseline"]["test"]["risk"]
    assert result["final"]["fields"]
