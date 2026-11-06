"""Engineering tests for the format-agnostic memory pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cetra.memory.decompose import OpenMetadataJudge
from cetra.memory.pipeline import run_source
from cetra.memory.sources import (
    JsonlRecordSource,
    LLMTextSource,
    SQCADLifecycleSource,
    TextLogSource,
    auto_source,
)


def test_jsonl_record_source_runs_end_to_end(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    rows = [
        {"trace_id": "t1", "action": "keep", "has_enough_context": "no", "needs_search": "yes"},
        {"trace_id": "t1", "action": "keep", "has_enough_context": "no", "needs_search": "no"},
        {"trace_id": "t2", "action": "archive", "has_enough_context": "yes", "needs_search": "yes"},
        {"trace_id": "t2", "action": "archive", "has_enough_context": "yes", "needs_search": "no"},
        {"trace_id": "t3", "action": "keep", "has_enough_context": "no", "needs_search": "yes"},
        {"trace_id": "t3", "action": "archive", "has_enough_context": "yes", "needs_search": "yes"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    source = JsonlRecordSource(path)
    run = run_source(source, judge=OpenMetadataJudge())
    assert run.errors == []
    assert run.events == 6
    assert run.atoms == 6
    assert run.graph["atoms"] == 6
    assert "action" in run.attribute_keys
    assert run.mean_grounding >= 0.9


def test_sqcad_like_json_source(tmp_path: Path) -> None:
    path = tmp_path / "sqcad.json"
    payload = {
        "rows": {
            "policy": [
                {"trace_id": "conv-1", "msg_id": "m1", "action": "keep", "full_value": 1.0, "keep": True},
                {"trace_id": "conv-1", "msg_id": "m2", "action": "archive", "full_value": 0.0, "keep": False},
                {"trace_id": "conv-1", "msg_id": "m3", "action": "keep", "full_value": 2.0, "keep": True},
            ]
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    source = SQCADLifecycleSource(path)
    run = run_source(source, judge=OpenMetadataJudge())
    assert run.errors == []
    assert run.events == 3
    assert run.atoms == 3
    assert "full_value" in run.attribute_keys


def test_text_log_source(tmp_path: Path) -> None:
    path = tmp_path / "run.log"
    path.write_text(
        "User query: what is AI agent?\n"
        "action: search\n"
        "observation: three pages returned\n",
        encoding="utf-8",
    )
    source = TextLogSource(path)
    run = run_source(source, judge=OpenMetadataJudge())
    assert run.errors == []
    assert run.events == 3
    assert run.atoms == 3


def test_llm_source_falls_back_without_llm(tmp_path: Path) -> None:
    class FakeClient:
        def chat_json(self, *args, **kwargs):
            return {"events": []}

    path = tmp_path / "unknown.txt"
    path.write_text("line one\nline two\n", encoding="utf-8")
    source = LLMTextSource(path, FakeClient())
    run = run_source(source, judge=OpenMetadataJudge())
    assert run.events == 2
    assert run.atoms == 2


def test_empty_source_returns_error_not_crash(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    run = run_source(JsonlRecordSource(path))
    assert run.events == 0
    assert run.errors


def test_auto_source_rejects_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        auto_source(path)


def test_directory_source_concatenates_supported_files(tmp_path: Path) -> None:
    (tmp_path / "a.log").write_text("action: search\nobservation: ok\n", encoding="utf-8")
    (tmp_path / "b.jsonl").write_text(
        json.dumps({"trace_id": "t", "action": "keep"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "ignore.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    run = run_source(auto_source(tmp_path), judge=OpenMetadataJudge())
    assert run.errors == []
    assert run.events >= 3
    assert run.atoms >= 3


def test_stackplanner_real_logs_if_present() -> None:
    log_dir = (
        Path(__file__).resolve().parents[1].parent
        / "PKU RA"
        / "项目代码"
        / "StackPlanner"
        / "代码仓库"
    )
    if not log_dir.exists():
        pytest.skip("StackPlanner logs are not available")
    run = run_source(auto_source(log_dir))
    assert run.errors == []
    assert run.events >= 18
    assert run.atoms >= 18
