"""Trace sources for heterogeneous real logs.

The pipeline should not care whether the input is a StackPlanner backend log,
a per-message memory-decision JSON, or a JSONL evaluation table.  A source only
has to produce ``Event`` objects; everything downstream is shared.

Supported sources:

- ``StackPlannerSource``: the three StackPlanner logs (backend/workflow/view).
- ``SQCADLifecycleSource``: SQCAD ``real_trace_named_lifecycle`` JSON.
- ``JsonlRecordSource``: one event per JSONL record (FinPlanRAG, CETRA, ...).
- ``TextLogSource``: one event per non-empty line (unknown text logs).
- ``GenericJsonSource``: list/dict JSON records.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Protocol

from .adapter import parse_stackplanner_logs
from .types import Event


class TraceSource(Protocol):
    name: str

    def events(self) -> list[Event]: ...


def _event(
    *,
    event_id: str,
    actor: str,
    kind: str,
    content: str,
    metadata: dict[str, Any] | None = None,
    episode: str = "default",
) -> Event:
    clean = {
        str(key): str(value)
        for key, value in (metadata or {}).items()
        if value is not None
    }
    return Event(
        event_id=event_id,
        timestamp="",
        actor=actor,
        kind=kind,
        content=content[:3000],
        metadata=clean,
        span=(0, 0),
        episode=episode,
    )


@dataclass
class StackPlannerSource:
    log_dir: str | Path
    name: str = "stackplanner"

    def events(self) -> list[Event]:
        return parse_stackplanner_logs(self.log_dir)


@dataclass
class SWEAgentSource:
    """SWE-agent ``.traj`` JSON (public long-horizon coding trajectories)."""

    path: str | Path
    max_messages: int = 200
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = Path(self.path).stem

    def events(self) -> list[Event]:
        payload = json.loads(Path(self.path).read_text(encoding="utf-8"))
        history = payload.get("history", [])
        if not isinstance(history, list):
            return []
        events: list[Event] = []
        episode = str(payload.get("instance_id") or Path(self.path).stem)
        for index, message in enumerate(history[: self.max_messages]):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", ""))
            message_type = str(message.get("message_type", ""))
            content = str(message.get("content", ""))
            if role == "system":
                events.append(
                    _event(
                        event_id=f"swe:{episode}:{index}",
                        actor="system",
                        kind="observation",
                        content=content,
                        metadata={"message_type": "system_prompt"},
                        episode=episode,
                    )
                )
            elif role == "user" and message_type == "observation":
                if index <= 1 or "We're currently solving" in content:
                    events.append(
                        _event(
                            event_id=f"swe:{episode}:{index}",
                            actor="user",
                            kind="user_message",
                            content=content,
                            metadata={
                                "task_type": "coding",
                                "output_format": "patch",
                                "issue": content[:800],
                            },
                            episode=episode,
                        )
                    )
                else:
                    events.append(
                        _event(
                            event_id=f"swe:{episode}:{index}",
                            actor="tool",
                            kind="tool_result",
                            content=content,
                            metadata={"observation": content[:800]},
                            episode=episode,
                        )
                    )
            elif role == "assistant":
                match = re.search(r"```(?:bash)?\s*\n(.*?)```", content, re.S)
                if match:
                    command = next(
                        (line.strip() for line in match.group(1).splitlines() if line.strip()),
                        "",
                    )
                    action = command.split()[0] if command else "command"
                    thought = content[: match.start()].strip()
                else:
                    command = ""
                    action = "think"
                    thought = content
                events.append(
                    _event(
                        event_id=f"swe:{episode}:{index}",
                        actor="agent",
                        kind="agent_action",
                        content=content,
                        metadata={
                            "action": action,
                            "command": command[:300],
                            "thought": thought[:800],
                            "message_type": message_type,
                        },
                        episode=episode,
                    )
                )
        return events


@dataclass
class SQCADLifecycleSource:
    path: str | Path
    policy: str | None = None
    max_rows: int = 300
    name: str = "sqcad_lifecycle"

    def events(self) -> list[Event]:
        payload = json.loads(Path(self.path).read_text(encoding="utf-8"))
        rows = payload.get("rows", {})
        if not isinstance(rows, dict) or not rows:
            return []
        policy = self.policy or next(iter(rows))
        records = rows.get(policy, [])
        if not isinstance(records, list):
            return []
        events: list[Event] = []
        for index, record in enumerate(records[: self.max_rows]):
            if not isinstance(record, dict):
                continue
            action = record.get("action") or record.get("keep")
            metadata = dict(record)
            metadata["policy"] = policy
            metadata["action"] = action
            trace_id = str(record.get("trace_id", f"row-{index}"))
            msg_id = str(record.get("msg_id", index))
            events.append(
                _event(
                    event_id=f"sqcad:{policy}:{index}",
                    actor="policy",
                    kind="agent_action",
                    content=f"{trace_id}/{msg_id} action={action}",
                    metadata=metadata,
                    episode=trace_id,
                )
            )
        return events


@dataclass
class JsonlRecordSource:
    path: str | Path
    max_rows: int = 300
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = Path(self.path).stem

    def events(self) -> list[Event]:
        events: list[Event] = []
        with Path(self.path).open(encoding="utf-8", errors="replace") as handle:
            for index, line in enumerate(handle):
                if index >= self.max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                action = record.get("action") or record.get("prediction") or record.get("policy")
                episode = str(
                    record.get("trace_id")
                    or record.get("episode_id")
                    or record.get("case_id")
                    or record.get("split")
                    or "record"
                )
                events.append(
                    _event(
                        event_id=f"{self.name}:{index}",
                        actor=str(record.get("actor", "record")),
                        kind="agent_action" if action is not None else "observation",
                        content=json.dumps(record, ensure_ascii=False)[:2000],
                        metadata=record,
                        episode=episode,
                    )
                )
        return events


@dataclass
class TextLogSource:
    path: str | Path
    max_lines: int = 500
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = Path(self.path).stem

    def events(self) -> list[Event]:
        events: list[Event] = []
        text = Path(self.path).read_text(encoding="utf-8", errors="replace")
        for index, line in enumerate(text.splitlines()):
            if index >= self.max_lines:
                break
            stripped = line.strip()
            if not stripped:
                continue
            kind = "agent_action"
            action = ""
            lower = stripped.lower()
            if "user" in lower and ("message" in lower or "query" in lower):
                kind = "user_message"
            elif "error" in lower:
                kind = "tool_result"
            elif "tool" in lower or "search" in lower:
                kind = "tool_result"
            elif "observation" in lower:
                kind = "observation"
            if "action" in lower:
                action = stripped[:120]
            events.append(
                _event(
                    event_id=f"{self.name}:{index}",
                    actor="agent",
                    kind=kind,
                    content=stripped,
                    metadata={"action": action} if action else {},
                    episode=self.name,
                )
            )
        return events


@dataclass
class LLMTextSource:
    """Unknown text log: LLM proposes events, text-line source is fallback."""

    path: str | Path
    client: Any
    max_lines: int = 500
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = Path(self.path).stem

    def events(self) -> list[Event]:
        from .llm_judge import parse_events_with_llm

        raw = Path(self.path).read_text(encoding="utf-8", errors="replace")
        events = parse_events_with_llm(raw, self.name or "unknown", self.client)
        if events:
            return events
        return TextLogSource(self.path, max_lines=self.max_lines, name=self.name).events()


@dataclass
class GenericJsonSource:
    path: str | Path
    max_records: int = 300
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = Path(self.path).stem

    def _records(self, payload: Any) -> Iterable[dict[str, Any]]:
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield item
        elif isinstance(payload, dict):
            for key in ("rows", "data", "records", "cases", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            yield item
                elif isinstance(value, dict):
                    for item in value.values():
                        if isinstance(item, list):
                            for sub in item:
                                if isinstance(sub, dict):
                                    yield sub
        return

    def events(self) -> list[Event]:
        payload = json.loads(Path(self.path).read_text(encoding="utf-8"))
        events: list[Event] = []
        for index, record in enumerate(self._records(payload)):
            if index >= self.max_records:
                break
            action = record.get("action") or record.get("prediction") or record.get("policy")
            episode = str(record.get("trace_id") or record.get("case_id") or self.name)
            events.append(
                _event(
                    event_id=f"{self.name}:{index}",
                    actor="record",
                    kind="agent_action" if action is not None else "observation",
                    content=json.dumps(record, ensure_ascii=False)[:2000],
                    metadata=record,
                    episode=episode,
                )
            )
        return events


@dataclass
class DirectorySource:
    """A directory of heterogeneous logs: concatenate each supported file."""

    path: str | Path
    max_records: int = 300
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = Path(self.path).name

    def events(self) -> list[Event]:
        events: list[Event] = []
        for file in sorted(Path(self.path).iterdir()):
            if not file.is_file():
                continue
            try:
                source = auto_source(file, max_records=self.max_records)
            except (ValueError, OSError, json.JSONDecodeError):
                continue
            for event in source.events():
                events.append(
                    replace(
                        event,
                        event_id=f"{file.stem}:{event.event_id}",
                        episode=f"{file.stem}:{event.episode}",
                    )
                )
                if len(events) >= self.max_records:
                    return events
        return events


def auto_source(
    path: str | Path,
    *,
    max_records: int = 300,
    llm_client: Any | None = None,
) -> TraceSource:
    """Pick a source adapter from the file shape."""

    path = Path(path)
    if path.is_dir():
        if any(
            (path / name).exists()
            for name in ("log.txt", "log_test.txt", "log_view.md")
        ):
            return StackPlannerSource(path)
        return DirectorySource(path, max_records=max_records)
    suffix = path.suffix.lower()
    if suffix == ".traj":
        return SWEAgentSource(path, max_messages=max_records)
    if suffix == ".jsonl":
        return JsonlRecordSource(path, max_rows=max_records)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("rows"), dict):
            return SQCADLifecycleSource(path, max_rows=max_records)
        return GenericJsonSource(path, max_records=max_records)
    if suffix in {".md", ".txt", ".log"}:
        if llm_client is not None:
            return LLMTextSource(path, llm_client, max_lines=max_records)
        return TextLogSource(path, max_lines=max_records)
    raise ValueError(f"unsupported trace format: {path.suffix}")
