"""Adapters from real StackPlanner logs to normalized events.

The two logs have different formats:

- ``log.txt`` is the backend log: one long line per log record, with
  ``request param details``, DST questions, and an interrupt.
- ``log_test.txt`` is the agent workflow log: human message, background
  investigation, planner response, reporter response.

The adapter is deliberately conservative.  It only emits an event when the
log line contains a clear marker; it never invents an action from free text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .types import Event


_TS = re.compile(r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)")


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _timestamp(line: str, fallback: str) -> str:
    match = _TS.match(line)
    return match.group(1) if match else fallback


def _event(
    events: list[Event],
    *,
    event_id: str,
    timestamp: str,
    actor: str,
    kind: str,
    content: str,
    metadata: dict[str, str] | None = None,
    span: tuple[int, int] = (0, 0),
    episode: str = "default",
) -> None:
    events.append(
        Event(
            event_id=event_id,
            timestamp=timestamp,
            actor=actor,
            kind=kind,
            content=content,
            metadata=dict(metadata or {}),
            span=span,
            episode=episode,
        )
    )


def _parse_backend_log(path: Path) -> list[Event]:
    """Parse the StackPlanner backend log (log.txt)."""

    lines = _read_text(path).splitlines()
    events: list[Event] = []
    first_request_seen = False
    skip_seen = False
    dst_seen = False

    for index, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        ts = _timestamp(line, f"line-{index}")

        if "request param details" in line:
            content_match = re.search(r"content='(.*?)'\)\]", line, re.S)
            thread_match = re.search(r"thread_id='([^']*)'", line)
            feedback_match = re.search(r"interrupt_feedback='([^']*)'", line)
            content = content_match.group(1) if content_match else ""
            thread = thread_match.group(1) if thread_match else ""
            feedback = feedback_match.group(1) if feedback_match else ""
            if not first_request_seen:
                first_request_seen = True
                _event(
                    events,
                    event_id=f"log:{index}:user",
                    timestamp=ts,
                    actor="user",
                    kind="user_message",
                    content=content,
                    metadata={
                        "thread_id": thread,
                        "task_type": "speech" if "讲话稿" in content else "qa",
                        "missing_slots": "yes" if "讲话稿" in content else "no",
                        "output_format": "speech" if "讲话稿" in content else "answer",
                    },
                    span=(index, index),
                    episode="log.txt",
                )
            if feedback.strip() == "[SKIP]" and not skip_seen:
                skip_seen = True
                _event(
                    events,
                    event_id=f"log:{index}:skip",
                    timestamp=ts,
                    actor="user",
                    kind="user_feedback",
                    content="[SKIP]",
                    metadata={"user_feedback": "skip", "thread_id": thread},
                    span=(index, index),
                    episode="log.txt",
                )
                _event(
                    events,
                    event_id=f"log:{index}:proceed",
                    timestamp=ts,
                    actor="agent",
                    kind="agent_action",
                    content="proceed after user skip",
                    metadata={"action": "proceed"},
                    span=(index, index),
                    episode="log.txt",
                )
            continue

        if ("DST问题" in line or "Fill the Question" in line) and not dst_seen:
            dst_seen = True
            _event(
                events,
                event_id=f"log:{index}:dst",
                timestamp=ts,
                actor="agent",
                kind="agent_action",
                content="ask clarification questions (DST)",
                metadata={
                    "action": "ask_clarification",
                    "missing_slots": "yes",
                    "needs_clarification": "yes",
                },
                span=(index, index),
                episode="log.txt",
            )
            _event(
                events,
                event_id=f"log:{index}:wait",
                timestamp=ts,
                actor="environment",
                kind="state_change",
                content="waiting for user feedback",
                metadata={"state": "waiting_for_user"},
                span=(index, index),
                episode="log.txt",
            )
            continue

    return events


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def _parse_workflow_log(path: Path) -> list[Event]:
    """Parse the agent workflow log (log_test.txt)."""

    lines = _read_text(path).splitlines()
    events: list[Event] = []
    section = ""
    in_sync = False
    sync_count = 0
    planner_buffer: list[str] | None = None

    for index, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        ts = _timestamp(line, f"line-{index}")

        if "Human Message" in line:
            section = "human"
            continue
        if section == "human" and line.strip() and not line.startswith("="):
            _event(
                events,
                event_id=f"test:{index}:user",
                timestamp=ts,
                actor="user",
                kind="user_message",
                content=stripped,
                metadata={"task_type": "qa", "missing_slots": "no", "output_format": "answer"},
                span=(index, index),
                episode="log_test.txt",
            )
            section = ""
            continue

        if "background investigation node is running" in line:
            _event(
                events,
                event_id=f"test:{index}:research",
                timestamp=ts,
                actor="agent",
                kind="agent_action",
                content="background investigation",
                metadata={"action": "research", "needs_search": "yes"},
                span=(index, index),
                episode="log_test.txt",
            )
            continue

        if stripped.startswith("sync ["):
            in_sync = True
            sync_count = 0
            continue
        if in_sync:
            if stripped == "]":
                in_sync = False
                _event(
                    events,
                    event_id=f"test:{index}:search_result",
                    timestamp=ts,
                    actor="tool",
                    kind="tool_result",
                    content=f"search returned {sync_count} pages",
                    metadata={
                        "source_count": str(sync_count),
                        "has_enough_context": "yes" if sync_count >= 2 else "no",
                    },
                    span=(index, index),
                    episode="log_test.txt",
                )
            elif '"title"' in stripped:
                sync_count += 1
            continue

        if "Planner response: {" in line:
            planner_buffer = [line[line.find("{") :]]
            continue
        if planner_buffer is not None:
            planner_buffer.append(line)
            if line.strip() == "}":
                payload = _extract_json_object("\n".join(planner_buffer))
                enough = payload.get("has_enough_context")
                _event(
                    events,
                    event_id=f"test:{index}:plan",
                    timestamp=ts,
                    actor="agent",
                    kind="agent_action",
                    content="planner response",
                    metadata={
                        "action": "plan",
                        "has_enough_context": "yes" if enough else "no",
                    },
                    span=(index - len(planner_buffer) + 1, index),
                    episode="log_test.txt",
                )
                if enough:
                    _event(
                        events,
                        event_id=f"test:{index}:enough",
                        timestamp=ts,
                        actor="environment",
                        kind="observation",
                        content="planner reports enough context",
                        metadata={"has_enough_context": "yes"},
                        span=(index, index),
                        episode="log_test.txt",
                    )
                planner_buffer = None
            continue

        if "Reporter write final report" in line:
            _event(
                events,
                event_id=f"test:{index}:report",
                timestamp=ts,
                actor="agent",
                kind="agent_action",
                content="write final report",
                metadata={"action": "write_report", "has_enough_context": "yes"},
                span=(index, index),
                episode="log_test.txt",
            )
            continue

        if "reporter response:" in line:
            _event(
                events,
                event_id=f"test:{index}:outcome",
                timestamp=ts,
                actor="agent",
                kind="outcome",
                content="final report written",
                metadata={"action": "write_report", "status": "success"},
                span=(index, index),
                episode="log_test.txt",
            )
            continue

        if "Async workflow completed successfully" in line:
            _event(
                events,
                event_id=f"test:{index}:success",
                timestamp=ts,
                actor="environment",
                kind="state_change",
                content="workflow completed",
                metadata={"state": "success"},
                span=(index, index),
                episode="log_test.txt",
            )
            continue

    return events


def _parse_log_view(path: Path) -> list[Event]:
    """Parse the human-readable CentralAgent trace (log_view.md).

    This trace is deliberately different from the other two: it contains
    repeated ``think -> delegate`` loops and no explicit task outcome.  The
    parser keeps one representative per decision signature and emits a
    ``loop_detected`` state change on the second occurrence.
    """

    lines = _read_text(path).splitlines()
    events: list[Event] = []
    query_seen = False
    seen_decisions: dict[tuple[str, str], int] = {}
    in_ai_message = False
    in_json = False
    json_buffer: list[str] = []

    def emit_decision(
        index: int,
        ts: str,
        action: str,
        signature: str,
        content: str,
        metadata: dict[str, str],
    ) -> None:
        key = (action, signature[:160])
        count = seen_decisions.get(key, 0) + 1
        seen_decisions[key] = count
        if count == 1:
            _event(
                events,
                event_id=f"view:{index}:{action}",
                timestamp=ts,
                actor="agent",
                kind="agent_action",
                content=content,
                metadata={**metadata, "action": action, "repeat_count": str(count)},
                span=(index, index),
                episode="log_view.md",
            )
        elif count == 2:
            _event(
                events,
                event_id=f"view:{index}:loop",
                timestamp=ts,
                actor="environment",
                kind="state_change",
                content=f"loop detected: {action}",
                metadata={
                    "state": "loop_detected",
                    "action": action,
                    "repeat_count": str(count),
                },
                span=(index, index),
                episode="log_view.md",
            )

    for index, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        ts = _timestamp(line, f"line-{index}")

        if "Starting async workflow with user input:" in line:
            content = line.split("user input:", 1)[1].strip()
            if content and not query_seen:
                query_seen = True
                _event(
                    events,
                    event_id=f"view:{index}:user",
                    timestamp=ts,
                    actor="user",
                    kind="user_message",
                    content=content,
                    metadata={"task_type": "opinion", "output_format": "answer"},
                    span=(index, index),
                    episode="log_view.md",
                )
            continue

        if "Name: central_think" in line:
            in_ai_message = True
            continue

        if in_ai_message and stripped == "```json":
            in_json = True
            json_buffer = []
            continue

        if in_json:
            if stripped == "```":
                try:
                    payload = json.loads("\\n".join(json_buffer))
                except json.JSONDecodeError:
                    payload = {}
                action = str(payload.get("action", "delegate"))
                params = payload.get("params", {})
                if isinstance(params, dict):
                    signature = str(params.get("task_description", "")) or str(payload.get("instruction", ""))
                else:
                    signature = str(params)
                emit_decision(
                    index,
                    ts,
                    action,
                    signature,
                    f"{action}: {signature[:200]}",
                    {"needs_search": "yes" if action == "delegate" else "no"},
                )
                in_json = False
                in_ai_message = False
            else:
                json_buffer.append(line)
            continue

        if stripped.startswith("决策结果:"):
            payload_text = stripped.split("决策结果:", 1)[1].strip()
            action = ""
            reasoning = ""
            try:
                import ast

                payload = ast.literal_eval(payload_text)
                action = str(payload.get("action", ""))
                reasoning = str(payload.get("reasoning", ""))
            except (ValueError, SyntaxError):
                match = re.search(r"'action':\s*'([^']+)'", payload_text)
                action = match.group(1) if match else ""
            if not action:
                continue
            # ``think`` has no action parameters; repeated think is a loop.
            signature = "" if action == "think" else reasoning
            emit_decision(
                index,
                ts,
                action,
                signature,
                f"{action}: {reasoning[:200]}",
                {"needs_search": "yes" if action == "delegate" else "no"},
            )
            continue

    return events


def parse_stackplanner_logs(log_dir: str | Path) -> list[Event]:
    """Parse all StackPlanner logs into a single event stream."""

    log_dir = Path(log_dir)
    events: list[Event] = []
    backend = log_dir / "log.txt"
    workflow = log_dir / "log_test.txt"
    view = log_dir / "log_view.md"
    if backend.exists():
        events.extend(_parse_backend_log(backend))
    if workflow.exists():
        events.extend(_parse_workflow_log(workflow))
    if view.exists():
        events.extend(_parse_log_view(view))
    return events
