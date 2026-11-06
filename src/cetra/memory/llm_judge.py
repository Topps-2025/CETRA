"""LLM-assisted trace parsing and attribute extraction.

The LLM is used where the deterministic adapter is hardcoded:

- unknown log formats -> ``parse_events_with_llm``
- new attribute schemas -> ``LLMJudge.attributes``

Every LLM output is validated against the atom/event schema; ungrounded
attributes are marked ``inferred`` and down-weighted.  The deterministic
adapter/judge remains the fallback, so the pipeline never depends on the API
being available.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from .decompose import DeterministicJudge
from .llm import DeepSeekClient
from .types import Atom, Attribute, Event

_ALLOWED_EVENT_KINDS = {
    "user_message",
    "agent_action",
    "tool_call",
    "tool_result",
    "observation",
    "user_feedback",
    "state_change",
    "outcome",
}
_ALLOWED_ACTORS = {"user", "agent", "tool", "environment"}
_KEY = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


def _event_from_payload(payload: dict[str, Any], index: int, source: str) -> Event | None:
    kind = str(payload.get("kind", "")).strip()
    actor = str(payload.get("actor", "")).strip()
    content = str(payload.get("content", "")).strip()
    if kind not in _ALLOWED_EVENT_KINDS or actor not in _ALLOWED_ACTORS or not content:
        return None
    metadata = payload.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    clean_metadata = {str(k): str(v) for k, v in metadata.items() if v is not None}
    action = payload.get("action")
    if action is not None and str(action).strip().lower() not in {"none", "null", ""}:
        clean_metadata["action"] = str(action)
    return Event(
        event_id=f"llm:{source}:{index}",
        timestamp=str(payload.get("timestamp", "")),
        actor=actor,
        kind=kind,
        content=content[:2000],
        metadata=clean_metadata,
        span=(0, 0),
        episode=source,
    )


def _dedupe_events(events: Iterable[Event]) -> list[Event]:
    """Collapse repeated think/delegate/activation cycles into one loop atom.

    The LLM parser tends to emit every cycle; this deterministic post-pass
    keeps the first occurrence and emits one ``loop_detected`` state change.
    It is a validation layer, not a replacement for the LLM.
    """

    out: list[Event] = []
    seen: dict[str, int] = {}
    loop_emitted = False
    for event in events:
        action = event.metadata.get("action", "")
        if action in {"think", "activate_central_agent"}:
            if event.metadata.get("state_completeness") == "no":
                key = action + ":missing_state"
            else:
                key = action + ":" + event.content[:60]
        elif action == "delegate":
            key = "delegate:" + event.metadata.get("task_description", event.content)[:120]
        else:
            out.append(event)
            continue
        count = seen.get(key, 0) + 1
        seen[key] = count
        if count == 1:
            out.append(event)
        elif not loop_emitted:
            loop_emitted = True
            out.append(
                Event(
                    event_id=event.event_id + ":loop",
                    timestamp=event.timestamp,
                    actor="environment",
                    kind="state_change",
                    content=f"loop detected: {action}",
                    metadata={
                        "state": "loop_detected",
                        "action": action,
                        "repeat_count": str(count),
                        "state_completeness": event.metadata.get("state_completeness", "no"),
                    },
                    span=event.span,
                    episode=event.episode,
                )
            )
    return out


def parse_events_with_llm(
    raw_text: str,
    source: str,
    client: DeepSeekClient,
    *,
    max_events: int = 60,
) -> list[Event]:
    """Extract a chronological event list from a raw log."""

    system = (
        "You are a strict log parser for agent memory. "
        "You must not invent events. Return JSON only."
    )
    user = f"""Source: {source}

Raw log (truncated):
---
{raw_text[:14000]}
---

Extract the chronological memory events. Return JSON:
{{
  "events": [
    {{
      "kind": "user_message|agent_action|tool_call|tool_result|observation|user_feedback|state_change|outcome",
      "actor": "user|agent|tool|environment",
      "content": "short factual description",
      "action": "action string or null",
      "metadata": {{"key": "value"}}
    }}
  ]
}}

Rules:
1. One event per decision, action, tool result, user feedback, or state change.
2. Do not repeat an identical think/delegate loop more than twice; add
   metadata "repeat_count" on the repeated event.
3. If the log shows missing query/state/memory (empty fields), add
   metadata "state_completeness": "no" to the relevant event.
4. If the trace has no successful outcome, add a final event with
   kind "state_change", metadata "state": "no_outcome".
5. Do not invent user messages, tools, or results that are not in the log.
6. Maximum {max_events} events.
"""
    try:
        payload = client.chat_json(system, user, max_tokens=2200)
    except Exception:
        return []
    events: list[Event] = []
    for index, item in enumerate(payload.get("events", []) or []):
        if isinstance(item, dict):
            event = _event_from_payload(item, index, source)
            if event is not None:
                events.append(event)
        if len(events) >= max_events:
            break
    return _dedupe_events(events)


class LLMJudge:
    """Attribute extraction with grounding validation."""

    def __init__(self, client: DeepSeekClient, fallback: DeterministicJudge | None = None):
        self.client = client
        self.fallback = fallback or DeterministicJudge()

    def attributes(self, atom: Atom) -> list[Attribute]:
        system = (
            "You extract typed attributes from one agent-memory atom. "
            "Every attribute must be grounded in the provided evidence ids "
            "or marked inferred. Return JSON only."
        )
        evidence = list(atom.evidence)
        known = {
            key: attr.value for key, attr in atom.attributes.items()
        }
        user = f"""Atom kind: {atom.kind}
Content: {atom.content[:3000]}
Action: {atom.action}
Outcome: {atom.outcome}
Evidence ids: {evidence}
Known attributes (hints): {json.dumps(known, ensure_ascii=False)}

Return JSON:
{{
  "attributes": [
    {{
      "key": "snake_case_key",
      "value": "value",
      "evidence": ["event_id"],
      "confidence": 0.0,
      "inferred": false
    }}
  ]
}}

Rules:
1. Use only evidence ids from the list above.
2. If a value is inferred but not directly grounded, set inferred=true and
   confidence <= 0.5.
3. Use short snake_case keys; do not invent entity names.
4. Include decision-relevant fields when present: task_type, missing_slots,
   user_feedback, has_enough_context, needs_search, output_format, action,
   state_completeness, failure_mode, repeat_count.
5. If the atom is a loop or missing-state failure, add
   failure_mode=loop|missing_state and state_completeness=no.
"""
        try:
            payload = self.client.chat_json(system, user, max_tokens=1200)
        except Exception:
            return self.fallback.attributes(atom)
        valid_ids = set(evidence)
        attributes: list[Attribute] = []
        for item in payload.get("attributes", []) or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            value = str(item.get("value", "")).strip()
            if not _KEY.match(key) or not value:
                continue
            if key == "action" and value.lower() in {"none", "null"}:
                continue
            if key in {"has_enough_context", "needs_search", "state_completeness"}:
                if value.lower() == "true":
                    value = "yes"
                elif value.lower() == "false":
                    value = "no"
            raw_evidence = item.get("evidence", []) or []
            if not isinstance(raw_evidence, list):
                raw_evidence = []
            grounded_evidence = tuple(
                str(e) for e in raw_evidence if str(e) in valid_ids
            )
            inferred = bool(item.get("inferred", False)) or not grounded_evidence
            try:
                confidence = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))
            if inferred:
                confidence = min(confidence, 0.5)
            attributes.append(
                Attribute(
                    key=key,
                    value=value[:500],
                    evidence=grounded_evidence,
                    confidence=confidence,
                    inferred=inferred,
                )
            )
        if not attributes:
            return self.fallback.attributes(atom)
        return attributes
