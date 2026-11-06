"""Atom -> attributes with evidence binding.

The LLM judge is represented by a small ``Judge`` protocol.  The prototype
implements a deterministic judge; a model-backed judge can be plugged in
without changing the downstream pipeline.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable, Protocol

from .types import Atom, Attribute, Event


class Judge(Protocol):
    def attributes(self, atom: Atom) -> list[Attribute]: ...


def _add(attrs: dict[str, Attribute], key: str, value: str, event_id: str, confidence: float = 0.9) -> None:
    if not value:
        return
    existing = attrs.get(key)
    if existing is None or confidence > existing.confidence:
        attrs[key] = Attribute(
            key=key,
            value=value,
            evidence=(event_id,),
            confidence=confidence,
        )


def _infer_from_text(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    lower = text.lower()
    if "讲话稿" in text or "speech" in lower:
        out["task_type"] = "speech"
        out["output_format"] = "speech"
    elif "report" in lower or "报告" in text:
        out["task_type"] = "report"
        out["output_format"] = "report"
    elif "ai agent" in lower or "what is" in lower:
        out["task_type"] = "qa"
        out["output_format"] = "answer"
    if "[skip]" in lower:
        out["user_feedback"] = "skip"
    if "只改" in text or "revise" in lower or "修改第" in text:
        out["user_feedback"] = "revise"
    if "缺" in text or "未给" in text or "没有给" in text:
        out["missing_slots"] = "yes"
    if "上下文不足" in text or "not enough" in lower:
        out["has_enough_context"] = "no"
    if "上下文足够" in text or "enough context" in lower:
        out["has_enough_context"] = "yes"
    return out


class DeterministicJudge:
    """A small, inspectable judge used for the real-trace smoke test."""

    def attributes(self, atom: Atom) -> list[Attribute]:
        attrs: dict[str, Attribute] = {}
        for event_id in atom.evidence:
            # Metadata is the most reliable source.
            event = _EVENT_INDEX.get(event_id)
            if event is not None:
                for key, value in event.metadata.items():
                    if key in {
                        "task_type", "missing_slots", "user_feedback",
                        "has_enough_context", "needs_search", "output_format",
                        "action", "source_count", "state",
                    }:
                        _add(attrs, key, value, event_id, 0.95)
            inferred = _infer_from_text(atom.content)
            for key, value in inferred.items():
                _add(attrs, key, value, event_id, 0.8)
        return list(attrs.values())


class OpenMetadataJudge:
    """Schema-open judge for non-StackPlanner traces.

    It accepts any scalar metadata key that looks like a snake_case field.
    This removes the StackPlanner-specific allowlist without letting the judge
    invent values: every attribute still points to the event id that carried
    it.
    """

    _KEY = re.compile(r"^[a-z][a-z0-9_]{0,40}$")

    def attributes(self, atom: Atom) -> list[Attribute]:
        attrs: dict[str, Attribute] = {}
        for event_id in atom.evidence:
            event = _EVENT_INDEX.get(event_id)
            if event is None:
                continue
            for key, value in event.metadata.items():
                if not self._KEY.match(key):
                    continue
                if isinstance(value, (str, int, float, bool)):
                    _add(attrs, key, str(value), event_id, 0.9)
        if atom.action:
            _add(attrs, "action", atom.action, atom.evidence[0] if atom.evidence else "", 0.9)
        return list(attrs.values())


_EVENT_INDEX: dict[str, Event] = {}


def set_event_index(events: Iterable[Event]) -> None:
    global _EVENT_INDEX
    _EVENT_INDEX = {event.event_id: event for event in events}


def deconstruct(atom: Atom, judge: Judge | None = None) -> Atom:
    """Return a copy of ``atom`` with attributes populated."""

    judge = judge or DeterministicJudge()
    attributes = {attr.key: attr for attr in judge.attributes(atom)}
    if not attributes and atom.action:
        attributes["action"] = Attribute(
            key="action", value=atom.action, evidence=atom.evidence, confidence=0.9
        )
    return replace(atom, attributes=attributes)


def grounding_score(atom: Atom) -> float:
    """Fraction of attributes with at least one evidence id."""

    if not atom.attributes:
        return 0.0
    grounded = sum(1 for attr in atom.attributes.values() if attr.evidence)
    return grounded / len(atom.attributes)


def deconstruct_all(atoms: Iterable[Atom], events: Iterable[Event], judge: Judge | None = None) -> list[Atom]:
    set_event_index(events)
    return [deconstruct(atom, judge) for atom in atoms]
