"""Trace parsing: turn real run logs into typed decision points.

The controlled benchmark supplies ``CaseRecord.factors`` directly.  Real logs
do not: they contain tool calls, errors, user corrections, and agent actions
interleaved with surface details (thread ids, request ids, timestamps).  This
module defines the boundary between the two.

A parser produces ``DecisionPoint`` objects.  A decision point keeps:

- the context that led to the action;
- the action taken;
- the observed outcome;
- decision-relevant factors;
- surface details (kept for provenance, not for rule induction);
- evidence event ids.

``StructuredTraceParser`` is deterministic and expects events with a small
amount of metadata.  An LLM can be plugged in behind the same interface later;
the rest of the pipeline does not need to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol

from .types import CaseRecord


# Factors that are allowed to enter rule induction.  Everything else is
# treated as surface detail and only kept in the narrative / provenance.
DEFAULT_DECISION_FACTORS = frozenset(
    {
        # tool-failure / operations domain
        "authority",
        "commit_state",
        "retry_safety",
        "resource_state",
        "reversibility",
        "retry_count",
        # long-horizon report / QA domain (StackPlanner traces)
        "task_type",
        "missing_slots",
        "user_feedback",
        "has_enough_context",
        "needs_search",
        "output_format",
        "style_role",
        "knowledge_base",
    }
)


@dataclass(frozen=True)
class TraceEvent:
    """One event in a run log."""

    event_id: str
    kind: str  # tool_call | tool_error | observation | user_feedback | state_change | agent_action
    actor: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionPoint:
    """A decision boundary extracted from a trace."""

    point_id: str
    context: str
    action: str
    outcome: str
    factors: dict[str, str]
    surface: dict[str, str] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()

    def to_case(self, *, split: str, family: str) -> CaseRecord:
        surface_text = " ".join(f"{key}={value}" for key, value in sorted(self.surface.items()))
        narrative = self.context
        if surface_text:
            narrative = f"{narrative} | surface: {surface_text}"
        if self.outcome:
            narrative = f"{narrative} | outcome: {self.outcome}"
        return CaseRecord(
            case_id=self.point_id,
            split=split,
            narrative=narrative,
            factors=dict(self.factors),
            action=self.action,
            family=family,
        )


class TraceParser(Protocol):
    def parse(self, events: Iterable[TraceEvent]) -> list[DecisionPoint]: ...


class StructuredTraceParser:
    """Deterministic parser for already-structured events.

    An event becomes a decision point when it is an ``agent_action`` or a
    ``tool_call`` and carries ``metadata['action']``.  The parser groups the
    preceding non-action events into the context, and attaches the next
    observation / error / user feedback as the outcome.

    This is deliberately conservative: if the log does not expose an action
    label, the parser does not invent one.  A model-backed parser can implement
    the same ``TraceParser`` interface and extract actions from free text.
    """

    def __init__(
        self,
        decision_factors: Iterable[str] | None = None,
        *,
        context_window: int = 4,
    ) -> None:
        self.decision_factors = frozenset(decision_factors or DEFAULT_DECISION_FACTORS)
        self.context_window = context_window

    def parse(self, events: Iterable[TraceEvent]) -> list[DecisionPoint]:
        items = list(events)
        points: list[DecisionPoint] = []
        pending: list[TraceEvent] = []

        for index, event in enumerate(items):
            is_action = event.kind in {"agent_action", "tool_call"} and "action" in event.metadata
            if not is_action:
                pending.append(event)
                if len(pending) > self.context_window:
                    pending = pending[-self.context_window :]
                continue

            action = event.metadata["action"]
            context = " ".join(item.content for item in pending if item.content).strip()
            outcome = ""
            evidence = [item.event_id for item in pending]
            evidence.append(event.event_id)
            if index + 1 < len(items):
                nxt = items[index + 1]
                if nxt.kind in {"observation", "tool_error", "user_feedback", "state_change"}:
                    outcome = nxt.content
                    evidence.append(nxt.event_id)

            factors = {
                key: value
                for key, value in event.metadata.items()
                if key != "action" and key in self.decision_factors
            }
            surface = {
                key: value
                for key, value in event.metadata.items()
                if key != "action" and key not in self.decision_factors
            }
            points.append(
                DecisionPoint(
                    point_id=event.event_id,
                    context=context,
                    action=action,
                    outcome=outcome,
                    factors=factors,
                    surface=surface,
                    evidence=tuple(evidence),
                )
            )
            pending = []

        return points


def decision_points_to_cases(
    points: Iterable[DecisionPoint],
    *,
    split: str,
    family: str,
) -> list[CaseRecord]:
    return [point.to_case(split=split, family=family) for point in points]
