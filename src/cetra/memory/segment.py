"""Boundary detection and segmentation of an event stream into atoms.

The production system would call an LLM judge for ambiguous boundaries.  This
prototype uses a deterministic judge with the same interface, so the rest of
the pipeline can be tested without an API key.  The deterministic scores are
also a useful calibration baseline for the LLM judge.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Atom, Event


@dataclass(frozen=True)
class SegmentConfig:
    min_len: int = 1
    max_len: int = 8
    lam: float = 0.25
    boundary_threshold: float = 0.45


@dataclass(frozen=True)
class BoundaryScore:
    index: int
    score: float
    kind: str
    rationale: str


_HIGH = {"user_message", "agent_action", "tool_result", "user_feedback", "state_change", "outcome"}
_MEDIUM = {"observation", "tool_call"}


def score_boundary(events: list[Event], index: int) -> BoundaryScore:
    """Score the boundary before ``events[index]``."""

    if index <= 0 or index >= len(events):
        return BoundaryScore(index, 0.0, "none", "edge")
    current = events[index]
    previous = events[index - 1]
    if current.kind in _HIGH:
        score = 0.9
    elif current.kind in _MEDIUM:
        score = 0.6
    else:
        score = 0.2
    if previous.kind == current.kind:
        score -= 0.25
    if current.kind == "observation" and previous.kind == "agent_action":
        score = 0.35
    if current.kind == "tool_result" and previous.kind in {"tool_call", "agent_action"}:
        score = 0.85
    if current.kind == "state_change" and previous.kind == "user_feedback":
        score = 0.9
    return BoundaryScore(index, max(0.0, score), current.kind, f"{previous.kind}->{current.kind}")


def _segment_score(events: list[Event], start: int, end: int, config: SegmentConfig) -> float:
    length = end - start
    if length < config.min_len or length > config.max_len:
        return float("-inf")
    boundary = score_boundary(events, end).score if end < len(events) else 0.0
    # Coherence: reward segments that contain an action/result pair.
    kinds = {event.kind for event in events[start:end]}
    coherence = 0.0
    if "agent_action" in kinds:
        coherence += 0.3
    if "tool_result" in kinds or "observation" in kinds:
        coherence += 0.2
    if "user_feedback" in kinds:
        coherence += 0.2
    return boundary + coherence


def dp_segment(events: list[Event], config: SegmentConfig | None = None) -> list[tuple[int, int]]:
    """Dynamic programming segmentation.

    Returns a list of half-open ``(start, end)`` spans.
    """

    config = config or SegmentConfig()
    n = len(events)
    if n == 0:
        return []
    best = [float("-inf")] * (n + 1)
    prev = [0] * (n + 1)
    best[0] = 0.0
    for end in range(1, n + 1):
        for start in range(max(0, end - config.max_len), end):
            if best[start] == float("-inf"):
                continue
            score = _segment_score(events, start, end, config)
            if score == float("-inf"):
                continue
            candidate = best[start] + score - config.lam
            if candidate > best[end]:
                best[end] = candidate
                prev[end] = start
    # Backtrack.
    cuts: list[tuple[int, int]] = []
    end = n
    while end > 0:
        start = prev[end]
        cuts.append((start, end))
        end = start
    cuts.reverse()
    return cuts


def _atom_kind(events: list[Event]) -> str:
    kinds = {event.kind for event in events}
    if "agent_action" in kinds or "user_feedback" in kinds or "state_change" in kinds:
        return "decision"
    if "tool_result" in kinds or "observation" in kinds:
        return "event"
    return "event"


def segment(
    events: list[Event],
    config: SegmentConfig | None = None,
) -> list[Atom]:
    """Turn an event stream into atoms."""

    config = config or SegmentConfig()
    cuts = dp_segment(events, config)
    atoms: list[Atom] = []
    for start, end in cuts:
        chunk = events[start:end]
        if not chunk:
            continue
        action = next(
            (event.metadata.get("action") for event in chunk
             if event.kind in {"agent_action", "tool_call"} and event.metadata.get("action")),
            None,
        )
        outcome = next(
            (event.content for event in chunk
             if event.kind in {"tool_result", "observation", "outcome"}),
            "",
        )
        context = " ".join(event.content for event in chunk[: max(0, len(chunk) - 1)])
        atom = Atom(
            atom_id=f"atom:{chunk[0].episode}:{start}-{end}",
            kind=_atom_kind(chunk),
            span=(chunk[0].span[0], chunk[-1].span[1]),
            content=" | ".join(event.content for event in chunk),
            context=context,
            action=action,
            outcome=outcome,
            evidence=tuple(event.event_id for event in chunk),
            confidence=min(1.0, 0.5 + 0.1 * len(chunk)),
            episode=chunk[0].episode,
        )
        atoms.append(atom)
    return atoms
