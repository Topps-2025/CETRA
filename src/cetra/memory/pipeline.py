"""Format-agnostic trace -> atoms -> graph -> (optional) rules pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from cetra.rsi.induction import induce_decision_tree
from cetra.rsi.types import CaseRecord

from .decompose import deconstruct_all, grounding_score, set_event_index
from .segment import segment
from .sources import TraceSource
from .types import Atom, Edge, MemoryGraph


@dataclass
class TraceRun:
    source: str
    events: int = 0
    atoms: int = 0
    graph: dict[str, Any] = field(default_factory=dict)
    attribute_keys: list[str] = field(default_factory=list)
    mean_grounding: float = 0.0
    inferred_ratio: float = 0.0
    event_kinds: dict[str, int] = field(default_factory=dict)
    rules: list[dict[str, Any]] = field(default_factory=list)
    sample_atoms: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "events": self.events,
            "atoms": self.atoms,
            "graph": self.graph,
            "attribute_keys": self.attribute_keys,
            "mean_grounding": round(self.mean_grounding, 6),
            "inferred_ratio": round(self.inferred_ratio, 6),
            "event_kinds": self.event_kinds,
            "rules": self.rules,
            "sample_atoms": self.sample_atoms,
            "errors": self.errors,
        }


def build_graph(atoms: list[Atom]) -> MemoryGraph:
    graph = MemoryGraph()
    for atom in atoms:
        graph.add_atom(atom)
    for previous, current in zip(atoms, atoms[1:]):
        graph.add_edge(
            Edge(
                src=previous.atom_id,
                dst=current.atom_id,
                kind="temporal",
                weight=1.0,
                evidence=previous.evidence + current.evidence,
            )
        )
        if previous.episode == current.episode:
            graph.add_edge(
                Edge(
                    src=previous.atom_id,
                    dst=current.atom_id,
                    kind="entity",
                    weight=0.8,
                    evidence=previous.evidence + current.evidence,
                )
            )
    for atom in atoms:
        if atom.action and atom.outcome:
            for other in atoms:
                if other.atom_id == atom.atom_id:
                    continue
                if other.episode == atom.episode and other.span[0] >= atom.span[0]:
                    graph.add_edge(
                        Edge(
                            src=atom.atom_id,
                            dst=other.atom_id,
                            kind="causal",
                            weight=0.8,
                            evidence=atom.evidence + other.evidence,
                        )
                    )
                    break
    return graph


def atoms_to_cases(atoms: Iterable[Atom]) -> list[CaseRecord]:
    cases: list[CaseRecord] = []
    for atom in atoms:
        action = atom.action or atom.attribute("action")
        if not action:
            continue
        factors = {
            key: attr.value
            for key, attr in atom.attributes.items()
            if key not in {"action", "outcome"} and attr.value
        }
        if not factors:
            continue
        cases.append(
            CaseRecord(
                case_id=atom.atom_id,
                split="all",
                narrative=atom.content,
                factors=factors,
                action=action,
                family=atom.episode,
            )
        )
    return cases


def induce_rules_from_atoms(atoms: Iterable[Atom]) -> list[dict[str, Any]]:
    cases = atoms_to_cases(atoms)
    if len(cases) < 3 or len({case.action for case in cases}) < 2:
        return []
    try:
        rules = induce_decision_tree(
            cases,
            max_depth=3,
            min_support=2,
            min_precision=0.8,
            min_gain=1e-9,
        )
    except Exception:
        return []
    return [
        {
            "conditions": [list(item) for item in rule.conditions],
            "action": rule.action,
            "support": rule.support,
            "precision": rule.precision,
        }
        for rule in rules[:20]
    ]


def run_source(
    source: TraceSource,
    *,
    judge: Any | None = None,
    include_rules: bool = True,
) -> TraceRun:
    run = TraceRun(source=getattr(source, "name", type(source).__name__))
    try:
        events = source.events()
    except Exception as exc:  # noqa: BLE001 - report, do not crash the batch
        run.errors.append(f"source error: {type(exc).__name__}: {exc}")
        return run
    run.events = len(events)
    if not events:
        run.errors.append("no events parsed")
        return run
    try:
        atoms = segment(events)
        set_event_index(events)
        atoms = deconstruct_all(atoms, events, judge=judge)
        graph = build_graph(atoms)
    except Exception as exc:  # noqa: BLE001
        run.errors.append(f"pipeline error: {type(exc).__name__}: {exc}")
        return run

    run.atoms = len(atoms)
    run.graph = graph.summary()
    run.attribute_keys = sorted({key for atom in atoms for key in atom.attributes})
    scores = [grounding_score(atom) for atom in atoms]
    run.mean_grounding = sum(scores) / len(scores) if scores else 0.0
    inferred = sum(
        1 for atom in atoms for attr in atom.attributes.values() if attr.inferred
    )
    total = sum(len(atom.attributes) for atom in atoms)
    run.inferred_ratio = inferred / total if total else 0.0
    run.event_kinds = {}
    for event in events:
        run.event_kinds[event.kind] = run.event_kinds.get(event.kind, 0) + 1
    if include_rules:
        run.rules = induce_rules_from_atoms(atoms)
    run.sample_atoms = [
        {
            "atom_id": atom.atom_id,
            "kind": atom.kind,
            "action": atom.action,
            "attributes": {
                key: attr.value for key, attr in list(atom.attributes.items())[:8]
            },
        }
        for atom in atoms[:5]
    ]
    return run


def save_runs(runs: Iterable[TraceRun], path: str | Path) -> None:
    payload = [run.to_dict() for run in runs]
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
