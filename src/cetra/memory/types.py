"""Core types for atomic graph memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cetra.rsi.types import CaseRecord, Rule


@dataclass(frozen=True)
class Event:
    """One normalized event in a raw agent memory stream."""

    event_id: str
    timestamp: str
    actor: str
    kind: str  # user_message | agent_action | tool_call | tool_result |
               # observation | user_feedback | state_change | outcome
    content: str
    metadata: dict[str, str] = field(default_factory=dict)
    span: tuple[int, int] = (0, 0)
    episode: str = "default"


@dataclass(frozen=True)
class Attribute:
    """A typed attribute extracted from an atom."""

    key: str
    value: str
    evidence: tuple[str, ...] = ()
    confidence: float = 0.0
    inferred: bool = False


@dataclass
class Atom:
    """A minimal, independently addressable memory unit."""

    atom_id: str
    kind: str  # event | decision | entity | episode | rule | chain | skill
    span: tuple[int, int]
    content: str
    context: str = ""
    action: str | None = None
    outcome: str | None = None
    attributes: dict[str, Attribute] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()
    confidence: float = 0.0
    version: int = 1
    parent: str | None = None
    episode: str = "default"

    def attribute(self, key: str, default: str = "") -> str:
        item = self.attributes.get(key)
        return item.value if item else default


@dataclass(frozen=True)
class Edge:
    """A typed edge between two atoms."""

    src: str
    dst: str
    kind: str  # temporal | causal | entity | precondition_effect |
               # evidence | version | scope
    weight: float = 1.0
    evidence: tuple[str, ...] = ()


@dataclass
class MemoryGraph:
    """A small typed property graph with adjacency queries."""

    atoms: dict[str, Atom] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_atom(self, atom: Atom) -> None:
        self.atoms[atom.atom_id] = atom

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def neighbors(self, atom_id: str, kind: str | None = None) -> list[Atom]:
        out: list[Atom] = []
        for edge in self.edges:
            if edge.src == atom_id and (kind is None or edge.kind == kind):
                target = self.atoms.get(edge.dst)
                if target is not None:
                    out.append(target)
        return out

    def nodes_of_kind(self, kind: str) -> list[Atom]:
        return [atom for atom in self.atoms.values() if atom.kind == kind]

    def summary(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for atom in self.atoms.values():
            by_kind[atom.kind] = by_kind.get(atom.kind, 0) + 1
        by_edge: dict[str, int] = {}
        for edge in self.edges:
            by_edge[edge.kind] = by_edge.get(edge.kind, 0) + 1
        return {
            "atoms": len(self.atoms),
            "edges": len(self.edges),
            "atoms_by_kind": by_kind,
            "edges_by_kind": by_edge,
        }


@dataclass
class ProgramState:
    """The current memory-construction program.

    ``fields`` is the field schema, ``operator`` is the rule learner,
    ``topology`` is the ordered operator list, and ``injected_rules`` are
    rules added by the self-evolution loop.
    """

    fields: tuple[str, ...]
    operator: str = "induce_single"
    params: dict[str, Any] = field(default_factory=dict)
    injected_rules: list[Rule] = field(default_factory=list)
    topology: tuple[str, ...] = (
        "decompose",
        "abduce",
        "induce",
        "predict",
    )
    history: list[dict[str, Any]] = field(default_factory=list)

    def clone(self) -> "ProgramState":
        return ProgramState(
            fields=tuple(self.fields),
            operator=self.operator,
            params=dict(self.params),
            injected_rules=list(self.injected_rules),
            topology=tuple(self.topology),
            history=list(self.history),
        )


@dataclass(frozen=True)
class Edit:
    """A typed field-level or operator-level evolution edit."""

    level: str  # field | operator | topology | param
    op: str
    target: str
    args: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "op": self.op,
            "target": self.target,
            "args": dict(self.args),
            "rationale": self.rationale,
        }


def case_to_atom(case: CaseRecord) -> Atom:
    """Adapt a labeled case to a decision atom (for the SE experiment)."""

    return Atom(
        atom_id=case.case_id,
        kind="decision",
        span=(0, 0),
        content=case.narrative,
        context=case.narrative,
        action=case.action,
        outcome="",
        attributes={
            key: Attribute(key=key, value=value, evidence=(case.case_id,), confidence=1.0)
            for key, value in case.factors.items()
        },
        evidence=(case.case_id,),
        confidence=1.0,
        episode=case.family,
    )
