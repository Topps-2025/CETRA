"""Schema-driven memory atoms.

Atom types are **data**, not Python subclasses.  A memory atom is a
``(schema, fields)`` pair; a schema is a list of field specs.  This lets the
same operators work on decision atoms, fact atoms, preference atoms, rules,
chains, and skills without hardcoding a class per type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from cetra.rsi.types import CaseRecord

from .types import Event


@dataclass(frozen=True)
class FieldValue:
    name: str
    value: Any
    type: str = "string"
    evidence: tuple[str, ...] = ()
    confidence: float = 1.0
    inferred: bool = False
    scope: str = "global"
    provenance: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.type,
            "evidence": list(self.evidence),
            "confidence": self.confidence,
            "inferred": self.inferred,
            "scope": self.scope,
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: str = "string"
    required: bool = False
    decomposable: bool = False
    description: str = ""
    scope: str = "global"


@dataclass(frozen=True)
class AtomSchema:
    name: str
    fields: tuple[FieldSpec, ...]
    parent: str | None = None
    description: str = ""

    def field_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.fields)

    def decomposable_fields(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.fields if spec.decomposable)


class SchemaRegistry:
    """A data-driven registry of atom schemas."""

    def __init__(self) -> None:
        self._schemas: dict[str, AtomSchema] = {}

    def add(self, schema: AtomSchema) -> None:
        self._schemas[schema.name] = schema

    def get(self, name: str) -> AtomSchema | None:
        return self._schemas.get(name)

    def names(self) -> list[str]:
        return sorted(self._schemas)

    @classmethod
    def default(cls) -> "SchemaRegistry":
        registry = cls()
        registry.add(
            AtomSchema(
                "event",
                (
                    FieldSpec("kind", required=True),
                    FieldSpec("actor"),
                    FieldSpec("content", decomposable=True),
                    FieldSpec("timestamp"),
                    FieldSpec("metadata", decomposable=True),
                ),
                description="raw/observation event",
            )
        )
        registry.add(
            AtomSchema(
                "decision",
                (
                    FieldSpec("context", decomposable=True),
                    FieldSpec("action", required=True),
                    FieldSpec("outcome", decomposable=True),
                    FieldSpec("factors", decomposable=True),
                    FieldSpec("scope"),
                ),
                parent="event",
                description="a choice point with action/outcome",
            )
        )
        registry.add(
            AtomSchema(
                "fact",
                (
                    FieldSpec("entity", required=True),
                    FieldSpec("relation", required=True),
                    FieldSpec("object"),
                    FieldSpec("value"),
                    FieldSpec("severity"),
                    FieldSpec("scope"),
                ),
                parent="event",
                description="semantic fact / constraint",
            )
        )
        registry.add(
            AtomSchema(
                "preference",
                (
                    FieldSpec("subject"),
                    FieldSpec("object", required=True),
                    FieldSpec("polarity", required=True),
                    FieldSpec("strength"),
                    FieldSpec("scope"),
                ),
                parent="fact",
                description="user preference",
            )
        )
        registry.add(
            AtomSchema(
                "rule",
                (
                    FieldSpec("conditions", required=True),
                    FieldSpec("action", required=True),
                    FieldSpec("effect"),
                    FieldSpec("scope"),
                    FieldSpec("evidence"),
                ),
                description="condition -> action",
            )
        )
        registry.add(
            AtomSchema(
                "chain",
                (
                    FieldSpec("rules", required=True),
                    FieldSpec("preconditions"),
                    FieldSpec("effects"),
                    FieldSpec("scope"),
                ),
                description="composed rule sequence",
            )
        )
        registry.add(
            AtomSchema(
                "skill",
                (
                    FieldSpec("contract", required=True),
                    FieldSpec("handler", required=True),
                    FieldSpec("evidence"),
                    FieldSpec("provenance"),
                ),
                description="executable skill artifact",
            )
        )
        return registry


@dataclass
class FieldAtom:
    atom_id: str
    schema: str
    fields: dict[str, FieldValue] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()
    scope: str = "global"
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = ""

    def get(self, name: str, default: Any = None) -> Any:
        item = self.fields.get(name)
        return item.value if item is not None else default

    def field(self, name: str) -> FieldValue | None:
        return self.fields.get(name)

    def set_field(self, value: FieldValue) -> None:
        self.fields[value.name] = value

    def to_case(self) -> CaseRecord | None:
        action = self.get("action")
        if not action:
            return None
        factors = {
            key: str(field.value)
            for key, field in self.fields.items()
            if key not in {"action", "kind", "actor", "content", "timestamp"}
            and field.value is not None
        }
        if not factors:
            return None
        return CaseRecord(
            case_id=self.atom_id,
            split="all",
            narrative=str(self.get("content", self.get("context", ""))),
            factors=factors,
            action=str(action),
            family=self.scope,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "schema": self.schema,
            "fields": {key: value.to_dict() for key, value in self.fields.items()},
            "evidence": list(self.evidence),
            "scope": self.scope,
            "parent": self.parent,
            "children": list(self.children),
            "confidence": self.confidence,
            "source": self.source,
        }


def event_to_field_atom(event: Event) -> FieldAtom:
    """Convert a normalized Event into a FieldAtom.

    If the event carries an ``action`` it becomes a decision atom; otherwise
    it is an event atom.  No type-specific subclass is created.
    """

    fields: dict[str, FieldValue] = {
        "kind": FieldValue("kind", event.kind, evidence=(event.event_id,)),
        "actor": FieldValue("actor", event.actor, evidence=(event.event_id,)),
        "content": FieldValue("content", event.content, evidence=(event.event_id,)),
        "timestamp": FieldValue("timestamp", event.timestamp, evidence=(event.event_id,)),
    }
    for key, value in event.metadata.items():
        if value is None:
            continue
        fields[key] = FieldValue(
            key,
            value,
            evidence=(event.event_id,),
            scope=event.episode,
        )
    action = event.metadata.get("action")
    schema = "decision" if action else "event"
    if schema == "decision":
        fields["action"] = FieldValue(
            "action", action, evidence=(event.event_id,), scope=event.episode
        )
        fields.setdefault(
            "context", FieldValue("context", event.content, evidence=(event.event_id,))
        )
    return FieldAtom(
        atom_id=event.event_id,
        schema=schema,
        fields=fields,
        evidence=(event.event_id,),
        scope=event.episode,
        source=event.episode,
    )


def events_to_field_atoms(events: Iterable[Event]) -> list[FieldAtom]:
    return [event_to_field_atom(event) for event in events]
