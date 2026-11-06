"""Entity/attribute decomposition.

This is deliberately separate from decision-point segmentation in ``trace.py``:

- ``trace.py``: along a trajectory, *where* does a decision happen?
- ``ontology.py``: for an entity node (a macaron, a ticket, a tool, an
  artifact), *what is it made of*?  Which components and attributes does it
  have?

The two processes are orthogonal.  A decision point may contain several
entities; each entity is decomposed into components and attributes; attributes
then feed abstraction (concepts/values), while decisions feed chain
reconstruction.

The deterministic decomposer accepts an explicit nested payload.  A model-backed
decomposer can implement the same ``EntityDecomposer`` protocol and use the
same output type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol


@dataclass(frozen=True)
class Attribute:
    name: str
    value: str
    evidence: str = ""


@dataclass(frozen=True)
class Component:
    name: str
    attributes: tuple[Attribute, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "attributes": [
                {"name": attr.name, "value": attr.value, "evidence": attr.evidence}
                for attr in self.attributes
            ],
        }


@dataclass(frozen=True)
class EntityNode:
    """A node decomposed into components and top-level attributes."""

    entity_id: str
    type: str
    components: tuple[Component, ...] = ()
    attributes: tuple[Attribute, ...] = ()
    evidence: tuple[str, ...] = ()

    def flatten_attributes(self) -> dict[str, str]:
        """Flatten to ``attr=value`` or ``component.attr=value``.

        The flat form is what the current rule learner consumes.  Keeping the
        nested form is useful for contracts and provenance.
        """

        flat: dict[str, str] = {}
        for component in self.components:
            for attr in component.attributes:
                flat[f"{component.name}.{attr.name}"] = attr.value
        for attr in self.attributes:
            flat[attr.name] = attr.value
        return flat

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "type": self.type,
            "components": [component.to_dict() for component in self.components],
            "attributes": [
                {"name": attr.name, "value": attr.value, "evidence": attr.evidence}
                for attr in self.attributes
            ],
            "evidence": list(self.evidence),
        }


class EntityDecomposer(Protocol):
    def decompose(self, entity_id: str, entity_type: str, payload: dict[str, Any]) -> EntityNode: ...


class StructuredEntityDecomposer:
    """Deterministic entity decomposition from an explicit payload.

    Example payload for a macaron::

        {
          "components": {
            "shell": {"texture": "crisp", "color": "pink"},
            "filling": {"flavor": "rose"}
          },
          "attributes": {"price": "8", "origin": "Paris"},
          "evidence": ["menu:item-42"]
        }
    """

    def decompose(self, entity_id: str, entity_type: str, payload: dict[str, Any]) -> EntityNode:
        components: list[Component] = []
        for name, attrs in (payload.get("components") or {}).items():
            component_attrs = tuple(
                Attribute(name=str(key), value=str(value))
                for key, value in (attrs or {}).items()
            )
            components.append(Component(name=str(name), attributes=component_attrs))

        attributes = tuple(
            Attribute(name=str(key), value=str(value))
            for key, value in (payload.get("attributes") or {}).items()
        )
        evidence = tuple(str(item) for item in payload.get("evidence", []))
        return EntityNode(
            entity_id=entity_id,
            type=entity_type,
            components=tuple(components),
            attributes=attributes,
            evidence=evidence,
        )


def flatten_entities(entities: Iterable[EntityNode], *, prefix: str = "") -> dict[str, str]:
    """Flatten several entity nodes into a single factor dict.

    Entity attributes are namespaced by entity id to avoid collisions when a
    decision point contains multiple entities.
    """

    flat: dict[str, str] = {}
    for entity in entities:
        entity_prefix = f"{prefix}{entity.entity_id}."
        for key, value in entity.flatten_attributes().items():
            flat[f"{entity_prefix}{key}"] = value
    return flat
