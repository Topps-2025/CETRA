"""Latent concept sets produced by abductive inference."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


def canonical_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return cleaned or "concept"


@dataclass
class Concept:
    name: str
    definition: str
    values: list[str]
    indicators: dict[str, list[str]] = field(default_factory=dict)
    evidence_case_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Concept" | None:
        name = canonical_name(str(payload.get("name", "")))
        definition = str(payload.get("definition", "")).strip()
        raw_values = payload.get("values") or payload.get("possible_values") or []
        if isinstance(raw_values, str):
            raw_values = [raw_values]
        values: list[str] = []
        for value in raw_values:
            text = str(value).strip()
            if text and text not in values:
                values.append(text)
        if len(values) < 2 or not definition:
            return None

        raw_indicators = payload.get("indicators") or {}
        indicators: dict[str, list[str]] = {}
        if isinstance(raw_indicators, dict):
            for value, phrases in raw_indicators.items():
                value_text = str(value).strip()
                if isinstance(phrases, str):
                    phrases = [phrases]
                if not isinstance(phrases, list):
                    continue
                indicators[value_text] = [str(p).strip() for p in phrases if str(p).strip()]
        elif isinstance(raw_indicators, list):
            # If the model gave a flat list, attach it to the first value only.
            indicators[values[0]] = [str(p).strip() for p in raw_indicators if str(p).strip()]

        evidence = payload.get("evidence_case_ids") or []
        if not isinstance(evidence, list):
            evidence = []
        return cls(
            name=name,
            definition=definition,
            values=values,
            indicators=indicators,
            evidence_case_ids=[str(x) for x in evidence],
        )

    def indicator_count(self) -> int:
        return sum(len(v) for v in self.indicators.values())


@dataclass
class ConceptSet:
    concepts: list[Concept]

    def __len__(self) -> int:
        return len(self.concepts)

    def __iter__(self):
        return iter(self.concepts)

    def names(self) -> list[str]:
        return [concept.name for concept in self.concepts]

    def factor_values(self) -> dict[str, tuple[str, ...]]:
        return {concept.name: tuple(concept.values) for concept in self.concepts}

    def to_dict(self) -> dict[str, Any]:
        return {"concepts": [concept.to_dict() for concept in self.concepts]}

    @classmethod
    def from_dict(cls, payload: Any) -> "ConceptSet":
        if isinstance(payload, dict):
            raw_concepts = payload.get("concepts") or []
        elif isinstance(payload, list):
            raw_concepts = payload
        else:
            raw_concepts = []
        merged: dict[str, Concept] = {}
        for item in raw_concepts:
            if not isinstance(item, dict):
                continue
            concept = Concept.from_dict(item)
            if concept is None:
                continue
            if concept.name in merged:
                existing = merged[concept.name]
                existing.indicators = {
                    **existing.indicators,
                    **{k: sorted(set(existing.indicators.get(k, [])) | set(v)) for k, v in concept.indicators.items()},
                }
                existing.evidence_case_ids = sorted(
                    set(existing.evidence_case_ids) | set(concept.evidence_case_ids)
                )
            else:
                merged[concept.name] = concept
        return cls(list(merged.values())[:12])

    def grounding_report(self) -> dict[str, Any]:
        total = len(self.concepts)
        with_indicators = sum(1 for concept in self.concepts if concept.indicator_count() > 0)
        with_evidence = sum(1 for concept in self.concepts if concept.evidence_case_ids)
        return {
            "num_concepts": total,
            "with_indicators": with_indicators,
            "with_evidence": with_evidence,
            "indicator_rate": with_indicators / total if total else 0.0,
            "evidence_rate": with_evidence / total if total else 0.0,
        }

    def compact(self) -> list[dict[str, Any]]:
        return [
            {
                "name": concept.name,
                "definition": concept.definition,
                "values": concept.values,
                "indicators": concept.indicators,
            }
            for concept in self.concepts
        ]


def concept_values_from_record(
    record: dict[str, Any], concept_set: ConceptSet
) -> dict[str, str]:
    """Read a grounding record's values, restricted to the concept set."""
    raw = record.get("values") or record.get("assignments") or {}
    values: dict[str, str] = {}
    for concept in concept_set:
        value = raw.get(concept.name)
        if isinstance(value, dict):
            value = value.get("value")
        if value is None:
            continue
        text = str(value).strip()
        if text in concept.values:
            values[concept.name] = text
    return values


def evidence_from_record(record: dict[str, Any]) -> dict[str, str]:
    raw = record.get("evidence") or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            value = value.get("evidence")
        if value is None:
            continue
        out[str(key)] = str(value)[:500]
    return out


def merge_concept_sets(*sets: ConceptSet) -> ConceptSet:
    merged: dict[str, Concept] = {}
    for concept_set in sets:
        for concept in concept_set:
            if concept.name not in merged:
                merged[concept.name] = concept
            else:
                existing = merged[concept.name]
                existing.evidence_case_ids = sorted(
                    set(existing.evidence_case_ids) | set(concept.evidence_case_ids)
                )
                for value, phrases in concept.indicators.items():
                    existing.indicators.setdefault(value, [])
                    existing.indicators[value] = sorted(set(existing.indicators[value]) | set(phrases))
    return ConceptSet(list(merged.values()))
