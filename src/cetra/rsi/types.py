"""Core types for the CETRA-RSI memory-construction harness.

The harness is a typed program ``M = (O, T, Theta)``:

- ``O``: a set of construction operators (decompose, abduce, reconstruct,
  induce rules, compose scope, predict);
- ``T``: the pipeline topology (an ordered list of operator names);
- ``Theta``: operator parameters.

A representation ``H`` is the output of running ``M`` on raw cases.  The inner
loop refines ``H``; the outer loop edits ``M``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CaseRecord:
    """A concrete raw-memory case (one ticket)."""

    case_id: str
    split: str
    narrative: str
    factors: dict[str, str]
    action: str
    family: str = "default"

    def key(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self.factors.items()))


@dataclass(frozen=True)
class Rule:
    """A conjunctive rule with optional threshold conditions and a scope label.

    ``conditions`` are equality tests (``concept == value``).  ``thresholds``
    are numeric lower bounds (``int(concept) >= value``); they let a
    synthesised operator express a split such as ``retry_count >= 4`` without
    materialising every observed value as a separate categorical rule.
    Rules whose scope contains ``"fallback"`` are consulted only when no
    ordinary rule matches.
    """

    conditions: tuple[tuple[str, str], ...]
    action: str
    scope: tuple[str, ...] = ()
    support: int = 0
    precision: float = 1.0
    thresholds: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        # Canonical ordering makes rules comparable and deduplicable.
        object.__setattr__(self, "conditions", tuple(sorted(self.conditions)))
        object.__setattr__(self, "scope", tuple(sorted(self.scope)))
        object.__setattr__(self, "thresholds", tuple(sorted(self.thresholds)))

    def key(self) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, int], ...], str, tuple[str, ...]]:
        return (self.conditions, self.thresholds, self.action, self.scope)


@dataclass
class Representation:
    """The memory representation ``H = (Z, phi, R, Pi, C, E)``."""

    concepts: tuple[str, ...] = ()
    values: dict[str, tuple[str, ...]] = field(default_factory=dict)
    rules: list[Rule] = field(default_factory=list)
    chains: list[tuple[str, ...]] = field(default_factory=list)
    evidence: dict[str, list[str]] = field(default_factory=dict)
    fallback: str | None = None

    def copy(self) -> "Representation":
        return Representation(
            concepts=tuple(self.concepts),
            values={k: tuple(v) for k, v in self.values.items()},
            rules=list(self.rules),
            chains=list(self.chains),
            evidence={k: list(v) for k, v in self.evidence.items()},
            fallback=self.fallback,
        )


@dataclass
class BuildState:
    """Mutable state threaded through the harness pipeline."""

    concepts: list[str] = field(default_factory=list)
    values: dict[str, tuple[str, ...]] = field(default_factory=dict)
    rules: list[Rule] = field(default_factory=list)
    chains: list[tuple[str, ...]] = field(default_factory=list)
    evidence: dict[str, list[str]] = field(default_factory=dict)
    fallback: str | None = None
    traces: dict[str, Any] = field(default_factory=dict)

    def to_representation(self) -> Representation:
        return Representation(
            concepts=tuple(self.concepts),
            values={k: tuple(v) for k, v in self.values.items()},
            rules=list(self.rules),
            chains=list(self.chains),
            evidence={k: list(v) for k, v in self.evidence.items()},
            fallback=self.fallback,
        )


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    description: str = ""


@dataclass
class Harness:
    """A typed memory-construction harness."""

    operators: dict[str, Any]
    pipeline: list[str]
    params: dict[str, Any]
    version: str = "h0"
    parent: str | None = None
    children: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self, version: str | None = None) -> "Harness":
        return Harness(
            operators=dict(self.operators),
            pipeline=list(self.pipeline),
            params={k: (dict(v) if isinstance(v, dict) else v) for k, v in self.params.items()},
            version=version or self.version,
            parent=self.version,
            children=0,
            metadata=dict(self.metadata),
        )

    def spec(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "parent": self.parent,
            "pipeline": list(self.pipeline),
            "operators": {
                name: {
                    "spec": {
                        "name": op.spec.name,
                        "inputs": list(op.spec.inputs),
                        "outputs": list(op.spec.outputs),
                        "description": op.spec.description,
                    }
                }
                for name, op in self.operators.items()
            },
            "params": dict(self.params),
        }

    def build(self, cases: list[CaseRecord]) -> Representation:
        from .operators import BuildState as _BuildState

        state = _BuildState()
        for name in self.pipeline:
            if name not in self.operators:
                raise ValueError(f"pipeline references missing operator: {name}")
            state = self.operators[name].run(state, cases, self.params)
        return state.to_representation()

    def predict(self, rep: Representation, case: CaseRecord) -> str:
        ordinary = [
            rule
            for rule in rep.rules
            if "fallback" not in rule.scope and _rule_matches(rule, case)
        ]
        if ordinary:
            # Most specific first; then precision; then support.
            best = max(
                ordinary,
                key=lambda r: (
                    len(r.conditions) + len(r.thresholds),
                    r.precision,
                    r.support,
                ),
            )
            return best.action
        fallback = [
            rule
            for rule in rep.rules
            if "fallback" in rule.scope and _rule_matches(rule, case)
        ]
        if fallback:
            best = max(
                fallback,
                key=lambda r: (
                    len(r.conditions) + len(r.thresholds),
                    r.precision,
                    r.support,
                ),
            )
            return best.action
        return rep.fallback or "noop"


def _as_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rule_matches(rule: Rule, case: CaseRecord) -> bool:
    if any(case.factors.get(concept) != value for concept, value in rule.conditions):
        return False
    for concept, threshold in rule.thresholds:
        observed = _as_int(case.factors.get(concept))
        if observed is None or observed < threshold:
            return False
    return True


@dataclass(frozen=True)
class Diagnosis:
    """A process-level counterexample with harness attribution."""

    case_id: str
    predicted: str
    truth: str
    kind: str
    missing_concepts: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class Edit:
    """A typed harness edit.

    ``template_id`` records which edit-policy template produced the edit.  It is
    audit metadata, not part of the harness semantics, and lets the meta loop
    assign credit to templates across generations.
    """

    kind: str
    target: str
    args: dict[str, Any] = field(default_factory=dict)
    template_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {"kind": self.kind, "target": self.target, "args": dict(self.args)}
        if self.template_id:
            payload["template_id"] = self.template_id
        return payload


@dataclass
class Metrics:
    """Evaluation metrics for a harness on one data split."""

    n: int
    correct: int
    risk: float
    cost: float
    mp: float = 0.0
    counterexample_resolution: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "correct": self.correct,
            "risk": round(self.risk, 6),
            "cost": round(self.cost, 6),
            "mp": round(self.mp, 6),
            "counterexample_resolution": round(self.counterexample_resolution, 6),
        }


@dataclass
class ArchiveEntry:
    harness: Harness
    representation: Representation
    metrics: dict[str, Metrics]
    generation: int
    inner_history: list[dict[str, Any]] = field(default_factory=list)
    edit_from_parent: Edit | None = None
    accepted: bool = True

    @property
    def mp(self) -> float:
        return self.metrics["test"].mp
