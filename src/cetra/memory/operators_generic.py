"""Memory operations as operators over schema-driven FieldAtoms.

No atom type is hardcoded here.  An operator consumes a list of atoms and
returns a list of atoms (plus metrics/trace).  Workflows are ordered lists of
operators; different schemas can use different workflows.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Protocol

from cetra.rsi.induction import DEFAULT_ACTION_EFFECTS, induce_decision_tree
from cetra.rsi.types import CaseRecord, Rule

from .decomp_program import MISSING, DecompositionProgram, apply_program
from .fields import FieldAtom, FieldValue, events_to_field_atoms
from .need import NeedEstimator, NeedScore
from .segment import segment
from .types import Event


@dataclass
class OperatorResult:
    atoms: list[FieldAtom]
    metrics: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)


class MemoryOperator(Protocol):
    name: str

    def apply(self, atoms: list[FieldAtom]) -> OperatorResult: ...


def segment_to_field_atoms(events: list[Event]) -> list[FieldAtom]:
    """Decision-point segmentation -> schema-driven field atoms.

    The DP segmenter groups events into decision/event spans; this adapter
    preserves the segment as one atom instead of treating every raw event as an
    independent memory unit.
    """

    legacy_atoms = segment(events)
    by_id = {event.event_id: event for event in events}
    atoms: list[FieldAtom] = []
    for atom in legacy_atoms:
        chunk = [by_id[event_id] for event_id in atom.evidence if event_id in by_id]
        fields: dict[str, FieldValue] = {
            "kind": FieldValue("kind", atom.kind, evidence=atom.evidence),
            "content": FieldValue("content", atom.content, evidence=atom.evidence),
        }
        if atom.context:
            fields["context"] = FieldValue(
                "context", atom.context, evidence=atom.evidence
            )
        if atom.action:
            fields["action"] = FieldValue(
                "action", atom.action, evidence=atom.evidence
            )
        if atom.outcome:
            fields["outcome"] = FieldValue(
                "outcome", atom.outcome, evidence=atom.evidence
            )
        for event in chunk:
            for key, value in event.metadata.items():
                if value is None or key == "action":
                    continue
                fields.setdefault(
                    key,
                    FieldValue(key, str(value), evidence=(event.event_id,), scope=event.episode),
                )
        if chunk:
            fields.setdefault(
                "timestamp",
                FieldValue(
                    "timestamp",
                    chunk[0].timestamp,
                    evidence=(chunk[0].event_id,),
                    scope=chunk[0].episode,
                ),
            )
        atoms.append(
            FieldAtom(
                atom_id=atom.atom_id,
                schema="decision" if atom.action else "event",
                fields=fields,
                evidence=atom.evidence,
                scope=atom.episode,
                confidence=atom.confidence,
                source=atom.episode,
            )
        )
    return atoms


class FieldDecomposer:
    """Deterministic field -> subfield decomposition.

    It handles dicts, JSON strings, key=value lists, SWE-agent commands, and
    file observations.  An LLM decomposer can implement the same interface.
    """

    _KEY_VALUE = re.compile(r"([a-z][a-z0-9_]{0,30})\s*[=:]\s*([^;|\n]+)")
    _FILE = re.compile(r"\[File:\s*([^\(\]]+?)\s*\((\d+)\s+lines total\)\]")

    def decompose(self, field_name: str, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return {str(key): val for key, val in value.items() if _scalar(val)}
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("{") or text.startswith("["):
                try:
                    payload = json.loads(text)
                    if isinstance(payload, dict):
                        return {str(key): val for key, val in payload.items() if _scalar(val)}
                except json.JSONDecodeError:
                    pass
            file_match = self._FILE.search(text)
            if file_match:
                return {
                    "file_path": file_match.group(1).strip(),
                    "line_count": int(file_match.group(2)),
                }
            pairs = self._KEY_VALUE.findall(text)
            structured = (
                ";" in text
                or "|" in text
                or bool(re.search(r"^\s*[a-z][a-z0-9_]{0,30}\s*=", text, re.M))
            )
            if len(pairs) >= 2 and structured:
                return {key: val.strip() for key, val in pairs}
            # SWE-agent command: "edit 1:1", "python reproduce.py".
            if field_name == "command" and text:
                parts = text.split(maxsplit=1)
                return {
                    "tool": parts[0],
                    "arguments": parts[1] if len(parts) > 1 else "",
                }
        return {}


def _scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


class DecomposeOperator:
    name = "decompose"

    DEFAULT_FIELDS = {"command", "observation", "metadata", "factors"}

    def __init__(
        self,
        *,
        need: NeedEstimator | None = None,
        decomposer: FieldDecomposer | None = None,
        fields: Iterable[str] | None = None,
        max_depth: int = 2,
    ) -> None:
        self.need = need or NeedEstimator(max_depth=max_depth)
        self.decomposer = decomposer or FieldDecomposer()
        self.fields = set(fields) if fields is not None else set(self.DEFAULT_FIELDS)
        self.max_depth = max_depth

    def _candidate_fields(self, atoms: list[FieldAtom]) -> list[str]:
        return sorted(
            {
                key
                for atom in atoms
                for key, value in atom.fields.items()
                if self.fields is None or key in self.fields
                if self.decomposer.decompose(key, value.value)
            }
        )

    def _score_field(
        self,
        scoring_atoms: list[FieldAtom],
        field_name: str,
        action_values: list[str],
        depth: int,
    ) -> tuple[NeedScore, dict[str, list[Any]]]:
        child_lists: dict[str, list[Any]] = {}
        parent_values: list[Any] = []
        for atom in scoring_atoms:
            value = atom.get(field_name)
            parent_values.append(value)
            for key, child in self.decomposer.decompose(field_name, value).items():
                child_lists.setdefault(key, []).append(child)
        for key in child_lists:
            if len(child_lists[key]) < len(scoring_atoms):
                child_lists[key].extend([""] * (len(scoring_atoms) - len(child_lists[key])))
        score = self.need.score(field_name, parent_values, child_lists, action_values, depth)
        return score, child_lists

    def apply(self, atoms: list[FieldAtom]) -> OperatorResult:
        decisions: list[dict[str, Any]] = []
        added = 0
        labeled = [atom for atom in atoms if atom.get("action") is not None]
        scoring_atoms = labeled if len(labeled) >= 3 else atoms
        action_values = [
            str(atom.get("action"))
            for atom in scoring_atoms
            if atom.get("action") is not None
        ]
        frontier = self._candidate_fields(scoring_atoms)
        seen: set[str] = set()
        depth = 0
        for depth in range(self.max_depth):
            if not frontier:
                break
            next_frontier: set[str] = set()
            for field_name in frontier:
                if field_name in seen:
                    continue
                score, _ = self._score_field(
                    scoring_atoms, field_name, action_values, depth
                )
                decisions.append(score.to_dict())
                seen.add(field_name)
                if not score.accepted:
                    continue
                # Apply to every atom that has the field (not only labeled atoms).
                for atom in atoms:
                    decomposed = self.decomposer.decompose(field_name, atom.get(field_name))
                    for key, value in decomposed.items():
                        atom.set_field(
                            FieldValue(
                                key,
                                value,
                                evidence=atom.evidence,
                                confidence=0.8,
                                scope=atom.scope,
                                provenance=(atom.atom_id,),
                            )
                        )
                        added += 1
                        next_frontier.add(key)
            frontier = sorted(next_frontier - seen)
        return OperatorResult(
            atoms=atoms,
            metrics={
                "decompose_evaluated": len(decisions),
                "decompose_accepted": sum(1 for item in decisions if item["accepted"]),
                "fields_added": added,
                "decompose_depth": depth + 1 if decisions else 0,
            },
            trace=decisions,
        )


class ProgramDecomposeOperator:
    """Apply a compiled DecompIR program to field atoms.

    This is the bridge between the research loop (LLM proposes DecompIR, SE
    evolves it) and the generic operator workflow.  Atoms are converted to
    cases only to reuse the interpreter; the output is again a field atom.
    """

    name = "decompose_program"

    def __init__(
        self,
        program: DecompositionProgram,
        *,
        client: Any | None = None,
    ) -> None:
        self.program = program
        self.client = client

    def apply(self, atoms: list[FieldAtom]) -> OperatorResult:
        cases: list[CaseRecord] = []
        for atom in atoms:
            content = str(atom.get("context") or atom.get("content") or "")
            factors = {key: str(value.value) for key, value in atom.fields.items()}
            for spec in self.program.factors:
                # Narrative/context programs can fall back to the atom text;
                # a command program must not run on atoms without a command.
                if spec.source in {"narrative", "context", "content"}:
                    factors.setdefault(spec.source, content)
            cases.append(
                CaseRecord(
                    case_id=atom.atom_id,
                    split=atom.scope or "default",
                    narrative=content,
                    factors=factors,
                    action=str(atom.get("action") or "__none__"),
                    family=atom.scope or "default",
                )
            )
        rebuilt, report = apply_program(
            self.program, cases, client=self.client, keep_base=True
        )
        added = 0
        for atom, case in zip(atoms, rebuilt):
            for spec in self.program.factors:
                value = case.factors.get(spec.name)
                if value in (None, MISSING, ""):
                    continue
                atom.set_field(
                    FieldValue(
                        spec.name,
                        value,
                        evidence=atom.evidence,
                        confidence=0.9 if spec.extractor != "llm" else 0.6,
                        scope=atom.scope,
                        provenance=(atom.atom_id,),
                    )
                )
                added += 1
        return OperatorResult(
            atoms=atoms,
            metrics={
                "decompose_program": True,
                "factors": list(self.program.factor_names()),
                "coverage": round(report.mean_coverage(), 4),
                "llm_calls": report.llm_calls,
                "fields_added": added,
            },
            trace=[report.to_dict()],
        )


class ProjectOperator:
    name = "project"

    def __init__(self, *, top_k: int = 8) -> None:
        self.top_k = top_k

    def apply(self, atoms: list[FieldAtom]) -> OperatorResult:
        # Mutual information proxy: entropy reduction of action per field.
        labeled = [atom for atom in atoms if atom.get("action") is not None]
        informative = _informative_keys(atoms)
        scores: list[tuple[float, str]] = []
        for key in sorted(informative):
            groups: dict[str, list[str]] = {}
            for atom in labeled:
                groups.setdefault(str(atom.get(key)), []).append(str(atom.get("action")))
            before = _entropy([str(atom.get("action")) for atom in labeled])
            after = sum(
                len(group) / len(labeled) * _entropy(group)
                for group in groups.values()
            )
            scores.append((before - after, key))
        scores.sort(reverse=True)
        keep = {key for _, key in scores[: self.top_k]}
        projected = 0
        for atom in atoms:
            atom.set_field(
                FieldValue(
                    "projected_fields",
                    sorted(keep),
                    evidence=atom.evidence,
                    confidence=0.9,
                    scope=atom.scope,
                )
            )
            projected += 1
        return OperatorResult(
            atoms=atoms,
            metrics={
                "projected_atoms": projected,
                "projected_fields": sorted(keep),
                "field_scores": [
                    {"field": key, "information_gain": round(score, 6)}
                    for score, key in scores
                ],
            },
        )


def _entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    import math
    from collections import Counter

    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _informative_keys(atoms: list[FieldAtom], *, max_ratio: float = 0.5) -> set[str]:
    """Drop identifier-like fields (unique context/timestamp/content)."""

    labeled = [atom for atom in atoms if atom.get("action") is not None]
    if not labeled:
        return set()
    limit = max(5, int(len(labeled) * max_ratio))
    keep: set[str] = set()
    for key in {key for atom in labeled for key in atom.fields}:
        if key in {"action", "kind", "actor", "content", "timestamp"}:
            continue
        values = [str(atom.get(key)) for atom in labeled if atom.get(key) is not None]
        if len(values) < 2:
            continue
        if len(set(values)) <= limit:
            keep.add(key)
    return keep


def _rule_from_atom(atom: FieldAtom) -> Rule | None:
    action = atom.get("action")
    conditions = atom.get("conditions")
    if not action or not conditions:
        return None
    try:
        cond = tuple(tuple(item) for item in conditions)
    except TypeError:
        return None
    return Rule(
        conditions=cond,
        action=str(action),
        scope=tuple(atom.get("scope", []) or []),
        support=int(atom.get("support", 0) or 0),
        precision=float(atom.get("precision", 1.0) or 1.0),
    )


def _atom_from_rule(rule: Rule, *, source: str = "abstract") -> FieldAtom:
    return FieldAtom(
        atom_id=f"rule:{hash(rule.key()) & 0xFFFFFFFF:x}",
        schema="rule",
        fields={
            "conditions": FieldValue("conditions", [list(item) for item in rule.conditions]),
            "action": FieldValue("action", rule.action),
            "scope": FieldValue("scope", list(rule.scope)),
            "support": FieldValue("support", rule.support),
            "precision": FieldValue("precision", rule.precision),
            "effect": FieldValue("effect", DEFAULT_ACTION_EFFECTS.get(rule.action)),
            "evidence": FieldValue("evidence", list(rule.scope)),
        },
        evidence=tuple(),
        scope=source,
    )


class AbstractOperator:
    name = "abstract"

    def __init__(self, *, min_support: int = 2, min_precision: float = 0.8) -> None:
        self.min_support = min_support
        self.min_precision = min_precision

    def apply(self, atoms: list[FieldAtom]) -> OperatorResult:
        keep = _informative_keys(atoms)
        cases = []
        for atom in atoms:
            case = atom.to_case()
            if case is None:
                continue
            factors = {key: value for key, value in case.factors.items() if key in keep}
            cases.append(replace(case, factors=factors))
        if len(cases) < 3 or len({case.action for case in cases}) < 2:
            return OperatorResult(
                atoms=atoms,
                metrics={"rules": 0, "reason": "not enough labeled decision atoms"},
            )
        rules = induce_decision_tree(
            cases,
            max_depth=3,
            min_support=self.min_support,
            min_precision=self.min_precision,
            min_gain=1e-9,
        )
        rule_atoms = [_atom_from_rule(rule) for rule in rules]
        return OperatorResult(
            atoms=atoms + rule_atoms,
            metrics={"rules": len(rule_atoms)},
            trace=[atom.to_dict() for atom in rule_atoms[:20]],
        )


class ComposeOperator:
    name = "compose"

    def apply(self, atoms: list[FieldAtom]) -> OperatorResult:
        rules = [_rule_from_atom(atom) for atom in atoms if atom.schema == "rule"]
        rules = [rule for rule in rules if rule is not None]
        chains: list[FieldAtom] = []
        effect_chains = 0
        for start in rules:
            effect = DEFAULT_ACTION_EFFECTS.get(start.action)
            if not effect:
                continue
            for nxt in rules:
                if nxt is start:
                    continue
                if effect in nxt.conditions:
                    conditions = list(start.conditions) + list(nxt.conditions)
                    chains.append(
                        FieldAtom(
                            atom_id=f"chain:{start.action}->{nxt.action}",
                            schema="chain",
                            fields={
                                "kind": FieldValue("kind", "effect_chain"),
                                "rules": FieldValue(
                                    "rules",
                                    [
                                        {"action": start.action, "conditions": [list(x) for x in start.conditions]},
                                        {"action": nxt.action, "conditions": [list(x) for x in nxt.conditions]},
                                    ],
                                ),
                                "preconditions": FieldValue(
                                    "preconditions", [list(x) for x in conditions]
                                ),
                                "effects": FieldValue(
                                    "effects", list(DEFAULT_ACTION_EFFECTS.get(nxt.action, ()))
                                ),
                                "scope": FieldValue("scope", list(start.scope) + list(nxt.scope)),
                            },
                            evidence=tuple(),
                            scope="chain",
                        )
                    )
                    effect_chains += 1
        # Generic sequence fallback: for traces without an effect map, keep the
        # observed action order as a candidate chain.  It is a candidate, not a
        # verified effect chain; the gate/evidence distinguishes them.
        by_episode: dict[str, list[FieldAtom]] = {}
        for atom in atoms:
            if atom.schema == "decision":
                by_episode.setdefault(atom.scope, []).append(atom)
        sequence_chains = 0
        for episode, decisions in by_episode.items():
            if len(decisions) < 2:
                continue
            actions = [str(atom.get("action")) for atom in decisions]
            chains.append(
                FieldAtom(
                    atom_id=f"chain:sequence:{episode}",
                    schema="chain",
                    fields={
                        "kind": FieldValue("kind", "sequence"),
                        "rules": FieldValue(
                            "rules",
                            [{"action": action, "atom_id": atom.atom_id}
                             for action, atom in zip(actions, decisions)],
                        ),
                        "preconditions": FieldValue("preconditions", []),
                        "effects": FieldValue("effects", [actions[-1]]),
                        "scope": FieldValue("scope", [episode]),
                        "evidence": FieldValue(
                            "evidence", [atom.atom_id for atom in decisions]
                        ),
                    },
                    evidence=tuple(
                        e for atom in decisions for e in atom.evidence
                    ),
                    scope="chain",
                )
            )
            sequence_chains += 1
        return OperatorResult(
            atoms=atoms + chains,
            metrics={
                "chains": len(chains),
                "effect_chains": effect_chains,
                "sequence_chains": sequence_chains,
            },
            trace=[atom.to_dict() for atom in chains[:10]],
        )


class ConsolidateOperator:
    name = "consolidate"

    def __init__(self, *, min_support: int = 2, min_precision: float = 0.8) -> None:
        self.min_support = min_support
        self.min_precision = min_precision

    def apply(self, atoms: list[FieldAtom]) -> OperatorResult:
        skills: list[FieldAtom] = []
        active = 0
        for atom in atoms:
            if atom.schema != "rule":
                continue
            support = int(atom.get("support", 0) or 0)
            precision = float(atom.get("precision", 0) or 0)
            valid = support >= self.min_support and precision >= self.min_precision
            status = "active" if valid else "candidate"
            if valid:
                active += 1
            conditions = atom.get("conditions", [])
            action = atom.get("action")
            reason = "passed support/precision" if valid else (
                f"support={support} < {self.min_support}"
                if support < self.min_support
                else f"precision={precision:.2f} < {self.min_precision}"
            )
            skills.append(
                FieldAtom(
                    atom_id=f"skill:{atom.atom_id}",
                    schema="skill",
                    fields={
                        "status": FieldValue("status", status),
                        "gate_reason": FieldValue("gate_reason", reason),
                        "contract": FieldValue(
                            "contract",
                            {
                                "when": conditions,
                                "outputs": [action],
                                "preconditions": conditions,
                                "scope": atom.get("scope", []),
                                "exceptions": [],
                            },
                        ),
                        "handler": FieldValue(
                            "handler", f"def run(case): return {action!r}"
                        ),
                        "evidence": FieldValue(
                            "evidence",
                            {
                                "support": support,
                                "precision": precision,
                            },
                        ),
                        "provenance": FieldValue(
                            "provenance",
                            {"source_rule": atom.atom_id, "version": "v1"},
                        ),
                    },
                    evidence=atom.evidence,
                    scope="skill",
                )
            )
        return OperatorResult(
            atoms=atoms + skills,
            metrics={
                "skills": len(skills),
                "skills_active": active,
                "skills_candidate": len(skills) - active,
            },
            trace=[atom.to_dict() for atom in skills],
        )


@dataclass
class Workflow:
    name: str
    operators: list[MemoryOperator]

    def run(self, atoms: list[FieldAtom]) -> OperatorResult:
        current = list(atoms)
        metrics: dict[str, Any] = {"operators": []}
        trace: list[dict[str, Any]] = []
        for operator in self.operators:
            result = operator.apply(current)
            current = result.atoms
            metrics["operators"].append({"name": operator.name, **result.metrics})
            trace.extend(result.trace)
        metrics["atoms"] = len(current)
        metrics["schemas"] = {}
        for atom in current:
            metrics["schemas"][atom.schema] = metrics["schemas"].get(atom.schema, 0) + 1
        return OperatorResult(atoms=current, metrics=metrics, trace=trace)


def default_workflow(
    *,
    max_depth: int = 2,
    need: NeedEstimator | None = None,
    decomposition_program: DecompositionProgram | None = None,
    client: Any | None = None,
) -> Workflow:
    if decomposition_program is not None:
        decompose: MemoryOperator = ProgramDecomposeOperator(
            decomposition_program, client=client
        )
    else:
        decompose = DecomposeOperator(need=need, max_depth=max_depth)
    return Workflow(
        name="decision_memory",
        operators=[
            decompose,
            ProjectOperator(top_k=8),
            AbstractOperator(min_support=2, min_precision=0.8),
            ComposeOperator(),
            ConsolidateOperator(min_support=2, min_precision=0.8),
        ],
    )


def run_events(
    events: list[Event],
    workflow: Workflow | None = None,
    *,
    segment_events: bool = True,
) -> OperatorResult:
    # Decision-point segmentation is the default: a raw event is not a
    # decision-sufficient memory unit.  ``segment_events=False`` keeps the
    # per-event baseline for ablation.
    atoms = (
        segment_to_field_atoms(events)
        if segment_events
        else events_to_field_atoms(events)
    )
    return (workflow or default_workflow()).run(atoms)
