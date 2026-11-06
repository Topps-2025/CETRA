"""Executable, auditable skills: contract + handler + evidence + provenance.

``consolidation.Skill`` is a rule with support/precision.  A long-term skill
needs more than that:

- a contract an LLM can read (when/inputs/outputs/preconditions/effects/scope/
  exceptions/failure modes);
- a handler an executor can run;
- evidence (validation cases, gate decision, transfer);
- provenance (source task/family, version, parent, retirement status).

This module is the landing point for those four pieces and a small registry
with retrieval/retirement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from .induction import DEFAULT_ACTION_EFFECTS
from .types import CaseRecord, Rule


def _condition_text(conditions: tuple[tuple[str, str], ...], thresholds: tuple[tuple[str, int], ...]) -> str:
    parts = [f"{key}={value}" for key, value in conditions]
    parts += [f"int({key}) >= {value}" for key, value in thresholds]
    return " and ".join(parts) if parts else "always"


def _safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_").lower() or "skill"


@dataclass(frozen=True)
class SkillContract:
    name: str
    when: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    preconditions: tuple[tuple[str, str], ...]
    thresholds: tuple[tuple[str, int], ...]
    effects: tuple[str, ...]
    scope: tuple[str, ...]
    exceptions: tuple[str, ...]
    failure_modes: tuple[str, ...]
    examples: tuple[dict[str, str], ...] = ()

    def to_markdown(self) -> str:
        lines = [
            f"# {self.name}",
            "",
            "## When",
            self.when,
            "",
            "## Inputs",
            ", ".join(self.inputs) if self.inputs else "—",
            "",
            "## Outputs",
            ", ".join(self.outputs) if self.outputs else "—",
            "",
            "## Preconditions",
            _condition_text(self.preconditions, self.thresholds),
            "",
            "## Effects",
            ", ".join(self.effects) if self.effects else "—",
            "",
            "## Scope",
            ", ".join(self.scope) if self.scope else "—",
            "",
            "## Exceptions",
            ", ".join(self.exceptions) if self.exceptions else "—",
            "",
            "## Failure modes",
            ", ".join(self.failure_modes) if self.failure_modes else "—",
        ]
        if self.examples:
            lines += ["", "## Examples"]
            for example in self.examples:
                lines.append(f"- `{json.dumps(example, ensure_ascii=False)}`")
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "when": self.when,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "preconditions": [list(item) for item in self.preconditions],
            "thresholds": [list(item) for item in self.thresholds],
            "effects": list(self.effects),
            "scope": list(self.scope),
            "exceptions": list(self.exceptions),
            "failure_modes": list(self.failure_modes),
            "examples": [dict(item) for item in self.examples],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkillContract":
        return cls(
            name=payload["name"],
            when=payload.get("when", ""),
            inputs=tuple(payload.get("inputs", [])),
            outputs=tuple(payload.get("outputs", [])),
            preconditions=tuple(tuple(item) for item in payload.get("preconditions", [])),
            thresholds=tuple(tuple(item) for item in payload.get("thresholds", [])),
            effects=tuple(payload.get("effects", [])),
            scope=tuple(payload.get("scope", [])),
            exceptions=tuple(payload.get("exceptions", [])),
            failure_modes=tuple(payload.get("failure_modes", [])),
            examples=tuple(dict(item) for item in payload.get("examples", [])),
        )


@dataclass(frozen=True)
class SkillHandler:
    kind: str  # rule | procedure | prompt
    source: str
    entrypoint: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "source": self.source, "entrypoint": self.entrypoint}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkillHandler":
        return cls(
            kind=payload.get("kind", "rule"),
            source=payload.get("source", ""),
            entrypoint=payload.get("entrypoint", "run"),
        )


@dataclass(frozen=True)
class SkillEvidence:
    support: int
    precision: float
    validation_cases: tuple[str, ...] = ()
    transfer_gain: float = 0.0
    coverage: float = 0.0
    source_family: str = ""
    gate_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "support": self.support,
            "precision": self.precision,
            "validation_cases": list(self.validation_cases),
            "transfer_gain": self.transfer_gain,
            "coverage": self.coverage,
            "source_family": self.source_family,
            "gate_reason": self.gate_reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkillEvidence":
        return cls(
            support=int(payload.get("support", 0)),
            precision=float(payload.get("precision", 0.0)),
            validation_cases=tuple(payload.get("validation_cases", [])),
            transfer_gain=float(payload.get("transfer_gain", 0.0)),
            coverage=float(payload.get("coverage", 0.0)),
            source_family=payload.get("source_family", ""),
            gate_reason=payload.get("gate_reason", ""),
        )


@dataclass(frozen=True)
class SkillProvenance:
    source_task: str
    source_family: str
    version: str
    parent: str | None = None
    created_at: str = field(default_factory=lambda: date.today().isoformat())
    author: str = "cetra"
    evidence_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_task": self.source_task,
            "source_family": self.source_family,
            "version": self.version,
            "parent": self.parent,
            "created_at": self.created_at,
            "author": self.author,
            "evidence_ids": list(self.evidence_ids),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkillProvenance":
        return cls(
            source_task=payload.get("source_task", ""),
            source_family=payload.get("source_family", ""),
            version=payload.get("version", "v1"),
            parent=payload.get("parent"),
            created_at=payload.get("created_at", ""),
            author=payload.get("author", "cetra"),
            evidence_ids=tuple(payload.get("evidence_ids", [])),
        )


@dataclass
class SkillArtifact:
    skill_id: str
    contract: SkillContract
    handler: SkillHandler
    evidence: SkillEvidence
    provenance: SkillProvenance
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "status": self.status,
            "contract": self.contract.to_dict(),
            "handler": self.handler.to_dict(),
            "evidence": self.evidence.to_dict(),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkillArtifact":
        return cls(
            skill_id=payload["skill_id"],
            status=payload.get("status", "active"),
            contract=SkillContract.from_dict(payload["contract"]),
            handler=SkillHandler.from_dict(payload.get("handler", {})),
            evidence=SkillEvidence.from_dict(payload.get("evidence", {})),
            provenance=SkillProvenance.from_dict(payload.get("provenance", {})),
        )

    def write_dir(self, root: str | Path) -> Path:
        directory = Path(root) / self.skill_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "SKILL.md").write_text(self.contract.to_markdown(), encoding="utf-8")
        (directory / "handler.py").write_text(self.handler.source, encoding="utf-8")
        (directory / "evidence.json").write_text(
            json.dumps(self.evidence.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (directory / "provenance.json").write_text(
            json.dumps(self.provenance.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return directory


def _handler_source(name: str, rule: Rule) -> str:
    conditions = dict(rule.conditions)
    thresholds = dict(rule.thresholds)
    return (
        f"def {name}(case):\n"
        f"    \"\"\"Return the skill action when the contract matches, else None.\"\"\"\n"
        f"    factors = getattr(case, 'factors', {{}})\n"
        f"    conditions = {conditions!r}\n"
        f"    thresholds = {thresholds!r}\n"
        f"    if any(factors.get(k) != v for k, v in conditions.items()):\n"
        f"        return None\n"
        f"    for k, v in thresholds.items():\n"
        f"        try:\n"
        f"            if int(factors.get(k, '')) < v:\n"
        f"                return None\n"
        f"        except (TypeError, ValueError):\n"
        f"            return None\n"
        f"    return {rule.action!r}\n"
    )


def _matches_rule(rule: Rule, case: CaseRecord) -> bool:
    if any(case.factors.get(key) != value for key, value in rule.conditions):
        return False
    for key, value in rule.thresholds:
        try:
            if int(case.factors.get(key, "")) < value:
                return False
        except (TypeError, ValueError):
            return False
    return True


def artifact_from_rule(
    rule: Rule,
    *,
    source_family: str,
    version: str = "v1",
    parent: str | None = None,
    validation_cases: Iterable[CaseRecord] | None = None,
    transfer_gain: float = 0.0,
    coverage: float = 0.0,
    gate_reason: str = "",
) -> SkillArtifact:
    """Project a rule into a contract + handler + evidence + provenance."""

    validation = list(validation_cases or [])
    matched_validation = [case for case in validation if _matches_rule(rule, case)]
    if not matched_validation and not gate_reason:
        gate_reason = "consolidated; no matched validation case"
    inputs = tuple(sorted({key for key, _ in rule.conditions} | {key for key, _ in rule.thresholds}))
    effect = DEFAULT_ACTION_EFFECTS.get(rule.action)
    effects = (f"{effect[0]}={effect[1]}",) if effect else (rule.action,)
    name = _safe_name(f"{source_family}_{rule.action}_{'_'.join(inputs)}")
    contract = SkillContract(
        name=name,
        when=f"when {_condition_text(rule.conditions, rule.thresholds)}",
        inputs=inputs,
        outputs=(rule.action,),
        preconditions=rule.conditions,
        thresholds=rule.thresholds,
        effects=effects,
        scope=rule.scope,
        exceptions=(),
        failure_modes=("precondition not observable", "scope mismatch", "negative transfer"),
        examples=tuple(
            {"factors": dict(case.factors), "action": case.action}
            for case in matched_validation[:2]
        ),
    )
    evidence = SkillEvidence(
        support=rule.support,
        precision=rule.precision,
        validation_cases=tuple(case.case_id for case in matched_validation[:8]),
        transfer_gain=transfer_gain,
        coverage=coverage,
        source_family=source_family,
        gate_reason=gate_reason,
    )
    provenance = SkillProvenance(
        source_task=source_family,
        source_family=source_family,
        version=version,
        parent=parent,
        evidence_ids=tuple(case.case_id for case in matched_validation[:8]),
    )
    skill_id = _safe_name(
        f"skill_{source_family}_{rule.action}_{rule.conditions}_{rule.thresholds}"
    )[:80]
    return SkillArtifact(
        skill_id=skill_id,
        contract=contract,
        handler=SkillHandler(kind="rule", source=_handler_source(name, rule), entrypoint=name),
        evidence=evidence,
        provenance=provenance,
    )


@dataclass
class SkillRegistry:
    """A persistent, versioned skill library with retrieval and retirement."""

    skills: dict[str, SkillArtifact] = field(default_factory=dict)

    def add(self, artifact: SkillArtifact) -> None:
        existing = self.skills.get(artifact.skill_id)
        if existing is not None and existing.status == "active":
            # Keep the newer version but preserve the parent link if missing.
            if artifact.provenance.parent is None:
                artifact.provenance = SkillProvenance(
                    **{**artifact.provenance.__dict__, "parent": existing.provenance.version}
                )
        self.skills[artifact.skill_id] = artifact

    def get(self, skill_id: str) -> SkillArtifact | None:
        return self.skills.get(skill_id)

    def active(self) -> list[SkillArtifact]:
        return [skill for skill in self.skills.values() if skill.status == "active"]

    def retrieve(self, query: dict[str, str], *, limit: int = 8) -> list[SkillArtifact]:
        """Return active skills whose preconditions match the query.

        This is a deterministic symbolic retriever: it is intentionally simple
        and inspectable.  An embedding/LLM retriever can be plugged in later.
        """

        matches: list[SkillArtifact] = []
        for skill in self.active():
            preconditions = dict(skill.contract.preconditions)
            thresholds = dict(skill.contract.thresholds)
            if all(query.get(key) == value for key, value in preconditions.items()):
                threshold_ok = True
                for key, value in thresholds.items():
                    try:
                        if int(query.get(key, "")) < value:
                            threshold_ok = False
                            break
                    except (TypeError, ValueError):
                        threshold_ok = False
                        break
                if threshold_ok:
                    matches.append(skill)
        matches.sort(
            key=lambda skill: (len(skill.contract.preconditions), skill.evidence.precision),
            reverse=True,
        )
        return matches[:limit]

    def retire(self, skill_id: str, reason: str = "") -> bool:
        skill = self.skills.get(skill_id)
        if skill is None:
            return False
        skill.status = "retired"
        if reason:
            skill.evidence = SkillEvidence(
                **{**skill.evidence.__dict__, "gate_reason": reason}
            )
        return True

    def save(self, path: str | Path) -> None:
        payload = {
            "skills": [skill.to_dict() for skill in self.skills.values()],
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SkillRegistry":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        registry = cls()
        for item in payload.get("skills", []):
            registry.skills[item["skill_id"]] = SkillArtifact.from_dict(item)
        return registry

    def write_dir(self, root: str | Path) -> list[Path]:
        return [skill.write_dir(root) for skill in self.skills.values()]


@dataclass
class SkillRuntime:
    """Minimal executor: retrieve a matching skill and run its handler.

    The handler source is generated by ``artifact_from_rule`` and only uses
    ``getattr`` / ``int``.  A production runtime would sandbox or import the
    handler; this small executor is enough to prove the contract is usable at
    inference time, not just a document.
    """

    registry: SkillRegistry

    def predict(self, case: CaseRecord) -> tuple[str | None, str | None]:
        for skill in self.registry.retrieve(case.factors):
            namespace: dict[str, Any] = {
                "__builtins__": {
                    "any": any,
                    "all": all,
                    "bool": bool,
                    "float": float,
                    "getattr": getattr,
                    "int": int,
                    "len": len,
                    "max": max,
                    "min": min,
                    "str": str,
                }
            }
            try:
                exec(skill.handler.source, namespace)
                handler = namespace.get(skill.handler.entrypoint)
                if handler is None:
                    continue
                action = handler(case)
            except Exception:
                continue
            if action is not None:
                return str(action), skill.skill_id
        return None, None
