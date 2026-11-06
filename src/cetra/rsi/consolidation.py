"""Experience-to-skill consolidation.

The harness produces rules.  A skill is a rule that has been promoted into a
reusable library, with provenance and a validation record.  Consolidation is a
separate mechanism from rule induction:

- which rules are stable enough to promote;
- how much support/precision they need;
- whether they transfer to a different family;
- how many skills to keep.

That mechanism is itself editable.  A loose policy keeps many rules and can
overfit; a cross-validated policy keeps fewer rules that transfer.  The demo
compares the two on a held-out family, which is the evidence that the
consolidation mechanism improved rather than the rule set merely growing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from .harness import risk
from .types import CaseRecord, Harness, Representation, Rule


@dataclass(frozen=True)
class Skill:
    skill_id: str
    conditions: tuple[tuple[str, str], ...]
    thresholds: tuple[tuple[str, int], ...]
    action: str
    scope: tuple[str, ...]
    support: int
    precision: float
    source_family: str
    evidence: str = ""

    def to_rule(self) -> Rule:
        return Rule(
            conditions=self.conditions,
            thresholds=self.thresholds,
            action=self.action,
            scope=self.scope,
            support=self.support,
            precision=self.precision,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "conditions": [list(item) for item in self.conditions],
            "thresholds": [list(item) for item in self.thresholds],
            "action": self.action,
            "scope": list(self.scope),
            "support": self.support,
            "precision": round(self.precision, 6),
            "source_family": self.source_family,
            "evidence": self.evidence,
        }


@dataclass
class ConsolidationPolicy:
    name: str = "c0"
    min_support: int = 2
    min_precision: float = 0.8
    max_skills: int = 16
    cross_validate: bool = False
    cross_min_precision: float = 0.8
    cross_min_support: int = 2
    version: str = "c0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "min_support": self.min_support,
            "min_precision": self.min_precision,
            "max_skills": self.max_skills,
            "cross_validate": self.cross_validate,
            "cross_min_precision": self.cross_min_precision,
            "cross_min_support": self.cross_min_support,
            "version": self.version,
        }


@dataclass
class SkillLibrary:
    skills: tuple[Skill, ...]
    policy: ConsolidationPolicy
    source_family: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_family": self.source_family,
            "policy": self.policy.to_dict(),
            "skills": [skill.to_dict() for skill in self.skills],
        }


def _skill_id(skill: Skill) -> str:
    payload = json.dumps(
        {
            "conditions": [list(item) for item in skill.conditions],
            "thresholds": [list(item) for item in skill.thresholds],
            "action": skill.action,
            "scope": list(skill.scope),
        },
        sort_keys=True,
    )
    return f"skill_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:10]}"


def _matches(rule: Rule, case: CaseRecord) -> bool:
    if any(case.factors.get(concept) != value for concept, value in rule.conditions):
        return False
    for concept, threshold in rule.thresholds:
        try:
            if int(case.factors.get(concept, "")) < threshold:
                return False
        except (TypeError, ValueError):
            return False
    return True


def consolidate(
    rep: Representation,
    policy: ConsolidationPolicy,
    *,
    source_family: str = "",
    validation_cases: Iterable[CaseRecord] | None = None,
) -> SkillLibrary:
    """Promote stable rules into a skill library."""

    validation = list(validation_cases or [])
    skills: list[Skill] = []
    seen: set[tuple] = set()

    for rule in rep.rules:
        if rule.support < policy.min_support or rule.precision < policy.min_precision:
            continue
        if policy.cross_validate and validation:
            matched = [case for case in validation if _matches(rule, case)]
            if len(matched) < policy.cross_min_support:
                continue
            correct = sum(1 for case in matched if case.action == rule.action)
            if correct / len(matched) < policy.cross_min_precision:
                continue
        key = (rule.conditions, rule.thresholds, rule.action, rule.scope)
        if key in seen:
            continue
        seen.add(key)
        skill = Skill(
            skill_id="pending",
            conditions=rule.conditions,
            thresholds=rule.thresholds,
            action=rule.action,
            scope=rule.scope,
            support=rule.support,
            precision=rule.precision,
            source_family=source_family,
            evidence=f"support={rule.support}, precision={rule.precision:.3f}",
        )
        skills.append(
            Skill(
                skill_id=_skill_id(skill),
                conditions=skill.conditions,
                thresholds=skill.thresholds,
                action=skill.action,
                scope=skill.scope,
                support=skill.support,
                precision=skill.precision,
                source_family=skill.source_family,
                evidence=skill.evidence,
            )
        )

    skills.sort(key=lambda item: (item.precision, item.support), reverse=True)
    return SkillLibrary(
        skills=tuple(skills[: policy.max_skills]),
        policy=policy,
        source_family=source_family,
    )


def apply_library(library: SkillLibrary) -> Representation:
    """Build a blank representation containing only the consolidated skills."""

    return Representation(
        concepts=(),
        values={},
        rules=[skill.to_rule() for skill in library.skills],
        chains=[],
        evidence={},
        fallback=None,
    )


def transfer_metrics(
    harness: Harness,
    library: SkillLibrary,
    cases: list[CaseRecord],
    *,
    source_rule_count: int = 0,
) -> dict[str, Any]:
    """Evaluate a skill library alone on a held-out family."""

    if not cases:
        return {
            "n": 0,
            "risk_with": 0.0,
            "risk_without": 0.0,
            "transfer_gain": 0.0,
            "coverage": 0.0,
            "compression": 0.0,
            "skills": len(library.skills),
        }
    blank = apply_library(library)
    risk_with = risk(harness, blank, cases)
    risk_without = risk(harness, Representation(), cases)
    covered = sum(
        1
        for case in cases
        if any(_matches(skill.to_rule(), case) for skill in library.skills)
    )
    gain = risk_without - risk_with
    return {
        "n": len(cases),
        "risk_with": risk_with,
        "risk_without": risk_without,
        "transfer_gain": gain,
        "gain_per_skill": gain / max(1, len(library.skills)),
        "coverage": covered / len(cases),
        "compression": len(library.skills) / max(1, source_rule_count),
        "skills": len(library.skills),
    }


def consolidate_artifacts(
    rep: Representation,
    policy: ConsolidationPolicy,
    *,
    source_family: str = "",
    validation_cases: Iterable[CaseRecord] | None = None,
    transfer_gain: float = 0.0,
    coverage: float = 0.0,
    registry: "SkillRegistry | None" = None,
) -> "SkillRegistry":
    """Consolidate rules into full skill artifacts (contract/handler/evidence).

    ``consolidate`` remains the rule-only primitive for backwards compatibility.
    This function is the landing point used by real-trace experiments.
    """

    from .skill import SkillRegistry, artifact_from_rule

    validation = list(validation_cases or [])
    library = consolidate(
        rep,
        policy,
        source_family=source_family,
        validation_cases=validation,
    )
    result = registry or SkillRegistry()
    for skill in library.skills:
        result.add(
            artifact_from_rule(
                skill.to_rule(),
                source_family=source_family,
                validation_cases=validation,
                transfer_gain=transfer_gain,
                coverage=coverage,
                gate_reason="consolidated",
            )
        )
    return result


def policy_c0() -> ConsolidationPolicy:
    return ConsolidationPolicy(
        name="c0_loose",
        min_support=2,
        min_precision=0.8,
        max_skills=16,
        cross_validate=False,
        version="c0",
    )


def policy_c1() -> ConsolidationPolicy:
    return ConsolidationPolicy(
        name="c1_cross_validated",
        min_support=2,
        min_precision=0.9,
        max_skills=8,
        cross_validate=True,
        cross_min_precision=0.9,
        cross_min_support=2,
        version="c1",
    )
