"""The edit policy ``U``: a first-class, editable improvement operator.

In the first CETRA-RSI prototype the proposer was a hard-coded Python function.
That makes the loop self-modifying at the harness level, but the mechanism that
proposes and selects edits remains host code.  This module reifies that
mechanism as data:

``U = (templates, statistics, budget, synthesis policy, meta rules)``.

The outer loop calls ``U``; the meta loop edits ``U``.  A template is a typed
recipe from a diagnosis kind to a candidate edit bundle.  Statistics record
which recipes worked.  Meta rules decide which meta-edits ``U`` proposes when
it diagnoses its own failures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .diagnosis_utils import error_counts
from .types import Diagnosis, Edit, Harness


# ---------------------------------------------------------------------------
# Templates and statistics
# ---------------------------------------------------------------------------


@dataclass
class TemplateStats:
    attempts: int = 0
    accepted: int = 0
    rejected: int = 0
    invalid: int = 0
    total_delta_mp: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.accepted / self.attempts if self.attempts else 0.0

    def ucb(self, exploration: float = 0.25) -> float:
        if not self.attempts:
            return 1.0 + exploration
        bonus = exploration * math.sqrt(math.log(max(2, self.attempts + 1)) / self.attempts)
        return self.success_rate + bonus

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "invalid": self.invalid,
            "total_delta_mp": round(self.total_delta_mp, 8),
            "success_rate": round(self.success_rate, 6),
        }


@dataclass(frozen=True)
class EditTemplate:
    """A typed recipe from a diagnosis kind to a candidate edit bundle."""

    template_id: str
    diagnosis_kind: str
    edit_kind: str
    target: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    wire_before: str | None = "predict"
    requires_operator: str | None = None
    forbids_operator: str | None = None
    synthesis: bool = False
    prior: float = 0.0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "diagnosis_kind": self.diagnosis_kind,
            "edit_kind": self.edit_kind,
            "target": self.target,
            "args": dict(self.args),
            "wire_before": self.wire_before,
            "requires_operator": self.requires_operator,
            "forbids_operator": self.forbids_operator,
            "synthesis": self.synthesis,
            "prior": self.prior,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EditTemplate":
        return cls(
            template_id=payload["template_id"],
            diagnosis_kind=payload["diagnosis_kind"],
            edit_kind=payload["edit_kind"],
            target=payload.get("target", ""),
            args=dict(payload.get("args", {})),
            wire_before=payload.get("wire_before", "predict"),
            requires_operator=payload.get("requires_operator"),
            forbids_operator=payload.get("forbids_operator"),
            synthesis=bool(payload.get("synthesis", False)),
            prior=float(payload.get("prior", 0.0)),
            description=payload.get("description", ""),
        )


@dataclass(frozen=True)
class MetaRule:
    """A rule that maps a meta-diagnosis to a proposed meta-edit.

    ``U`` starts with a small set of rules.  The meta loop can add, remove, or
    re-weight rules, which is what makes the improvement operator itself an
    object of improvement.
    """

    rule_id: str
    when: str
    then: str
    args: dict[str, Any] = field(default_factory=dict)
    min_count: int = 1
    enabled: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "when": self.when,
            "then": self.then,
            "args": dict(self.args),
            "min_count": self.min_count,
            "enabled": self.enabled,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MetaRule":
        return cls(
            rule_id=payload["rule_id"],
            when=payload["when"],
            then=payload["then"],
            args=dict(payload.get("args", {})),
            min_count=int(payload.get("min_count", 1)),
            enabled=bool(payload.get("enabled", True)),
            description=payload.get("description", ""),
        )


# ---------------------------------------------------------------------------
# Edit policy
# ---------------------------------------------------------------------------


@dataclass
class EditPolicy:
    templates: dict[str, list[EditTemplate]]
    stats: dict[str, TemplateStats] = field(default_factory=dict)
    meta_rules: list[MetaRule] = field(default_factory=list)
    meta_rule_library: dict[str, dict[str, Any]] = field(default_factory=dict)
    max_proposals: int = 4
    exploration: float = 0.25
    synthesis_enabled: bool = False
    synthesis_max_arity: int = 2
    synthesis_beam: int = 10
    synthesis_primitives: tuple[str, ...] = ("eq", "threshold", "fallback")
    gate_holdout_margin: float = 1e-9
    gate_mp_epsilon: float = 1e-12
    consolidation_enabled: bool = False
    consolidation_min_support: int = 2
    consolidation_min_precision: float = 0.8
    consolidation_max_skills: int = 12
    consolidation_cross_validate: bool = True
    version: str = "u0"
    parent: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def clone(self, version: str) -> "EditPolicy":
        return EditPolicy(
            templates={
                kind: [EditTemplate.from_dict(item.to_dict()) for item in items]
                for kind, items in self.templates.items()
            },
            stats={
                key: TemplateStats(
                    attempts=value.attempts,
                    accepted=value.accepted,
                    rejected=value.rejected,
                    invalid=value.invalid,
                    total_delta_mp=value.total_delta_mp,
                )
                for key, value in self.stats.items()
            },
            meta_rules=[MetaRule.from_dict(item.to_dict()) for item in self.meta_rules],
            meta_rule_library={
                key: dict(value) for key, value in self.meta_rule_library.items()
            },
            max_proposals=self.max_proposals,
            exploration=self.exploration,
            synthesis_enabled=self.synthesis_enabled,
            synthesis_max_arity=self.synthesis_max_arity,
            synthesis_beam=self.synthesis_beam,
            synthesis_primitives=tuple(self.synthesis_primitives),
            gate_holdout_margin=self.gate_holdout_margin,
            gate_mp_epsilon=self.gate_mp_epsilon,
            consolidation_enabled=self.consolidation_enabled,
            consolidation_min_support=self.consolidation_min_support,
            consolidation_min_precision=self.consolidation_min_precision,
            consolidation_max_skills=self.consolidation_max_skills,
            consolidation_cross_validate=self.consolidation_cross_validate,
            version=version,
            parent=self.version,
            history=list(self.history),
        )

    def ensure_stats(self, template_id: str) -> TemplateStats:
        return self.stats.setdefault(template_id, TemplateStats())

    def add_template(self, template: EditTemplate) -> None:
        self.templates.setdefault(template.diagnosis_kind, [])
        if any(item.template_id == template.template_id for item in self.templates[template.diagnosis_kind]):
            return
        self.templates[template.diagnosis_kind].append(template)
        self.ensure_stats(template.template_id)

    def remove_template(self, template_id: str) -> bool:
        for kind, templates in self.templates.items():
            remaining = [item for item in templates if item.template_id != template_id]
            if len(remaining) != len(templates):
                self.templates[kind] = remaining
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "parent": self.parent,
            "templates": {
                kind: [item.to_dict() for item in items]
                for kind, items in self.templates.items()
            },
            "stats": {key: value.to_dict() for key, value in self.stats.items()},
            "meta_rules": [rule.to_dict() for rule in self.meta_rules],
            "meta_rule_library": {
                key: dict(value) for key, value in self.meta_rule_library.items()
            },
            "max_proposals": self.max_proposals,
            "exploration": self.exploration,
            "synthesis_enabled": self.synthesis_enabled,
            "synthesis_max_arity": self.synthesis_max_arity,
            "synthesis_beam": self.synthesis_beam,
            "synthesis_primitives": list(self.synthesis_primitives),
            "gate_holdout_margin": self.gate_holdout_margin,
            "gate_mp_epsilon": self.gate_mp_epsilon,
            "consolidation_enabled": self.consolidation_enabled,
            "consolidation_min_support": self.consolidation_min_support,
            "consolidation_min_precision": self.consolidation_min_precision,
            "consolidation_max_skills": self.consolidation_max_skills,
            "consolidation_cross_validate": self.consolidation_cross_validate,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EditPolicy":
        return cls(
            templates={
                kind: [EditTemplate.from_dict(item) for item in items]
                for kind, items in payload.get("templates", {}).items()
            },
            stats={
                key: TemplateStats(
                    attempts=value.get("attempts", 0),
                    accepted=value.get("accepted", 0),
                    rejected=value.get("rejected", 0),
                    invalid=value.get("invalid", 0),
                    total_delta_mp=value.get("total_delta_mp", 0.0),
                )
                for key, value in payload.get("stats", {}).items()
            },
            meta_rules=[MetaRule.from_dict(item) for item in payload.get("meta_rules", [])],
            meta_rule_library={
                key: dict(value)
                for key, value in payload.get("meta_rule_library", {}).items()
            },
            max_proposals=int(payload.get("max_proposals", 4)),
            exploration=float(payload.get("exploration", 0.25)),
            synthesis_enabled=bool(payload.get("synthesis_enabled", False)),
            synthesis_max_arity=int(payload.get("synthesis_max_arity", 2)),
            synthesis_beam=int(payload.get("synthesis_beam", 10)),
            synthesis_primitives=tuple(payload.get("synthesis_primitives", ("eq", "threshold", "fallback"))),
            gate_holdout_margin=float(payload.get("gate_holdout_margin", 1e-9)),
            gate_mp_epsilon=float(payload.get("gate_mp_epsilon", 1e-12)),
            consolidation_enabled=bool(payload.get("consolidation_enabled", False)),
            consolidation_min_support=int(payload.get("consolidation_min_support", 2)),
            consolidation_min_precision=float(payload.get("consolidation_min_precision", 0.8)),
            consolidation_max_skills=int(payload.get("consolidation_max_skills", 12)),
            consolidation_cross_validate=bool(payload.get("consolidation_cross_validate", True)),
            version=payload.get("version", "u0"),
            parent=payload.get("parent"),
            history=list(payload.get("history", [])),
        )


# ---------------------------------------------------------------------------
# Seed policy
# ---------------------------------------------------------------------------


def seed_policy(*, version: str = "u0") -> EditPolicy:
    """A deliberately weak starting policy.

    It can add the known ``compose_scope`` operator and tune the seed learner,
    but it has no synthesis template and no meta-rule that would enable one.
    That is the gap the meta loop must discover and close.
    """

    templates: list[EditTemplate] = [
        EditTemplate(
            template_id="t_scope_add",
            diagnosis_kind="operator_gap",
            edit_kind="ADD_OPERATOR",
            target="compose_scope",
            forbids_operator="compose_scope",
            wire_before="predict",
            prior=0.2,
            description="add the known scope-composition operator",
        ),
        EditTemplate(
            template_id="t_scope_add_scope_error",
            diagnosis_kind="scope_error",
            edit_kind="ADD_OPERATOR",
            target="compose_scope",
            forbids_operator="compose_scope",
            wire_before="predict",
            prior=0.2,
            description="add scope composition when a rule fails outside its scope",
        ),
        EditTemplate(
            template_id="t_tune_support",
            diagnosis_kind="missing_rule",
            edit_kind="TUNE",
            target="induce_single",
            args={"key": "min_support", "value": 1},
            wire_before=None,
            prior=0.0,
            description="lower the support threshold",
        ),
        EditTemplate(
            template_id="t_tune_support_concept",
            diagnosis_kind="missing_concept",
            edit_kind="TUNE",
            target="induce_single",
            args={"key": "min_support", "value": 1},
            wire_before=None,
            prior=0.0,
            description="lower the support threshold after a concept is added",
        ),
        EditTemplate(
            template_id="t_tune_precision",
            diagnosis_kind="wrong_granularity",
            edit_kind="TUNE",
            target="induce_single",
            args={"key": "min_precision", "value": 0.4},
            wire_before=None,
            prior=0.0,
            description="relax rule purity",
        ),
        EditTemplate(
            template_id="t_compose_tune",
            diagnosis_kind="composition_error",
            edit_kind="TUNE",
            target="compose_scope",
            args={"key": "max_conditions", "value": 3},
            wire_before=None,
            requires_operator="compose_scope",
            prior=0.0,
            description="increase the arity of the known composition operator",
        ),
        EditTemplate(
            template_id="t_fallback_action",
            diagnosis_kind="fallback_gap",
            edit_kind="ADD_FALLBACK",
            target="",
            args={"action": "escalate"},
            wire_before=None,
            prior=-0.1,
            description="set a global fallback action",
        ),
        EditTemplate(
            template_id="t_synth_operator_gap",
            diagnosis_kind="operator_gap",
            edit_kind="SYNTHESIZE_OPERATOR",
            target="operator_gap",
            args={"scope_hint": "ordinary"},
            wire_before="predict",
            synthesis=True,
            prior=-0.5,
            description="synthesize a new typed operator for an operator gap",
        ),
        EditTemplate(
            template_id="t_synth_missing_rule",
            diagnosis_kind="missing_rule",
            edit_kind="SYNTHESIZE_OPERATOR",
            target="missing_rule",
            args={"scope_hint": "ordinary"},
            wire_before="predict",
            synthesis=True,
            prior=-0.4,
            description="synthesize a rule program when no ordinary rule covers a case",
        ),
        EditTemplate(
            template_id="t_synth_composition_error",
            diagnosis_kind="composition_error",
            edit_kind="SYNTHESIZE_OPERATOR",
            target="composition_error",
            args={"scope_hint": "ordinary"},
            wire_before="predict",
            synthesis=True,
            prior=-0.4,
            description="synthesize a composition program when the known operator is too weak",
        ),
        EditTemplate(
            template_id="t_synth_fallback_gap",
            diagnosis_kind="fallback_gap",
            edit_kind="SYNTHESIZE_OPERATOR",
            target="fallback_gap",
            args={"scope_hint": "fallback"},
            wire_before="predict",
            synthesis=True,
            prior=-0.5,
            description="synthesize a scoped fallback operator",
        ),
    ]
    grouped: dict[str, list[EditTemplate]] = {}
    for template in templates:
        grouped.setdefault(template.diagnosis_kind, []).append(template)

    policy = EditPolicy(
        templates=grouped,
        stats={template.template_id: TemplateStats() for template in templates},
        meta_rules=[
            MetaRule(
                rule_id="mr_template_rejected",
                when="template_rejected",
                then="UPDATE_PRIOR",
                args={"delta": -0.1},
                description="downweight a template that is repeatedly rejected",
            ),
            MetaRule(
                rule_id="mr_no_template",
                when="no_template",
                then="ADD_TEMPLATE",
                description="add a template when a diagnosis kind has no recipe",
            ),
            MetaRule(
                rule_id="mr_synthesis_failed",
                when="synthesis_failed",
                then="INCREASE_SYNTHESIS_DEPTH",
                description="widen the synthesis grammar after a synthesis failure",
            ),
            MetaRule(
                rule_id="mr_budget_exhausted",
                when="budget_exhausted",
                then="INCREASE_BUDGET",
                description="increase the proposal budget when search is exhausted",
            ),
            MetaRule(
                rule_id="mr_acquire_meta_rule",
                when="no_meta_rule",
                then="ADD_META_RULE",
                description="acquire a rule from the meta-rule library when a diagnosis has none",
            ),
        ],
        meta_rule_library={
            "synthesis_missing": {
                "rule_id": "mr_synthesis_missing",
                "when": "synthesis_missing",
                "then": "ENABLE_SYNTHESIS",
                "args": {"max_arity": 3, "beam": 16},
                "description": "enable open operator synthesis when a gap repeats",
            },
            "unsolved_family": {
                "rule_id": "mr_unsolved_family",
                "when": "unsolved_family",
                "then": "ENABLE_SYNTHESIS",
                "args": {"max_arity": 3, "beam": 16},
                "description": "enable synthesis when a family is unsolved",
            },
            "consolidation_missing": {
                "rule_id": "mr_consolidation_missing",
                "when": "consolidation_missing",
                "then": "ENABLE_CONSOLIDATION",
                "args": {
                    "min_support": 2,
                    "min_precision": 0.9,
                    "max_skills": 8,
                    "cross_validate": True,
                },
                "description": "consolidate stable rules into a skill library",
            },
            "synthesis_proven": {
                "rule_id": "mr_synthesis_proven",
                "when": "synthesis_proven",
                "then": "PROMOTE_SYNTHESIS",
                "args": {},
                "description": "rank synthesis above known-operator templates",
            },
            "depth_limited": {
                "rule_id": "mr_depth_limited",
                "when": "depth_limited",
                "then": "INCREASE_SYNTHESIS_DEPTH",
                "args": {},
                "description": "widen the synthesis grammar when a family is partly solved",
            },
            "cost_high": {
                "rule_id": "mr_cost_high",
                "when": "cost_high",
                "then": "DECREASE_SYNTHESIS_BEAM",
                "args": {},
                "description": "reduce search cost when every family is solved",
            },
        },
        version=version,
    )
    return policy


# ---------------------------------------------------------------------------
# Proposal
# ---------------------------------------------------------------------------


def _instantiate(template: EditTemplate) -> list[Edit]:
    if template.edit_kind == "ADD_OPERATOR":
        edits = [
            Edit(
                "ADD_OPERATOR",
                template.target,
                dict(template.args),
                template_id=template.template_id,
            )
        ]
        if template.wire_before:
            edits.append(
                Edit(
                    "REWIRE",
                    template.target,
                    {"before": template.wire_before},
                    template_id=template.template_id,
                )
            )
        return edits
    return [
        Edit(
            template.edit_kind,
            template.target,
            dict(template.args),
            template_id=template.template_id,
        )
    ]


def propose_with_policy(
    harness: Harness,
    errors: list[Diagnosis],
    policy: EditPolicy,
    *,
    generation: int,
) -> list[list[Edit]]:
    """Rank templates by UCB and return candidate edit bundles."""

    kinds = error_counts(errors)
    candidates: list[tuple[float, list[Edit], str]] = []
    for kind, count in sorted(kinds.items(), key=lambda item: (-item[1], item[0])):
        for template in policy.templates.get(kind, []):
            if template.synthesis and not policy.synthesis_enabled:
                continue
            if template.requires_operator and template.requires_operator not in harness.operators:
                continue
            if template.forbids_operator and template.forbids_operator in harness.operators:
                continue
            stats = policy.ensure_stats(template.template_id)
            score = stats.ucb(policy.exploration) + template.prior + 0.01 * count
            candidates.append((score, _instantiate(template), template.template_id))

    # If a gap has no usable template but synthesis is enabled, synthesize.
    if policy.synthesis_enabled:
        for kind in ("operator_gap", "fallback_gap"):
            if kind in kinds and not any(
                template.diagnosis_kind == kind and not template.synthesis
                for template in policy.templates.get(kind, [])
            ):
                template_id = f"t_synth_{kind}_fallback"
                candidates.append(
                    (
                        1.0,
                        [
                            Edit(
                                "SYNTHESIZE_OPERATOR",
                                kind,
                                {"scope_hint": "fallback" if kind == "fallback_gap" else "ordinary"},
                                template_id=template_id,
                            )
                        ],
                        template_id,
                    )
                )

    candidates.sort(key=lambda item: item[0], reverse=True)
    bundles: list[list[Edit]] = []
    seen: set[tuple[str, ...]] = set()
    for _, edits, _ in candidates:
        key = tuple(edit.to_dict().get("kind", "") + ":" + edit.target for edit in edits)
        if key in seen:
            continue
        seen.add(key)
        bundles.append(edits)
        if len(bundles) >= policy.max_proposals:
            break
    return bundles


def update_stats(policy: EditPolicy, record: Any) -> None:
    """Assign credit from one edit record to the templates that produced it."""

    template_ids = getattr(record, "template_ids", []) or []
    if not template_ids:
        template_ids = [
            edit.get("template_id")
            for edit in getattr(record, "edits", [])
            if edit.get("template_id")
        ]
    for template_id in dict.fromkeys(template_ids):
        stats = policy.ensure_stats(template_id)
        stats.attempts += 1
        if getattr(record, "accepted", False):
            stats.accepted += 1
            stats.total_delta_mp += float(getattr(record, "delta_mp", 0.0))
        elif str(getattr(record, "reason", "")).startswith("invalid harness"):
            stats.invalid += 1
        else:
            stats.rejected += 1


def describe_policy(policy: EditPolicy) -> dict[str, Any]:
    return {
        "version": policy.version,
        "parent": policy.parent,
        "synthesis_enabled": policy.synthesis_enabled,
        "synthesis_max_arity": policy.synthesis_max_arity,
        "synthesis_beam": policy.synthesis_beam,
        "synthesis_primitives": list(policy.synthesis_primitives),
        "max_proposals": policy.max_proposals,
        "consolidation_enabled": policy.consolidation_enabled,
        "consolidation_min_support": policy.consolidation_min_support,
        "consolidation_min_precision": policy.consolidation_min_precision,
        "consolidation_max_skills": policy.consolidation_max_skills,
        "consolidation_cross_validate": policy.consolidation_cross_validate,
        "templates": {
            kind: [item.template_id for item in items]
            for kind, items in policy.templates.items()
        },
        "stats": {key: value.to_dict() for key, value in policy.stats.items()},
        "meta_rules": [rule.to_dict() for rule in policy.meta_rules],
        "meta_rule_library": sorted(policy.meta_rule_library),
    }
