"""The meta loop: improving the improvement operator ``U``.

The outer loop in :mod:`cetra.rsi.loop` edits the harness ``M`` with a fixed
policy ``U``.  This module makes ``U`` itself the object of improvement:

1. run ``U`` on a meta-training suite and collect its edit log;
2. diagnose *why* ``U`` failed (missing capability, rejected template,
   synthesis failure, exhausted budget, high cost);
3. propose typed meta-edits to ``U``;
4. evaluate the candidate ``U`` by the meta-metaproductivity ``MMP`` on a
   disjoint meta-validation suite;
5. keep the candidate only when ``MMP`` improves and the anchor does not
   degrade.

A one-step meta-rollout makes self-edits meaningful: if ``U`` adds a meta-rule
that says "when synthesis is missing, enable synthesis", the gate evaluates the
*consequences* of that rule, not just the immediate policy object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from .benchmarks import FamilyDataset
from .consolidation import ConsolidationPolicy, consolidate, transfer_metrics
from .harness import make_seed_harness
from .loop import CETRARSILoop, RunReport
from .policy import EditPolicy, MetaRule, TemplateStats, seed_policy, update_stats


# ---------------------------------------------------------------------------
# Meta diagnosis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetaDiagnosis:
    kind: str
    count: int
    evidence: tuple[str, ...] = ()
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "count": self.count,
            "evidence": list(self.evidence),
            "detail": self.detail,
        }


def _records(reports: dict[str, RunReport]) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for family, report in reports.items():
        for record in report.edit_log:
            rows.append((family, record))
    return rows


def meta_diagnose(reports: dict[str, RunReport], policy: EditPolicy) -> list[MetaDiagnosis]:
    """Attribute failures of ``U`` to actionable meta-diagnosis kinds."""

    counts: dict[str, int] = {}
    evidence: dict[str, list[str]] = {}
    accepted_by_family = {
        family: any(record.accepted for record in report.edit_log)
        for family, report in reports.items()
    }
    solved_by_family = {
        family: report.best_metrics["test"].mp > 1e-9
        for family, report in reports.items()
    }

    def add(kind: str, item: str) -> None:
        counts[kind] = counts.get(kind, 0) + 1
        evidence.setdefault(kind, []).append(item)

    for family, record in _records(reports):
        has_gap = bool({"operator_gap", "fallback_gap"} & set(record.diagnosis_kinds))
        tried_synthesis = any(
            edit.get("kind") == "SYNTHESIZE_OPERATOR" for edit in record.edits
        ) or record.synthesis is not None
        if (
            not record.accepted
            and tried_synthesis
            and not accepted_by_family.get(family, False)
            and not solved_by_family.get(family, False)
        ):
            add("synthesis_failed", f"{family}:{record.child_version}")
        if not record.accepted and record.template_ids and not tried_synthesis:
            add("template_rejected", f"{family}:{record.child_version}")
        if has_gap and not policy.synthesis_enabled:
            add("synthesis_missing", f"{family}:{record.child_version}")
        if not record.accepted and str(record.reason).startswith("invalid harness"):
            add("template_invalid", f"{family}:{record.child_version}")

    for family, report in reports.items():
        accepted = any(record.accepted for record in report.edit_log)
        if not accepted and report.best_metrics["test"].mp <= 1e-9:
            add("unsolved_family", family)
        if (
            not accepted
            and len(report.edit_log) >= policy.max_proposals
        ):
            add("budget_exhausted", family)

    if reports and all(
        report.best_metrics["test"].mp > 1e-9 for report in reports.values()
    ):
        if policy.max_proposals > 4 or policy.synthesis_beam > 8:
            add("cost_high", "all families solved with a wide search budget")
        if not policy.consolidation_enabled:
            add("consolidation_missing", "all families solved; skills are not consolidated")

    # A family can be partly solved by a shallow synthesised operator while
    # still having high held-out risk.  That is evidence that the *grammar
    # depth* is the bottleneck, not the template ranking.
    if policy.synthesis_enabled and policy.synthesis_max_arity < 4:
        for family, report in reports.items():
            if report.best_metrics["test"].risk > 0.25:
                add("depth_limited", f"{family}: risk={report.best_metrics['test'].risk:.3f}")

    # Synthesis has produced accepted edits but is still ranked below the
    # known-operator templates.  Promote it so the loop stops paying for
    # proposals that are known to fail.
    for templates in policy.templates.values():
        for template in templates:
            stats = policy.stats.get(template.template_id, TemplateStats())
            if template.synthesis and stats.accepted > 0 and template.prior < 0.5:
                add("synthesis_proven", template.template_id)

    # Credit assignment: a template that is repeatedly rejected while another
    # is accepted is evidence that U's proposal ranking is wrong, not that the
    # task is impossible.
    for template_id, stats in policy.stats.items():
        if stats.rejected >= 2 and stats.accepted == 0:
            add("credit_assignment", f"{template_id}: {stats.rejected} rejected, 0 accepted")
        elif stats.accepted > 0 and stats.rejected >= 2:
            add("credit_assignment", f"{template_id}: {stats.accepted} accepted, {stats.rejected} rejected")

    diagnoses = [
        MetaDiagnosis(kind=kind, count=count, evidence=tuple(evidence.get(kind, [])[:6]))
        for kind, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    # A diagnosis with no matching meta-rule is an *acquisition* opportunity:
    # U has a generic "acquire from the library" rule, but it has not yet
    # acquired the specific rule.  This is the self-referential step.
    existing = {rule.when for rule in policy.meta_rules if rule.enabled}
    for kind, count in sorted(counts.items()):
        if kind == "no_meta_rule" or kind in existing:
            continue
        if kind not in policy.meta_rule_library:
            continue
        diagnoses.append(
            MetaDiagnosis(
                kind="no_meta_rule",
                count=count,
                evidence=(kind,),
                detail=kind,
            )
        )
    return diagnoses


# ---------------------------------------------------------------------------
# Meta edits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetaEdit:
    kind: str
    target: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target": self.target,
            "args": dict(self.args),
            "rationale": self.rationale,
        }


def _replace_template(policy: EditPolicy, template_id: str, **changes: Any) -> None:
    for kind, templates in policy.templates.items():
        updated: list[Any] = []
        for template in templates:
            if template.template_id != template_id:
                updated.append(template)
                continue
            payload = template.to_dict()
            payload.update(changes)
            updated.append(type(template).from_dict(payload))
        policy.templates[kind] = updated


def apply_meta_edit(policy: EditPolicy, edit: MetaEdit) -> EditPolicy:
    """Apply one typed meta-edit to a clone of ``policy``."""

    child = policy.clone(policy.version)

    if edit.kind == "ENABLE_SYNTHESIS":
        child.synthesis_enabled = True
        child.synthesis_max_arity = int(edit.args.get("max_arity", 3))
        primitives = edit.args.get("primitives", ("eq", "threshold", "fallback"))
        child.synthesis_primitives = tuple(primitives)
        child.synthesis_beam = int(edit.args.get("beam", max(12, child.synthesis_beam)))
    elif edit.kind == "INCREASE_SYNTHESIS_DEPTH":
        child.synthesis_max_arity = min(4, child.synthesis_max_arity + 1)
        child.synthesis_beam = min(32, child.synthesis_beam + 8)
    elif edit.kind == "INCREASE_SYNTHESIS_BEAM":
        child.synthesis_beam = min(32, child.synthesis_beam + 8)
    elif edit.kind == "DECREASE_SYNTHESIS_BEAM":
        child.synthesis_beam = max(4, child.synthesis_beam - 4)
    elif edit.kind == "INCREASE_BUDGET":
        child.max_proposals = min(16, child.max_proposals + 2)
    elif edit.kind == "DECREASE_BUDGET":
        child.max_proposals = max(2, child.max_proposals - 2)
        child.synthesis_beam = max(4, child.synthesis_beam - 4)
    elif edit.kind == "UPDATE_PRIOR":
        delta = float(edit.args.get("delta", -0.1))
        for template_id in edit.args.get("template_ids", [edit.target]):
            for templates in child.templates.values():
                for template in templates:
                    if template.template_id == template_id:
                        _replace_template(child, template_id, prior=template.prior + delta)
    elif edit.kind == "PROMOTE_SYNTHESIS":
        for templates in child.templates.values():
            for template in templates:
                if template.synthesis:
                    _replace_template(child, template.template_id, prior=1.0)
                elif template.template_id in {
                    "t_scope_add",
                    "t_scope_add_scope_error",
                    "t_tune_support",
                    "t_tune_support_concept",
                }:
                    _replace_template(child, template.template_id, prior=-0.5)
    elif edit.kind == "REMOVE_TEMPLATE":
        child.remove_template(edit.target)
    elif edit.kind == "ADD_TEMPLATE":
        from .policy import EditTemplate

        child.add_template(EditTemplate.from_dict(edit.args["template"]))
    elif edit.kind == "ADD_META_RULE":
        rule = MetaRule.from_dict(edit.args["rule"])
        if not any(item.rule_id == rule.rule_id for item in child.meta_rules):
            child.meta_rules.append(rule)
    elif edit.kind == "REMOVE_META_RULE":
        child.meta_rules = [item for item in child.meta_rules if item.rule_id != edit.target]
    elif edit.kind == "SET_GATE_MARGIN":
        child.gate_holdout_margin = float(edit.args.get("holdout_margin", child.gate_holdout_margin))
        child.gate_mp_epsilon = float(edit.args.get("mp_epsilon", child.gate_mp_epsilon))
    elif edit.kind == "ENABLE_CONSOLIDATION":
        child.consolidation_enabled = True
        child.consolidation_min_support = int(edit.args.get("min_support", 2))
        child.consolidation_min_precision = float(edit.args.get("min_precision", 0.9))
        child.consolidation_max_skills = int(edit.args.get("max_skills", 8))
        child.consolidation_cross_validate = bool(edit.args.get("cross_validate", True))
    else:
        raise ValueError(f"unknown meta-edit kind: {edit.kind}")

    child.history.append({"edit": edit.to_dict(), "parent": policy.version})
    return child


def apply_meta_edits(policy: EditPolicy, edits: list[MetaEdit], version: str) -> EditPolicy:
    child = policy.clone(version)
    for edit in edits:
        child = apply_meta_edit(child, edit)
        child.version = version
    child.parent = policy.version
    return child


# ---------------------------------------------------------------------------
# Meta proposal
# ---------------------------------------------------------------------------


def propose_meta_edits(
    diagnoses: list[MetaDiagnosis],
    policy: EditPolicy,
) -> list[list[MetaEdit]]:
    """Turn meta-diagnoses into candidate meta-edit bundles."""

    by_kind = {item.kind: item for item in diagnoses}
    bundles: list[list[MetaEdit]] = []
    seen: set[tuple[str, ...]] = set()

    def add(bundle: list[MetaEdit]) -> None:
        key = tuple(edit.kind + ":" + edit.target for edit in bundle)
        if key not in seen:
            seen.add(key)
            bundles.append(bundle)

    # 1. Rules that U already has.
    for diagnosis in diagnoses:
        if diagnosis.kind == "no_meta_rule":
            continue
        for rule in policy.meta_rules:
            if not rule.enabled or rule.when != diagnosis.kind:
                continue
            if diagnosis.count < rule.min_count:
                continue
            args = dict(rule.args)
            add(
                [
                    MetaEdit(
                        kind=rule.then,
                        target=rule.rule_id,
                        args=args,
                        rationale=f"meta-rule {rule.rule_id}",
                    )
                ]
            )

    # 2. Rule acquisition: U has the generic mr_acquire_meta_rule, but no
    # specific rule for this diagnosis.  Look the rule up in U's own library
    # and acquire it, then use it immediately.  This is the self-referential
    # step that makes U_{t+1}=U_t(U_t,D_t^meta) concrete.
    for diagnosis in diagnoses:
        if diagnosis.kind != "no_meta_rule":
            continue
        missing = diagnosis.detail
        payload = policy.meta_rule_library.get(missing)
        if not payload:
            continue
        add(
            [
                MetaEdit(
                    kind="ADD_META_RULE",
                    target=payload["rule_id"],
                    args={"rule": payload},
                    rationale=f"acquire a meta-rule for {missing} from U's library",
                ),
                MetaEdit(
                    kind=payload["then"],
                    target="",
                    args=dict(payload.get("args", {})),
                    rationale=f"use the newly acquired rule for {missing}",
                ),
            ]
        )

    # 3. Once synthesis has produced accepted edits, put it first.
    if "synthesis_proven" in by_kind:
        add(
            [
                MetaEdit(
                    kind="PROMOTE_SYNTHESIS",
                    target="",
                    args={},
                    rationale="synthesis has accepted edits; stop trying known-operator templates first",
                )
            ]
        )

    # 4. A depth-for-beam trade: widen the grammar but narrow the beam, so the
    # extra expressivity does not cost more search.
    if "depth_limited" in by_kind and "cost_high" in by_kind:
        add(
            [
                MetaEdit(
                    kind="INCREASE_SYNTHESIS_DEPTH",
                    target="",
                    args={},
                    rationale="trade beam width for grammar depth",
                ),
                MetaEdit(
                    kind="DECREASE_SYNTHESIS_BEAM",
                    target="",
                    args={},
                    rationale="keep the search cost flat",
                ),
            ]
        )

    # 4. Consolidation: once the harness solves the families, promote stable
    # rules into a transferable skill library.
    if "consolidation_missing" in by_kind:
        add(
            [
                MetaEdit(
                    kind="ENABLE_CONSOLIDATION",
                    target="",
                    args={
                        "min_support": 2,
                        "min_precision": 0.9,
                        "max_skills": 8,
                        "cross_validate": True,
                    },
                    rationale="all families solved; consolidate experience into skills",
                )
            ]
        )

    # 5. Cost reduction is tried before widening the search: once every family
    # is solved, the next improvement should come from a cheaper improvement
    # process, not from more compute.
    if "cost_high" in by_kind:
        add(
            [
                MetaEdit(
                    kind="DECREASE_SYNTHESIS_BEAM",
                    target="",
                    args={},
                    rationale="all families solved; shrink the synthesis beam",
                )
            ]
        )

    # 4. Depth / beam / budget responses.
    if "depth_limited" in by_kind:
        add(
            [
                MetaEdit(
                    kind="INCREASE_SYNTHESIS_DEPTH",
                    target="",
                    args={},
                    rationale="a family is partly solved but still high-risk; widen the grammar",
                )
            ]
        )
    if "synthesis_failed" in by_kind:
        add(
            [
                MetaEdit(
                    kind="INCREASE_SYNTHESIS_DEPTH",
                    target="",
                    args={},
                    rationale="the grammar was too shallow for the remaining gap",
                ),
                MetaEdit(
                    kind="INCREASE_SYNTHESIS_BEAM",
                    target="",
                    args={},
                    rationale="keep more candidate programs",
                ),
            ]
        )
    if "template_rejected" in by_kind:
        rejected = by_kind["template_rejected"]
        add(
            [
                MetaEdit(
                    kind="UPDATE_PRIOR",
                    target="",
                    args={"delta": -0.1, "template_ids": []},
                    rationale=f"downweight templates rejected {rejected.count} times",
                )
            ]
        )
    if "credit_assignment" in by_kind:
        rejected_ids = [
            template_id
            for template_id, stats in policy.stats.items()
            if stats.rejected >= 2 and stats.accepted == 0
        ]
        good_ids = [
            template_id
            for template_id, stats in policy.stats.items()
            if stats.accepted > 0
        ]
        bundle: list[MetaEdit] = []
        if rejected_ids:
            bundle.append(
                MetaEdit(
                    kind="UPDATE_PRIOR",
                    target="",
                    args={"delta": -0.3, "template_ids": rejected_ids},
                    rationale="demote templates that repeatedly failed",
                )
            )
        if good_ids:
            bundle.append(
                MetaEdit(
                    kind="UPDATE_PRIOR",
                    target="",
                    args={"delta": 0.2, "template_ids": good_ids},
                    rationale="promote templates with accepted edits",
                )
            )
        if bundle:
            add(bundle)
    if "budget_exhausted" in by_kind:
        add([MetaEdit(kind="INCREASE_BUDGET", target="", args={}, rationale="search was exhausted")])

    return bundles


# ---------------------------------------------------------------------------
# Policy evaluation and meta gate
# ---------------------------------------------------------------------------


@dataclass
class PolicyEvaluation:
    policy_version: str
    mmp: float
    mp_by_family: dict[str, float]
    risk_by_family: dict[str, float]
    coverage: int
    cost: float
    total_proposals: int
    accepted_edits: int
    skill_gain_by_family: dict[str, float] = field(default_factory=dict)
    skill_gain_total: float = 0.0
    reports: dict[str, RunReport] = field(default_factory=dict)

    def to_dict(self, include_reports: bool = False) -> dict[str, Any]:
        payload = {
            "policy_version": self.policy_version,
            "mmp": round(self.mmp, 8),
            "mp_by_family": {key: round(value, 8) for key, value in self.mp_by_family.items()},
            "risk_by_family": {key: round(value, 6) for key, value in self.risk_by_family.items()},
            "coverage": self.coverage,
            "cost": round(self.cost, 6),
            "total_proposals": self.total_proposals,
            "accepted_edits": self.accepted_edits,
            "skill_gain_by_family": {
                key: round(value, 6) for key, value in self.skill_gain_by_family.items()
            },
            "skill_gain_total": round(self.skill_gain_total, 6),
        }
        if include_reports:
            payload["reports"] = {name: report.to_dict() for name, report in self.reports.items()}
        return payload


def evaluate_policy(
    policy: EditPolicy,
    suite: dict[str, FamilyDataset],
    *,
    split: str = "val",
    generations: int = 1,
) -> PolicyEvaluation:
    """Run ``policy`` on every family and compute cost-normalized MMP."""

    mp_by_family: dict[str, float] = {}
    risk_by_family: dict[str, float] = {}
    skill_gain_by_family: dict[str, float] = {}
    reports: dict[str, RunReport] = {}
    total_cost = 0.0
    total_proposals = 0
    accepted = 0
    skill_gain_total = 0.0

    for name, family in suite.items():
        dataset = family.split(split)
        # Reserve the last third of the compositional split for the skill
        # transfer test.  It is disjoint from the cases used for MP and from
        # the gate's iid split.
        comp = list(dataset.compositional)
        cut = max(1, int(len(comp) * 0.67)) if comp else 0
        mp_cases = comp[:cut]
        skill_cases = comp[cut:]
        loop_dataset = replace(
            dataset,
            compositional=mp_cases,
            surface=mp_cases or dataset.surface,
        )

        harness = make_seed_harness(concept_pool=family.spec.concepts)
        # Each family gets an independent policy clone: the capability under
        # test is U's search policy, not a library of family-specific programs.
        run_policy = policy.clone(policy.version)
        loop = CETRARSILoop(loop_dataset, seed_harness=harness, policy=run_policy)
        report = loop.run(generations=generations)
        reports[name] = report
        mp = max(0.0, report.best_metrics["test"].mp)
        mp_by_family[name] = mp
        risk_by_family[name] = report.best_metrics["test"].risk

        if policy.consolidation_enabled and skill_cases:
            cpol = ConsolidationPolicy(
                name="meta_c",
                min_support=policy.consolidation_min_support,
                min_precision=policy.consolidation_min_precision,
                max_skills=policy.consolidation_max_skills,
                cross_validate=policy.consolidation_cross_validate,
                cross_min_precision=policy.consolidation_min_precision,
                cross_min_support=policy.consolidation_min_support,
            )
            def _heldout_mp(entry: Any) -> float:
                if "iid" in entry.metrics:
                    return entry.metrics["iid"].mp
                return entry.mp

            best = max(report.archive, key=_heldout_mp)
            library = consolidate(
                best.representation,
                cpol,
                source_family=name,
                validation_cases=dataset.val,
            )
            metrics = transfer_metrics(
                best.harness,
                library,
                skill_cases,
                source_rule_count=len(best.representation.rules),
            )
            gain = max(0.0, float(metrics["transfer_gain"]))
            skill_gain_by_family[name] = gain
            skill_gain_total += gain

        n_cases = (
            len(loop_dataset.seed)
            + len(loop_dataset.val)
            + len(loop_dataset.iid)
            + len(loop_dataset.compositional)
        )
        depth = len(report.best_spec.get("pipeline", []))
        synthesis_attempts = sum(1 for record in report.edit_log if record.synthesis)
        # Deterministic cost proxy: proposals + construction + synthesis search.
        # The synthesis beam is charged because it controls how many programs
        # the synthesizer enumerates and evaluates.
        cost = (
            len(report.edit_log)
            + 0.001 * n_cases * max(1, depth)
            + 0.02 * synthesis_attempts * policy.synthesis_beam
        )
        total_cost += cost
        total_proposals += len(report.edit_log)
        accepted += sum(1 for record in report.edit_log if record.accepted)

    mmp = (sum(mp_by_family.values()) + 0.001 * skill_gain_total) / max(total_cost, 1e-9)
    coverage = sum(1 for value in mp_by_family.values() if value > 1e-9)
    return PolicyEvaluation(
        policy_version=policy.version,
        mmp=mmp,
        mp_by_family=mp_by_family,
        risk_by_family=risk_by_family,
        coverage=coverage,
        cost=total_cost,
        total_proposals=total_proposals,
        accepted_edits=accepted,
        skill_gain_by_family=skill_gain_by_family,
        skill_gain_total=skill_gain_total,
        reports=reports,
    )


@dataclass
class MetaGateDecision:
    accepted: bool
    delta_mmp: float
    coverage_ok: bool
    anchor_ok: bool
    reason: str
    old: PolicyEvaluation | None = None
    new: PolicyEvaluation | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "delta_mmp": round(self.delta_mmp, 8),
            "coverage_ok": self.coverage_ok,
            "anchor_ok": self.anchor_ok,
            "reason": self.reason,
            "old_mmp": round(self.old.mmp, 8) if self.old else None,
            "new_mmp": round(self.new.mmp, 8) if self.new else None,
        }


def meta_gate(
    old: PolicyEvaluation,
    new: PolicyEvaluation,
    *,
    margin: float = 1e-7,
    anchor_margin: float = 0.05,
) -> MetaGateDecision:
    delta = new.mmp - old.mmp
    coverage_ok = new.coverage >= old.coverage
    old_risk = max(old.risk_by_family.values(), default=0.0)
    new_risk = max(new.risk_by_family.values(), default=0.0)
    anchor_ok = new_risk <= old_risk + anchor_margin
    reasons: list[str] = []
    if delta <= margin:
        reasons.append("MMP did not improve")
    if not coverage_ok:
        reasons.append("coverage decreased")
    if not anchor_ok:
        reasons.append("anchor risk degraded")
    return MetaGateDecision(
        accepted=not reasons,
        delta_mmp=delta,
        coverage_ok=coverage_ok,
        anchor_ok=anchor_ok,
        reason="accepted" if not reasons else "; ".join(reasons),
        old=old,
        new=new,
    )


# ---------------------------------------------------------------------------
# Meta loop
# ---------------------------------------------------------------------------


@dataclass
class MetaArchiveEntry:
    policy: EditPolicy
    evaluation: PolicyEvaluation
    generation: int
    parent: str | None = None
    edits: list[MetaEdit] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy.version,
            "parent": self.parent,
            "generation": self.generation,
            "edits": [edit.to_dict() for edit in self.edits],
            "evaluation": self.evaluation.to_dict(),
            "policy": self.policy.to_dict(),
        }


@dataclass
class MetaArchive:
    entries: list[MetaArchiveEntry] = field(default_factory=list)

    def add(
        self,
        policy: EditPolicy,
        evaluation: PolicyEvaluation,
        *,
        generation: int,
        parent: str | None = None,
        edits: list[MetaEdit] | None = None,
    ) -> MetaArchiveEntry:
        entry = MetaArchiveEntry(
            policy=policy,
            evaluation=evaluation,
            generation=generation,
            parent=parent,
            edits=list(edits or []),
        )
        self.entries.append(entry)
        return entry

    def best(self) -> MetaArchiveEntry:
        return max(self.entries, key=lambda entry: entry.evaluation.mmp)

    def to_list(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.entries]


@dataclass
class MetaRecord:
    generation: int
    policy_version: str
    parent_version: str
    diagnoses: list[dict[str, Any]]
    edits: list[dict[str, Any]]
    decision: dict[str, Any]
    evaluation: dict[str, Any]
    rollout_edits: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "policy_version": self.policy_version,
            "parent_version": self.parent_version,
            "diagnoses": self.diagnoses,
            "edits": self.edits,
            "rollout_edits": self.rollout_edits,
            "decision": self.decision,
            "evaluation": self.evaluation,
        }


@dataclass
class MetaReport:
    initial: PolicyEvaluation
    final: PolicyEvaluation
    test: PolicyEvaluation
    records: list[MetaRecord]
    mmp_curve: list[dict[str, Any]]
    final_policy: EditPolicy
    meta_archive: MetaArchive = field(default_factory=MetaArchive)

    def to_dict(self, include_reports: bool = False) -> dict[str, Any]:
        return {
            "initial": self.initial.to_dict(include_reports),
            "final": self.final.to_dict(include_reports),
            "test": self.test.to_dict(include_reports),
            "records": [record.to_dict() for record in self.records],
            "mmp_curve": self.mmp_curve,
            "final_policy": self.final_policy.to_dict(),
            "meta_archive": self.meta_archive.to_list(),
        }


class MetaRSILoop:
    """Run the full meta loop: ``U_{t+1} = U_t(U_t, D_t^meta)``."""

    def __init__(
        self,
        suite: dict[str, FamilyDataset],
        *,
        policy: EditPolicy | None = None,
        generations: int = 1,
        meta_generations: int = 3,
        max_candidates: int = 4,
    ) -> None:
        self.suite = suite
        self.policy = policy or seed_policy()
        self.generations = generations
        self.meta_generations = meta_generations
        self.max_candidates = max_candidates

    def _run_train(self, policy: EditPolicy) -> dict[str, RunReport]:
        reports: dict[str, RunReport] = {}
        for name, family in self.suite.items():
            dataset = family.meta_train
            harness = make_seed_harness(concept_pool=family.spec.concepts)
            loop = CETRARSILoop(
                dataset,
                seed_harness=harness,
                policy=policy.clone(policy.version),
            )
            report = loop.run(generations=self.generations)
            reports[name] = report
            # Credit assignment is part of the meta-training signal: U sees
            # which of its templates actually produced accepted edits.
            for record in report.edit_log:
                update_stats(policy, record)
        return reports

    def _rollout(
        self,
        policy: EditPolicy,
        train_reports: dict[str, RunReport],
    ) -> tuple[EditPolicy, list[MetaEdit]]:
        """Apply one step of the policy's own meta-rules, if any."""

        diagnoses = meta_diagnose(train_reports, policy)
        bundles = propose_meta_edits(diagnoses, policy)
        if not bundles:
            return policy, []
        version = f"{policy.version}r"
        return apply_meta_edits(policy, bundles[0], version), bundles[0]

    def run(self) -> MetaReport:
        initial = evaluate_policy(self.policy, self.suite, split="val", generations=self.generations)
        current = self.policy
        last_eval = initial
        archive = MetaArchive()
        archive.add(self.policy, initial, generation=0)
        records: list[MetaRecord] = []
        curve: list[dict[str, Any]] = [
            {
                "stage": "init",
                "policy_version": current.version,
                "mmp": round(initial.mmp, 8),
                "coverage": initial.coverage,
                "cost": round(initial.cost, 6),
                "synthesis_enabled": current.synthesis_enabled,
            }
        ]

        for generation in range(1, self.meta_generations + 1):
            train_reports = self._run_train(current)
            diagnoses = meta_diagnose(train_reports, current)
            bundles = propose_meta_edits(diagnoses, current)[: self.max_candidates]

            best_candidate: EditPolicy | None = None
            best_eval: PolicyEvaluation | None = None
            best_decision: MetaGateDecision | None = None
            best_bundle: list[MetaEdit] | None = None
            best_rollout: list[MetaEdit] = []

            for index, bundle in enumerate(bundles):
                candidate_version = f"u{generation}_{index}"
                candidate = apply_meta_edits(current, bundle, candidate_version)
                # One-step lookahead: evaluate the consequences of the rules
                # the candidate just added, not only the immediate object.
                rolled, rollout_edits = self._rollout(candidate, train_reports)
                evaluation = evaluate_policy(
                    rolled,
                    self.suite,
                    split="val",
                    generations=self.generations,
                )
                decision = meta_gate(last_eval, evaluation)
                if decision.accepted and (
                    best_decision is None or evaluation.mmp > best_eval.mmp  # type: ignore[union-attr]
                ):
                    best_candidate = rolled
                    best_eval = evaluation
                    best_decision = decision
                    best_bundle = bundle
                    best_rollout = rollout_edits

            if best_candidate is None or best_eval is None or best_decision is None:
                records.append(
                    MetaRecord(
                        generation=generation,
                        policy_version=current.version,
                        parent_version=current.version,
                        diagnoses=[item.to_dict() for item in diagnoses],
                        edits=[edit.to_dict() for edit in bundles[0]] if bundles else [],
                        decision={"accepted": False, "reason": "no candidate improved MMP"},
                        evaluation=last_eval.to_dict(),
                    )
                )
                break

            current = best_candidate
            last_eval = best_eval
            archive.add(
                current,
                best_eval,
                generation=generation,
                parent=current.parent,
                edits=best_bundle or [],
            )
            records.append(
                MetaRecord(
                    generation=generation,
                    policy_version=current.version,
                    parent_version=current.parent or "u0",
                    diagnoses=[item.to_dict() for item in diagnoses],
                    edits=[edit.to_dict() for edit in (best_bundle or [])],
                    rollout_edits=[edit.to_dict() for edit in best_rollout],
                    decision=best_decision.to_dict(),
                    evaluation=best_eval.to_dict(),
                )
            )
            curve.append(
                {
                    "stage": f"meta_{generation}",
                    "policy_version": current.version,
                    "mmp": round(best_eval.mmp, 8),
                    "coverage": best_eval.coverage,
                    "cost": round(best_eval.cost, 6),
                    "synthesis_enabled": current.synthesis_enabled,
                }
            )

        final = evaluate_policy(current, self.suite, split="val", generations=self.generations)
        test = evaluate_policy(current, self.suite, split="test", generations=self.generations)
        curve.append(
            {
                "stage": "final_val",
                "policy_version": current.version,
                "mmp": round(final.mmp, 8),
                "coverage": final.coverage,
                "cost": round(final.cost, 6),
                "synthesis_enabled": current.synthesis_enabled,
            }
        )
        curve.append(
            {
                "stage": "final_test",
                "policy_version": current.version,
                "mmp": round(test.mmp, 8),
                "coverage": test.coverage,
                "cost": round(test.cost, 6),
                "synthesis_enabled": current.synthesis_enabled,
            }
        )
        return MetaReport(
            initial=initial,
            final=final,
            test=test,
            records=records,
            mmp_curve=curve,
            final_policy=current,
            meta_archive=archive,
        )
