"""The CETRA-RSI outer loop.

Inner loop: counterexample-guided representation refinement (CEGAR).
Outer loop: typed harness edits, triple gate, archive, parent selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cases import Dataset
from .diagnosis_utils import error_counts, error_kinds
from .edits import apply_edit_bundle
from .gate import GateDecision, accept_edit
from .harness import (
    diagnose,
    evaluate,
    inner_refine,
    make_seed_harness,
    risk,
)
from .policy import EditPolicy, describe_policy, propose_with_policy, seed_policy, update_stats
from .synthesis import OperatorSynthesizer, SynthesisContext
from .types import (
    ArchiveEntry,
    CaseRecord,
    Edit,
    Harness,
    Metrics,
    Representation,
)


@dataclass
class EditRecord:
    generation: int
    parent_version: str
    child_version: str
    edits: list[dict[str, Any]]
    accepted: bool
    reason: str
    delta_in: float
    delta_ho: float
    delta_mp: float
    parent_metrics: dict[str, Metrics] = field(default_factory=dict)
    child_metrics: dict[str, Metrics] = field(default_factory=dict)
    inner_history: list[dict[str, Any]] = field(default_factory=list)
    diagnosis_kinds: list[str] = field(default_factory=list)
    template_ids: list[str] = field(default_factory=list)
    policy_version: str = "u0"
    synthesis: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "parent_version": self.parent_version,
            "child_version": self.child_version,
            "edits": self.edits,
            "accepted": self.accepted,
            "reason": self.reason,
            "delta_in": round(self.delta_in, 6),
            "delta_ho": round(self.delta_ho, 6),
            "delta_mp": round(self.delta_mp, 6),
            "parent_metrics": {
                name: metrics.to_dict() for name, metrics in self.parent_metrics.items()
            },
            "child_metrics": {
                name: metrics.to_dict() for name, metrics in self.child_metrics.items()
            },
            "inner_history": self.inner_history,
            "diagnosis_kinds": self.diagnosis_kinds,
            "template_ids": self.template_ids,
            "policy_version": self.policy_version,
            "synthesis": self.synthesis,
        }


@dataclass
class RunReport:
    dataset_summary: dict[str, Any]
    seed_version: str
    seed_metrics: dict[str, Metrics]
    seed_inner: list[dict[str, Any]]
    archive: list[ArchiveEntry]
    edit_log: list[EditRecord]
    best_version: str
    best_metrics: dict[str, Metrics]
    best_spec: dict[str, Any]
    gate_split: str = "iid"
    gate_mp_split: str = "iid"
    policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_split": self.gate_split,
            "gate_mp_split": self.gate_mp_split,
            "policy": self.policy,
            "dataset_summary": self.dataset_summary,
            "seed": {
                "version": self.seed_version,
                "metrics": {name: metrics.to_dict() for name, metrics in self.seed_metrics.items()},
                "inner_history": self.seed_inner,
            },
            "archive": [
                {
                    "version": entry.harness.version,
                    "parent": entry.harness.parent,
                    "generation": entry.generation,
                    "accepted": entry.accepted,
                    "metrics": {
                        name: metrics.to_dict() for name, metrics in entry.metrics.items()
                    },
                    "edit_from_parent": entry.edit_from_parent.to_dict()
                    if entry.edit_from_parent
                    else None,
                    "inner_history": entry.inner_history,
                }
                for entry in self.archive
            ],
            "edit_log": [record.to_dict() for record in self.edit_log],
            "best": {
                "version": self.best_version,
                "metrics": {
                    name: metrics.to_dict() for name, metrics in self.best_metrics.items()
                },
                "spec": self.best_spec,
            },
        }


class CETRARSILoop:
    """Run the complete inner/outer CETRA-RSI loop."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        seed_harness: Harness | None = None,
        max_inner_rounds: int = 3,
        cost_scale: float = 1.0,
        policy: EditPolicy | None = None,
        synthesizer: OperatorSynthesizer | None = None,
        gate_mp_split: str = "iid",
        strict_mp: bool = True,
        fit_cases: list[CaseRecord] | None = None,
    ) -> None:
        self.dataset = dataset
        self.train = dataset.search
        # Callers running a clean held-out protocol can pass a fit set that
        # excludes the validation split.  The default keeps the historical
        # controlled-demo behavior.
        self.fit = list(fit_cases) if fit_cases is not None else dataset.fit
        self.val = dataset.val
        self.test = dataset.compositional or dataset.surface
        self.seed_harness = seed_harness or make_seed_harness()
        self.max_inner_rounds = max_inner_rounds
        self.cost_scale = cost_scale
        self.gate_split = "iid"  # held-out gate, not the inner-fit val split
        self.gate_mp_split = gate_mp_split
        self.strict_mp = strict_mp
        self.seed_test_risk: float | None = None
        self.seed_risk_by_split: dict[str, float] = {}
        self.archive: list[ArchiveEntry] = []
        self.edit_log: list[EditRecord] = []
        self.policy = policy or seed_policy()
        self.synthesizer = synthesizer or OperatorSynthesizer(
            max_arity=self.policy.synthesis_max_arity,
            beam=self.policy.synthesis_beam,
            primitives=self.policy.synthesis_primitives,
        )

    def metrics_for(self, harness: Harness, rep: Representation) -> dict[str, Metrics]:
        """Public evaluation hook used by the demo and by external probes."""

        if self.seed_test_risk is None:
            raise RuntimeError("call bootstrap() or run() before metrics_for()")
        return self._metrics(harness, rep)

    def _splits(self) -> dict[str, list[CaseRecord]]:
        return {
            "train": self.train,
            "val": self.val,
            "test": self.test,
            "iid": self.dataset.iid,
            "surface": self.dataset.surface,
            "compositional": self.dataset.compositional,
        }

    def _metrics(self, harness: Harness, rep: Representation) -> dict[str, Metrics]:
        assert self.seed_test_risk is not None
        splits = self._splits()
        return {
            name: evaluate(
                harness,
                rep,
                cases,
                seed_risk=self.seed_risk_by_split.get(name, self.seed_test_risk),
                cost_scale=self.cost_scale,
            )
            for name, cases in splits.items()
        }

    def bootstrap(self) -> ArchiveEntry:
        seed_rep, inner = inner_refine(
            self.seed_harness,
            self.train,
            self.val,
            fit_cases=self.fit,
            max_rounds=self.max_inner_rounds,
        )
        self.seed_risk_by_split = {
            name: risk(self.seed_harness, seed_rep, cases)
            for name, cases in self._splits().items()
        }
        self.seed_test_risk = self.seed_risk_by_split["test"]
        metrics = self._metrics(self.seed_harness, seed_rep)
        entry = ArchiveEntry(
            harness=self.seed_harness,
            representation=seed_rep,
            metrics=metrics,
            generation=0,
            inner_history=inner,
        )
        self.archive.append(entry)
        return entry

    def _selection_mp(self, entry: ArchiveEntry) -> float:
        metrics = entry.metrics
        if self.gate_mp_split in metrics:
            return metrics[self.gate_mp_split].mp
        return entry.mp

    def select_parent(self) -> ArchiveEntry:
        # Exploit the highest held-out MP; break ties toward less-explored
        # lineages.  The selection split is configurable so the test set is not
        # used for search.
        return max(
            self.archive,
            key=lambda entry: (self._selection_mp(entry), -entry.harness.children),
        )

    def run(self, generations: int = 2) -> RunReport:
        seed_entry = self.bootstrap()

        for generation in range(1, generations + 1):
            parent = self.select_parent()
            parent_errors = diagnose(parent.harness, parent.representation, self.val)
            bundles = propose_with_policy(
                parent.harness,
                parent_errors,
                self.policy,
                generation=generation,
            )
            accepted_this_generation = False

            for index, bundle in enumerate(bundles):
                version = f"h{generation}_{index}"
                child: Harness | None = None
                child_rep: Representation | None = None
                child_metrics: dict[str, Metrics] = {}
                inner_history: list[dict[str, Any]] = []
                synthesis_payload: dict[str, Any] | None = None
                record_edits = [edit.to_dict() for edit in bundle]
                decision: GateDecision
                try:
                    if len(bundle) == 1 and bundle[0].kind == "SYNTHESIZE_OPERATOR":
                        assert self.seed_test_risk is not None
                        edit = bundle[0]
                        ctx = SynthesisContext(
                            harness=parent.harness,
                            representation=parent.representation,
                            diagnosis=parent_errors,
                            fit_cases=self.fit,
                            val_cases=self.val,
                            iid_cases=self.dataset.iid,
                            test_cases=self.test,
                            parent_metrics=parent.metrics,
                            parent_errors=parent_errors,
                            seed_risk=self.seed_test_risk,
                            version=version,
                            gate_split=self.gate_split,
                            mp_split=self.gate_mp_split,
                            require_mp=self.strict_mp,
                            scope_hint=str(edit.args.get("scope_hint", "ordinary")),
                        )
                        result = self.synthesizer.synthesize(ctx)
                        if result is None:
                            decision = GateDecision(
                                accepted=False,
                                delta_in=0.0,
                                delta_ho=0.0,
                                delta_mp=0.0,
                                reason="synthesis produced no candidate",
                                gate_split=self.gate_split,
                                parent=parent.metrics,
                                child={},
                            )
                        else:
                            synthesis_payload = result.to_dict()
                            child = result.child
                            child_rep = result.child_rep
                            decision = result.decision or GateDecision(
                                accepted=False,
                                delta_in=0.0,
                                delta_ho=0.0,
                                delta_mp=0.0,
                                reason="synthesis gate missing",
                                gate_split=self.gate_split,
                                parent=parent.metrics,
                                child={},
                            )
                            if child is not None and child_rep is not None:
                                child_metrics = self._metrics(child, child_rep)
                            record_edits = [
                                Edit(
                                    "ADD_OPERATOR",
                                    result.operator_name,
                                    {"program": result.program.to_dict()},
                                    template_id=edit.template_id,
                                ).to_dict(),
                                Edit(
                                    "REWIRE",
                                    result.operator_name,
                                    {"before": "predict"},
                                    template_id=edit.template_id,
                                ).to_dict(),
                            ]
                    else:
                        child = apply_edit_bundle(parent.harness, bundle, version)
                        child_rep, inner_history = inner_refine(
                            child,
                            self.train,
                            self.val,
                            fit_cases=self.fit,
                            max_rounds=self.max_inner_rounds,
                        )
                        child_metrics = self._metrics(child, child_rep)
                        child_errors = diagnose(child, child_rep, self.val)
                        decision = accept_edit(
                            parent.metrics,
                            child_metrics,
                            parent_errors,
                            child_errors,
                            gate_split=self.gate_split,
                            mp_split=self.gate_mp_split,
                            require_mp=self.strict_mp,
                        )
                except Exception as exc:  # invalid harness / unexecutable edit
                    decision = GateDecision(
                        accepted=False,
                        delta_in=0.0,
                        delta_ho=0.0,
                        delta_mp=0.0,
                        reason=f"invalid harness: {exc}",
                        gate_split=self.gate_split,
                        parent=parent.metrics,
                        child={},
                    )

                record = EditRecord(
                    generation=generation,
                    parent_version=parent.harness.version,
                    child_version=version,
                    edits=record_edits,
                    accepted=decision.accepted,
                    reason=decision.reason,
                    delta_in=decision.delta_in,
                    delta_ho=decision.delta_ho,
                    delta_mp=decision.delta_mp,
                    parent_metrics=parent.metrics,
                    child_metrics=child_metrics,
                    inner_history=inner_history,
                    diagnosis_kinds=sorted(error_kinds(parent_errors)),
                    template_ids=[
                        edit.template_id for edit in bundle if edit.template_id
                    ],
                    policy_version=self.policy.version,
                    synthesis=synthesis_payload,
                )
                self.edit_log.append(record)
                update_stats(self.policy, record)

                if decision.accepted and child is not None and child_rep is not None:
                    parent.harness.children += 1
                    edit_from_parent = (
                        Edit(
                            "ADD_OPERATOR",
                            record_edits[0]["target"],
                            dict(record_edits[0].get("args", {})),
                        )
                        if record_edits and record_edits[0]["kind"] == "ADD_OPERATOR"
                        else bundle[0]
                    )
                    entry = ArchiveEntry(
                        harness=child,
                        representation=child_rep,
                        metrics=child_metrics,
                        generation=generation,
                        inner_history=inner_history,
                        edit_from_parent=edit_from_parent,
                        accepted=True,
                    )
                    self.archive.append(entry)
                    accepted_this_generation = True
                    break  # one accepted child per generation

            if not accepted_this_generation:
                # Keep the archive unchanged; later generations can still try.
                continue

        best = max(
            self.archive,
            key=lambda entry: (self._selection_mp(entry), -entry.harness.children),
        )
        return RunReport(
            dataset_summary={
                "seed": len(self.dataset.seed),
                "induction": len(self.dataset.induction),
                "val": len(self.val),
                "test_compositional": len(self.test),
                "iid": len(self.dataset.iid),
                "surface": len(self.dataset.surface),
                "counterfactual": len(self.dataset.counterfactual),
            },
            seed_version=seed_entry.harness.version,
            seed_metrics=seed_entry.metrics,
            seed_inner=seed_entry.inner_history,
            archive=list(self.archive),
            edit_log=list(self.edit_log),
            best_version=best.harness.version,
            best_metrics=best.metrics,
            best_spec=best.harness.spec(),
            gate_split=self.gate_split,
            gate_mp_split=self.gate_mp_split,
            policy=describe_policy(self.policy),
        )


def error_summary(errors) -> dict[str, int]:
    return error_counts(errors)
