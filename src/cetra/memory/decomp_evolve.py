"""Joint self-evolution of a decomposition program and its memory rules.

This is the co-evolution loop the deterministic prototype was missing:

1. an LLM/heuristic proposes a DecompIR program (factor schema + extractors);
2. the program is executed once and scored by coverage / decision gain / need;
3. the SE layer searches factor subsets and re-derives scoped rules from the
   decomposed cases;
4. an edit is accepted only if held-out risk improves without negative
   transfer, and the accepted subset becomes a new program version.

The LLM proposes the hypothesis space; SE decides what survives.  Neither a
prompt alone nor a fixed splitter can do both.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from cetra.rsi.types import CaseRecord, Rule

from .decomp_program import (
    MISSING,
    client_available,
    DecompositionProgram,
    ExtractionReport,
    apply_program,
)
from .evolve import Evaluation, _gate, evaluate
from .recompose import derive_scoped_rule
from .types import ProgramState


@dataclass
class ProgramScore:
    coverage: float
    decision_gain: float
    compression: float
    cost: float
    need: float
    fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage": round(self.coverage, 4),
            "decision_gain": round(self.decision_gain, 4),
            "compression": round(self.compression, 4),
            "cost": round(self.cost, 4),
            "need": round(self.need, 4),
            "fields": list(self.fields),
        }


@dataclass
class PreparedSplits:
    program: DecompositionProgram
    fit: list[CaseRecord]
    dev: list[CaseRecord]
    test: list[CaseRecord]
    all_rule: list[CaseRecord]
    report: ExtractionReport
    raw_lengths: dict[str, int] = field(default_factory=dict)


@dataclass
class JointEvaluation:
    program: DecompositionProgram
    fields: tuple[str, ...]
    dev: Evaluation
    test: Evaluation
    score: ProgramScore
    extraction: ExtractionReport
    rule: Rule | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "program": self.program.to_dict(),
            "fields": list(self.fields),
            "dev": _eval_dict(self.dev),
            "test": _eval_dict(self.test),
            "score": self.score.to_dict(),
            "extraction": self.extraction.to_dict(),
            "rule": _rule_dict(self.rule),
        }


def _eval_dict(evaluation: Evaluation) -> dict[str, Any]:
    return {
        "risk": round(evaluation.risk, 4),
        "correct": evaluation.correct,
        "n": evaluation.n,
        "predictions": dict(evaluation.predictions),
        "truths": dict(evaluation.truths),
    }


def _rule_dict(rule: Rule | None) -> dict[str, Any] | None:
    if rule is None:
        return None
    return {
        "conditions": [list(item) for item in rule.conditions],
        "action": rule.action,
        "scope": list(rule.scope),
        "support": rule.support,
        "precision": round(rule.precision, 4),
    }


def _raw(case: CaseRecord, source_field: str) -> str:
    if source_field in case.factors:
        return str(case.factors[source_field])
    if source_field == "narrative":
        return case.narrative
    return ""


def _entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def score_program(
    program: DecompositionProgram,
    fit_cases: list[CaseRecord],
    *,
    fields: Iterable[str] | None = None,
    source_field: str = "narrative",
    report: ExtractionReport | None = None,
) -> ProgramScore:
    selected = tuple(fields if fields is not None else program.factor_names())
    actions = [case.action for case in fit_cases]
    before = _entropy(actions)
    groups: dict[tuple[str, ...], list[str]] = {}
    for case in fit_cases:
        key = tuple(str(case.factors.get(name, MISSING)) for name in selected)
        groups.setdefault(key, []).append(case.action)
    after = sum(len(group) / len(fit_cases) * _entropy(group) for group in groups.values()) if fit_cases else 0.0
    decision_gain = max(0.0, before - after) if selected else 0.0

    raw_len = sum(len(_raw(case, source_field)) for case in fit_cases)
    child_len = sum(
        len(str(case.factors.get(name, "")))
        for case in fit_cases
        for name in selected
        if case.factors.get(name) not in (None, "", MISSING)
    )
    compression = (
        max(0.0, (raw_len - child_len) / max(1, raw_len)) * 0.5 if selected else 0.0
    )
    factor_costs = sum(spec.cost for spec in program.factors if spec.name in set(selected))
    cost = factor_costs + 0.2 * len(selected)
    need = (decision_gain + compression) / (cost + 1e-9) if selected else 0.0
    coverage = 0.0
    if report is not None and selected:
        coverage = sum(report.coverage.get(name, 0.0) for name in selected) / len(selected)
    return ProgramScore(
        coverage=coverage,
        decision_gain=decision_gain,
        compression=compression,
        cost=cost,
        need=need,
        fields=tuple(selected),
    )


def prepare_program(
    program: DecompositionProgram,
    *,
    fit: list[CaseRecord],
    dev: list[CaseRecord],
    test: list[CaseRecord],
    all_rule_cases: list[CaseRecord] | None = None,
    source_field: str = "narrative",
    client: Any | None = None,
    base_fields: tuple[str, ...] = (),
) -> PreparedSplits:
    """Execute a program once over all splits; subsets reuse the extraction."""

    unique: dict[str, CaseRecord] = {}
    for case in list(fit) + list(dev) + list(test) + list(all_rule_cases or []):
        unique.setdefault(case.case_id, case)
    cases = list(unique.values())
    rebuilt, report = apply_program(
        program, cases, client=client, keep_base=False
    )
    if base_fields:
        for case in rebuilt:
            original = unique[case.case_id]
            for name in base_fields:
                if name in original.factors:
                    case.factors[name] = original.factors[name]
    by_id = {case.case_id: case for case in rebuilt}
    return PreparedSplits(
        program=program,
        fit=[by_id[case.case_id] for case in fit],
        dev=[by_id[case.case_id] for case in dev],
        test=[by_id[case.case_id] for case in test],
        all_rule=[by_id[case.case_id] for case in (all_rule_cases or [])],
        report=report,
        raw_lengths={case.case_id: len(_raw(case, source_field)) for case in cases},
    )


def evaluate_prepared(
    prepared: PreparedSplits,
    program: DecompositionProgram,
    *,
    fields: Iterable[str],
    source_field: str = "narrative",
    base_fields: tuple[str, ...] = (),
    operator: str = "induce_single",
    feedback_case: CaseRecord | None = None,
) -> JointEvaluation:
    selected = tuple(dict.fromkeys(tuple(base_fields) + tuple(fields)))
    rule: Rule | None = None
    if feedback_case is not None and prepared.all_rule:
        feedback = next(
            (case for case in prepared.all_rule if case.case_id == feedback_case.case_id),
            None,
        )
        if feedback is not None and selected:
            rule = derive_scoped_rule(
                feedback,
                prepared.all_rule,
                fields=selected,
                min_support=2,
                min_precision=0.9,
            )
    state = ProgramState(
        fields=selected,
        operator=operator,
        params={"min_support": 2},
        injected_rules=[rule] if rule is not None else [],
    )
    dev_eval = evaluate(state, prepared.fit, prepared.dev)
    test_eval = evaluate(state, prepared.fit, prepared.test)
    score = score_program(
        program,
        prepared.fit,
        fields=selected,
        source_field=source_field,
        report=prepared.report,
    )
    return JointEvaluation(
        program=program,
        fields=selected,
        dev=dev_eval,
        test=test_eval,
        score=score,
        extraction=prepared.report,
        rule=rule,
    )


def candidate_subsets(
    fields: Iterable[str],
    *,
    max_size: int = 4,
    max_candidates: int = 300,
) -> list[tuple[str, ...]]:
    names = list(dict.fromkeys(fields))
    subsets: list[tuple[str, ...]] = []
    for size in range(1, min(max_size, len(names)) + 1):
        for combo in itertools.combinations(names, size):
            subsets.append(tuple(combo))
            if len(subsets) >= max_candidates:
                return subsets
    return subsets


def _complexity(item: JointEvaluation) -> tuple[int, int]:
    """Occam tie-break: fewer rule conditions, then fewer fields."""

    conditions = len(item.rule.conditions) if item.rule is not None else 99
    return conditions, len(item.fields)


def _joint_gate(
    before_dev: Evaluation,
    after_dev: Evaluation,
    before_test: Evaluation,
    after_test: Evaluation,
    *,
    before_complexity: tuple[int, int] = (99, 99),
    after_complexity: tuple[int, int] = (99, 99),
) -> tuple[bool, str]:
    """Gate for joint (decomposition, rule) edits.

    Test is the held-out split and must never degrade.  If test is tied, a
    lower dev risk is still a valid joint gain: it means the edit improved the
    validation fit without costing held-out performance.  This matters when a
    spurious first proposal already achieved the same test risk.
    """

    if after_dev.risk > before_dev.risk + 1e-9:
        return False, "dev risk degraded"
    if after_test.risk > before_test.risk + 1e-9:
        return False, "test risk degraded (negative transfer)"
    for case_id, truth in before_dev.truths.items():
        if (
            before_dev.predictions.get(case_id) == truth
            and after_dev.predictions.get(case_id) != truth
        ):
            return False, f"negative transfer on {case_id}"
    for case_id, truth in before_test.truths.items():
        if (
            before_test.predictions.get(case_id) == truth
            and after_test.predictions.get(case_id) != truth
        ):
            return False, f"negative transfer on {case_id}"
    if (
        after_test.risk >= before_test.risk - 1e-9
        and after_dev.risk >= before_dev.risk - 1e-9
    ):
        if after_complexity < before_complexity:
            return True, "accepted: simpler rule/field set at equal risk"
        return False, "no joint gain"
    return True, "accepted: no test regression and dev/test joint gain"


def _feedback_from_result(
    result: dict[str, Any],
    dev_cases: list[CaseRecord],
    source_field: str,
) -> list[dict[str, Any]]:
    by_id = {case.case_id: case for case in dev_cases}
    dev = result.get("final", {}).get("dev", {})
    predictions = dev.get("predictions", {})
    truths = dev.get("truths", {})
    feedback: list[dict[str, Any]] = []
    for case_id, predicted in predictions.items():
        truth = truths.get(case_id)
        if truth is None or predicted == truth or case_id not in by_id:
            continue
        case = by_id[case_id]
        feedback.append(
            {
                "case_id": case_id,
                "raw": _raw(case, source_field)[:1200],
                "action": truth,
                "predicted": predicted,
            }
        )
    return feedback


def run_iterative_se(
    *,
    fit: list[CaseRecord],
    dev: list[CaseRecord],
    test: list[CaseRecord],
    initial_candidates: list[tuple[str, DecompositionProgram]],
    all_rule_cases: list[CaseRecord] | None = None,
    feedback_case: CaseRecord | None = None,
    source_field: str = "narrative",
    client: Any | None = None,
    base_fields: tuple[str, ...] = (),
    operator: str = "induce_single",
    max_fields: int = 4,
    beam: int = 8,
    rounds: int = 2,
) -> dict[str, Any]:
    """Run propose -> gate -> counterexample feedback -> revise -> gate."""

    from .llm_decomposer import LLMDecomposer  # local import avoids a cycle

    history: list[dict[str, Any]] = []
    candidates = list(initial_candidates)
    accepted = False
    for round_index in range(max(1, rounds)):
        result = run_joint_se(
            fit=fit,
            dev=dev,
            test=test,
            candidates=candidates,
            all_rule_cases=all_rule_cases,
            feedback_case=feedback_case,
            source_field=source_field,
            client=client,
            base_fields=base_fields,
            operator=operator,
            max_fields=max_fields,
            beam=beam,
        )
        history.append(result)
        accepted = bool(result.get("accepted"))
        if accepted or round_index + 1 >= rounds:
            break
        if not client_available(client):
            break
        current = DecompositionProgram.from_dict(
            result["proposals"][0]["program"]
        ) if result.get("proposals") else None
        if current is None or not current.factors:
            break
        feedback = _feedback_from_result(result, dev, source_field)
        if not feedback:
            break
        revised = LLMDecomposer(client).revise(
            current,
            fit,
            feedback,
            source_field=source_field,
            metrics=result["final"].get("score", {}),
        )
        if revised is None or revised.program_id == current.program_id:
            break
        candidates = [("llm_revise", revised)]
    return {
        "accepted": accepted,
        "rounds": history,
        "final": history[-1] if history else None,
        "revisions": max(0, len(history) - 1),
    }


def run_joint_se(
    *,
    fit: list[CaseRecord],
    dev: list[CaseRecord],
    test: list[CaseRecord],
    candidates: list[tuple[str, DecompositionProgram]],
    all_rule_cases: list[CaseRecord] | None = None,
    feedback_case: CaseRecord | None = None,
    source_field: str = "narrative",
    client: Any | None = None,
    base_fields: tuple[str, ...] = (),
    operator: str = "induce_single",
    max_fields: int = 4,
    beam: int = 8,
    min_need: float = 0.02,
    min_coverage: float = 0.5,
) -> dict[str, Any]:
    """Propose -> decompose -> select factors -> derive rules -> gate."""

    baseline_program = DecompositionProgram(
        program_id="dp:empty",
        factors=[],
        rationale="no decomposition",
        source="baseline",
    )
    baseline_prepared = prepare_program(
        baseline_program,
        fit=fit,
        dev=dev,
        test=test,
        all_rule_cases=all_rule_cases,
        source_field=source_field,
        client=client,
        base_fields=base_fields,
    )
    baseline = evaluate_prepared(
        baseline_prepared,
        baseline_program,
        fields=base_fields,
        source_field=source_field,
        base_fields=base_fields,
        operator=operator,
        feedback_case=feedback_case,
    )

    records: list[dict[str, Any]] = []
    best = baseline
    best_program = baseline_program
    best_proposal = "baseline"
    subset_total = 0

    for proposal_name, program in candidates:
        prepared = prepare_program(
            program,
            fit=fit,
            dev=dev,
            test=test,
            all_rule_cases=all_rule_cases,
            source_field=source_field,
            client=client,
            base_fields=base_fields,
        )
        evaluated: list[JointEvaluation] = []
        subsets = candidate_subsets(program.factor_names(), max_size=max_fields)
        subset_total += len(subsets)
        for subset in subsets:
            restricted = program.restrict(
                subset, rationale=f"{program.rationale} | select {','.join(subset)}"
            )
            evaluated.append(
                evaluate_prepared(
                    prepared,
                    restricted,
                    fields=subset,
                    source_field=source_field,
                    base_fields=base_fields,
                    operator=operator,
                    feedback_case=feedback_case,
                )
            )
        # Search on dev; test is only used by the gate.
        evaluated.sort(key=lambda item: (item.dev.risk, -item.score.need))
        for item in evaluated[:beam]:
            accepted, reason = _joint_gate(
                best.dev,
                item.dev,
                best.test,
                item.test,
                before_complexity=_complexity(best),
                after_complexity=_complexity(item),
            )
            if item.score.need < min_need:
                accepted, reason = False, f"need {item.score.need:.3f} < {min_need}"
            if item.score.coverage < min_coverage:
                accepted, reason = (
                    False,
                    f"coverage {item.score.coverage:.3f} < {min_coverage}",
                )
            records.append(
                {
                    "proposal": proposal_name,
                    "program": item.program.to_dict(),
                    "fields": list(item.fields),
                    "dev_risk": round(item.dev.risk, 4),
                    "test_risk": round(item.test.risk, 4),
                    "need": round(item.score.need, 4),
                    "coverage": round(item.score.coverage, 4),
                    "accepted": accepted,
                    "reason": reason,
                    "rule": _rule_dict(item.rule),
                }
            )
            if accepted and (
                (
                    round(item.test.risk, 9),
                    round(item.dev.risk, 9),
                    _complexity(item),
                )
                < (
                    round(best.test.risk, 9),
                    round(best.dev.risk, 9),
                    _complexity(best),
                )
            ):
                best = item
                best_program = item.program
                best_proposal = proposal_name

    accepted_records = [record for record in records if record["accepted"]]
    return {
        "baseline": baseline.to_dict(),
        "proposals": [
            {"name": name, "program": program.to_dict()} for name, program in candidates
        ],
        "search": {
            "max_fields": max_fields,
            "beam": beam,
            "subsets": subset_total,
            "gated": len(records),
            "accepted": len(accepted_records),
        },
        "candidates": records,
        "final": best.to_dict(),
        "final_program": best_program.to_dict(),
        "final_proposal": best_proposal,
        "accepted": bool(accepted_records),
    }
