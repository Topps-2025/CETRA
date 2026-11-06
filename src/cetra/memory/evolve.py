"""Field-level and operator-level self-evolution for memory construction.

The loop is deliberately small and inspectable:

1. build a baseline program (fields + operator + rules);
2. evaluate on dev/test splits;
3. propose a field edit and/or a rule edit from a dev counterexample;
4. gate the edit on held-out risk and negative transfer;
5. repeat until no edit is accepted.

It is a real SE loop in the sense that the accepted edit changes the program's
field schema/topology and the memory rules, while the outer loop itself stays
fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from cetra.rsi.harness import make_seed_harness, make_tree_harness, risk
from cetra.rsi.types import CaseRecord, Representation, Rule

from .recompose import derive_scoped_rule
from .types import Edit, ProgramState


@dataclass
class Evaluation:
    risk: float
    correct: int
    n: int
    predictions: dict[str, str]
    truths: dict[str, str]


def build_program(state: ProgramState, fit_cases: list[CaseRecord]) -> tuple[object, Representation]:
    # An empty field set must not silently fall back to the default concept
    # pool: the baseline program really has no factors to read.
    concept_pool = state.fields or ("__none__",)
    if state.operator == "induce_tree":
        harness = make_tree_harness(concept_pool=concept_pool)
    else:
        harness = make_seed_harness(concept_pool=concept_pool)
    harness.params.update(state.params)
    rep = harness.build(fit_cases)
    existing = {rule.key() for rule in rep.rules}
    for rule in state.injected_rules:
        if rule.key() not in existing:
            rep.rules.append(rule)
            existing.add(rule.key())
    return harness, rep


def evaluate(
    state: ProgramState,
    fit_cases: list[CaseRecord],
    cases: list[CaseRecord],
) -> Evaluation:
    harness, rep = build_program(state, fit_cases)
    predictions = {case.case_id: harness.predict(rep, case) for case in cases}
    truths = {case.case_id: case.action for case in cases}
    correct = sum(1 for case in cases if predictions[case.case_id] == case.action)
    return Evaluation(
        risk=1.0 - correct / len(cases) if cases else 0.0,
        correct=correct,
        n=len(cases),
        predictions=predictions,
        truths=truths,
    )


def _gate(
    before_dev: Evaluation,
    after_dev: Evaluation,
    before_test: Evaluation,
    after_test: Evaluation,
    *,
    min_test_gain: float = 1e-9,
    max_dev_regression: float = 1e-9,
) -> tuple[bool, str]:
    if after_dev.risk > before_dev.risk + max_dev_regression:
        return False, "dev risk degraded"
    if after_test.risk > before_test.risk + 1e-9:
        return False, "test risk degraded (negative transfer)"
    # Per-case negative transfer: an edit must not flip a previously correct
    # case to wrong, even if the aggregate risk improves.
    for case_id, truth in before_dev.truths.items():
        if before_dev.predictions.get(case_id) == truth and after_dev.predictions.get(case_id) != truth:
            return False, f"negative transfer on {case_id}"
    for case_id, truth in before_test.truths.items():
        if before_test.predictions.get(case_id) == truth and after_test.predictions.get(case_id) != truth:
            return False, f"negative transfer on {case_id}"
    if before_test.risk - after_test.risk <= min_test_gain:
        return False, "no held-out gain"
    return True, "accepted: held-out gain without dev regression"


def run_se_loop(
    *,
    fit_cases: list[CaseRecord],
    dev_cases: list[CaseRecord],
    test_cases: list[CaseRecord],
    feedback_case: CaseRecord,
    all_rule_cases: list[CaseRecord],
    base_fields: tuple[str, ...] = ("has_enough_context", "missing_slots"),
) -> dict:
    """Run one field edit and one operator/rule edit, with a naive-rule control."""

    state = ProgramState(fields=base_fields, operator="induce_single", params={"min_support": 2})
    baseline_dev = evaluate(state, fit_cases, dev_cases)
    baseline_test = evaluate(state, fit_cases, test_cases)

    edits: list[dict] = []

    # ------------------------------------------------------------------
    # Edit 1: field-level. Add user_feedback to the schema.
    # ------------------------------------------------------------------
    if "user_feedback" not in state.fields:
        state.fields = tuple(sorted(set(state.fields) | {"user_feedback"}))
        state.history.append(
            {
                "level": "field",
                "op": "ADD_FIELD",
                "target": "user_feedback",
                "rationale": f"counterexample {feedback_case.case_id} is not representable",
            }
        )
        edits.append(
            {
                "edit": Edit(
                    level="field",
                    op="ADD_FIELD",
                    target="user_feedback",
                    rationale=f"counterexample {feedback_case.case_id} is not representable",
                ).to_dict(),
                "accepted": True,
                "reason": "schema can now represent the counterexample",
            }
        )

    dev_after_field = evaluate(state, fit_cases, dev_cases)
    test_after_field = evaluate(state, fit_cases, test_cases)

    # ------------------------------------------------------------------
    # Edit 2: operator/rule-level. Derive a scoped rule from dev feedback.
    # ------------------------------------------------------------------
    rule = derive_scoped_rule(
        feedback_case,
        all_rule_cases,
        fields=state.fields,
        min_support=2,
        min_precision=0.9,
    )
    if rule is not None:
        state.injected_rules.append(rule)
        state.topology = tuple(state.topology[:-1]) + ("inject_scoped_rule", "predict")
        dev_after_rule = evaluate(state, fit_cases, dev_cases)
        test_after_rule = evaluate(state, fit_cases, test_cases)
        accepted, reason = _gate(
            dev_after_field,
            dev_after_rule,
            test_after_field,
            test_after_rule,
        )
        if not accepted:
            state.injected_rules.pop()
            state.topology = tuple(state.topology[:-2]) + ("predict",)
        edits.append(
            {
                "edit": Edit(
                    level="operator",
                    op="ADD_SCOPED_RULE",
                    target="inject_scoped_rule",
                    args={"rule": rule.key()},
                    rationale=f"derived from {feedback_case.case_id}",
                ).to_dict(),
                "accepted": accepted,
                "reason": reason,
                "rule": {
                    "conditions": [list(item) for item in rule.conditions],
                    "action": rule.action,
                    "support": rule.support,
                    "precision": rule.precision,
                },
            }
        )

    # ------------------------------------------------------------------
    # Negative control: an unscoped rule. It should be rejected because
    # it causes negative transfer on the dev case with a different action.
    # ------------------------------------------------------------------
    naive = Rule(
        conditions=(("user_feedback", feedback_case.factors["user_feedback"]),),
        action=feedback_case.action,
        support=2,
        precision=0.66,
        scope=("naive",),
    )
    state_naive = state.clone()
    state_naive.injected_rules.append(naive)
    naive_dev = evaluate(state_naive, fit_cases, dev_cases)
    naive_test = evaluate(state_naive, fit_cases, test_cases)
    naive_accepted, naive_reason = _gate(
        dev_after_field,
        naive_dev,
        test_after_field,
        naive_test,
    )
    edits.append(
        {
            "edit": Edit(
                level="operator",
                op="ADD_RULE",
                target="naive_unscoped_rule",
                args={"conditions": [list(item) for item in naive.conditions]},
                rationale="negative control: no scope condition",
            ).to_dict(),
            "accepted": naive_accepted,
            "reason": naive_reason,
            "rule": {
                "conditions": [list(item) for item in naive.conditions],
                "action": naive.action,
                "support": naive.support,
                "precision": naive.precision,
            },
        }
    )

    final_dev = evaluate(state, fit_cases, dev_cases)
    final_test = evaluate(state, fit_cases, test_cases)
    return {
        "baseline": {
            "fields": list(base_fields),
            "operator": "induce_single",
            "dev": baseline_dev.__dict__,
            "test": baseline_test.__dict__,
        },
        "field_edit": {
            "fields": list(state.fields),
            "dev": dev_after_field.__dict__,
            "test": test_after_field.__dict__,
        },
        "final": {
            "fields": list(state.fields),
            "operator": state.operator,
            "topology": list(state.topology),
            "rules": [
                {
                    "conditions": [list(item) for item in rule.conditions],
                    "action": rule.action,
                    "support": rule.support,
                    "precision": rule.precision,
                }
                for rule in state.injected_rules
            ],
            "dev": final_dev.__dict__,
            "test": final_test.__dict__,
        },
        "edits": edits,
    }
