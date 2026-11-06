"""Seed harness, evaluation, diagnosis, and inner CEGAR refinement."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .operators import make_operator
from .types import (
    BuildState,
    CaseRecord,
    Diagnosis,
    Harness,
    Metrics,
    Representation,
)


CONCEPT_POOL = (
    "authority",
    "commit_state",
    "retry_safety",
    "resource_state",
    "reversibility",
)


def make_seed_harness(
    concept_pool: tuple[str, ...] | list[str] | None = None,
    *,
    version: str = "h0_seed",
) -> Harness:
    """The seed harness: no ``compose_scope`` and a single-condition learner."""

    operators = {
        name: make_operator(name)
        for name in ("decompose", "abduce", "reconstruct", "induce_single", "predict")
    }
    return Harness(
        operators=operators,
        pipeline=["decompose", "abduce", "reconstruct", "induce_single", "predict"],
        params={
            "concept_pool": list(concept_pool or CONCEPT_POOL),
            "min_support": 2,
            "min_precision": 0.5,
            "compose_min_support": 2,
            "compose_min_precision": 0.9,
            "max_conditions": 3,
            "min_gain": 0.01,
        },
        version=version,
        metadata={"description": "single-condition learner without compose_scope"},
    )


def make_compose_harness(version: str = "h1_compose_scope") -> Harness:
    """A harness with the missing ``compose_scope`` operator wired in."""

    child = make_seed_harness().clone(version=version)
    child.operators["compose_scope"] = make_operator("compose_scope")
    if "compose_scope" not in child.pipeline:
        index = child.pipeline.index("predict")
        child.pipeline.insert(index, "compose_scope")
    child.metadata = {"description": "seed + compose_scope before predict"}
    return child


def make_tree_harness(
    concept_pool: tuple[str, ...] | list[str] | None = None,
    *,
    version: str = "h0_tree",
) -> Harness:
    """A harness for real traces: multi-condition rules instead of single ones.

    The controlled suite intentionally starts from a one-condition learner so
    the ``compose_scope`` gap is visible.  Real traces need conjunctions from
    the start; this harness wires a depth-bounded decision tree before predict.
    """

    operators = {
        name: make_operator(name)
        for name in ("decompose", "abduce", "reconstruct", "induce_tree", "predict")
    }
    return Harness(
        operators=operators,
        pipeline=["decompose", "abduce", "reconstruct", "induce_tree", "predict"],
        params={
            "concept_pool": list(concept_pool or CONCEPT_POOL),
            "min_support": 2,
            "min_precision": 0.8,
            "tree_max_depth": 3,
            "min_gain": 1e-9,
        },
        version=version,
        metadata={"description": "multi-condition tree learner for real traces"},
    )


def predict_all(
    harness: Harness,
    rep: Representation,
    cases: list[CaseRecord],
) -> list[tuple[CaseRecord, str]]:
    return [(case, harness.predict(rep, case)) for case in cases]


def risk(harness: Harness, rep: Representation, cases: list[CaseRecord]) -> float:
    if not cases:
        return 0.0
    correct = sum(1 for case in cases if harness.predict(rep, case) == case.action)
    return 1.0 - correct / len(cases)


def evaluate(
    harness: Harness,
    rep: Representation,
    cases: list[CaseRecord],
    *,
    seed_risk: float,
    cost_scale: float = 1.0,
    counterexamples_resolved: float = 0.0,
) -> Metrics:
    if not cases:
        return Metrics(n=0, correct=0, risk=0.0, cost=0.0, mp=0.0)
    correct = sum(1 for case in cases if harness.predict(rep, case) == case.action)
    risk_value = 1.0 - correct / len(cases)
    # A simple, deterministic cost proxy: pipeline depth plus rule count.
    cost = cost_scale * len(cases) * (len(harness.pipeline) + 0.25 * len(rep.rules))
    mp = (seed_risk - risk_value) / max(cost, 1e-9)
    return Metrics(
        n=len(cases),
        correct=correct,
        risk=risk_value,
        cost=cost,
        mp=mp,
        counterexample_resolution=counterexamples_resolved,
    )


def _single_rule_matches(rep: Representation, case: CaseRecord) -> list:
    matches = []
    for rule in rep.rules:
        if len(rule.conditions) != 1:
            continue
        if all(case.factors.get(concept) == value for concept, value in rule.conditions):
            matches.append(rule)
    return matches


def _is_composition_error(rep: Representation, case: CaseRecord) -> bool:
    """True when the case has single-condition coverage but no correct rule."""

    single = _single_rule_matches(rep, case)
    if not single:
        return False
    return all(rule.action != case.action for rule in single)


def _is_fallback_gap(rep: Representation, case: CaseRecord) -> bool:
    """True when a rare branch has no ordinary rule and a scoped default would help.

    This is intentionally narrow: it is a *proposal signal*, not a claim that
    the branch is truly a fallback case.  The outer gate still has to validate
    the resulting harness on held-out data.
    """

    if _single_rule_matches(rep, case):
        return False
    return (
        case.factors.get("commit_state") == "failed"
        and case.factors.get("retry_safety") == "unsafe"
    )


def diagnose(
    harness: Harness,
    rep: Representation,
    cases: list[CaseRecord],
) -> list[Diagnosis]:
    errors: list[Diagnosis] = []
    for case in cases:
        predicted = harness.predict(rep, case)
        if predicted == case.action:
            continue
        missing = tuple(
            concept for concept in case.factors if concept not in rep.concepts
        )
        if missing:
            kind = "missing_concept"
            detail = f"case needs concepts {missing}"
        elif _is_composition_error(rep, case):
            kind = "operator_gap" if "compose_scope" not in harness.operators else "composition_error"
            detail = "single-condition rules match but none predicts the expert action"
        elif _is_fallback_gap(rep, case):
            kind = "fallback_gap"
            detail = "rare failed/unsafe branch has no ordinary rule; a scoped default may help"
        else:
            kind = "missing_rule"
            detail = "no rule covers this case"
        errors.append(
            Diagnosis(
                case_id=case.case_id,
                predicted=predicted,
                truth=case.action,
                kind=kind,
                missing_concepts=missing,
                detail=detail,
            )
        )
    return errors


def add_concept(
    harness: Harness,
    rep: Representation,
    concept: str,
    cases: list[CaseRecord],
) -> Representation:
    """Add one concept to the representation and re-induce rules."""

    child = rep.copy()
    child.concepts = tuple(sorted(set(child.concepts) | {concept}))
    child.values[concept] = tuple(
        sorted({case.factors[concept] for case in cases if concept in case.factors})
    )
    # Keep the existing rules; induce only the new concept's rules.  Re-inducing
    # every concept on the larger fit set would wash out the seed rules.
    state = BuildState(
        concepts=[concept],
        values={concept: child.values[concept]},
        rules=list(child.rules),
        chains=list(child.chains),
        evidence={key: list(value) for key, value in child.evidence.items()},
        fallback=child.fallback,
    )
    learner = "induce_tree" if "induce_tree" in harness.operators else "induce_single"
    state = harness.operators[learner].run(state, cases, harness.params)
    # Restore the full concept set for composition and prediction.
    state.concepts = list(child.concepts)
    state.values = dict(child.values)
    if "compose_scope" in harness.operators:
        state = harness.operators["compose_scope"].run(state, cases, harness.params)
    return state.to_representation()


def inner_refine(
    harness: Harness,
    train: list[CaseRecord],
    val: list[CaseRecord],
    *,
    fit_cases: list[CaseRecord] | None = None,
    max_rounds: int = 3,
) -> tuple[Representation, list[dict[str, Any]]]:
    """Inner CEGAR: add concepts until validation risk stops improving.

    This loop deliberately cannot add a new operator; it can only refine the
    representation inside the fixed harness.  On the latent dataset it adds
    ``reversibility`` and then plateaus on the interaction cases.
    """

    fit_cases = fit_cases or train
    # The seed harness is intentionally built on the search split so the
    # compose_scope gap stays visible.  A tree harness needs the fuller fit set
    # to induce multi-condition rules.
    build_cases = fit_cases if "induce_tree" in harness.operators else train
    rep = harness.build(build_cases)
    history: list[dict[str, Any]] = []
    current_risk = risk(harness, rep, val)
    history.append(
        {
            "round": 0,
            "action": "bootstrap",
            "concepts": list(rep.concepts),
            "risk": current_risk,
            "rules": len(rep.rules),
        }
    )

    for round_index in range(1, max_rounds + 1):
        errors = diagnose(harness, rep, val)
        missing = sorted(
            {
                concept
                for error in errors
                for concept in error.missing_concepts
                if concept not in rep.concepts
            }
        )
        if not missing:
            break
        best: tuple[float, str, Representation] | None = None
        for concept in missing:
            candidate = add_concept(harness, rep, concept, fit_cases)
            candidate_risk = risk(harness, candidate, val)
            if best is None or candidate_risk < best[0]:
                best = (candidate_risk, concept, candidate)
        if best is None:
            break
        candidate_risk, concept, candidate = best
        # Accept a risk improvement, or a missing-concept counterexample
        # resolution that is risk-neutral.  The latter is an enabler: the
        # concept may only pay off once the outer loop adds a composition
        # operator, so rejecting it here would make the harness-limited gap
        # invisible.
        resolves_missing = any(concept in error.missing_concepts for error in errors)
        if candidate_risk > current_risk + 1e-9:
            break
        if candidate_risk >= current_risk - 1e-9 and not resolves_missing:
            break
        previous_risk = current_risk
        current_risk, concept, rep = candidate_risk, concept, candidate
        history.append(
            {
                "round": round_index,
                "action": "ADD_CONCEPT",
                "concept": concept,
                "concepts": list(rep.concepts),
                "risk": current_risk,
                "risk_delta": current_risk - previous_risk,
                "rules": len(rep.rules),
                "reason": "risk" if candidate_risk < previous_risk else "counterexample",
            }
        )
    return rep, history


def majority_action(cases: list[CaseRecord]) -> str:
    if not cases:
        return "noop"
    return Counter(case.action for case in cases).most_common(1)[0][0]
