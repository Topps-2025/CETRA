"""Typed construction operators for the CETRA-RSI harness.

The seed harness contains the operators needed to decompose cases, abduce
concepts, reconstruct chains, and induce **single-condition** rules.  It does
not contain ``ComposeScope``, so it cannot express interactions such as
``commit_state=failed AND reversibility=irreversible AND authority=owner``.
That missing operator is the harness-limited gap used by the demo.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Any

from .types import BuildState, CaseRecord, OperatorSpec, Rule


class Operator:
    """Base class for typed operators."""

    spec: OperatorSpec

    def run(
        self,
        state: BuildState,
        cases: list[CaseRecord],
        params: dict[str, Any],
    ) -> BuildState:
        raise NotImplementedError


def _majority(actions: list[str]) -> tuple[str, int]:
    counts = Counter(actions)
    action, count = counts.most_common(1)[0]
    return action, count


def _matches(case: CaseRecord, conditions: tuple[tuple[str, str], ...]) -> bool:
    return all(case.factors.get(concept) == value for concept, value in conditions)


@dataclass
class DecomposeOperator(Operator):
    """Split raw cases into decision points and evidence spans."""

    spec: OperatorSpec = field(
        default_factory=lambda: OperatorSpec(
            name="decompose",
            inputs=("raw_case",),
            outputs=("decision_point", "evidence"),
            description="segment a ticket at its decision point and attach the narrative as evidence",
        )
    )

    def run(self, state, cases, params):
        for case in cases:
            state.evidence.setdefault(case.case_id, []).append(case.narrative)
        state.traces["n_cases"] = len(cases)
        state.traces["decision_points"] = {
            case.case_id: {"action": case.action, "factors": dict(case.factors)} for case in cases
        }
        return state


@dataclass
class AbduceOperator(Operator):
    """Propose concepts that are supported by at least two values in the data."""

    spec: OperatorSpec = field(
        default_factory=lambda: OperatorSpec(
            name="abduce",
            inputs=("decision_point", "evidence"),
            outputs=("concept", "value_set"),
            description="select supported concepts and their value sets",
        )
    )

    def run(self, state, cases, params):
        pool = list(params.get("concept_pool", []))
        min_support = int(params.get("min_support", 2))
        selected: list[str] = []
        for concept in pool:
            values = sorted({case.factors[concept] for case in cases if concept in case.factors})
            support = sum(1 for case in cases if concept in case.factors)
            if len(values) >= 2 and support >= min_support:
                selected.append(concept)
                state.values[concept] = tuple(values)
        state.concepts = selected
        actions = [case.action for case in cases]
        state.fallback = _majority(actions)[0] if actions else None
        state.traces["abduced"] = list(selected)
        return state


@dataclass
class ReconstructOperator(Operator):
    """Build a simple concept-level chain for each case."""

    spec: OperatorSpec = field(
        default_factory=lambda: OperatorSpec(
            name="reconstruct",
            inputs=("concept", "evidence"),
            outputs=("chain",),
            description="reconstruct an attribute chain per case",
        )
    )

    def run(self, state, cases, params):
        chains: list[tuple[str, ...]] = []
        for case in cases:
            chain = tuple(f"{concept}={case.factors[concept]}" for concept in state.concepts)
            chains.append((case.case_id, *chain, case.action))
        state.chains = chains
        return state


@dataclass
class InduceSingleOperator(Operator):
    """Induce one-condition rules (the seed harness's rule learner).

    A single-condition learner cannot represent an interaction: if the same
    concept value is followed by different actions, it must pick a majority.
    """

    spec: OperatorSpec = field(
        default_factory=lambda: OperatorSpec(
            name="induce_single",
            inputs=("concept", "chain"),
            outputs=("rule",),
            description="induce one-condition rules by majority action per concept value",
        )
    )

    def run(self, state, cases, params):
        min_support = int(params.get("min_support", 2))
        min_precision = float(params.get("min_precision", 0.5))
        existing = list(state.rules)
        rules: list[Rule] = []
        seen: set[tuple[tuple[tuple[str, str], ...], str]] = {
            (rule.conditions, rule.action) for rule in existing
        }
        for concept in state.concepts:
            groups: dict[str, list[str]] = {}
            for case in cases:
                if concept in case.factors:
                    groups.setdefault(case.factors[concept], []).append(case.action)
            for value, actions in sorted(groups.items()):
                if len(actions) < min_support:
                    continue
                action, count = _majority(actions)
                precision = count / len(actions)
                if precision < min_precision:
                    continue
                key = (((concept, value),), action)
                if key in seen:
                    continue
                seen.add(key)
                rules.append(
                    Rule(
                        conditions=((concept, value),),
                        action=action,
                        support=len(actions),
                        precision=precision,
                    )
                )
        state.rules = existing + rules
        if cases:
            state.fallback = _majority([case.action for case in cases])[0]
        state.traces["single_rules"] = len(state.rules)
        return state


@dataclass
class ComposeScopeOperator(Operator):
    """Compose conjunctive, scoped rules (the operator missing from the seed).

    For each single-condition rule, greedily add conditions while the purity of
    the matched subset improves.  The result is a scoped conjunction such as
    ``commit_state=failed AND reversibility=irreversible AND authority=owner``.
    """

    spec: OperatorSpec = field(
        default_factory=lambda: OperatorSpec(
            name="compose_scope",
            inputs=("rule", "concept"),
            outputs=("scoped_rule",),
            description="compose two or more conditions into a scoped conjunctive rule",
        )
    )

    def run(self, state, cases, params):
        min_support = int(params.get("compose_min_support", params.get("min_support", 2)))
        max_conditions = int(params.get("max_conditions", 3))
        min_precision = float(params.get("compose_min_precision", 0.9))

        composed: list[Rule] = []
        seen: set[tuple[tuple[tuple[str, str], ...], str]] = set()

        def purity(subset: list[CaseRecord]) -> tuple[str, float, int]:
            if not subset:
                return ("", 0.0, 0)
            action, count = _majority([case.action for case in subset])
            return action, count / len(subset), len(subset)

        def values_for(concept: str) -> tuple[str, ...]:
            observed = {
                case.factors[concept] for case in cases if concept in case.factors
            }
            return tuple(sorted(set(state.values.get(concept, ())) | observed))

        # Enumerate all conjunctions of 2..max_conditions concepts.  The
        # search space is small; enumeration avoids the greedy path
        # dependence that misses interactions such as
        # (commit_state=failed, reversibility=irreversible, authority=owner).
        for arity in range(2, max_conditions + 1):
            for concepts in combinations(state.concepts, arity):
                value_lists = [values_for(concept) for concept in concepts]
                for values in product(*value_lists):
                    conditions = tuple(zip(concepts, values))
                    subset = [case for case in cases if _matches(case, conditions)]
                    if len(subset) < min_support:
                        continue
                    action, precision, support = purity(subset)
                    if precision < min_precision:
                        continue

                    # Keep only rules that are not redundant with a more
                    # general parent (same action and no precision gain).
                    redundant = False
                    for drop in range(arity):
                        parent_conditions = tuple(
                            condition
                            for index, condition in enumerate(conditions)
                            if index != drop
                        )
                        parent_subset = [
                            case for case in cases if _matches(case, parent_conditions)
                        ]
                        if not parent_subset:
                            continue
                        parent_action, parent_precision, _ = purity(parent_subset)
                        if parent_action == action and parent_precision >= precision - 1e-9:
                            redundant = True
                            break
                    if redundant:
                        continue

                    scope = tuple(f"{c}={v}" for c, v in conditions[1:])
                    rule = Rule(
                        conditions=conditions,
                        action=action,
                        scope=scope,
                        support=support,
                        precision=precision,
                    )
                    if (rule.conditions, rule.action) in seen:
                        continue
                    seen.add((rule.conditions, rule.action))
                    composed.append(rule)

        # Keep the single-condition rules as fallbacks; add composed rules.
        state.rules = list(state.rules) + composed
        state.traces["composed_rules"] = len(composed)
        return state


@dataclass
class InduceTreeOperator(Operator):
    """Induce multi-condition rules with a small decision tree.

    This is the landing point for real decisions that need conjunctions, such
    as ``missing_slots=yes AND user_feedback=none -> ask_clarification``.
    """

    spec: OperatorSpec = field(
        default_factory=lambda: OperatorSpec(
            name="induce_tree",
            inputs=("concept", "chain"),
            outputs=("rule",),
            description="induce conjunctive rules with a depth-bounded decision tree",
        )
    )

    def run(self, state, cases, params):
        from .induction import induce_decision_tree

        max_depth = int(params.get("tree_max_depth", 3))
        min_support = int(params.get("min_support", 2))
        min_precision = float(params.get("min_precision", 0.8))
        min_gain = float(params.get("min_gain", 1e-9))
        induced = induce_decision_tree(
            cases,
            max_depth=max_depth,
            min_support=min_support,
            min_precision=min_precision,
            min_gain=min_gain,
        )
        existing = list(state.rules)
        seen = {(rule.conditions, rule.thresholds, rule.action) for rule in existing}
        for rule in induced:
            key = (rule.conditions, rule.thresholds, rule.action)
            if key in seen:
                continue
            seen.add(key)
            existing.append(rule)
        state.rules = existing
        if cases:
            state.fallback = _majority([case.action for case in cases])[0]
        state.traces["tree_rules"] = len(induced)
        return state


@dataclass
class NoopInduceOperator(Operator):
    """A deliberately weak rule learner used to test gate rejection."""

    spec: OperatorSpec = field(
        default_factory=lambda: OperatorSpec(
            name="noop_induce",
            inputs=("concept",),
            outputs=("rule",),
            description="induce no rules; used as a negative edit",
        )
    )

    def run(self, state, cases, params):
        state.rules = []
        if cases:
            state.fallback = _majority([case.action for case in cases])[0]
        state.traces["noop_induce"] = True
        return state


@dataclass
class PredictOperator(Operator):
    """Placeholder for the prediction stage.

    Prediction is implemented by ``Harness.predict``; this operator keeps the
    pipeline typed and records the rule count.
    """

    spec: OperatorSpec = field(
        default_factory=lambda: OperatorSpec(
            name="predict",
            inputs=("rule", "case"),
            outputs=("action",),
            description="apply the most specific matching rule",
        )
    )

    def run(self, state, cases, params):
        state.traces["n_rules"] = len(state.rules)
        return state


OPERATOR_CLASSES: dict[str, type[Operator]] = {
    "decompose": DecomposeOperator,
    "abduce": AbduceOperator,
    "reconstruct": ReconstructOperator,
    "induce_single": InduceSingleOperator,
    "induce_tree": InduceTreeOperator,
    "compose_scope": ComposeScopeOperator,
    "noop_induce": NoopInduceOperator,
    "predict": PredictOperator,
}


def make_operator(name: str) -> Operator:
    if name not in OPERATOR_CLASSES:
        raise KeyError(f"unknown operator: {name}")
    return OPERATOR_CLASSES[name]()
