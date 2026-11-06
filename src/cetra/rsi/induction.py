"""Multi-condition rule induction and chain construction.

The seed harness in ``operators.py`` learns one-condition majority rules.  That
cannot express real decisions such as::

    missing_slots=yes and user_feedback=none -> ask_clarification

This module adds a small decision-tree learner and a chain builder.  Both are
deterministic; an LLM/ILP learner can replace them behind the same interfaces.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .types import CaseRecord, Rule


# ---------------------------------------------------------------------------
# Decision-tree rule induction
# ---------------------------------------------------------------------------


def _entropy(actions: list[str]) -> float:
    if not actions:
        return 0.0
    counts = Counter(actions)
    total = len(actions)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _majority(actions: list[str]) -> tuple[str, float]:
    counts = Counter(actions)
    action, count = counts.most_common(1)[0]
    return action, count / len(actions)


def _information_gain(
    cases: list[CaseRecord],
    factor: str,
) -> float:
    parent = _entropy([case.action for case in cases])
    groups: dict[str, list[CaseRecord]] = {}
    for case in cases:
        value = case.factors.get(factor)
        if value is None:
            continue
        groups.setdefault(value, []).append(case)
    if len(groups) < 2:
        return 0.0
    weighted = sum(
        (len(group) / len(cases)) * _entropy([case.action for case in group])
        for group in groups.values()
        if group
    )
    return parent - weighted


def induce_decision_tree(
    cases: Iterable[CaseRecord],
    *,
    max_depth: int = 3,
    min_support: int = 2,
    min_precision: float = 0.8,
    min_gain: float = 1e-9,
) -> list[Rule]:
    """Induce conjunctive rules with a small ID3-style tree.

    Leaves become rules.  A leaf is emitted when it is pure enough, has too
    little support to split, or reaches the depth limit.  The caller can still
    filter by support/precision; the tree deliberately returns all leaves so
    the downstream gate sees the full candidate set.
    """

    items = list(cases)
    if not items:
        return []

    factors = sorted({key for case in items for key in case.factors})
    rules: list[Rule] = []
    seen: set[tuple[tuple[tuple[str, str], ...], str]] = set()

    def build(subset: list[CaseRecord], path: tuple[tuple[str, str], ...], depth: int) -> None:
        actions = [case.action for case in subset]
        action, precision = _majority(actions)
        if (
            precision >= min_precision
            or depth >= max_depth
            or len(subset) < max(2 * min_support, min_support + 1)
        ):
            key = (path, action)
            if key not in seen:
                seen.add(key)
                rules.append(
                    Rule(
                        conditions=path,
                        action=action,
                        support=len(subset),
                        precision=precision,
                    )
                )
            return

        used = {name for name, _ in path}
        # Only split on a factor that is present for every case in this subset.
        # This avoids silently dropping cases with missing factors; a richer
        # learner can add an explicit "unknown" branch later.
        available = [
            factor
            for factor in factors
            if factor not in used and all(factor in case.factors for case in subset)
        ]
        if not available:
            key = (path, action)
            if key not in seen:
                seen.add(key)
                rules.append(
                    Rule(conditions=path, action=action, support=len(subset), precision=precision)
                )
            return

        gains = [(_information_gain(subset, factor), factor) for factor in available]
        gain, factor = max(gains, key=lambda item: (item[0], item[1]))
        if gain <= min_gain:
            key = (path, action)
            if key not in seen:
                seen.add(key)
                rules.append(
                    Rule(conditions=path, action=action, support=len(subset), precision=precision)
                )
            return

        values = sorted({case.factors.get(factor) for case in subset if case.factors.get(factor) is not None})
        for value in values:
            child = [case for case in subset if case.factors.get(factor) == value]
            if len(child) < min_support:
                continue
            build(child, path + ((factor, value),), depth + 1)

    build(items, (), 0)
    rules.sort(key=lambda rule: (rule.precision, rule.support, len(rule.conditions)), reverse=True)
    return rules


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------


DEFAULT_ACTION_EFFECTS: dict[str, tuple[str, str]] = {
    # long-horizon report / QA
    "ask_clarification": ("missing_slots", "no"),
    "proceed": ("missing_slots", "no"),
    "research": ("has_enough_context", "yes"),
    "replan": ("user_feedback", "none"),
    # tool failure recovery
    "rollback": ("commit_state", "committed"),
    "compensate": ("commit_state", "committed"),
    "retry": ("commit_state", "committed"),
    "request_authorization": ("authority", "owner"),
    "wait": ("resource_state", "free"),
    "escalate": ("done", "yes"),
    "write_report": ("done", "yes"),
    "answer": ("done", "yes"),
}


@dataclass(frozen=True)
class Chain:
    chain_id: str
    rule_ids: tuple[int, ...]
    actions: tuple[str, ...]
    conditions: tuple[tuple[str, str], ...]
    effect: str

    def to_dict(self) -> dict[str, object]:
        return {
            "chain_id": self.chain_id,
            "rule_ids": list(self.rule_ids),
            "actions": list(self.actions),
            "conditions": [list(item) for item in self.conditions],
            "effect": self.effect,
        }


def build_chains(
    rules: Iterable[Rule],
    *,
    action_effects: Mapping[str, tuple[str, str]] | None = None,
    max_length: int = 3,
    max_chains: int = 32,
) -> list[Chain]:
    """Connect rules when one rule's effect satisfies another's condition.

    The effect map is domain knowledge (or an explicit rule field in a later
    version).  If no effect is known, the action name is used as the effect and
    no chain is built across it.  This is deliberately conservative: a chain is
    a candidate for the gate, not an accepted fact.
    """

    items = list(rules)
    effects = dict(action_effects or DEFAULT_ACTION_EFFECTS)

    def effect_of(rule: Rule) -> tuple[str, str] | None:
        return effects.get(rule.action)

    chains: list[Chain] = []
    seen: set[tuple[int, ...]] = set()

    def extend(path: list[int]) -> None:
        if len(path) >= max_length or len(chains) >= max_chains:
            return
        last = items[path[-1]]
        effect = effect_of(last)
        if effect is None:
            return
        for index, rule in enumerate(items):
            if index in path:
                continue
            if effect in rule.conditions:
                new_path = path + [index]
                key = tuple(new_path)
                if key in seen:
                    continue
                seen.add(key)
                conditions: list[tuple[str, str]] = []
                for rule_index in new_path:
                    for condition in items[rule_index].conditions:
                        if condition not in conditions:
                            conditions.append(condition)
                chains.append(
                    Chain(
                        chain_id="chain_" + "_".join(str(i) for i in new_path),
                        rule_ids=tuple(new_path),
                        actions=tuple(items[i].action for i in new_path),
                        conditions=tuple(conditions),
                        effect=str(effect),
                    )
                )
                extend(new_path)

    for start in range(len(items)):
        extend([start])

    chains.sort(key=lambda chain: (len(chain.actions), chain.chain_id), reverse=True)
    return chains[:max_chains]
