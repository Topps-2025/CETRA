"""Rule/chain/skill recomposition.

The key operation in the SE experiment is ``derive_scoped_rule``: given a
counterexample, find a minimal set of conditions that keeps precision high and
excludes cases with a different action.  The resulting rule is a scoped memory
atom; it becomes a skill only if it passes the consolidation gate.
"""

from __future__ import annotations

from typing import Iterable

from cetra.rsi.types import CaseRecord, Rule


def _matches(case: CaseRecord, conditions: tuple[tuple[str, str], ...]) -> bool:
    return all(case.factors.get(key) == value for key, value in conditions)


def _precision(case: CaseRecord, conditions: tuple[tuple[str, str], ...],
               cases: list[CaseRecord]) -> tuple[float, int]:
    matched = [item for item in cases if _matches(item, conditions)]
    if not matched:
        return 0.0, 0
    correct = sum(1 for item in matched if item.action == case.action)
    return correct / len(matched), len(matched)


def derive_scoped_rule(
    case: CaseRecord,
    cases: Iterable[CaseRecord],
    fields: Iterable[str],
    *,
    min_support: int = 2,
    min_precision: float = 0.9,
    blocked_values: Iterable[str] = ("unknown", "other", "__missing__"),
) -> Rule | None:
    """Greedily derive a minimal scoped rule from one counterexample.

    The search starts from the most discriminative condition of ``case`` and
    adds conditions until precision is high enough or no condition helps.
    Abstention values are not allowed as hard-rule conditions: a rule such as
    ``context=unknown -> replan`` is usually a proxy, not a decision factor.
    """

    items = list(cases)
    field_set = set(fields)
    blocked = {str(value) for value in blocked_values}
    candidates = [
        (key, value)
        for key, value in case.factors.items()
        if key in field_set and str(value) not in blocked
    ]
    if not candidates:
        return None

    # Rank single conditions by precision, then support.
    ranked = []
    for key, value in candidates:
        conditions = ((key, value),)
        precision, support = _precision(case, conditions, items)
        ranked.append((precision, support, key, value))
    ranked.sort(reverse=True)

    chosen: list[tuple[str, str]] = []
    best: tuple[float, int, tuple[tuple[str, str], ...]] | None = None

    def consider(conditions: list[tuple[str, str]]) -> None:
        nonlocal best
        cond = tuple(sorted(conditions))
        precision, support = _precision(case, cond, items)
        if support < min_support:
            return
        if best is None or (precision, support) > (best[0], best[1]):
            best = (precision, support, cond)

    for _, _, key, value in ranked:
        chosen.append((key, value))
        consider(chosen)
        if best is not None and best[0] >= min_precision:
            break

    # If no single condition reached precision, add the condition that most
    # improves precision.
    if best is None or best[0] < min_precision:
        remaining = [item for item in candidates if item not in chosen]
        while remaining:
            scored = []
            for key, value in remaining:
                precision, support = _precision(case, tuple(sorted(chosen + [(key, value)])), items)
                scored.append((precision, support, key, value))
            scored.sort(reverse=True)
            precision, support, key, value = scored[0]
            chosen.append((key, value))
            consider(chosen)
            remaining = [item for item in remaining if item != (key, value)]
            if best is not None and best[0] >= min_precision:
                break

    if best is None:
        return None
    precision, support, conditions = best
    if support < min_support:
        return None
    # A scoped rule is a hard memory atom.  Returning a low-precision rule and
    # relying on a later aggregate-risk gate is not enough: on small dev/test
    # sets it can look like a gain while encoding a spurious proxy.  Refuse it
    # here, so the caller can try a larger factor subset instead.
    if precision < min_precision:
        return None
    return Rule(
        conditions=conditions,
        action=case.action,
        support=support,
        precision=round(precision, 6),
        scope=("feedback",),
    )
