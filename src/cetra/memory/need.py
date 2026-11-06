"""A 'need' measure for recursive deconstruction.

A field is decomposed only if the decomposition earns its keep.  The primary
signal is decision information: how much does splitting the field reduce the
entropy of the action?  The cost is the number of new fields plus depth.  If
no action labels are available, a compression proxy (description-length
reduction per new field) is used.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


def _entropy(values: Iterable[str]) -> float:
    items = list(values)
    if not items:
        return 0.0
    counts = Counter(items)
    total = len(items)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


@dataclass(frozen=True)
class NeedScore:
    field: str
    gain: float
    cost: float
    need: float
    action_entropy_before: float
    action_entropy_after: float
    accepted: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "gain": round(self.gain, 6),
            "cost": round(self.cost, 6),
            "need": round(self.need, 6),
            "action_entropy_before": round(self.action_entropy_before, 6),
            "action_entropy_after": round(self.action_entropy_after, 6),
            "accepted": self.accepted,
            "reason": self.reason,
        }


@dataclass
class NeedEstimator:
    min_gain: float = 0.01
    min_need: float = 0.05
    field_cost: float = 0.2
    depth_cost: float = 0.1
    max_depth: int = 3

    def score(
        self,
        field_name: str,
        parent_values: list[Any],
        child_fields: dict[str, list[Any]],
        action_values: list[str] | None = None,
        depth: int = 0,
    ) -> NeedScore:
        child_names = list(child_fields)
        cost = self.field_cost * max(1, len(child_names)) + self.depth_cost * depth
        before = after = 0.0
        decision_gain = 0.0
        if action_values and len(action_values) == len(parent_values):
            parent_groups: dict[str, list[str]] = {}
            child_groups: dict[tuple[str, ...], list[str]] = {}
            for index, action in enumerate(action_values):
                parent_groups.setdefault(str(parent_values[index]), []).append(action)
                child_key = tuple(str(child_fields[name][index]) for name in child_names)
                child_groups.setdefault(child_key, []).append(action)
            before = sum(
                len(group) / len(action_values) * _entropy(group)
                for group in parent_groups.values()
            )
            after = sum(
                len(group) / len(action_values) * _entropy(group)
                for group in child_groups.values()
            )
            decision_gain = max(0.0, before - after)
        # Structural/compression gain: a long composite value split into
        # shorter typed fields is useful even if it does not change action
        # entropy (e.g. parsing a command into tool + arguments).
        parent_len = sum(len(str(value)) for value in parent_values)
        child_len = sum(
            len(str(value)) for values in child_fields.values() for value in values
        )
        compression_gain = (
            max(0.0, (parent_len - child_len) / max(1, parent_len)) * 0.5
        )
        gain = decision_gain + compression_gain
        need = gain / (cost + 1e-9)
        accepted = (
            gain > self.min_gain
            and need > self.min_need
            and depth < self.max_depth
            and len(child_names) >= 1
        )
        reason = "accepted" if accepted else (
            "depth limit" if depth >= self.max_depth
            else "low gain" if gain <= self.min_gain
            else "low need"
        )
        return NeedScore(
            field=field_name,
            gain=gain,
            cost=cost,
            need=need,
            action_entropy_before=before,
            action_entropy_after=after,
            accepted=accepted,
            reason=reason,
        )
