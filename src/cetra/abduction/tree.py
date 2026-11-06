"""A small discrete decision-tree learner over abduced concepts.

The tree is a canonical recursive partition of the concept space.  It is used as
a strong symbolic rule learner: if the abduced concepts are sufficient, a shallow
tree should recover the hidden policy.  This separates concept discovery from
LLM rule induction.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TreeNode:
    action: str | None = None
    feature: str | None = None
    children: dict[str, "TreeNode"] = field(default_factory=dict)
    samples: int = 0

    def is_leaf(self) -> bool:
        return self.feature is None


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
        if count > 0
    )


def _majority(records: list[dict[str, Any]]) -> str:
    counts = Counter(record["action"] for record in records)
    return counts.most_common(1)[0][0]


def build_tree(
    records: list[dict[str, Any]],
    features: list[str],
    *,
    max_depth: int = 6,
    min_samples: int = 2,
    depth: int = 0,
) -> TreeNode:
    if not records:
        return TreeNode(action="noop", samples=0)
    counts = Counter(record["action"] for record in records)
    node = TreeNode(action=counts.most_common(1)[0][0], samples=len(records))
    if len(counts) == 1 or depth >= max_depth or len(records) < min_samples:
        return node

    parent_entropy = _entropy(counts)
    best_feature: str | None = None
    best_gain = 1e-9
    for feature in features:
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            value = record["values"].get(feature, "unknown")
            groups.setdefault(value, []).append(record)
        if len(groups) <= 1:
            continue
        weighted = sum(
            (len(group) / len(records)) * _entropy(Counter(r["action"] for r in group))
            for group in groups.values()
        )
        gain = parent_entropy - weighted
        if gain > best_gain:
            best_gain = gain
            best_feature = feature

    if best_feature is None:
        return node

    node.feature = best_feature
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        value = record["values"].get(best_feature, "unknown")
        groups.setdefault(value, []).append(record)
    for value, group in groups.items():
        node.children[value] = build_tree(
            group,
            [feature for feature in features if feature != best_feature],
            max_depth=max_depth,
            min_samples=min_samples,
            depth=depth + 1,
        )
    return node


def predict_tree(tree: TreeNode, values: dict[str, str]) -> str:
    node = tree
    while not node.is_leaf():
        assert node.feature is not None
        value = values.get(node.feature, "unknown")
        if value in node.children:
            node = node.children[value]
        elif "unknown" in node.children:
            node = node.children["unknown"]
        else:
            break
    return node.action or "noop"


def tree_stats(tree: TreeNode) -> dict[str, Any]:
    def walk(node: TreeNode) -> tuple[int, int, int]:
        if node.is_leaf():
            return 1, 0, node.samples
        leaves = 0
        depth = 0
        samples = 0
        for child in node.children.values():
            child_leaves, child_depth, child_samples = walk(child)
            leaves += child_leaves
            depth = max(depth, child_depth + 1)
            samples += child_samples
        return leaves, depth, samples

    leaves, depth, samples = walk(tree)
    return {"leaves": leaves, "depth": depth, "samples": samples}


def tree_to_conditions(tree: TreeNode) -> list[dict[str, Any]]:
    """Export root-to-leaf paths as rules (condition, action)."""
    rules: list[dict[str, Any]] = []

    def walk(node: TreeNode, condition: dict[str, str]) -> None:
        if node.is_leaf():
            if node.action is not None:
                rules.append({"condition": dict(condition), "action": node.action})
            return
        assert node.feature is not None
        for value, child in node.children.items():
            walk(child, {**condition, node.feature: value})

    walk(tree, {})
    return rules
