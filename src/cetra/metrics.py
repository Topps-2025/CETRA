"""Evaluation metrics for the memory abstraction experiment."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any, Iterable

from .cases import ACTIONS, Case
from .pipeline import Prediction


def is_correct(prediction: Prediction) -> bool:
    return prediction.valid and prediction.predicted_action == prediction.ground_truth


def accuracy(predictions: Iterable[Prediction]) -> float:
    predictions = list(predictions)
    if not predictions:
        return 0.0
    return sum(is_correct(p) for p in predictions) / len(predictions)


def bootstrap_ci(
    values: list[bool], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(sum(sample) / n)
    estimates.sort()
    lower = estimates[int((alpha / 2) * n_boot)]
    upper = estimates[int((1 - alpha / 2) * n_boot) - 1]
    return (lower, upper)


def mcnemar_exact(correct_a: list[bool], correct_b: list[bool]) -> dict[str, Any]:
    """Exact McNemar test for two paired binary outcomes."""
    if len(correct_a) != len(correct_b):
        raise ValueError("paired outcomes must have the same length")
    b = sum(1 for a, c in zip(correct_a, correct_b) if a and not c)
    c = sum(1 for a, c in zip(correct_a, correct_b) if not a and c)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return {"b": b, "c": c, "p_value": min(1.0, 2 * tail)}


def summarize_predictions(
    predictions: list[Prediction],
    cases: list[Case],
    n_boot: int = 2000,
) -> dict[str, Any]:
    split_of = {case.case_id: case.split for case in cases}
    by_method: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        by_method[prediction.method].append(prediction)

    result: dict[str, Any] = {}
    for method, items in sorted(by_method.items()):
        overall_values = [is_correct(p) for p in items]
        overall_ci = bootstrap_ci(overall_values, n_boot=n_boot)
        by_split: dict[str, Any] = {}
        for split in sorted({split_of[p.case_id] for p in items}):
            subset = [p for p in items if split_of[p.case_id] == split]
            values = [is_correct(p) for p in subset]
            ci = bootstrap_ci(values, n_boot=n_boot)
            by_split[split] = {
                "n": len(subset),
                "accuracy": sum(values) / len(values) if values else 0.0,
                "ci95": list(ci),
                "invalid": sum(1 for p in subset if not p.valid),
            }
        result[method] = {
            "n": len(items),
            "accuracy": sum(overall_values) / len(overall_values) if overall_values else 0.0,
            "ci95": list(overall_ci),
            "invalid": sum(1 for p in items if not p.valid),
            "by_split": by_split,
            "avg_prompt_tokens": (
                sum(int(p.usage.get("prompt_tokens", 0) or 0) for p in items) / len(items)
                if items
                else 0.0
            ),
            "avg_completion_tokens": (
                sum(int(p.usage.get("completion_tokens", 0) or 0) for p in items) / len(items)
                if items
                else 0.0
            ),
            "avg_latency_s": sum(p.latency_s for p in items) / len(items) if items else 0.0,
        }
    return result


def counterfactual_metrics(
    predictions: list[Prediction],
    cf_pairs: list[tuple[Case, Case]],
) -> dict[str, Any]:
    by_case = {p.case_id: p for p in predictions}
    rows = []
    for base, perturbed in cf_pairs:
        base_pred = by_case.get(base.case_id)
        pert_pred = by_case.get(perturbed.case_id)
        if base_pred is None or pert_pred is None:
            continue
        gt_changes = base.action != perturbed.action
        pred_changes = base_pred.predicted_action != pert_pred.predicted_action
        rows.append(
            {
                "pair_id": base.pair_id,
                "changed_factor": base.changed_factor,
                "gt_changes": gt_changes,
                "pred_changes": pred_changes,
                "consistent": gt_changes == pred_changes,
                "both_correct": is_correct(base_pred) and is_correct(pert_pred),
                "base_gt": base.action,
                "pert_gt": perturbed.action,
                "base_pred": base_pred.predicted_action,
                "pert_pred": pert_pred.predicted_action,
            }
        )
    if not rows:
        return {"n_pairs": 0}
    sensitivity_rows = [row for row in rows if row["gt_changes"]]
    invariance_rows = [row for row in rows if not row["gt_changes"]]
    return {
        "n_pairs": len(rows),
        "pair_consistency": sum(row["consistent"] for row in rows) / len(rows),
        "both_correct": sum(row["both_correct"] for row in rows) / len(rows),
        "sensitivity": (
            sum(row["pred_changes"] for row in sensitivity_rows) / len(sensitivity_rows)
            if sensitivity_rows
            else None
        ),
        "invariance": (
            sum(not row["pred_changes"] for row in invariance_rows) / len(invariance_rows)
            if invariance_rows
            else None
        ),
        "n_sensitivity": len(sensitivity_rows),
        "n_invariance": len(invariance_rows),
        "rows": rows,
    }


def accuracy_by_action(predictions: Iterable[Prediction]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.ground_truth].append(prediction)
    result: dict[str, dict[str, float]] = {}
    for action in sorted(grouped):
        items = grouped[action]
        result[action] = {
            "n": len(items),
            "accuracy": sum(is_correct(p) for p in items) / len(items),
        }
    return result


def paired_comparison(
    predictions: list[Prediction], method_a: str, method_b: str
) -> dict[str, Any]:
    by_method: dict[str, dict[str, Prediction]] = defaultdict(dict)
    for prediction in predictions:
        by_method[prediction.method][prediction.case_id] = prediction
    common = sorted(set(by_method[method_a]) & set(by_method[method_b]))
    correct_a = [is_correct(by_method[method_a][case_id]) for case_id in common]
    correct_b = [is_correct(by_method[method_b][case_id]) for case_id in common]
    test = mcnemar_exact(correct_a, correct_b)
    return {
        "method_a": method_a,
        "method_b": method_b,
        "n": len(common),
        "accuracy_a": sum(correct_a) / len(common) if common else 0.0,
        "accuracy_b": sum(correct_b) / len(common) if common else 0.0,
        **test,
    }


def action_distribution(cases: Iterable[Case]) -> dict[str, int]:
    counts = Counter(case.action for case in cases)
    return {action: counts.get(action, 0) for action in ACTIONS}
