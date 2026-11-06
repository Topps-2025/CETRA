"""Metrics for abductive concept discovery, grounding, rules, and QA."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any

from ..llm import DeepSeekClient
from ..pipeline import Prediction
from ..rules import RuleSet
from .concepts import ConceptSet
from .latent_cases import LatentCase
from .prompts import CONCEPT_JUDGE_SYSTEM, concept_judge_user

HIDDEN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "authority": ("authority", "access", "permission", "auth", "role", "owner", "delegate"),
    "commit_state": ("commit", "state", "status", "settle", "progress", "transaction"),
    "retry_safety": ("retry", "safety", "idempot", "duplicate", "repeat", "dedup"),
    "resource_state": ("resource", "capacity", "availability", "lock", "quota", "worker"),
    "reversibility": ("revers", "undo", "rollback", "compensat", "permanent", "irreversible"),
}


def grounding_stats(records: list[dict[str, Any]], num_concepts: int) -> dict[str, Any]:
    total_slots = len(records) * num_concepts
    unknown = sum(len(record.get("unknown_concepts", [])) for record in records)
    known = sum(len(record.get("known_concepts", [])) for record in records)
    grounded_known = sum(len(record.get("grounded_known", [])) for record in records)
    return {
        "cases": len(records),
        "num_concepts": num_concepts,
        "known_rate": known / total_slots if total_slots else 0.0,
        "unknown_rate": unknown / total_slots if total_slots else 0.0,
        "evidence_grounded_rate": grounded_known / known if known else 0.0,
        "avg_known_per_case": known / len(records) if records else 0.0,
    }


def keyword_concept_fidelity(
    concept_set: ConceptSet, hidden_concepts: dict[str, list[str]]
) -> dict[str, Any]:
    proposed_text = {
        concept.name: (
            concept.name
            + " "
            + concept.definition
            + " "
            + " ".join(
                phrase for phrases in concept.indicators.values() for phrase in phrases
            )
        ).lower()
        for concept in concept_set
    }
    matches: dict[str, str | None] = {}
    matched_proposed: set[str] = set()
    for hidden, keywords in HIDDEN_KEYWORDS.items():
        best_name = None
        best_score = 0
        for name, text in proposed_text.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > best_score:
                best_score = score
                best_name = name
        matches[hidden] = best_name if best_score > 0 else None
        if best_name:
            matched_proposed.add(best_name)
    return {
        "matches": matches,
        "hidden_coverage": sum(1 for value in matches.values() if value) / len(hidden_concepts),
        "proposed_precision": len(matched_proposed) / len(concept_set) if len(concept_set) else 0.0,
        "matched_proposed": sorted(matched_proposed),
    }


def llm_concept_fidelity(
    client: DeepSeekClient,
    concept_set: ConceptSet,
    hidden_concepts: dict[str, list[str]],
) -> dict[str, Any]:
    try:
        payload, _ = client.chat_json(
            [
                {"role": "system", "content": CONCEPT_JUDGE_SYSTEM},
                {"role": "user", "content": concept_judge_user(concept_set, hidden_concepts)},
            ],
            max_tokens=4096,
            reasoning_effort="low",
            phase="concept_judge",
        )
        matches = payload.get("matches") or {}
        normalized = {}
        for hidden in hidden_concepts:
            value = matches.get(hidden)
            if isinstance(value, str) and value.strip().lower() not in {"", "none", "null"}:
                normalized[hidden] = value.strip()
            else:
                normalized[hidden] = None
        precision_matches = payload.get("precision_matches") or []
        precision_matches = [str(x) for x in precision_matches if str(x).strip()]
        return {
            "matches": normalized,
            "hidden_coverage": sum(1 for value in normalized.values() if value) / len(hidden_concepts),
            "proposed_precision": (
                len(set(precision_matches)) / len(concept_set) if len(concept_set) else 0.0
            ),
            "matched_proposed": sorted(set(precision_matches)),
        }
    except Exception as exc:  # noqa: BLE001
        fallback = keyword_concept_fidelity(concept_set, hidden_concepts)
        fallback["error"] = str(exc)[:300]
        return fallback


def _entropy(counts: Counter | dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
        if count > 0
    )


def _mutual_information(table: dict[str, dict[str, int]]) -> float:
    total = sum(sum(row.values()) for row in table.values())
    if total <= 0:
        return 0.0
    row_sums = Counter({key: sum(row.values()) for key, row in table.items()})
    col_sums: Counter = Counter()
    for row in table.values():
        for value, count in row.items():
            col_sums[value] += count
    mi = 0.0
    for hidden, row in table.items():
        for value, count in row.items():
            p_joint = count / total
            p_hidden = row_sums[hidden] / total
            p_value = col_sums[value] / total
            if p_joint > 0:
                mi += p_joint * math.log2(p_joint / (p_hidden * p_value))
    return max(0.0, mi)


def _nmi(table: dict[str, dict[str, int]]) -> float:
    row_sums = Counter({key: sum(row.values()) for key, row in table.items()})
    col_sums: Counter = Counter()
    for row in table.values():
        for value, count in row.items():
            col_sums[value] += count
    h_hidden = _entropy(row_sums)
    h_value = _entropy(col_sums)
    if h_hidden <= 0 or h_value <= 0:
        return 0.0
    return _mutual_information(table) / math.sqrt(h_hidden * h_value)


def value_alignment(
    concept_set: ConceptSet,
    records: list[dict[str, Any]],
    cases: list[LatentCase],
    matches: dict[str, str | None],
) -> dict[str, Any]:
    """Check whether a proposed concept preserves the hidden value distinctions.

    Purity is the fraction of cases whose proposed value agrees with the majority
    proposed value for their hidden value.  NMI detects value collapse: if owner
    and delegate both map to "authenticated", NMI drops even though each row is
    individually pure.
    """
    by_id = {record["case_id"]: record for record in records}
    per_hidden: dict[str, Any] = {}
    for hidden, proposed in matches.items():
        if not proposed:
            continue
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        total = 0
        for case in cases:
            record = by_id.get(case.case_id)
            if record is None:
                continue
            value = record["values"].get(proposed, "unknown")
            if value == "unknown":
                continue
            hidden_value = case.factors.get(hidden, "unknown")
            counts[hidden_value][value] += 1
            total += 1
        if total == 0:
            continue
        purity = sum(max(row.values()) for row in counts.values()) / total
        injective = sum(1 for row in counts.values() if len(row) == 1)
        per_hidden[hidden] = {
            "proposed": proposed,
            "total": total,
            "hidden_values": len(counts),
            "purity": purity,
            "injective_values": injective,
            "injective_rate": injective / len(counts) if counts else 0.0,
            "nmi": _nmi(counts),
            "table": {key: dict(value) for key, value in counts.items()},
        }
    if not per_hidden:
        return {
            "per_hidden": {},
            "avg_purity": 0.0,
            "avg_injective_rate": 0.0,
            "avg_nmi": 0.0,
        }
    return {
        "per_hidden": per_hidden,
        "avg_purity": sum(item["purity"] for item in per_hidden.values()) / len(per_hidden),
        "avg_injective_rate": sum(item["injective_rate"] for item in per_hidden.values())
        / len(per_hidden),
        "avg_nmi": sum(item["nmi"] for item in per_hidden.values()) / len(per_hidden),
    }


def rule_set_stats(
    rule_set: RuleSet,
    records: list[dict[str, Any]],
    cases: list[LatentCase],
) -> dict[str, Any]:
    by_id = {record["case_id"]: record for record in records}
    correct = covered = 0
    for case in cases:
        record = by_id.get(case.case_id)
        values = record["values"] if record else {}
        action, matched, _ = rule_set.predict(values)
        covered += int(action is not None)
        correct += int(action == case.action)
    n = len(cases)
    return {
        "num_rules": len(rule_set),
        "coverage": covered / n if n else 0.0,
        "accuracy": correct / n if n else 0.0,
        "accuracy_on_covered": correct / covered if covered else 0.0,
        "avg_condition_length": (
            sum(len(rule.condition) for rule in rule_set) / len(rule_set) if len(rule_set) else 0.0
        ),
        "avg_rule_precision": (
            sum(rule.precision or 0.0 for rule in rule_set) / len(rule_set)
            if len(rule_set)
            else 0.0
        ),
    }


def counterfactual_metrics(
    predictions: list[Prediction],
    cf_pairs: list[tuple[LatentCase, LatentCase]],
) -> dict[str, Any]:
    by_case = {prediction.case_id: prediction for prediction in predictions}
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
                "both_correct": (
                    base_pred.predicted_action == base.action
                    and pert_pred.predicted_action == perturbed.action
                ),
            }
        )
    if not rows:
        return {"n_pairs": 0}
    sensitivity = [row for row in rows if row["gt_changes"]]
    invariance = [row for row in rows if not row["gt_changes"]]
    return {
        "n_pairs": len(rows),
        "pair_consistency": sum(row["consistent"] for row in rows) / len(rows),
        "both_correct": sum(row["both_correct"] for row in rows) / len(rows),
        "sensitivity": (
            sum(row["pred_changes"] for row in sensitivity) / len(sensitivity)
            if sensitivity
            else None
        ),
        "invariance": (
            sum(not row["pred_changes"] for row in invariance) / len(invariance)
            if invariance
            else None
        ),
        "n_sensitivity": len(sensitivity),
        "n_invariance": len(invariance),
        "rows": rows,
    }


def qa_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"n": 0, "accuracy": 0.0, "valid_rate": 0.0}
    by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_concept[result.get("answer_concept", "unknown")].append(result)
    return {
        "n": len(results),
        "accuracy": sum(1 for result in results if result.get("correct")) / len(results),
        "valid_rate": sum(1 for result in results if result.get("valid")) / len(results),
        "by_concept": {
            concept: {
                "n": len(items),
                "accuracy": sum(1 for item in items if item.get("correct")) / len(items),
            }
            for concept, items in sorted(by_concept.items())
        },
    }


def compression_stats(concept_set: ConceptSet, rule_set: RuleSet) -> dict[str, Any]:
    concept_chars = len(json.dumps(concept_set.to_dict(), ensure_ascii=False))
    rule_chars = len(json.dumps(rule_set.to_dict(), ensure_ascii=False))
    return {
        "num_concepts": len(concept_set),
        "num_rules": len(rule_set),
        "concept_chars": concept_chars,
        "rule_chars": rule_chars,
        "total_chars": concept_chars + rule_chars,
    }
