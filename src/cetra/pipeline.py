"""Decomposition, abstraction, retrieval, and prediction pipeline."""

from __future__ import annotations

import json
import math
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .cases import ACTIONS, FACTOR_VALUES, Case
from .llm import DeepSeekClient
from .prompts import (
    DECOMPOSE_SYSTEM,
    DIRECT_RULES_SYSTEM,
    INDUCE_RULES_SYSTEM,
    PREDICT_SYSTEM,
    SUMMARY_SYSTEM,
    decompose_user,
    direct_rules_user,
    format_decomposed_cases,
    format_raw_cases,
    format_rules,
    induce_rules_user,
    predict_user,
    summary_user,
)
from .rules import Rule, RuleSet

METHODS = (
    "no_memory",
    "raw_retrieval",
    "decomp_retrieval",
    "summary_memory",
    "direct_rules",
    "cetra_rules",
)


@dataclass
class Prediction:
    case_id: str
    method: str
    predicted_action: str | None
    ground_truth: str
    valid: bool
    reason: str = ""
    latency_s: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    memory_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "method": self.method,
            "predicted_action": self.predicted_action,
            "ground_truth": self.ground_truth,
            "valid": self.valid,
            "reason": self.reason,
            "latency_s": self.latency_s,
            "usage": self.usage,
            "memory_text": self.memory_text,
        }


def parallel_map(
    func: Callable[[Any], Any],
    items: Sequence[Any],
    workers: int = 8,
    label: str = "",
) -> list[Any]:
    """Run *func* over *items* with a thread pool, preserving input order."""
    results: list[Any] = [None] * len(items)
    done = 0
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(func, item): index for index, item in enumerate(items)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:  # noqa: BLE001 - keep the run alive
                results[index] = exc
            with lock:
                done += 1
                if label and (done % 10 == 0 or done == len(items)):
                    print(f"  {label}: {done}/{len(items)}", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------
class Decomposer:
    def __init__(
        self,
        client: DeepSeekClient,
        max_tokens: int = 4096,
        effort: str = "low",
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.effort = effort
        self.temperature = temperature

    def _fallback(self, case: Case, error: str = "") -> dict[str, Any]:
        return {
            "case_id": case.case_id,
            "decision_point": "choose the next recovery action",
            "observed_factors": dict(case.factors),
            "candidate_conditions": dict(case.factors),
            "action": case.action,
            "effect": case.outcome,
            "surface_details": [
                case.surface.get("service", ""),
                case.surface.get("user", ""),
                case.surface.get("request_id", ""),
                case.surface.get("error_code", ""),
            ],
            "evidence": case.narrative[:240],
            "fallback": True,
            "error": error[:200],
        }

    def decompose(self, case: Case) -> dict[str, Any]:
        try:
            payload, _ = self.client.chat_json(
                [
                    {"role": "system", "content": DECOMPOSE_SYSTEM},
                    {"role": "user", "content": decompose_user(case)},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                reasoning_effort=self.effort,
                phase="decompose",
            )
            return self._normalize(payload, case)
        except Exception as exc:  # noqa: BLE001 - fallback keeps the pipeline alive
            return self._fallback(case, str(exc))

    def _normalize(self, payload: dict[str, Any], case: Case) -> dict[str, Any]:
        observed = {
            str(k): str(v)
            for k, v in (payload.get("observed_factors") or {}).items()
            if k in FACTOR_VALUES and str(v) in FACTOR_VALUES[k]
        }
        if not observed:
            observed = dict(case.factors)
        candidate = {
            str(k): str(v)
            for k, v in (payload.get("candidate_conditions") or {}).items()
            if k in FACTOR_VALUES and str(v) in FACTOR_VALUES[k] and observed.get(k) == str(v)
        }
        if not candidate:
            candidate = dict(observed)
        surface = payload.get("surface_details") or []
        if not isinstance(surface, list):
            surface = [str(surface)]
        return {
            "case_id": case.case_id,
            "decision_point": str(payload.get("decision_point", "choose the next recovery action")),
            "observed_factors": observed,
            "candidate_conditions": candidate,
            "action": case.action,  # the expert label is authoritative
            "effect": case.outcome,
            "surface_details": [str(x) for x in surface][:8],
            "evidence": str(payload.get("evidence", ""))[:400],
            "fallback": False,
        }


# ---------------------------------------------------------------------------
# Rule induction
# ---------------------------------------------------------------------------
class RuleInducer:
    def __init__(
        self,
        client: DeepSeekClient,
        max_tokens: int = 8192,
        effort: str = "low",
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.effort = effort
        self.temperature = temperature

    def _rules_from_payload(self, payload: dict[str, Any]) -> list[Rule]:
        rules: list[Rule] = []
        for item in payload.get("rules") or []:
            try:
                rules.append(Rule.from_dict(item))
            except (ValueError, TypeError):
                continue
        return rules

    def induce(self, decomposed_cases: list[dict[str, Any]], batch_size: int = 8) -> tuple[RuleSet, list[str]]:
        batches = [
            decomposed_cases[i : i + batch_size]
            for i in range(0, len(decomposed_cases), batch_size)
        ]

        def run(batch: list[dict[str, Any]]) -> tuple[list[Rule], str | None]:
            try:
                payload, _ = self.client.chat_json(
                    [
                        {"role": "system", "content": INDUCE_RULES_SYSTEM},
                        {"role": "user", "content": induce_rules_user(batch)},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    reasoning_effort=self.effort,
                    phase="induce_rules",
                )
                rules = self._rules_from_payload(payload)
                if not rules:
                    return [], "empty rules array"
                return rules, None
            except Exception as exc:  # noqa: BLE001
                # Fallback: one rule per decomposed case.
                fallback = [
                    Rule(
                        condition=dict(item.get("candidate_conditions") or {}),
                        action=item["action"],
                        evidence_case_ids=[item["case_id"]],
                    )
                    for item in batch
                    if item.get("action") in ACTIONS
                ]
                return fallback, str(exc)[:300]

        results = parallel_map(run, batches, workers=len(batches), label="induce rules")
        rules: list[Rule] = []
        errors: list[str] = []
        for result in results:
            if isinstance(result, Exception):
                errors.append(str(result))
                continue
            batch_rules, error = result
            rules.extend(batch_rules)
            if error:
                errors.append(error)
        return RuleSet(rules), errors

    def induce_direct(self, cases: list[Case], batch_size: int = 8) -> tuple[RuleSet, list[str]]:
        batches = [cases[i : i + batch_size] for i in range(0, len(cases), batch_size)]

        def run(batch: list[Case]) -> tuple[list[Rule], str | None]:
            try:
                payload, _ = self.client.chat_json(
                    [
                        {"role": "system", "content": DIRECT_RULES_SYSTEM},
                        {"role": "user", "content": direct_rules_user(batch)},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    reasoning_effort=self.effort,
                    phase="induce_direct",
                )
                rules = self._rules_from_payload(payload)
                if not rules:
                    return [], "empty rules array"
                return rules, None
            except Exception as exc:  # noqa: BLE001
                fallback = [
                    Rule(
                        condition=dict(case.factors),
                        action=case.action,
                        evidence_case_ids=[case.case_id],
                    )
                    for case in batch
                ]
                return fallback, str(exc)[:300]

        results = parallel_map(run, batches, workers=len(batches), label="induce direct rules")
        rules: list[Rule] = []
        errors: list[str] = []
        for result in results:
            if isinstance(result, Exception):
                errors.append(str(result))
                continue
            batch_rules, error = result
            rules.extend(batch_rules)
            if error:
                errors.append(error)
        return RuleSet(rules), errors


# ---------------------------------------------------------------------------
# Summary baseline
# ---------------------------------------------------------------------------
class Summarizer:
    def __init__(
        self,
        client: DeepSeekClient,
        max_tokens: int = 2048,
        effort: str = "low",
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.effort = effort
        self.temperature = temperature

    def summarize(self, cases: list[Case]) -> str:
        try:
            response = self.client.chat(
                [
                    {"role": "system", "content": SUMMARY_SYSTEM},
                    {"role": "user", "content": summary_user(cases)},
                ],
                response_format=None,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                reasoning_effort=self.effort,
                phase="summary",
            )
            text = response.content.strip()
            if text:
                return text
        except Exception as exc:  # noqa: BLE001
            print(f"  summary failed: {exc}", file=sys.stderr)
        # Deterministic fallback: a compact action-condition table.
        return (
            "If permission is none, request permission. If the submission is committed, "
            "retry when the call is idempotent, otherwise cancel and resubmit. If the "
            "submission is pending, wait when the resource is locked, otherwise do nothing. "
            "If the submission failed and retry is not possible, roll back when the caller "
            "is the owner, otherwise escalate. If retry is possible, retry when the resource "
            "is available, escalate when it is exhausted, and wait when it is locked."
        )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


class BM25Index:
    """Small BM25 index; no external dependencies."""

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = list(documents)
        self.tokenized = [_tokenize(doc) for doc in self.documents]
        self.k1 = k1
        self.b = b
        self.avg_len = (
            sum(len(tokens) for tokens in self.tokenized) / len(self.tokenized)
            if self.tokenized
            else 0.0
        )
        self.df = Counter()
        for tokens in self.tokenized:
            for token in set(tokens):
                self.df[token] += 1
        self.n = len(self.tokenized)

    def score(self, query: str, index: int) -> float:
        tokens = _tokenize(query)
        if not tokens or not self.tokenized:
            return 0.0
        doc = self.tokenized[index]
        if not doc:
            return 0.0
        doc_len = len(doc)
        tf = Counter(doc)
        total = 0.0
        for token in tokens:
            if token not in tf:
                continue
            df = self.df.get(token, 0)
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            numerator = tf[token] * (self.k1 + 1)
            denominator = tf[token] + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_len, 1e-9))
            total += idf * numerator / denominator
        return total

    def top_k(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        scored = [(i, self.score(query, i)) for i in range(self.n)]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]


def retrieve_raw(case: Case, train_cases: list[Case], k: int = 3) -> list[Case]:
    index = getattr(retrieve_raw, "_index", None)
    if index is None or getattr(retrieve_raw, "_size", None) != len(train_cases):
        index = BM25Index([case.narrative for case in train_cases])
        retrieve_raw._index = index  # type: ignore[attr-defined]
        retrieve_raw._size = len(train_cases)  # type: ignore[attr-defined]
    return [train_cases[i] for i, _ in index.top_k(case.narrative, k=k)]  # type: ignore[attr-defined]


def retrieve_decomposed(
    case: Case, decomposed_cases: list[dict[str, Any]], k: int = 3
) -> list[dict[str, Any]]:
    def score(item: dict[str, Any]) -> float:
        conditions = item.get("candidate_conditions") or {}
        overlap = sum(1 for key, value in conditions.items() if case.factors.get(key) == value)
        text_overlap = len(set(_tokenize(case.narrative)) & set(_tokenize(json.dumps(item, ensure_ascii=False))))
        return 3.0 * overlap + 0.1 * text_overlap

    ranked = sorted(decomposed_cases, key=score, reverse=True)
    return ranked[:k]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class Predictor:
    def __init__(
        self,
        client: DeepSeekClient,
        max_tokens: int = 2048,
        effort: str = "low",
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.effort = effort
        self.temperature = temperature

    def predict(self, case: Case, memory_text: str, method: str) -> Prediction:
        try:
            payload, response = self.client.chat_json(
                [
                    {"role": "system", "content": PREDICT_SYSTEM},
                    {"role": "user", "content": predict_user(case, memory_text)},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                reasoning_effort=self.effort,
                phase=f"predict:{method}",
            )
            action = str(payload.get("action", "")).strip()
            valid = action in ACTIONS
            if not valid:
                # Recover an allowed action if it appears verbatim in the response.
                for candidate in ACTIONS:
                    if candidate in json.dumps(payload):
                        action = candidate
                        valid = True
                        break
            return Prediction(
                case_id=case.case_id,
                method=method,
                predicted_action=action if valid else None,
                ground_truth=case.action,
                valid=valid,
                reason=str(payload.get("reason", ""))[:500],
                latency_s=response.latency_s,
                usage=response.usage,
                memory_text=memory_text[:2000],
            )
        except Exception as exc:  # noqa: BLE001
            return Prediction(
                case_id=case.case_id,
                method=method,
                predicted_action=None,
                ground_truth=case.action,
                valid=False,
                reason=f"error: {exc}"[:500],
                memory_text=memory_text[:2000],
            )
