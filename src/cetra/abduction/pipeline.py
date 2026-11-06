"""Abductive concept proposal, grounding, rule induction, prediction, and RSI."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterable

from ..cases import Case
from ..llm import DeepSeekClient
from ..pipeline import BM25Index, Prediction, parallel_map
from ..prompts import format_rules
from ..rules import Rule, RuleSet
from .concepts import Concept, ConceptSet, concept_values_from_record, evidence_from_record
from .latent_cases import ACTIONS, LatentCase
from .prompts import (
    ABDUCT_PREDICT_SYSTEM,
    CONCEPT_GROUNDING_SYSTEM,
    CONCEPT_PROPOSAL_SYSTEM,
    CONCEPT_REVISION_SYSTEM,
    QA_SYSTEM,
    RULE_INDUCTION_SYSTEM,
    SUMMARY_SYSTEM,
    abduct_predict_user,
    concept_grounding_user,
    concept_proposal_user,
    concept_revision_user,
    latent_summary_user,
    qa_user,
    rule_induction_user,
)


def default_concept_set() -> ConceptSet:
    """Safety net if the LLM returns no usable concepts."""
    return ConceptSet(
        [
            Concept("access", "who is allowed to act", ["owner", "delegate", "none"]),
            Concept("state", "whether a prior attempt took effect", ["settled", "queued", "errored"]),
            Concept("retry_safety", "whether repeating the operation is safe", ["safe", "unsafe"]),
            Concept("capacity", "whether the target resource can accept work", ["free", "locked", "exhausted"]),
            Concept("undoability", "whether the operation can be reversed", ["reversible", "irreversible"]),
        ]
    )


def to_rule_case(case: LatentCase, values: dict[str, str]) -> Case:
    return Case(
        case_id=case.case_id,
        split=case.split,
        factors=dict(values),
        surface={},
        action=case.action,
        outcome=case.outcome,
        narrative=case.narrative,
    )


def normalize_grounding_record(
    record: dict[str, Any],
    case: LatentCase,
    concept_set: ConceptSet,
) -> dict[str, Any]:
    values = concept_values_from_record(record, concept_set)
    for concept in concept_set:
        values.setdefault(concept.name, "unknown")
    evidence = evidence_from_record(record)
    normalized_narrative = re.sub(r"\s+", " ", case.narrative.lower())
    evidence_grounded: dict[str, bool] = {}
    for name, quote in evidence.items():
        quote_norm = re.sub(r"\s+", " ", quote.lower()).strip()
        evidence_grounded[name] = bool(quote_norm) and quote_norm[:80] in normalized_narrative
    known = [name for name, value in values.items() if value != "unknown"]
    grounded_known = [name for name in known if evidence_grounded.get(name, False)]
    return {
        "case_id": case.case_id,
        "values": values,
        "evidence": evidence,
        "evidence_grounded": evidence_grounded,
        "known_concepts": known,
        "unknown_concepts": [name for name, value in values.items() if value == "unknown"],
        "grounded_known": grounded_known,
        "narrative": case.narrative,
        "action": case.action,
        "outcome": case.outcome,
    }


class ConceptProposer:
    def __init__(
        self,
        client: DeepSeekClient,
        max_tokens: int = 16384,
        effort: str = "low",
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.effort = effort
        self.temperature = temperature

    def propose(self, cases: list[LatentCase]) -> tuple[ConceptSet, str]:
        messages = [
            {"role": "system", "content": CONCEPT_PROPOSAL_SYSTEM},
            {"role": "user", "content": concept_proposal_user(cases)},
        ]
        last_error = ""
        for attempt, max_tokens in enumerate((self.max_tokens, 32768), start=1):
            try:
                payload, response = self.client.chat_json(
                    messages,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    reasoning_effort=self.effort,
                    phase="abduct_concepts",
                )
                concept_set = ConceptSet.from_dict(payload)
                if len(concept_set) > 0:
                    return concept_set, response.content
                last_error = "empty concepts"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                print(
                    f"  concept proposal attempt {attempt} failed: {exc[:200]}",
                    file=sys.stderr,
                )
                # Ask for a shorter answer on the retry.
                if attempt == 1:
                    messages = messages + [
                        {
                            "role": "user",
                            "content": (
                                "Your previous answer was truncated or invalid. "
                                "Return at most 5 concepts, at most 2 indicators per value, "
                                "and definitions under 20 words. Return valid compact JSON."
                            ),
                        }
                    ]
        print(f"  concept proposal failed ({last_error}); using fallback", file=sys.stderr)
        return default_concept_set(), f"error: {last_error}"

    def revise(self, concept_set: ConceptSet, errors: list[dict[str, Any]]) -> tuple[ConceptSet, str]:
        try:
            payload, response = self.client.chat_json(
                [
                    {"role": "system", "content": CONCEPT_REVISION_SYSTEM},
                    {"role": "user", "content": concept_revision_user(concept_set, errors)},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                reasoning_effort=self.effort,
                phase="abduct_revise",
            )
            revised = ConceptSet.from_dict(payload)
            if len(revised) == 0:
                return concept_set, response.content
            return revised, response.content
        except Exception as exc:  # noqa: BLE001
            print(f"  concept revision failed: {exc}", file=sys.stderr)
            return concept_set, f"error: {exc}"


class ConceptGrounder:
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

    def ground(
        self,
        concept_set: ConceptSet,
        cases: list[LatentCase],
        batch_size: int = 4,
        workers: int = 8,
    ) -> list[dict[str, Any]]:
        batches = [cases[i : i + batch_size] for i in range(0, len(cases), batch_size)]

        def run(batch: list[LatentCase]) -> list[dict[str, Any]]:
            try:
                payload, _ = self.client.chat_json(
                    [
                        {"role": "system", "content": CONCEPT_GROUNDING_SYSTEM},
                        {"role": "user", "content": concept_grounding_user(concept_set, batch)},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    reasoning_effort=self.effort,
                    phase="abduct_ground",
                )
                raw_records = payload.get("records") or payload.get("results") or []
                if isinstance(raw_records, dict):
                    raw_records = [raw_records]
                by_id = {
                    str(record.get("case_id")): record
                    for record in raw_records
                    if isinstance(record, dict) and record.get("case_id")
                }
                return [
                    normalize_grounding_record(by_id.get(case.case_id, {}), case, concept_set)
                    for case in batch
                ]
            except Exception:  # noqa: BLE001
                return [
                    normalize_grounding_record({"case_id": case.case_id}, case, concept_set)
                    for case in batch
                ]

        results = parallel_map(run, batches, workers=workers, label="ground")
        records: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                continue
            records.extend(result)
        return records


def oracle_ground(concept_set: ConceptSet, cases: list[LatentCase]) -> list[dict[str, Any]]:
    records = []
    for case in cases:
        values = {concept.name: case.factors.get(concept.name, "unknown") for concept in concept_set}
        records.append(
            {
                "case_id": case.case_id,
                "values": values,
                "evidence": {name: "oracle" for name in values},
                "evidence_grounded": {name: True for name in values},
                "known_concepts": [name for name, value in values.items() if value != "unknown"],
                "unknown_concepts": [name for name, value in values.items() if value == "unknown"],
                "grounded_known": [name for name, value in values.items() if value != "unknown"],
                "narrative": case.narrative,
                "action": case.action,
                "outcome": case.outcome,
            }
        )
    return records


class AbductiveRuleInducer:
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

    def induce(
        self,
        concept_set: ConceptSet,
        records: list[dict[str, Any]],
        cases: list[LatentCase],
        batch_size: int = 8,
        min_precision: float = 0.8,
        workers: int = 8,
    ) -> RuleSet:
        by_id = {record["case_id"]: record for record in records}
        grounded_cases = []
        for case in cases:
            record = by_id.get(case.case_id)
            if record is None:
                continue
            grounded_cases.append(
                {
                    "case_id": case.case_id,
                    "values": {k: v for k, v in record["values"].items() if v != "unknown"},
                    "action": case.action,
                    "outcome": case.outcome,
                }
            )
        batches = [grounded_cases[i : i + batch_size] for i in range(0, len(grounded_cases), batch_size)]

        def run(batch: list[dict[str, Any]]) -> tuple[list[Rule], str | None]:
            try:
                payload, _ = self.client.chat_json(
                    [
                        {"role": "system", "content": RULE_INDUCTION_SYSTEM},
                        {"role": "user", "content": rule_induction_user(concept_set, batch)},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    reasoning_effort=self.effort,
                    phase="abduct_rules",
                )
                rules = []
                for item in payload.get("rules") or []:
                    try:
                        rules.append(Rule.from_dict(item, factor_values=concept_set.factor_values()))
                    except (ValueError, TypeError):
                        continue
                if rules:
                    return rules, None
                return self._fallback(batch), "empty rules array"
            except Exception as exc:  # noqa: BLE001
                return self._fallback(batch), str(exc)[:300]

        results = parallel_map(run, batches, workers=workers, label="abduct rules")
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

        rule_set = RuleSet(rules, factor_values=concept_set.factor_values())
        rule_cases = [to_rule_case(case, by_id[case.case_id]["values"]) for case in cases if case.case_id in by_id]
        rule_set.refine(rule_cases, min_precision=min_precision, max_depth=3)
        return rule_set

    @staticmethod
    def _fallback(batch: list[dict[str, Any]]) -> list[Rule]:
        rules = []
        for item in batch:
            rules.append(
                Rule(
                    condition=dict(item.get("values") or {}),
                    action=item["action"],
                    evidence_case_ids=[item["case_id"]],
                )
            )
        return rules


class AbductivePredictor:
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

    def predict(self, case: LatentCase, memory_text: str, method: str) -> Prediction:
        try:
            payload, response = self.client.chat_json(
                [
                    {"role": "system", "content": ABDUCT_PREDICT_SYSTEM},
                    {"role": "user", "content": abduct_predict_user(case, memory_text)},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                reasoning_effort=self.effort,
                phase=f"abduct_predict:{method}",
            )
            action = str(payload.get("action", "")).strip()
            valid = action in ACTIONS
            if not valid:
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

    def answer_qa(self, item: dict[str, Any], memory_text: str, method: str) -> dict[str, Any]:
        try:
            payload, response = self.client.chat_json(
                [
                    {"role": "system", "content": QA_SYSTEM},
                    {"role": "user", "content": qa_user(item, memory_text)},
                ],
                max_tokens=512,
                temperature=self.temperature,
                reasoning_effort=self.effort,
                phase=f"qa:{method}",
            )
            answer = str(payload.get("concept", "")).strip()
            valid = answer in item["options"]
            return {
                "pair_id": item["pair_id"],
                "method": method,
                "answer": answer,
                "valid": valid,
                "correct": valid and answer == item["answer_concept"],
                "usage": response.usage,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "pair_id": item["pair_id"],
                "method": method,
                "answer": "",
                "valid": False,
                "correct": False,
                "error": str(exc)[:300],
                "usage": {},
            }


class LatentSummarizer:
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

    def summarize(self, cases: list[LatentCase]) -> str:
        try:
            response = self.client.chat(
                [
                    {"role": "system", "content": SUMMARY_SYSTEM},
                    {"role": "user", "content": latent_summary_user(cases)},
                ],
                response_format=None,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                reasoning_effort=self.effort,
                phase="latent_summary",
            )
            text = response.content.strip()
            if text:
                return text
        except Exception as exc:  # noqa: BLE001
            print(f"  latent summary failed: {exc}", file=sys.stderr)
        return "No summary available."


# ---------------------------------------------------------------------------
# Retrieval and memory formatting
# ---------------------------------------------------------------------------
class LatentMemory:
    def __init__(self, train_cases: list[LatentCase], top_k: int = 3) -> None:
        self.train_cases = train_cases
        self.top_k = top_k
        self.index = BM25Index([case.narrative for case in train_cases])

    def raw_cases(self, case: LatentCase) -> list[LatentCase]:
        return [self.train_cases[i] for i, _ in self.index.top_k(case.narrative, k=self.top_k)]

    def raw_cases_from_text(self, text: str, k: int = 2) -> list[LatentCase]:
        return [self.train_cases[i] for i, _ in self.index.top_k(text, k=k)]

    def format_raw(self, cases: Iterable[LatentCase]) -> str:
        parts = []
        for case in cases:
            parts.append(
                f"[{case.case_id}] action={case.action}; outcome={case.outcome}\n{case.narrative}"
            )
        return "\n\n".join(parts) if parts else "None."


# ---------------------------------------------------------------------------
# RSI loop
# ---------------------------------------------------------------------------
@dataclass
class RSIteration:
    iteration: int
    concept_set: ConceptSet
    num_rules: int
    val_accuracy: float
    val_coverage: float
    val_score: float
    accepted: bool
    revised_from_errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "concepts": self.concept_set.to_dict(),
            "num_concepts": len(self.concept_set),
            "num_rules": self.num_rules,
            "val_accuracy": self.val_accuracy,
            "val_coverage": self.val_coverage,
            "val_score": self.val_score,
            "accepted": self.accepted,
            "revised_from_errors": self.revised_from_errors,
        }


class RSILoop:
    """Iteratively revise the abstraction representation using validation errors."""

    def __init__(
        self,
        proposer: ConceptProposer,
        grounder: ConceptGrounder,
        inducer: AbductiveRuleInducer,
        max_iterations: int = 2,
        workers: int = 8,
    ) -> None:
        self.proposer = proposer
        self.grounder = grounder
        self.inducer = inducer
        self.max_iterations = max_iterations
        self.workers = workers

    def evaluate_val(
        self,
        rule_set: RuleSet,
        val_records: list[dict[str, Any]],
        val_cases: list[LatentCase],
    ) -> tuple[float, float, list[dict[str, Any]]]:
        by_id = {record["case_id"]: record for record in val_records}
        correct = covered = 0
        errors: list[dict[str, Any]] = []
        for case in val_cases:
            record = by_id.get(case.case_id)
            values = record["values"] if record else {}
            action, matched, _ = rule_set.predict(values)
            covered += int(action is not None)
            correct += int(action == case.action)
            if action != case.action:
                errors.append(
                    {
                        "case_id": case.case_id,
                        "narrative": case.narrative,
                        "true_action": case.action,
                        "predicted_action": action,
                        "matched_rules": format_rules(matched[:5]),
                        "concept_values": values,
                    }
                )
        n = len(val_cases)
        accuracy = correct / n if n else 0.0
        coverage = covered / n if n else 0.0
        return accuracy, coverage, errors

    def run(
        self,
        seed_cases: list[LatentCase],
        rule_train_cases: list[LatentCase],
        val_cases: list[LatentCase],
    ) -> tuple[ConceptSet, RuleSet, list[RSIteration], dict[str, Any]]:
        concept_set, raw_proposal = self.proposer.propose(seed_cases)
        best_concepts = concept_set
        best_rules: RuleSet | None = None
        best_score = -1.0
        best_val_accuracy = 0.0
        best_val_coverage = 0.0
        history: list[RSIteration] = []

        for iteration in range(self.max_iterations + 1):
            train_records = self.grounder.ground(
                concept_set, rule_train_cases, workers=self.workers
            )
            val_records = self.grounder.ground(concept_set, val_cases, workers=self.workers)
            rule_set = self.inducer.induce(concept_set, train_records, rule_train_cases)
            val_accuracy, val_coverage, errors = self.evaluate_val(rule_set, val_records, val_cases)
            score = val_accuracy + 0.05 * val_coverage
            accepted = score > best_score
            history.append(
                RSIteration(
                    iteration=iteration,
                    concept_set=concept_set,
                    num_rules=len(rule_set),
                    val_accuracy=val_accuracy,
                    val_coverage=val_coverage,
                    val_score=score,
                    accepted=accepted,
                    revised_from_errors=len(errors),
                )
            )
            if accepted:
                best_concepts = concept_set
                best_rules = rule_set
                best_score = score
                best_val_accuracy = val_accuracy
                best_val_coverage = val_coverage

            if iteration >= self.max_iterations:
                break
            if not errors:
                break
            revised, _ = self.proposer.revise(concept_set, errors[:8])
            if len(revised) == 0:
                break
            concept_set = revised

        assert best_rules is not None
        summary = {
            "iterations": len(history),
            "best_val_accuracy": best_val_accuracy,
            "best_val_coverage": best_val_coverage,
            "best_score": best_score,
            "raw_proposal": raw_proposal,
        }
        return best_concepts, best_rules, history, summary
