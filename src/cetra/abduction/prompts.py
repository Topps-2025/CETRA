"""Prompts for abductive memory abstraction.

The prompts deliberately separate three questions:

1. What latent concepts could explain the observed actions?  (abduction)
2. Which concept values are grounded in each narrative?        (grounding)
3. What rules over the grounded concepts predict the actions?  (abstraction)
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from .concepts import ConceptSet
from .latent_cases import ACTION_DESCRIPTIONS, ACTIONS, LatentCase

ACTION_BLOCK = "\n".join(f"- {action}: {ACTION_DESCRIPTIONS[action]}" for action in ACTIONS)


def _concept_block(concept_set: ConceptSet) -> str:
    return json.dumps(concept_set.compact(), ensure_ascii=False, indent=2)


def _case_for_proposal(case: LatentCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "narrative": case.narrative,
        "action": case.action,
        "outcome": case.outcome,
    }


def _case_for_grounding(case: LatentCase) -> dict[str, Any]:
    # Deliberately no action: grounding must come from the narrative.
    return {"case_id": case.case_id, "narrative": case.narrative}


CONCEPT_PROPOSAL_SYSTEM = f"""You are analyzing a memory of interaction cases. Each case
contains a narrative, the action taken, and the outcome. The reasons for the
actions are latent and are not labeled. Your job is abductive: propose a small
set of latent concepts that best explain why different actions were taken.

Use world knowledge to interpret observable phrases. Do not use the action itself
as an indicator. Do not propose surface attributes such as service, user, ticket
id, error code, or timestamp as concepts. A good concept:
- has mutually exclusive values, as fine-grained as the behavior requires;
- if three situations lead to three different actions, do not collapse them
  into a binary distinction; if a binary distinction is enough, prefer it;
- is supported by observable indicators in the narratives;
- would change the predicted action if its value changed.

Keep the answer compact: at most 6 concepts, at most 3 indicators per value,
definitions under 25 words. Do not include evidence_case_ids.

Allowed actions:
{ACTION_BLOCK}

Return JSON:
{{"concepts": [
  {{
    "name": "short_snake_case_name",
    "definition": "what the concept means",
    "values": ["value_a", "value_b"],
    "indicators": {{"value_a": ["observable phrase"], "value_b": ["observable phrase"]}}
  }}
]}}
No markdown fences."""


def concept_proposal_user(cases: Iterable[LatentCase]) -> str:
    payload = [_case_for_proposal(case) for case in cases]
    return f"""Memory sample:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Propose the latent concepts now."""


CONCEPT_GROUNDING_SYSTEM = """You are grounding proposed latent concepts to case
narratives. For each case and each concept, choose exactly one of the concept's
values, or "unknown" if the narrative gives no evidence. Use only the narrative;
do not infer from what action you think was taken. Quote a short evidence span
from the narrative for every non-unknown value.

Return JSON:
{"records": [
  {
    "case_id": "case id",
    "values": {"concept_name": "value or unknown"},
    "evidence": {"concept_name": "short quote from the narrative"}
  }
]}
No markdown fences."""


def concept_grounding_user(concept_set: ConceptSet, cases: Iterable[LatentCase]) -> str:
    payload = [_case_for_grounding(case) for case in cases]
    return f"""Latent concepts:
{_concept_block(concept_set)}

Cases:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Ground the concepts for every case now."""


CONCEPT_REVISION_SYSTEM = """You previously proposed latent concepts to explain a
memory of cases. The induced rules made errors on a validation set. Revise the
concept set abductively: add missing latent distinctions, remove concepts that
are ungrounded or irrelevant, split or merge concepts, or refine values and
indicators. In particular, if errors show that cases assigned to the same value
require different actions, split that value into finer values. Focus on
distinctions that explain the errors, not on memorizing individual cases. Keep
the same JSON schema and return the full revised concept set. No markdown
fences."""


def concept_revision_user(
    concept_set: ConceptSet,
    errors: list[dict[str, Any]],
) -> str:
    return f"""Current concepts:
{_concept_block(concept_set)}

Validation errors (the true action is shown only for revision, never for test):
{json.dumps(errors, ensure_ascii=False, indent=2)}

Return the full revised concept set as JSON:
{{"concepts": [...]}}"""


RULE_INDUCTION_SYSTEM = """You are inducing reusable IF-THEN rules over a set of
latent concepts. Conditions may use only the concept names and values in the
provided vocabulary. Omit a concept to mean "any value". Do not use surface
details. Prefer the smallest set of rules that explains the observed actions and
covers as many cases as possible.

Return JSON:
{"rules": [
  {
    "condition": {"concept_name": "value"},
    "action": "one allowed action",
    "effect": "expected effect",
    "scope": "when the rule applies",
    "evidence_case_ids": ["case id"],
    "confidence": 0.0
  }
]}
No markdown fences."""


def rule_induction_user(
    concept_set: ConceptSet,
    grounded_cases: list[dict[str, Any]],
) -> str:
    compact = [
        {
            "case_id": item["case_id"],
            "concepts": item["values"],
            "action": item["action"],
            "outcome": item.get("outcome", "success"),
        }
        for item in grounded_cases
    ]
    return f"""Concept vocabulary:
{_concept_block(concept_set)}

Allowed actions:
{ACTION_BLOCK}

Grounded cases:
{json.dumps(compact, ensure_ascii=False, indent=2)}

Induce the rules now."""


ABDUCT_PREDICT_SYSTEM = f"""You are an expert operations agent. Choose exactly one
action for the new case from the allowed list. Use the retrieved memory when it
is relevant. If matching rules are provided, follow the most specific matching
rule unless the case clearly contradicts its conditions.

Allowed actions:
{ACTION_BLOCK}

Return JSON: {{"action": "one allowed action", "reason": "brief justification"}}.
No markdown fences."""


def abduct_predict_user(case: LatentCase, memory_text: str) -> str:
    return f"""New case:
{case.narrative}

Retrieved memory:
{memory_text}

Choose the best action and return JSON."""


QA_SYSTEM = """You are answering a contrastive question about two cases. The cases
differ in one latent concept, which is why their actions differ. Choose the
latent concept that best explains the difference from the options. Return JSON:
{"concept": "one option"}.
No markdown fences."""


def qa_user(item: dict[str, Any], memory_text: str) -> str:
    return f"""Case A:
{item['narratives'][0]}

Case B:
{item['narratives'][1]}

Action A: {item['actions'][0]}
Action B: {item['actions'][1]}

Options: {json.dumps(item['options'], ensure_ascii=False)}

Retrieved memory:
{memory_text}

Which latent concept best explains the difference? Return JSON."""


CONCEPT_JUDGE_SYSTEM = """You are comparing a set of proposed latent concepts to a
set of hidden reference concepts. For each hidden concept, find the proposed
concept that best matches its meaning, or "none". Two concepts match if they
would assign the same distinctions to the cases, even if their names differ.
Return JSON:
{"matches": {"hidden_name": "proposed_name_or_none"},
 "precision_matches": ["proposed_name", "..."],
 "notes": "brief"}
No markdown fences."""


def concept_judge_user(concept_set: ConceptSet, hidden_concepts: dict[str, list[str]]) -> str:
    hidden_block = json.dumps(
        [
            {"name": name, "values": values, "definition": f"hidden concept {name}"}
            for name, values in hidden_concepts.items()
        ],
        ensure_ascii=False,
        indent=2,
    )
    return f"""Hidden reference concepts:
{hidden_block}

Proposed concepts:
{_concept_block(concept_set)}

Map each hidden concept to the best proposed concept or "none". Also list the
proposed concepts that genuinely match a hidden concept (for precision)."""


SUMMARY_SYSTEM = """You are writing a concise operations memo from a set of
interaction cases. The memo will be shown to an agent that must choose an action
for new cases. State the conditions under which each action is appropriate. Do
not mention case ids, service names, user names, request ids, error codes, or
timestamps. Keep the memo under 350 words. Return plain text, not JSON."""


def latent_summary_user(cases: Iterable[LatentCase]) -> str:
    payload = [
        {
            "narrative": case.narrative,
            "action": case.action,
            "outcome": case.outcome,
        }
        for case in cases
    ]
    return f"""Interaction cases:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Write the operations memo now."""
