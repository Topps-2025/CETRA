"""Prompt templates for the memory decomposition / abstraction experiment."""

from __future__ import annotations

import json
from typing import Any, Iterable

from .cases import ACTION_DESCRIPTIONS, ACTIONS, FACTOR_VALUES, Case
from .rules import Rule


def _factor_vocabulary() -> str:
    return json.dumps(FACTOR_VALUES, ensure_ascii=False)


def _action_vocabulary() -> str:
    lines = [f"- {action}: {ACTION_DESCRIPTIONS[action]}" for action in ACTIONS]
    return "\n".join(lines)


DECOMPOSE_SYSTEM = f"""You are a careful analyst of expert decision cases.
Decompose each case into a decision point, the factors that were present, the
action taken, and the observed effect. Separate decision-relevant conditions
from surface details.

Allowed factor names and values:
{_factor_vocabulary()}

Allowed actions:
{_action_vocabulary()}

Return a single JSON object. Do not use markdown fences. Never invent a factor
name or value outside the allowed vocabulary."""


def decompose_user(case: Case) -> str:
    payload = {
        "case_id": case.case_id,
        "narrative": case.narrative,
        "factors": case.factors,
        "expert_action": case.action,
        "outcome": case.outcome,
    }
    return f"""Case (the expert action is included for analysis):
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return JSON with exactly these keys:
{{
  "case_id": "{case.case_id}",
  "decision_point": "short description of the moment at which the action is chosen",
  "observed_factors": {{"factor": "value"}},
  "candidate_conditions": {{"factor": "value"}},
  "action": "one allowed action",
  "effect": "observed effect of the action",
  "surface_details": ["details that should not enter a reusable rule"],
  "evidence": "short quote or paraphrase from the case"
}}

`candidate_conditions` should be the subset of observed factors that you believe
explains why this action was chosen in this case. Omit factors that look like
surface details or that do not change the action."""


INDUCE_RULES_SYSTEM = f"""You are inducing reusable IF-THEN rules from decomposed
expert cases. A rule must be predictive across cases, not a restatement of a
single case. Prefer the smallest set of rules that explains the expert actions.

Allowed factor names and values:
{_factor_vocabulary()}

Allowed actions:
{_action_vocabulary()}

Rules:
- Conditions may use only the allowed factor names and values.
- Omit a factor to mean "any value".
- Do not use service names, user names, request ids, error codes, timestamps,
  or other surface details in conditions.
- Every rule must have one allowed action.
- Cover as many cases as possible while keeping the rule consistent.

Return one JSON object with a `rules` array. Do not use markdown fences."""


def induce_rules_user(decomposed_cases: Iterable[dict[str, Any]]) -> str:
    cases = list(decomposed_cases)
    compact = [
        {
            "case_id": item.get("case_id"),
            "candidate_conditions": item.get("candidate_conditions", {}),
            "action": item.get("action"),
            "effect": item.get("effect"),
            "evidence": item.get("evidence", ""),
        }
        for item in cases
    ]
    return f"""Decomposed training cases:
{json.dumps(compact, ensure_ascii=False, indent=2)}

Return JSON:
{{
  "rules": [
    {{
      "condition": {{"factor": "value"}},
      "action": "one allowed action",
      "effect": "expected effect",
      "scope": "when this rule applies",
      "evidence_case_ids": ["case id", "..."],
      "confidence": 0.0
    }}
  ]
}}

The rules should collectively explain the actions in the cases above. If two
cases share an action but differ in a factor that does not matter, omit that
factor from the condition."""


DIRECT_RULES_SYSTEM = f"""You are inducing reusable IF-THEN rules directly from raw
expert cases, without an explicit decomposition step. Prefer the smallest set of
predictive rules.

Allowed factor names and values:
{_factor_vocabulary()}

Allowed actions:
{_action_vocabulary()}

Rules:
- Conditions may use only the allowed factor names and values.
- Omit a factor to mean "any value".
- Do not use service names, user names, request ids, error codes, timestamps, or
  other surface details in conditions.
- Cover as many cases as possible while keeping the rule consistent.

Return one JSON object with a `rules` array. Do not use markdown fences."""


def direct_rules_user(cases: Iterable[Case]) -> str:
    compact = [
        {
            "case_id": case.case_id,
            "narrative": case.narrative,
            "factors": case.factors,
            "action": case.action,
            "outcome": case.outcome,
        }
        for case in cases
    ]
    return f"""Raw training cases:
{json.dumps(compact, ensure_ascii=False, indent=2)}

Return JSON:
{{
  "rules": [
    {{
      "condition": {{"factor": "value"}},
      "action": "one allowed action",
      "effect": "expected effect",
      "scope": "when this rule applies",
      "evidence_case_ids": ["case id", "..."],
      "confidence": 0.0
    }}
  ]
}}"""


SUMMARY_SYSTEM = """You are writing a concise operations memo from a set of expert
cases. The memo will be shown to an agent that must choose an action for new
cases. State the conditions under which each action is appropriate. Do not
mention case ids, service names, user names, request ids, error codes, or
timestamps. Keep the memo under 300 words. Return plain text, not JSON."""


def summary_user(cases: Iterable[Case]) -> str:
    compact = [
        {
            "narrative": case.narrative,
            "factors": case.factors,
            "action": case.action,
            "outcome": case.outcome,
        }
        for case in cases
    ]
    return f"""Expert cases:
{json.dumps(compact, ensure_ascii=False, indent=2)}

Write the operations memo now."""


PREDICT_SYSTEM = f"""You are an expert operations agent. Choose exactly one action
for the new case from the allowed list.

The retrieved memory contains rules that were validated on expert cases. If one
or more rules match the new case, follow the action of the most specific
matching rule. Do not override a matching rule with general prior knowledge.
Override a rule only when its condition clearly does not apply to the case.
The structured factors in the case are authoritative.

Allowed actions:
{_action_vocabulary()}

Return a single JSON object with keys `action` and `reason`. `action` must be
exactly one of the allowed action names. Do not use markdown fences."""


def predict_user(case: Case, memory_text: str) -> str:
    return f"""New case:
{case.narrative}

Structured factors:
{json.dumps(case.factors, ensure_ascii=False)}

Retrieved memory:
{memory_text}

Choose the best action. Return JSON:
{{"action": "one allowed action", "reason": "brief justification"}}"""


def format_raw_case(case: Case, include_action: bool = True) -> str:
    action_line = f"Expert action: {case.action}\n" if include_action else ""
    return (
        f"[{case.case_id}] factors={json.dumps(case.factors, ensure_ascii=False)}; "
        f"outcome={case.outcome}\n{action_line}{case.narrative}"
    )


def format_raw_cases(cases: Iterable[Case]) -> str:
    return "\n\n".join(format_raw_case(case, include_action=True) for case in cases)


def format_decomposed(item: dict[str, Any]) -> str:
    return (
        f"[{item.get('case_id')}] conditions="
        f"{json.dumps(item.get('candidate_conditions', {}), ensure_ascii=False)}; "
        f"action={item.get('action')}; effect={item.get('effect', '')}"
    )


def format_decomposed_cases(items: Iterable[dict[str, Any]]) -> str:
    return "\n".join(format_decomposed(item) for item in items)


def format_rule(rule: Rule, index: int | None = None) -> str:
    prefix = f"R{index}. " if index is not None else ""
    condition = ", ".join(f"{k}={v}" for k, v in rule.condition.items()) or "any case"
    stats = []
    if rule.precision is not None:
        stats.append(f"precision={rule.precision:.2f}")
    if rule.coverage is not None:
        stats.append(f"coverage={rule.coverage}")
    suffix = f" ({'; '.join(stats)})" if stats else ""
    return f"{prefix}IF {condition} THEN {rule.action}{suffix}"


def format_rules(rules: Iterable[Rule]) -> str:
    rules = list(rules)
    if not rules:
        return "No rule matched this case."
    return "\n".join(format_rule(rule, i) for i, rule in enumerate(rules, start=1))
