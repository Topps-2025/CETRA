"""Tests for rule matching, priority learning, refinement, and symbolic metrics."""

from __future__ import annotations

from cetra.cases import Case, all_combinations, generate_dataset, split_cases
from cetra.rules import Rule, RuleSet


def oracle_rules() -> RuleSet:
    rules: list[Rule] = [Rule({"permission": "none"}, "request_permission")]
    for permission in ("owner", "delegate"):
        for idempotent in ("yes", "no"):
            rules.append(
                Rule(
                    {
                        "permission": permission,
                        "submission_status": "committed",
                        "idempotent": idempotent,
                    },
                    "retry" if idempotent == "yes" else "cancel_and_resubmit",
                )
            )
        for resource in ("available", "locked", "exhausted"):
            rules.append(
                Rule(
                    {
                        "permission": permission,
                        "submission_status": "pending",
                        "resource_state": resource,
                    },
                    "wait" if resource == "locked" else "noop",
                )
            )
        rules.append(
            Rule(
                {
                    "permission": permission,
                    "submission_status": "failed",
                    "retryable": "no",
                },
                "rollback" if permission == "owner" else "escalate",
            )
        )
        for resource in ("available", "locked", "exhausted"):
            rules.append(
                Rule(
                    {
                        "permission": permission,
                        "submission_status": "failed",
                        "retryable": "yes",
                        "resource_state": resource,
                    },
                    "retry"
                    if resource == "available"
                    else "escalate"
                    if resource == "exhausted"
                    else "wait",
                )
            )
    return RuleSet(rules)


def test_oracle_rules_are_perfect() -> None:
    rule_set = oracle_rules()
    metrics = rule_set.evaluate_symbolic(all_combinations())
    assert metrics["coverage"] == 1.0
    assert metrics["accuracy_all"] == 1.0
    assert metrics["factor_sensitivity"]["f1"] == 1.0


def test_priority_override_beats_general_rule() -> None:
    payload = generate_dataset()
    cases = [Case.from_dict(item) for item in payload["cases"]]
    train = split_cases(cases, "train")
    rule_set = RuleSet(
        [
            Rule({"permission": "none"}, "request_permission"),
            Rule({"submission_status": "committed", "idempotent": "yes"}, "retry"),
        ]
    )
    rule_set.learn_priorities(train)
    factors = {
        "permission": "none",
        "submission_status": "committed",
        "idempotent": "yes",
        "resource_state": "available",
        "retryable": "yes",
    }
    assert rule_set.predict(factors)[0] == "request_permission"


def test_refine_removes_low_precision_rules() -> None:
    payload = generate_dataset()
    cases = [Case.from_dict(item) for item in payload["cases"]]
    train = split_cases(cases, "train")
    rule_set = RuleSet([Rule({"submission_status": "committed"}, "retry")])
    rule_set.refine(train, min_precision=0.8)
    assert all((rule.precision or 0.0) >= 0.8 for rule in rule_set)
    assert all((rule.coverage or 0) > 0 for rule in rule_set)


def test_oracle_counterfactual_consistency() -> None:
    payload = generate_dataset()
    cases = [Case.from_dict(item) for item in payload["cases"]]
    cf = split_cases(cases, "counterfactual")
    by_pair: dict[str, list[Case]] = {}
    for case in cf:
        by_pair.setdefault(case.pair_id or "", []).append(case)
    pairs = []
    for pair in by_pair.values():
        base, perturbed = sorted(pair, key=lambda c: 0 if c.pair_role == "base" else 1)
        pairs.append((base, perturbed))
    metrics = oracle_rules().evaluate_symbolic(all_combinations(), pairs)
    assert metrics["counterfactual"]["pair_consistency"] == 1.0


def test_rule_json_round_trip_preserves_statistics() -> None:
    rule = Rule(
        {"permission": "none"},
        "request_permission",
        precision=1.0,
        coverage=14,
        priority=14.3,
    )
    restored = Rule.from_dict(rule.to_dict())
    assert restored.condition == rule.condition
    assert restored.action == rule.action
    assert restored.precision == rule.precision
    assert restored.coverage == rule.coverage
    assert restored.priority == rule.priority
