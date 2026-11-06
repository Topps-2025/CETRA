"""Tests for the controlled case generator and ground-truth policy."""

from __future__ import annotations

from collections import Counter

from cetra.cases import (
    ACTIONS,
    Case,
    all_combinations,
    generate_dataset,
    ground_truth_action,
    split_cases,
)


def test_ground_truth_policy_examples() -> None:
    assert ground_truth_action(
        {
            "permission": "none",
            "submission_status": "committed",
            "idempotent": "yes",
            "resource_state": "available",
            "retryable": "yes",
        }
    ) == "request_permission"
    assert ground_truth_action(
        {
            "permission": "owner",
            "submission_status": "committed",
            "idempotent": "yes",
            "resource_state": "available",
            "retryable": "yes",
        }
    ) == "retry"
    assert ground_truth_action(
        {
            "permission": "delegate",
            "submission_status": "committed",
            "idempotent": "no",
            "resource_state": "available",
            "retryable": "yes",
        }
    ) == "cancel_and_resubmit"
    assert ground_truth_action(
        {
            "permission": "owner",
            "submission_status": "pending",
            "idempotent": "yes",
            "resource_state": "locked",
            "retryable": "yes",
        }
    ) == "wait"
    assert ground_truth_action(
        {
            "permission": "owner",
            "submission_status": "failed",
            "idempotent": "yes",
            "resource_state": "available",
            "retryable": "yes",
        }
    ) == "retry"


def test_dataset_sizes_and_splits() -> None:
    payload = generate_dataset()
    cases = [Case.from_dict(item) for item in payload["cases"]]
    assert len(cases) == 99
    assert len(split_cases(cases, "train")) == 48
    assert len(split_cases(cases, "iid")) == 12
    assert len(split_cases(cases, "surface")) == 9
    assert len(split_cases(cases, "compositional")) == 12
    assert len(split_cases(cases, "counterfactual")) == 18


def test_train_covers_all_actions_and_factor_values() -> None:
    payload = generate_dataset()
    cases = [Case.from_dict(item) for item in payload["cases"]]
    train = split_cases(cases, "train")
    assert {case.action for case in train} == set(ACTIONS)
    for factor, values in payload["meta"]["factor_values"].items():
        seen = {case.factors[factor] for case in train}
        assert seen == set(values), (factor, seen)


def test_compositional_combinations_are_unseen() -> None:
    payload = generate_dataset()
    cases = [Case.from_dict(item) for item in payload["cases"]]
    train_combos = {tuple(sorted(case.factors.items())) for case in split_cases(cases, "train")}
    compositional = split_cases(cases, "compositional")
    assert all(tuple(sorted(case.factors.items())) not in train_combos for case in compositional)


def test_counterfactual_pairs_change_exactly_one_factor() -> None:
    payload = generate_dataset()
    cases = [Case.from_dict(item) for item in payload["cases"]]
    cf = split_cases(cases, "counterfactual")
    by_pair: dict[str, list[Case]] = {}
    for case in cf:
        by_pair.setdefault(case.pair_id or "", []).append(case)
    assert len(by_pair) == 9
    for pair in by_pair.values():
        assert len(pair) == 2
        base, perturbed = sorted(pair, key=lambda c: 0 if c.pair_role == "base" else 1)
        diffs = [key for key in base.factors if base.factors[key] != perturbed.factors[key]]
        assert len(diffs) == 1
        assert diffs[0] == base.changed_factor


def test_action_distribution_is_non_degenerate() -> None:
    payload = generate_dataset()
    cases = [Case.from_dict(item) for item in payload["cases"]]
    counts = Counter(case.action for case in cases)
    assert len(counts) == len(ACTIONS)
    assert min(counts.values()) >= 3
