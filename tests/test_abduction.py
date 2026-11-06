"""Tests for the abductive memory-abstraction components."""

from __future__ import annotations

from cetra.abduction.concepts import Concept, ConceptSet
from cetra.abduction.latent_cases import (
    LatentCase,
    cases_by_ids,
    generate_latent_dataset,
)
from cetra.abduction.metrics import value_alignment
from cetra.abduction.tree import build_tree, predict_tree
from cetra.rules import Rule, RuleSet


def test_latent_dataset_structure() -> None:
    payload = generate_latent_dataset()
    meta = payload["meta"]
    assert meta["splits"]["train"] == 48
    assert meta["splits"]["counterfactual"] == 36
    assert len(meta["seed_case_ids"]) == 12
    assert len(meta["val_case_ids"]) == 12
    assert len(meta["qa"]) == 12


def test_seed_is_biased_but_validation_has_failed_and_irreversible() -> None:
    payload = generate_latent_dataset()
    cases = [LatentCase.from_dict(item) for item in payload["cases"]]
    seed = cases_by_ids(cases, payload["meta"]["seed_case_ids"])
    val = cases_by_ids(cases, payload["meta"]["val_case_ids"])
    assert all(case.factors["commit_state"] != "failed" for case in seed)
    assert all(case.factors["reversibility"] == "reversible" for case in seed)
    assert any(case.factors["commit_state"] == "failed" for case in val)
    assert any(case.factors["reversibility"] == "irreversible" for case in val)


def test_oracle_tree_is_strong() -> None:
    payload = generate_latent_dataset()
    cases = [LatentCase.from_dict(item) for item in payload["cases"]]
    train = cases_by_ids(cases, payload["meta"]["train_case_ids"])
    test = [case for case in cases if case.split != "train"]
    rows = [{"values": case.factors, "action": case.action} for case in train]
    tree = build_tree(rows, list(payload["meta"]["latent_values"]), max_depth=6, min_samples=2)
    correct = sum(
        predict_tree(tree, case.factors) == case.action for case in test
    )
    assert correct / len(test) > 0.85


def test_value_alignment_detects_owner_delegate_collapse() -> None:
    concept_set = ConceptSet(
        [Concept("auth", "binary auth", ["authenticated", "unauthenticated"])]
    )
    cases = []
    records = []
    for index, authority in enumerate(("owner", "delegate", "none")):
        case = LatentCase(
            case_id=f"c{index}",
            split="test",
            factors={
                "authority": authority,
                "commit_state": "failed",
                "retry_safety": "safe",
                "resource_state": "free",
                "reversibility": "irreversible",
            },
            surface={},
            action="rollback",
            outcome="success",
            narrative="",
        )
        cases.append(case)
        records.append(
            {
                "case_id": case.case_id,
                "values": {
                    "auth": "unauthenticated" if authority == "none" else "authenticated"
                },
            }
        )
    alignment = value_alignment(concept_set, records, cases, {"authority": "auth"})
    assert alignment["avg_purity"] == 1.0
    assert alignment["avg_nmi"] < 1.0  # owner and delegate are collapsed


def test_dynamic_rule_vocabulary() -> None:
    rule_set = RuleSet(
        [Rule({"foo": "bar"}, "retry")],
        factor_values={"foo": ("bar", "baz")},
    )
    assert rule_set.predict({"foo": "bar"})[0] == "retry"
    assert rule_set.predict({"foo": "baz"})[0] is None
