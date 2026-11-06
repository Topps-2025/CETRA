"""Tests for the CETRA-RSI harness, inner CEGAR, and triple gate."""

from __future__ import annotations

from cetra.rsi import (
    CETRARSILoop,
    Edit,
    accept_edit,
    apply_edit_bundle,
    diagnose,
    inner_refine,
    load_latent_dataset,
    make_compose_harness,
    make_seed_harness,
)
from cetra.rsi.harness import risk


def test_seed_harness_is_harness_limited() -> None:
    ds = load_latent_dataset(val_size=24)
    seed = make_seed_harness()
    rep = seed.build(ds.seed)
    assert "reversibility" not in rep.concepts

    refined, history = inner_refine(seed, ds.seed, ds.val, fit_cases=ds.fit)
    assert "reversibility" in refined.concepts
    assert any(item.get("action") == "ADD_CONCEPT" for item in history)
    # Adding the concept is risk-neutral inside the fixed harness: the
    # single-condition learner still cannot express the interaction.
    assert risk(seed, refined, ds.val) <= risk(seed, rep, ds.val) + 1e-9


def test_compose_scope_fixes_the_interaction() -> None:
    ds = load_latent_dataset(val_size=24)
    seed = make_seed_harness()
    seed_rep, _ = inner_refine(seed, ds.seed, ds.val, fit_cases=ds.fit)
    seed_risk = risk(seed, seed_rep, ds.val)

    child = make_compose_harness()
    child_rep, _ = inner_refine(child, ds.seed, ds.val, fit_cases=ds.fit)
    child_risk = risk(child, child_rep, ds.val)

    assert child_risk < seed_risk
    assert any(
        ("commit_state", "failed") in rule.conditions
        and ("reversibility", "irreversible") in rule.conditions
        for rule in child_rep.rules
    )


def test_add_operator_and_rewire() -> None:
    seed = make_seed_harness()
    child = apply_edit_bundle(
        seed,
        [
            Edit("ADD_OPERATOR", "compose_scope", {}),
            Edit("REWIRE", "compose_scope", {"before": "predict"}),
        ],
        "h_test",
    )
    assert "compose_scope" in child.operators
    assert child.pipeline.index("compose_scope") < child.pipeline.index("predict")


def test_missing_operator_edit_is_rejected() -> None:
    seed = make_seed_harness()
    try:
        apply_edit_bundle(
            seed,
            [Edit("REMOVE_OPERATOR", "compose_scope", {})],
            "h_invalid",
        )
    except ValueError as exc:
        assert "missing operator" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("removing a missing operator should fail")


def test_gate_rejects_negative_control() -> None:
    ds = load_latent_dataset(val_size=24)
    loop = CETRARSILoop(ds)
    report = loop.run(generations=1)
    best = max(report.archive, key=lambda entry: entry.mp)

    bundle = [Edit("REPLACE_OPERATOR", "induce_single", {"with": "noop_induce"})]
    if "compose_scope" in best.harness.operators:
        bundle.insert(0, Edit("REMOVE_OPERATOR", "compose_scope", {}))
    bad = apply_edit_bundle(best.harness, bundle, "h_bad")
    bad_rep, _ = inner_refine(bad, ds.seed, ds.val, fit_cases=ds.fit)
    bad_metrics = loop.metrics_for(bad, bad_rep)
    parent_errors = diagnose(best.harness, best.representation, ds.val)
    child_errors = diagnose(bad, bad_rep, ds.val)
    decision = accept_edit(
        best.metrics,
        bad_metrics,
        parent_errors,
        child_errors,
        gate_split="iid",
    )
    assert not decision.accepted
