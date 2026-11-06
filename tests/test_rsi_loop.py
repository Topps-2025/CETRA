"""End-to-end tests for the CETRA-RSI outer loop."""

from __future__ import annotations

from cetra.rsi import CETRARSILoop, load_latent_dataset


def test_loop_accepts_compose_scope_and_improves_mp() -> None:
    ds = load_latent_dataset(seed=20261026, val_size=24)
    loop = CETRARSILoop(ds)
    report = loop.run(generations=1)

    accepted = [record for record in report.edit_log if record.accepted]
    assert accepted, "the compose-scope edit should be accepted on the main seed"
    assert accepted[0].edits[0]["kind"] == "ADD_OPERATOR"
    assert accepted[0].edits[0]["target"] == "compose_scope"

    seed_test = report.seed_metrics["test"]
    best_test = report.best_metrics["test"]
    assert best_test.correct >= seed_test.correct
    assert best_test.mp > seed_test.mp


def test_loop_reports_rejected_edit() -> None:
    ds = load_latent_dataset(seed=20261026, val_size=24)
    loop = CETRARSILoop(ds)
    report = loop.run(generations=2)

    assert any(record.accepted for record in report.edit_log)
    assert any(not record.accepted for record in report.edit_log)
    # Rejected edits must carry a reason and the gate deltas.
    for record in report.edit_log:
        if not record.accepted:
            assert record.reason
            assert record.delta_mp <= 1e-9


def test_archive_keeps_seed_and_child() -> None:
    ds = load_latent_dataset(seed=20261026, val_size=24)
    loop = CETRARSILoop(ds)
    report = loop.run(generations=1)

    versions = [entry.harness.version for entry in report.archive]
    assert "h0_seed" in versions
    assert len(versions) >= 2
    assert report.best_version in versions
