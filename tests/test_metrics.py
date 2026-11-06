"""Tests for accuracy, bootstrap CI, and McNemar exact test."""

from __future__ import annotations

from cetra.metrics import bootstrap_ci, is_correct, mcnemar_exact
from cetra.pipeline import Prediction


def make_prediction(predicted: str | None, truth: str, valid: bool = True) -> Prediction:
    return Prediction(
        case_id="c1",
        method="test",
        predicted_action=predicted,
        ground_truth=truth,
        valid=valid,
    )


def test_is_correct() -> None:
    assert is_correct(make_prediction("retry", "retry"))
    assert not is_correct(make_prediction("wait", "retry"))
    assert not is_correct(make_prediction(None, "retry", valid=False))


def test_bootstrap_ci_contains_mean() -> None:
    values = [True, True, False, True, False, True, True, False]
    low, high = bootstrap_ci(values, n_boot=500, seed=1)
    mean = sum(values) / len(values)
    assert low <= mean <= high
    assert 0.0 <= low <= high <= 1.0


def test_mcnemar_exact() -> None:
    # A wins 10 discordant pairs, B wins 0: strongly significant.
    result = mcnemar_exact([True] * 10, [False] * 10)
    assert result["b"] == 10
    assert result["c"] == 0
    assert result["p_value"] < 0.01
    # No discordant pairs: p = 1.
    result = mcnemar_exact([True, False], [True, False])
    assert result["b"] == 0
    assert result["c"] == 0
    assert result["p_value"] == 1.0
