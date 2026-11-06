"""Small helpers shared by the proposer, gate, and loop."""

from __future__ import annotations

from collections import Counter

from .types import Diagnosis


def error_kinds(errors: list[Diagnosis]) -> set[str]:
    return {error.kind for error in errors}


def error_counts(errors: list[Diagnosis]) -> dict[str, int]:
    return dict(Counter(error.kind for error in errors))


def resolved_fraction(
    parent_errors: list[Diagnosis],
    child_errors: list[Diagnosis],
) -> float:
    """Fraction of the parent's failing cases that the child resolves."""

    parent_ids = {error.case_id for error in parent_errors}
    if not parent_ids:
        return 0.0
    child_ids = {error.case_id for error in child_errors}
    return len(parent_ids - child_ids) / len(parent_ids)
