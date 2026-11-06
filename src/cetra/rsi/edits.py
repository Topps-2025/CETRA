"""Typed harness edits and the deterministic edit proposer.

The proposer is intentionally simple: it maps process-level diagnoses to a
small set of typed edits.  It plays the role of the LLM proposer in the
framework; the interface is the same, so a model-backed proposer can be
plugged in without changing the loop.
"""

from __future__ import annotations

from typing import Any

from .diagnosis_utils import error_kinds
from .operators import make_operator
from .synthesis import ProgramOperator
from .types import Diagnosis, Edit, Harness


def apply_edit(harness: Harness, edit: Edit, version: str) -> Harness:
    """Apply one typed edit and return a child harness.

    Raises ``ValueError`` for edits that would make the harness unexecutable;
    the loop treats such children as invalid and records a rejection.
    """

    child = harness.clone(version=version)

    if edit.kind == "ADD_OPERATOR":
        name = edit.target
        if name in child.operators:
            raise ValueError(f"operator already exists: {name}")
        program = edit.args.get("program")
        if program:
            # A synthesized operator is carried as a program payload, not as a
            # name in the host registry.  This is the open-language path.
            child.operators[name] = ProgramOperator.from_dict(program)
        else:
            child.operators[name] = make_operator(name)
        # Adding an operator does not wire it in; REWIRE does.
    elif edit.kind == "REPLACE_OPERATOR":
        if edit.target not in child.operators:
            raise ValueError(f"cannot replace missing operator: {edit.target}")
        replacement = edit.args.get("with")
        if not replacement:
            raise ValueError("REPLACE_OPERATOR requires args['with']")
        child.operators[edit.target] = make_operator(replacement)
    elif edit.kind == "REWIRE":
        name = edit.target
        if name not in child.operators:
            raise ValueError(f"cannot wire missing operator: {name}")
        before = edit.args.get("before", "predict")
        if before not in child.pipeline:
            raise ValueError(f"unknown pipeline anchor: {before}")
        if name in child.pipeline:
            child.pipeline.remove(name)
        child.pipeline.insert(child.pipeline.index(before), name)
    elif edit.kind == "TUNE":
        key = edit.args.get("key")
        value = edit.args.get("value")
        if not key:
            raise ValueError("TUNE requires args['key']")
        if key in ("min_support", "min_precision", "max_conditions", "min_gain"):
            child.params[key] = value
        else:
            child.params.setdefault("operator_params", {})[key] = value
    elif edit.kind == "ADD_FALLBACK":
        child.params["fallback_action"] = edit.args.get("action")
    elif edit.kind == "REMOVE_OPERATOR":
        if edit.target not in child.operators:
            raise ValueError(f"cannot remove missing operator: {edit.target}")
        del child.operators[edit.target]
        if edit.target in child.pipeline:
            child.pipeline.remove(edit.target)
    else:
        raise ValueError(f"unknown edit kind: {edit.kind}")

    child.metadata = dict(child.metadata)
    child.metadata["last_edit"] = edit.to_dict()
    return child


def propose_edits(
    harness: Harness,
    errors: list[Diagnosis],
    *,
    generation: int,
) -> list[list[Edit]]:
    """Map diagnoses to candidate **edit bundles**.

    A bundle is applied atomically; this is necessary because adding an
    operator and wiring it into the pipeline are two edits that only make
    sense together.  The gate evaluates the resulting harness as one child.

    The first generation targets the missing operator.  Later generations tune
    it and, for demonstration, propose a deliberately bad bundle that the gate
    should reject.
    """

    kinds = error_kinds(errors)

    needs_composition = bool(
        {"operator_gap", "composition_error", "missing_rule"} & kinds
    )
    if needs_composition and "compose_scope" not in harness.operators:
        return [
            [
                Edit("ADD_OPERATOR", "compose_scope", {}),
                Edit("REWIRE", "compose_scope", {"before": "predict"}),
            ]
        ]

    if "composition_error" in kinds and "compose_scope" in harness.operators:
        return [
            [
                Edit("TUNE", "compose_scope", {"key": "max_conditions", "value": 3}),
                Edit("TUNE", "compose_scope", {"key": "min_gain", "value": 0.001}),
            ]
        ]

    if errors:
        return [[Edit("TUNE", "induce_single", {"key": "min_support", "value": 2})]]

    # No errors: a small parameter change, plus a deliberately bad bundle.
    bundles: list[list[Edit]] = [
        [Edit("TUNE", "induce_single", {"key": "min_support", "value": 2})]
    ]
    if generation >= 2:
        bundles.append(
            [
                Edit(
                    "REPLACE_OPERATOR",
                    "induce_single",
                    {"with": "noop_induce"},
                )
            ]
        )
    return bundles


def apply_edit_bundle(
    harness: Harness,
    bundle: list[Edit],
    version: str,
) -> Harness:
    """Apply a bundle atomically and return the resulting child."""

    child = harness
    for edit in bundle:
        child = apply_edit(child, edit, version)
    child.parent = harness.version
    child.metadata = dict(child.metadata)
    child.metadata["edit_bundle"] = [edit.to_dict() for edit in bundle]
    return child


def edit_summary(edit: Edit) -> dict[str, Any]:
    return edit.to_dict()
