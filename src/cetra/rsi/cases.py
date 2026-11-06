"""Concrete cases for the CETRA-RSI demo.

We reuse the controlled latent-case generator from the abduction experiment.
Each case is a realistic tool-failure ticket with five hidden factors and one
expert action.  The generator gives us three held-out families:

- ``iid``: same mechanism, new surface details;
- ``surface``: new surface details and different combinations;
- ``compositional``: unseen factor combinations.

The demo uses these as the search/validation/test families for metaproductivity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cetra.abduction.latent_cases import LatentCase, generate_latent_dataset

from .types import CaseRecord


@dataclass
class Dataset:
    seed: list[CaseRecord]
    induction: list[CaseRecord]
    val: list[CaseRecord]
    iid: list[CaseRecord]
    surface: list[CaseRecord]
    compositional: list[CaseRecord]
    counterfactual: list[CaseRecord]
    all_cases: list[CaseRecord] = field(default_factory=list)

    @property
    def search(self) -> list[CaseRecord]:
        """The seed cases used to bootstrap the representation."""

        return self.seed

    @property
    def fit(self) -> list[CaseRecord]:
        """Cases available to the inner refinement after a diagnosis."""

        return self.seed + self.val + self.induction

    def heldout_families(self) -> dict[str, list[CaseRecord]]:
        return {
            "iid": self.iid,
            "surface": self.surface,
            "compositional": self.compositional,
        }


def _to_record(case: LatentCase) -> CaseRecord:
    return CaseRecord(
        case_id=case.case_id,
        split=case.split,
        narrative=case.narrative,
        factors=dict(case.factors),
        action=case.action,
        family=case.split,
    )


def load_latent_dataset(seed: int = 20261026, val_size: int = 12) -> Dataset:
    payload = generate_latent_dataset(seed=seed, val_size=val_size)
    cases = [LatentCase.from_dict(item) for item in payload["cases"]]
    records = [_to_record(case) for case in cases]
    by_id = {record.case_id: record for record in records}
    val_ids = list(payload["meta"]["val_case_ids"])

    train_records = [record for record in records if record.split == "train"]
    val = [by_id[case_id] for case_id in val_ids if case_id in by_id]
    seed_ids = set(payload["meta"]["seed_case_ids"])
    seed = [by_id[case_id] for case_id in seed_ids if case_id in by_id]
    val_id_set = {record.case_id for record in val}
    induction = [
        record
        for record in train_records
        if record.case_id not in seed_ids and record.case_id not in val_id_set
    ]
    iid = [record for record in records if record.split == "iid"]
    surface = [record for record in records if record.split == "surface"]
    compositional = [record for record in records if record.split == "compositional"]
    counterfactual = [record for record in records if record.split == "counterfactual"]
    return Dataset(
        seed=seed,
        induction=induction,
        val=val,
        iid=iid,
        surface=surface,
        compositional=compositional,
        counterfactual=counterfactual,
        all_cases=records,
    )


def select_demo_cases(dataset: Dataset, per_family: int = 2) -> list[CaseRecord]:
    """Select a small, diverse set of concrete cases for the report."""

    selected: list[CaseRecord] = []
    for family in (dataset.val, dataset.iid, dataset.surface, dataset.compositional):
        # Prefer cases that exercise the failed/irreversible interaction.
        preferred = [
            case
            for case in family
            if case.factors.get("commit_state") == "failed"
            and case.factors.get("reversibility") == "irreversible"
        ]
        pool = preferred + [case for case in family if case not in preferred]
        selected.extend(pool[:per_family])
    seen: set[str] = set()
    unique: list[CaseRecord] = []
    for case in selected:
        if case.case_id in seen:
            continue
        seen.add(case.case_id)
        unique.append(case)
    return unique


def write_cases(path: str | Path, cases: list[CaseRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "case_id": case.case_id,
            "family": case.family,
            "narrative": case.narrative,
            "factors": case.factors,
            "action": case.action,
        }
        for case in cases
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dataset_summary(dataset: Dataset) -> dict[str, Any]:
    return {
        "seed": len(dataset.seed),
        "induction": len(dataset.induction),
        "val": len(dataset.val),
        "iid": len(dataset.iid),
        "surface": len(dataset.surface),
        "compositional": len(dataset.compositional),
        "counterfactual": len(dataset.counterfactual),
    }
