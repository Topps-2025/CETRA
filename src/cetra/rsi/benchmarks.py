"""Controlled harness-limited task families for the full RSI demo.

Each family has a hidden decision policy.  The seed harness contains only
single-condition rule induction, so it can memorise surface values but cannot
express the interaction that the family is built around.  A family therefore
has a *known* missing operator:

- ``scope``: conjunction of categorical conditions;
- ``threshold``: a numeric threshold interacting with categorical conditions;
- ``fallback``: a scoped default rule used when no ordinary rule matches;
- ``composition``: threshold + conjunction + authority, requiring arity 3.

The families are deliberately small and controlled.  They are a mechanism
test, not an external-validity claim.  The concrete narratives keep the demo
readable and give the evidence operator something to attach.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, replace
from typing import Callable

from .cases import Dataset
from .types import CaseRecord


# ---------------------------------------------------------------------------
# Narrative phrases
# ---------------------------------------------------------------------------

PHRASES: dict[str, dict[str, tuple[str, ...]]] = {
    "authority": {
        "owner": (
            "The caller is the registered owner of the account.",
            "The session belongs to the account owner.",
        ),
        "delegate": (
            "The caller is an authorized delegate on the account.",
            "The request is authenticated as a trusted delegate.",
        ),
        "none": (
            "The request arrived without a valid session.",
            "No valid credential accompanied the request.",
        ),
    },
    "commit_state": {
        "committed": (
            "The ledger already shows the charge.",
            "The prior attempt is recorded as settled.",
        ),
        "pending": (
            "The job is still queued.",
            "The prior attempt has not settled yet.",
        ),
        "failed": (
            "The last attempt was rejected with an error.",
            "The prior attempt did not go through.",
        ),
    },
    "retry_safety": {
        "safe": (
            "The client attached a unique deduplication key.",
            "The call carries an idempotency token.",
        ),
        "unsafe": (
            "The client did not attach a deduplication key.",
            "Each attempt would create a separate charge.",
        ),
    },
    "resource_state": {
        "free": (
            "The worker pool has free capacity.",
            "A worker slot is open.",
        ),
        "locked": (
            "Another worker is holding the reservation.",
            "The lock is currently held.",
        ),
        "exhausted": (
            "The quota is exhausted.",
            "No capacity remains in the pool.",
        ),
    },
    "reversibility": {
        "reversible": (
            "A rollback path exists for this operation.",
            "The side effects can be reversed.",
        ),
        "irreversible": (
            "The side effects are permanent.",
            "No compensating action can reverse this operation.",
        ),
    },
    "retry_count": {
        "0": ("This is the first attempt.", "No retry has been made."),
        "1": ("This is the second attempt.", "One retry has already been made."),
        "2": ("This is the third attempt.", "Two retries have already been made."),
        "3": ("This is the fourth attempt.", "Three retries have already been made."),
        "4": ("This is the fifth attempt.", "Four retries have already been made."),
        "5": ("This is the sixth attempt.", "Five retries have already been made."),
        "6": ("This is the seventh attempt.", "Six retries have already been made."),
        "7": ("This is the eighth attempt.", "Seven retries have already been made."),
        "8": ("This is the ninth attempt.", "Eight retries have already been made."),
        "9": ("This is the tenth attempt.", "Nine retries have already been made."),
        "10": ("This is the eleventh attempt.", "Ten retries have already been made."),
        "11": ("This is the twelfth attempt.", "Eleven retries have already been made."),
    },
}

SERVICES = ("atlas-billing", "nimbus-auth", "orion-orders", "vega-payments")
USERS = ("alice", "bob", "carol", "dave", "erin", "frank")


# ---------------------------------------------------------------------------
# Hidden policies
# ---------------------------------------------------------------------------


def scope_action(f: dict[str, str]) -> str:
    if f["authority"] == "none":
        return "request_authorization"
    if f["commit_state"] == "committed":
        return "retry" if f["retry_safety"] == "safe" else "compensate"
    if f["commit_state"] == "pending":
        return "wait" if f["resource_state"] == "locked" else "noop"
    if f["reversibility"] == "irreversible":
        return "rollback" if f["authority"] == "owner" else "escalate"
    if f["resource_state"] == "free":
        return "retry"
    if f["resource_state"] == "exhausted":
        return "escalate"
    return "wait"


def threshold_action(f: dict[str, str]) -> str:
    if f["authority"] == "none":
        return "request_authorization"
    n = int(f["retry_count"])
    if n >= 4:
        if f["retry_safety"] == "safe" and f["resource_state"] == "free":
            return "retry"
        return "escalate"
    if n == 0:
        return "noop"
    if f["retry_safety"] == "safe":
        return "retry"
    return "wait"


def fallback_action(f: dict[str, str]) -> str:
    if f["authority"] == "none":
        return "request_authorization"
    if f["commit_state"] == "failed":
        if f["retry_safety"] == "safe":
            return "retry"
        return "escalate" if f["authority"] == "delegate" else "rollback"
    if f["commit_state"] == "pending":
        return "wait" if f["resource_state"] == "locked" else "noop"
    return "noop"


def composition_action(f: dict[str, str]) -> str:
    if f["authority"] == "none":
        return "request_authorization"
    n = int(f["retry_count"])
    if n >= 3 and f["reversibility"] == "irreversible":
        return "rollback" if f["authority"] == "owner" else "escalate"
    if n >= 3 and f["retry_safety"] == "safe":
        return "retry"
    if n >= 3:
        return "wait"
    return "noop"


def composition_hard_action(f: dict[str, str]) -> str:
    """A four-way interaction: threshold, reversibility, authority, resource."""

    if f["authority"] == "none":
        return "request_authorization"
    n = int(f["retry_count"])
    if n >= 3 and f["reversibility"] == "irreversible":
        if f["authority"] == "owner":
            return "rollback" if f["resource_state"] == "free" else "escalate"
        return "escalate" if f["retry_safety"] == "unsafe" else "wait"
    if n >= 3 and f["retry_safety"] == "safe":
        return "retry"
    if n >= 3:
        return "wait"
    return "noop"


# ---------------------------------------------------------------------------
# Family specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilySpec:
    name: str
    factors: dict[str, tuple[str, ...]]
    action_fn: Callable[[dict[str, str]], str]
    required_operator: str
    description: str
    numeric: tuple[str, ...] = ()

    @property
    def concepts(self) -> tuple[str, ...]:
        return tuple(self.factors)

    def combinations(self) -> list[dict[str, str]]:
        keys = list(self.factors)
        return [
            dict(zip(keys, values))
            for values in itertools.product(*(self.factors[key] for key in keys))
        ]


FAMILIES: dict[str, FamilySpec] = {
    "scope": FamilySpec(
        name="scope",
        factors={
            "authority": ("owner", "delegate", "none"),
            "commit_state": ("committed", "pending", "failed"),
            "retry_safety": ("safe", "unsafe"),
            "resource_state": ("free", "locked", "exhausted"),
            "reversibility": ("reversible", "irreversible"),
            "retry_count": ("0", "1", "2", "3", "4", "5"),
        },
        action_fn=scope_action,
        required_operator="scope_conjunction",
        description="failed AND irreversible AND authority selects rollback vs escalate",
    ),
    "threshold": FamilySpec(
        name="threshold",
        factors={
            "authority": ("owner", "delegate", "none"),
            "retry_count": ("0", "1", "2", "3", "4", "5"),
            "retry_safety": ("safe", "unsafe"),
            "resource_state": ("free", "locked", "exhausted"),
            "commit_state": ("committed", "pending", "failed"),
        },
        action_fn=threshold_action,
        required_operator="threshold_split",
        description="retry_count >= 4 must be split before interacting with safety/resource",
        numeric=("retry_count",),
    ),
    "fallback": FamilySpec(
        name="fallback",
        factors={
            "authority": ("owner", "delegate", "none"),
            "commit_state": ("committed", "pending", "failed"),
            "retry_safety": ("safe", "unsafe"),
            "resource_state": ("free", "locked", "exhausted"),
            "reversibility": ("reversible", "irreversible"),
            "retry_count": ("0", "1", "2", "3", "4", "5"),
        },
        action_fn=fallback_action,
        required_operator="scoped_fallback",
        description="the failed+unsafe+delegate branch is too rare for an ordinary rule",
    ),
    "composition": FamilySpec(
        name="composition",
        factors={
            "authority": ("owner", "delegate", "none"),
            "retry_count": ("0", "1", "2", "3", "4", "5"),
            "retry_safety": ("safe", "unsafe"),
            "reversibility": ("reversible", "irreversible"),
            "resource_state": ("free", "locked", "exhausted"),
            "commit_state": ("committed", "pending", "failed"),
        },
        action_fn=composition_action,
        required_operator="threshold_scope_conjunction",
        description="threshold >= 3 interacts with reversibility and authority (arity 3)",
        numeric=("retry_count",),
    ),
    "composition_hard": FamilySpec(
        name="composition_hard",
        factors={
            "authority": ("owner", "delegate", "none"),
            "retry_count": ("0", "1", "2", "3", "4", "5"),
            "retry_safety": ("safe", "unsafe"),
            "reversibility": ("reversible", "irreversible"),
            "resource_state": ("free", "locked", "exhausted"),
            "commit_state": ("committed", "pending", "failed"),
        },
        action_fn=composition_hard_action,
        required_operator="threshold_scope_resource_conjunction",
        description="threshold >= 3 x reversibility x authority x resource (arity 4)",
        numeric=("retry_count",),
    ),
}


# ---------------------------------------------------------------------------
# Case generation
# ---------------------------------------------------------------------------


@dataclass
class FamilyDataset:
    spec: FamilySpec
    meta_train: Dataset
    meta_val: Dataset
    meta_test: Dataset

    def split(self, name: str) -> Dataset:
        if name == "train":
            return self.meta_train
        if name == "val":
            return self.meta_val
        if name == "test":
            return self.meta_test
        raise KeyError(name)


def _surface(rng: random.Random, family: str, index: int) -> dict[str, str]:
    return {
        "service": rng.choice(SERVICES),
        "user": rng.choice(USERS),
        "request_id": f"REQ-{family[:2].upper()}-{index:05d}",
        "timestamp": f"2026-11-{rng.randrange(1, 29):02d}T{rng.randrange(0, 24):02d}:00:00Z",
    }


def _narrative(factors: dict[str, str], surface: dict[str, str], rng: random.Random) -> str:
    sentences: list[str] = []
    for concept, value in factors.items():
        choices = PHRASES.get(concept, {}).get(value)
        if choices:
            sentences.append(rng.choice(choices))
    rng.shuffle(sentences)
    body = " ".join(sentences)
    return (
        f"Ticket {surface['request_id']} reached the {surface['service']} desk from "
        f"{surface['user']} at {surface['timestamp']}. {body}"
    )


def _make_cases(
    spec: FamilySpec,
    combinations: list[dict[str, str]],
    *,
    split: str,
    rng: random.Random,
    target: int,
    prefix: str,
) -> list[CaseRecord]:
    """Build ``target`` cases from ``combinations``.

    The combinations are disjoint across splits.  If a split has fewer
    combinations than ``target``, we resample within the split with fresh
    surface details; this keeps the split disjoint while reaching a usable
    evaluation size.
    """

    if not combinations:
        return []
    pool = list(combinations)
    rng.shuffle(pool)
    selected: list[dict[str, str]] = []
    while len(selected) < target:
        if len(selected) < len(pool):
            selected.append(pool[len(selected)])
        else:
            selected.append(rng.choice(pool))

    cases: list[CaseRecord] = []
    for index, factors in enumerate(selected):
        surface = _surface(rng, spec.name, index)
        cases.append(
            CaseRecord(
                case_id=f"{prefix}-{index:04d}",
                split=split,
                narrative=_narrative(factors, surface, rng),
                factors=dict(factors),
                action=spec.action_fn(factors),
                family=spec.name,
            )
        )
    return cases


def _partition(items: list[dict[str, str]], ratios: tuple[float, ...]) -> list[list[dict[str, str]]]:
    n = len(items)
    cuts: list[int] = []
    consumed = 0
    for ratio in ratios[:-1]:
        take = max(1, int(round(n * ratio)))
        take = min(take, n - consumed - (len(ratios) - len(cuts) - 1))
        cuts.append(consumed + take)
        consumed += take
    bounds = [0, *cuts, n]
    return [items[bounds[i] : bounds[i + 1]] for i in range(len(ratios))]


def _make_dataset(
    spec: FamilySpec,
    combinations: list[dict[str, str]],
    *,
    rng: random.Random,
    prefix: str,
) -> Dataset:
    # seed / val / induction / iid / compositional are disjoint combinations.
    parts = _partition(
        list(combinations),
        (0.30, 0.20, 0.15, 0.20, 0.15),
    )
    seed_cases = _make_cases(
        spec, parts[0], split="train", rng=rng, target=max(12, len(parts[0])), prefix=f"{prefix}-s"
    )
    val_cases = _make_cases(
        spec, parts[1], split="val", rng=rng, target=max(10, len(parts[1])), prefix=f"{prefix}-v"
    )
    induction_cases = _make_cases(
        spec, parts[2], split="train", rng=rng, target=max(8, len(parts[2])), prefix=f"{prefix}-i"
    )
    iid_cases = _make_cases(
        spec, parts[3], split="iid", rng=rng, target=max(10, len(parts[3])), prefix=f"{prefix}-d"
    )
    comp_cases = _make_cases(
        spec,
        parts[4],
        split="compositional",
        rng=rng,
        target=max(10, len(parts[4])),
        prefix=f"{prefix}-c",
    )
    all_cases = seed_cases + val_cases + induction_cases + iid_cases + comp_cases
    return Dataset(
        seed=seed_cases,
        induction=induction_cases,
        val=val_cases,
        iid=iid_cases,
        surface=iid_cases,
        compositional=comp_cases,
        counterfactual=[],
        all_cases=all_cases,
    )


DEFAULT_FAMILIES = ("scope", "threshold", "composition", "composition_hard")
SHIFT_FAMILIES = {"threshold", "composition", "composition_hard"}
VAL_SHIFT = {"3": "6", "4": "7", "5": "8"}
TEST_SHIFT = {"3": "9", "4": "10", "5": "11"}


def _shift_cases(
    cases: list[CaseRecord],
    spec: FamilySpec,
    mapping: dict[str, str],
    rng: random.Random,
) -> list[CaseRecord]:
    """Replace a numeric value with an unseen value and recompute the label.

    This is the generalisation test for threshold operators: a rule that
    memorises ``retry_count=4`` cannot cover ``retry_count=10``; a genuine
    ``retry_count>=4`` operator can.
    """

    shifted: list[CaseRecord] = []
    for case in cases:
        factors = dict(case.factors)
        value = factors.get("retry_count")
        if value in mapping:
            factors["retry_count"] = mapping[value]
        surface = {
            "service": rng.choice(SERVICES),
            "user": rng.choice(USERS),
            "request_id": case.case_id,
            "timestamp": f"2026-11-{rng.randrange(1, 29):02d}T{rng.randrange(0, 24):02d}:00:00Z",
        }
        shifted.append(
            replace(
                case,
                factors=factors,
                action=spec.action_fn(factors),
                narrative=_narrative(factors, surface, rng),
            )
        )
    return shifted


def _shift_dataset(
    dataset: Dataset,
    spec: FamilySpec,
    mapping: dict[str, str],
    rng: random.Random,
) -> Dataset:
    seed = _shift_cases(dataset.seed, spec, mapping, rng)
    induction = _shift_cases(dataset.induction, spec, mapping, rng)
    val = _shift_cases(dataset.val, spec, mapping, rng)
    iid = _shift_cases(dataset.iid, spec, mapping, rng)
    comp = _shift_cases(dataset.compositional, spec, mapping, rng)
    return Dataset(
        seed=seed,
        induction=induction,
        val=val,
        iid=iid,
        surface=iid,
        compositional=comp,
        counterfactual=[],
        all_cases=seed + induction + val + iid + comp,
    )


def build_family(spec: FamilySpec, *, seed: int = 20261102) -> FamilyDataset:
    combinations = spec.combinations()
    rng = random.Random(seed)
    rng.shuffle(combinations)
    n = len(combinations)
    train_end = int(n * 0.50)
    val_end = int(n * 0.75)
    train_combos = combinations[:train_end]
    val_combos = combinations[train_end:val_end]
    test_combos = combinations[val_end:]

    meta_train = _make_dataset(
        spec, train_combos, rng=random.Random(seed + 11), prefix=f"{spec.name}-tr"
    )
    meta_val = _make_dataset(
        spec, val_combos, rng=random.Random(seed + 23), prefix=f"{spec.name}-va"
    )
    meta_test = _make_dataset(
        spec, test_combos, rng=random.Random(seed + 37), prefix=f"{spec.name}-te"
    )

    if spec.name in SHIFT_FAMILIES:
        # Validation and test use different unseen numeric values, so a
        # value-memorising operator cannot transfer from one to the other.
        meta_val = _shift_dataset(meta_val, spec, VAL_SHIFT, random.Random(seed + 41))
        meta_test = _shift_dataset(meta_test, spec, TEST_SHIFT, random.Random(seed + 43))

    return FamilyDataset(
        spec=spec,
        meta_train=meta_train,
        meta_val=meta_val,
        meta_test=meta_test,
    )


def build_suite(*, seed: int = 20261102, families: tuple[str, ...] | None = None) -> dict[str, FamilyDataset]:
    names = families or DEFAULT_FAMILIES
    return {name: build_family(FAMILIES[name], seed=seed) for name in names}


def suite_summary(suite: dict[str, FamilyDataset]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for name, family in suite.items():
        summary[name] = {
            "train_seed": len(family.meta_train.seed),
            "train_val": len(family.meta_train.val),
            "train_iid": len(family.meta_train.iid),
            "train_comp": len(family.meta_train.compositional),
            "val_seed": len(family.meta_val.seed),
            "val_comp": len(family.meta_val.compositional),
            "test_seed": len(family.meta_test.seed),
            "test_comp": len(family.meta_test.compositional),
        }
    return summary
