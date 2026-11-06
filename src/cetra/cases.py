"""Synthetic case generator for the memory abstraction experiment.

The domain is deliberately controlled: each case has five decision factors and
one expert action.  Surface details (service, user, request id, error code,
timestamp) are distractors.  The ground-truth action is a deterministic tree,
which lets us measure whether an abstraction preserves the decision-relevant
conditions rather than surface correlations.
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

FACTOR_VALUES: dict[str, tuple[str, ...]] = {
    "permission": ("owner", "delegate", "none"),
    "submission_status": ("committed", "pending", "failed"),
    "idempotent": ("yes", "no"),
    "resource_state": ("available", "locked", "exhausted"),
    "retryable": ("yes", "no"),
}

FACTOR_NAMES: tuple[str, ...] = tuple(FACTOR_VALUES)

ACTIONS: tuple[str, ...] = (
    "request_permission",
    "retry",
    "cancel_and_resubmit",
    "wait",
    "noop",
    "rollback",
    "escalate",
)

ACTION_DESCRIPTIONS: dict[str, str] = {
    "request_permission": "ask the caller to obtain the required permission",
    "retry": "retry the same request",
    "cancel_and_resubmit": "cancel the prior submission and submit a fresh request",
    "wait": "wait for the resource or submission state to change",
    "noop": "take no further action",
    "rollback": "roll back the failed operation",
    "escalate": "escalate to a human or higher-level operator",
}

TRAIN_SERVICES = ("atlas-billing", "nimbus-auth", "orion-orders", "vega-payments", "lumen-search", "zephyr-mail")
TEST_SERVICES = ("quartz-billing", "helios-auth", "meridian-orders", "solstice-payments", "cinder-search", "aurora-mail")
TRAIN_USERS = ("alice", "bob", "carol", "dave", "erin", "frank")
TEST_USERS = ("grace", "heidi", "ivan", "judy", "mallory", "oscar")
ERROR_CODES = ("E-1042", "E-2087", "E-3310", "E-4415", "E-5521", "E-6634")


def ground_truth_action(factors: dict[str, str]) -> str:
    """Deterministic expert policy used to label every case."""
    if factors["permission"] == "none":
        return "request_permission"
    if factors["submission_status"] == "committed":
        return "retry" if factors["idempotent"] == "yes" else "cancel_and_resubmit"
    if factors["submission_status"] == "pending":
        return "wait" if factors["resource_state"] == "locked" else "noop"
    # submission_status == "failed"
    if factors["retryable"] == "no":
        return "rollback" if factors["permission"] == "owner" else "escalate"
    if factors["resource_state"] == "available":
        return "retry"
    if factors["resource_state"] == "exhausted":
        return "escalate"
    return "wait"


def all_combinations() -> list[dict[str, str]]:
    keys = list(FACTOR_VALUES)
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(FACTOR_VALUES[k] for k in keys))
    ]


@dataclass
class Case:
    case_id: str
    split: str
    factors: dict[str, str]
    surface: dict[str, str]
    action: str
    outcome: str
    narrative: str
    pair_id: str | None = None
    pair_role: str | None = None
    changed_factor: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Case":
        return cls(**payload)

    def context_text(self) -> str:
        factors = ", ".join(f"{k}={v}" for k, v in self.factors.items())
        return f"{self.narrative}\nStructured factors: {factors}"


def _surface(rng: random.Random, test_pool: bool) -> dict[str, str]:
    services = TEST_SERVICES if test_pool else TRAIN_SERVICES
    users = TEST_USERS if test_pool else TRAIN_USERS
    return {
        "service": rng.choice(services),
        "user": rng.choice(users),
        "request_id": f"REQ-{rng.randrange(100000, 999999)}",
        "error_code": rng.choice(ERROR_CODES),
        "timestamp": f"2026-10-{rng.randrange(1, 29):02d}T{rng.randrange(0, 24):02d}:{rng.randrange(0, 60):02d}:00Z",
    }


def _narrative(factors: dict[str, str], surface: dict[str, str], rng: random.Random) -> str:
    template = rng.randrange(3)
    if template == 0:
        text = (
            f"Request {surface['request_id']} arrived at the {surface['service']} gateway "
            f"from {surface['user']} at {surface['timestamp']}. The submission is currently "
            f"{factors['submission_status']}. The operation is {factors['idempotent']} "
            f"idempotent. The target resource is {factors['resource_state']}. The caller holds "
            f"{factors['permission']} permission. Retry is {factors['retryable']}."
        )
    elif template == 1:
        text = (
            f"At {surface['timestamp']}, {surface['user']} submitted {surface['request_id']} "
            f"through {surface['service']}. Submission status: {factors['submission_status']}. "
            f"Idempotency: {factors['idempotent']}. Resource state: "
            f"{factors['resource_state']}. Permission: {factors['permission']}. Retryable: "
            f"{factors['retryable']}."
        )
    else:
        text = (
            f"The {surface['service']} service is handling ticket {surface['request_id']} for "
            f"{surface['user']} (opened {surface['timestamp']}). The previous submission state "
            f"is {factors['submission_status']}; the call is {factors['idempotent']} idempotent; "
            f"the resource is {factors['resource_state']}; the caller is "
            f"{factors['permission']}; the operation is {factors['retryable']} retryable."
        )
    return f"{text} The ticket also mentions error code {surface['error_code']} and a latency spike."


def _make_case(
    case_id: str,
    split: str,
    factors: dict[str, str],
    rng: random.Random,
    *,
    test_pool: bool,
    action: str | None = None,
    pair_id: str | None = None,
    pair_role: str | None = None,
    changed_factor: str | None = None,
    surface: dict[str, str] | None = None,
) -> Case:
    surface = surface or _surface(rng, test_pool=test_pool)
    return Case(
        case_id=case_id,
        split=split,
        factors=dict(factors),
        surface=surface,
        action=action or ground_truth_action(factors),
        outcome="success",
        narrative=_narrative(factors, surface, rng),
        pair_id=pair_id,
        pair_role=pair_role,
        changed_factor=changed_factor,
    )


def _sample_train_combinations(
    combos: list[dict[str, str]], size: int, rng: random.Random
) -> list[dict[str, str]]:
    """Sample a train split that covers every action and every factor value."""
    all_actions = set(ACTIONS)
    all_values = {(k, v) for k, values in FACTOR_VALUES.items() for v in values}
    for _ in range(2000):
        sample = rng.sample(combos, size)
        actions = {ground_truth_action(c) for c in sample}
        values = {(k, v) for c in sample for k, v in c.items()}
        if actions == all_actions and values == all_values:
            return sample
    # Deterministic fallback: one case per action, then greedy factor coverage.
    selected: list[dict[str, str]] = []
    for action in ACTIONS:
        candidate = next(c for c in combos if ground_truth_action(c) == action)
        if candidate not in selected:
            selected.append(candidate)
    remaining = [c for c in combos if c not in selected]
    rng.shuffle(remaining)
    while len(selected) < size and remaining:
        selected.append(remaining.pop())
    return selected


def _pick_counterfactual_pairs(
    base_combos: list[dict[str, str]], rng: random.Random, n_pairs: int
) -> list[tuple[dict[str, str], dict[str, str], str]]:
    """Return (base, perturbed, changed_factor) pairs, mixing changed/unchanged actions."""
    if n_pairs <= 0:
        return []
    changed: list[tuple[dict[str, str], dict[str, str], str]] = []
    unchanged: list[tuple[dict[str, str], dict[str, str], str]] = []
    for base in base_combos:
        base_action = ground_truth_action(base)
        for factor in FACTOR_NAMES:
            for value in FACTOR_VALUES[factor]:
                if value == base[factor]:
                    continue
                perturbed = dict(base)
                perturbed[factor] = value
                item = (base, perturbed, factor)
                if ground_truth_action(perturbed) != base_action:
                    changed.append(item)
                else:
                    unchanged.append(item)
    rng.shuffle(changed)
    rng.shuffle(unchanged)
    # Aim for roughly two thirds sensitivity pairs and one third invariance pairs.
    n_changed = min(len(changed), max(1, (2 * n_pairs + 2) // 3))
    n_unchanged = min(len(unchanged), n_pairs - n_changed)
    picked = changed[:n_changed] + unchanged[:n_unchanged]
    if len(picked) < n_pairs:
        picked += changed[n_changed : n_changed + (n_pairs - len(picked))]
    return picked[:n_pairs]


def generate_dataset(
    seed: int = 20261025,
    train_size: int = 48,
    iid_size: int = 12,
    surface_size: int = 9,
    compositional_size: int = 12,
    n_cf_pairs: int = 9,
) -> dict[str, Any]:
    """Generate train / iid / surface / compositional / counterfactual splits."""
    rng = random.Random(seed)
    combos = all_combinations()
    train_combos = _sample_train_combinations(combos, size=train_size, rng=rng)
    train_set = {tuple(sorted(c.items())) for c in train_combos}

    cases: list[Case] = []
    counter = 0

    def next_id(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}-{counter:03d}"

    for factors in train_combos:
        cases.append(
            _make_case(next_id("TR"), "train", factors, rng, test_pool=False, action=ground_truth_action(factors))
        )

    # IID: factor combinations seen in training, but new surface details.
    iid_combos = rng.sample(train_combos, min(iid_size, len(train_combos)))
    for factors in iid_combos:
        cases.append(
            _make_case(next_id("IID"), "iid", factors, rng, test_pool=True, action=ground_truth_action(factors))
        )

    # Surface shift: seen factor combinations, unseen service/user pools.
    surface_pool = [c for c in train_combos if c not in iid_combos]
    surface_combos = rng.sample(surface_pool, min(surface_size, len(surface_pool)))
    for factors in surface_combos:
        cases.append(
            _make_case(next_id("SURF"), "surface", factors, rng, test_pool=True, action=ground_truth_action(factors))
        )

    # Compositional: factor combinations not present in training.
    comp_pool = [c for c in combos if tuple(sorted(c.items())) not in train_set]
    comp_combos = rng.sample(comp_pool, min(compositional_size, len(comp_pool)))
    for factors in comp_combos:
        cases.append(
            _make_case(next_id("COMP"), "compositional", factors, rng, test_pool=True, action=ground_truth_action(factors))
        )

    # Counterfactual pairs: new base combinations, each perturbed in one factor.
    used = {tuple(sorted(c.items())) for c in iid_combos + surface_combos + comp_combos}
    cf_pool = [c for c in comp_pool if tuple(sorted(c.items())) not in used]
    rng.shuffle(cf_pool)
    pairs = _pick_counterfactual_pairs(cf_pool, rng, n_pairs=n_cf_pairs)
    for index, (base, perturbed, factor) in enumerate(pairs, start=1):
        pair_id = f"CF-{index:02d}"
        surface = _surface(rng, test_pool=True)
        cases.append(
            _make_case(
                f"{pair_id}-base",
                "counterfactual",
                base,
                rng,
                test_pool=True,
                action=ground_truth_action(base),
                pair_id=pair_id,
                pair_role="base",
                changed_factor=factor,
                surface=surface,
            )
        )
        cases.append(
            _make_case(
                f"{pair_id}-pert",
                "counterfactual",
                perturbed,
                rng,
                test_pool=True,
                action=ground_truth_action(perturbed),
                pair_id=pair_id,
                pair_role="perturbed",
                changed_factor=factor,
                surface=surface,
            )
        )

    meta = {
        "seed": seed,
        "num_cases": len(cases),
        "splits": {split: sum(1 for c in cases if c.split == split) for split in sorted({c.split for c in cases})},
        "factor_values": {k: list(v) for k, v in FACTOR_VALUES.items()},
        "actions": list(ACTIONS),
        "ground_truth": "deterministic decision tree; see src/cetra/cases.py:ground_truth_action",
    }
    return {"meta": meta, "cases": [c.to_dict() for c in cases]}


def save_dataset(dataset: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_dataset(path: str | Path) -> tuple[list[Case], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Case.from_dict(item) for item in payload["cases"]], payload["meta"]


def split_cases(cases: Iterable[Case], split: str) -> list[Case]:
    return [case for case in cases if case.split == split]


def all_action_counts(cases: Iterable[Case]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.action] = counts.get(case.action, 0) + 1
    return dict(sorted(counts.items()))
