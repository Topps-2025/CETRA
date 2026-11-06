"""Controlled memory with hidden latent concepts.

Each case has five latent factors.  The narrative never names the factors
directly; it only gives observable indicators.  The mapping from indicators to
latent values requires world knowledge, which is exactly the abductive step the
new experiment is meant to test.
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

LATENT_VALUES: dict[str, tuple[str, ...]] = {
    "authority": ("owner", "delegate", "none"),
    "commit_state": ("committed", "pending", "failed"),
    "retry_safety": ("safe", "unsafe"),
    "resource_state": ("free", "locked", "exhausted"),
    "reversibility": ("reversible", "irreversible"),
}

LATENT_NAMES: tuple[str, ...] = tuple(LATENT_VALUES)

HIDDEN_CONCEPT_LABELS: dict[str, str] = {
    "authority": "authority",
    "commit_state": "commit state",
    "retry_safety": "retry safety",
    "resource_state": "resource state",
    "reversibility": "reversibility",
}

ACTIONS: tuple[str, ...] = (
    "request_authorization",
    "retry",
    "compensate",
    "wait",
    "noop",
    "rollback",
    "escalate",
)

ACTION_DESCRIPTIONS: dict[str, str] = {
    "request_authorization": "ask the caller to obtain valid authorization",
    "retry": "retry the same operation",
    "compensate": "apply a compensating action to undo or offset the prior effect",
    "wait": "wait for the resource or prior attempt to change state",
    "noop": "take no further action",
    "rollback": "roll back the failed operation",
    "escalate": "escalate to a human or higher-level operator",
}

TRAIN_SERVICES = ("atlas-billing", "nimbus-auth", "orion-orders", "vega-payments", "lumen-search", "zephyr-mail")
TEST_SERVICES = ("quartz-billing", "helios-auth", "meridian-orders", "solstice-payments", "cinder-search", "aurora-mail")
TRAIN_USERS = ("alice", "bob", "carol", "dave", "erin", "frank")
TEST_USERS = ("grace", "heidi", "ivan", "judy", "mallory", "oscar")
ERROR_CODES = ("E-1042", "E-2087", "E-3310", "E-4415", "E-5521", "E-6634")

# Observable indicators.  These phrases avoid the latent factor names; the
# mapping to a latent value requires world knowledge.
INDICATORS: dict[str, dict[str, tuple[str, ...]]] = {
    "authority": {
        "owner": (
            "The caller is the registered owner of the account.",
            "The session belongs to the account owner.",
            "The request is authenticated as the account holder.",
        ),
        "delegate": (
            "The caller is an authorized delegate on the account.",
            "The session belongs to a delegate with delegated rights.",
            "The request is authenticated as a trusted delegate.",
        ),
        "none": (
            "The request arrived without a valid session.",
            "The caller could not be authenticated.",
            "No valid credential accompanied the request.",
        ),
    },
    "commit_state": {
        "committed": (
            "The ledger already shows the charge.",
            "The job is marked complete in the ledger.",
            "The prior attempt is recorded as settled.",
        ),
        "pending": (
            "The job is still queued.",
            "The submission is awaiting confirmation.",
            "The prior attempt has not settled yet.",
        ),
        "failed": (
            "The last attempt was rejected with an error.",
            "The job failed before taking effect.",
            "The prior attempt did not go through.",
        ),
    },
    "retry_safety": {
        "safe": (
            "The client attached a unique deduplication key.",
            "The call carries an idempotency token.",
            "The request includes a one-time request key.",
        ),
        "unsafe": (
            "The client did not attach a deduplication key.",
            "Each attempt would create a separate charge.",
            "There is no idempotency token on the call.",
        ),
    },
    "resource_state": {
        "free": (
            "The worker pool has free capacity.",
            "The target resource is available.",
            "A worker slot is open.",
        ),
        "locked": (
            "Another worker is holding the reservation.",
            "The target resource is locked by another job.",
            "The lock is currently held.",
        ),
        "exhausted": (
            "The quota is exhausted.",
            "No capacity remains in the pool.",
            "The resource budget is used up.",
        ),
    },
    "reversibility": {
        "reversible": (
            "The operation can be undone with a compensating action.",
            "A rollback path exists for this operation.",
            "The side effects can be reversed.",
        ),
        "irreversible": (
            "The operation has been written to the audit log and cannot be undone.",
            "The side effects are permanent.",
            "No compensating action can reverse this operation.",
        ),
    },
}


def hidden_action(factors: dict[str, str]) -> str:
    """The latent expert policy.  Only used for labels and evaluation."""
    if factors["authority"] == "none":
        return "request_authorization"
    if factors["commit_state"] == "committed":
        return "retry" if factors["retry_safety"] == "safe" else "compensate"
    if factors["commit_state"] == "pending":
        return "wait" if factors["resource_state"] == "locked" else "noop"
    # failed
    if factors["reversibility"] == "irreversible":
        return "rollback" if factors["authority"] == "owner" else "escalate"
    if factors["resource_state"] == "free":
        return "retry"
    if factors["resource_state"] == "exhausted":
        return "escalate"
    return "wait"


def all_combinations() -> list[dict[str, str]]:
    keys = list(LATENT_VALUES)
    return [dict(zip(keys, values)) for values in itertools.product(*(LATENT_VALUES[k] for k in keys))]


@dataclass
class LatentCase:
    case_id: str
    split: str
    factors: dict[str, str]  # hidden labels; never shown to the model
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
    def from_dict(cls, payload: dict[str, Any]) -> "LatentCase":
        return cls(**payload)

    def model_input(self) -> str:
        """What the model is allowed to see: narrative + action/outcome for training."""
        return self.narrative


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


def _indicator_sentences(factors: dict[str, str], rng: random.Random) -> list[tuple[str, str]]:
    return [(concept, rng.choice(INDICATORS[concept][factors[concept]])) for concept in LATENT_NAMES]


def _compose_narrative(
    surface: dict[str, str],
    sentences: list[tuple[str, str]],
) -> str:
    body = " ".join(sentence for _, sentence in sentences)
    return (
        f"Ticket {surface['request_id']} reached the {surface['service']} desk from "
        f"{surface['user']} at {surface['timestamp']}. {body} "
        f"A note mentions error code {surface['error_code']} and a latency spike."
    )


def _make_case(
    case_id: str,
    split: str,
    factors: dict[str, str],
    rng: random.Random,
    *,
    test_pool: bool,
    surface: dict[str, str] | None = None,
    sentences: list[tuple[str, str]] | None = None,
) -> tuple[LatentCase, list[tuple[str, str]]]:
    surface = surface or _surface(rng, test_pool=test_pool)
    generated = sentences is None
    if generated:
        sentences = _indicator_sentences(factors, rng)
        # Shuffle sentence order once per case; counterfactual pairs reuse it.
        rng.shuffle(sentences)
    assert sentences is not None
    if len(sentences) != len(LATENT_NAMES):
        raise ValueError("sentence list must cover every latent concept")
    case = LatentCase(
        case_id=case_id,
        split=split,
        factors=dict(factors),
        surface=surface,
        action=hidden_action(factors),
        outcome="success",
        narrative=_compose_narrative(surface, sentences),
    )
    return case, sentences


def _sample_covering(
    pool: list[dict[str, str]],
    size: int,
    rng: random.Random,
    *,
    required_values: dict[str, Iterable[str]] | None = None,
    required_actions: Iterable[str] | None = None,
) -> list[dict[str, str]]:
    """Sample a subset that covers the requested values/actions when possible."""
    required_values = {k: set(v) for k, v in (required_values or {}).items()}
    required_actions = set(required_actions or [])

    for _ in range(3000):
        sample = rng.sample(pool, min(size, len(pool)))
        values = {(k, v) for case in sample for k, v in case.items()}
        actions = {hidden_action(case) for case in sample}
        ok_values = all((k, v) in values for k, vs in required_values.items() for v in vs)
        ok_actions = required_actions.issubset(actions)
        if ok_values and ok_actions:
            return sample

    # Greedy fallback: start with cases covering required actions, then fill.
    selected: list[dict[str, str]] = []
    for action in required_actions:
        candidate = next((c for c in pool if hidden_action(c) == action and c not in selected), None)
        if candidate is not None:
            selected.append(candidate)
    remaining = [c for c in pool if c not in selected]
    rng.shuffle(remaining)
    while len(selected) < min(size, len(pool)) and remaining:
        selected.append(remaining.pop())
    return selected


def _pick_cf_pairs(
    base_combos: list[dict[str, str]], rng: random.Random, n_pairs: int
) -> list[tuple[dict[str, str], dict[str, str], str]]:
    changed: list[tuple[dict[str, str], dict[str, str], str]] = []
    unchanged: list[tuple[dict[str, str], dict[str, str], str]] = []
    for base in base_combos:
        base_action = hidden_action(base)
        for factor in LATENT_NAMES:
            for value in LATENT_VALUES[factor]:
                if value == base[factor]:
                    continue
                perturbed = dict(base)
                perturbed[factor] = value
                item = (base, perturbed, factor)
                if hidden_action(perturbed) != base_action:
                    changed.append(item)
                else:
                    unchanged.append(item)
    rng.shuffle(changed)
    rng.shuffle(unchanged)
    n_changed = min(len(changed), max(1, (2 * n_pairs + 2) // 3))
    n_unchanged = min(len(unchanged), n_pairs - n_changed)
    picked = changed[:n_changed] + unchanged[:n_unchanged]
    if len(picked) < n_pairs:
        picked += changed[n_changed : n_changed + (n_pairs - len(picked))]
    return picked[:n_pairs]


def explanatory_concepts(factors: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """Return latent concepts whose single-factor intervention changes the action."""
    base_action = hidden_action(factors)
    acceptable: list[str] = []
    alternatives: dict[str, str] = {}
    for concept in LATENT_NAMES:
        for value in LATENT_VALUES[concept]:
            if value == factors[concept]:
                continue
            perturbed = dict(factors)
            perturbed[concept] = value
            action = hidden_action(perturbed)
            if action != base_action:
                acceptable.append(concept)
                alternatives[concept] = action
                break
    return acceptable, alternatives


def qa_items(
    cf_pairs: list[tuple[LatentCase, LatentCase]], limit: int = 12
) -> list[dict[str, Any]]:
    """Contrastive QA: two cases differ in one latent concept; name the concept.

    This directly tests whether the abstraction identifies the latent dimension
    responsible for a behavioral difference, rather than merely predicting an
    action.
    """
    items: list[dict[str, Any]] = []
    for base, perturbed in cf_pairs:
        if base.action == perturbed.action:
            continue
        items.append(
            {
                "pair_id": base.pair_id,
                "question": (
                    f"Case A led to {base.action}; Case B led to {perturbed.action}. "
                    "Which latent concept best explains the difference?"
                ),
                "options": [HIDDEN_CONCEPT_LABELS[name] for name in LATENT_NAMES],
                "answer_concept": HIDDEN_CONCEPT_LABELS[base.changed_factor or ""],
                "changed_factor": base.changed_factor,
                "narratives": [base.narrative, perturbed.narrative],
                "actions": [base.action, perturbed.action],
            }
        )
        if len(items) >= limit:
            break
    return items


def generate_latent_dataset(
    seed: int = 20261026,
    train_size: int = 48,
    iid_size: int = 12,
    surface_size: int = 9,
    compositional_size: int = 12,
    n_cf_pairs: int = 18,
    seed_size: int = 12,
    val_size: int = 12,
) -> dict[str, Any]:
    rng = random.Random(seed)
    combos = all_combinations()
    train_combos = _sample_covering(
        combos,
        train_size,
        rng,
        required_values={k: v for k, v in LATENT_VALUES.items()},
        required_actions=ACTIONS,
    )
    train_set = {tuple(sorted(c.items())) for c in train_combos}

    # Seed: no failed / irreversible cases.  The initial abductive model cannot
    # know that reversibility matters until validation errors reveal it.
    seed_pool = [
        c
        for c in train_combos
        if c["commit_state"] != "failed" and c["reversibility"] == "reversible"
    ]
    seed_combos = _sample_covering(
        seed_pool,
        seed_size,
        rng,
        required_values={
            "authority": LATENT_VALUES["authority"],
            "commit_state": ("committed", "pending"),
            "retry_safety": LATENT_VALUES["retry_safety"],
            "resource_state": LATENT_VALUES["resource_state"],
        },
    )
    remaining = [c for c in train_combos if c not in seed_combos]
    val_pool = [
        c
        for c in remaining
        if c["commit_state"] == "failed" or c["reversibility"] == "irreversible"
    ]
    val_combos = _sample_covering(
        val_pool,
        val_size,
        rng,
        required_values={
            "commit_state": ("failed",),
            "reversibility": ("irreversible",),
            "authority": ("owner", "delegate"),
        },
        required_actions=("rollback", "escalate"),
    )
    induction_combos = [c for c in remaining if c not in val_combos]
    # If val_pool is too small, fill from remaining so the partitions cover train.
    if len(induction_combos) > train_size - seed_size - val_size:
        induction_combos = induction_combos[: train_size - seed_size - val_size]

    cases: list[LatentCase] = []
    counter = 0

    def next_id(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}-{counter:03d}"

    seed_ids: list[str] = []
    val_ids: list[str] = []
    induction_ids: list[str] = []
    for factors in seed_combos:
        case, _ = _make_case(next_id("SEED"), "train", factors, rng, test_pool=False)
        cases.append(case)
        seed_ids.append(case.case_id)
    for factors in val_combos:
        case, _ = _make_case(next_id("VAL"), "train", factors, rng, test_pool=False)
        cases.append(case)
        val_ids.append(case.case_id)
    for factors in induction_combos:
        case, _ = _make_case(next_id("IND"), "train", factors, rng, test_pool=False)
        cases.append(case)
        induction_ids.append(case.case_id)

    iid_combos = rng.sample(train_combos, min(iid_size, len(train_combos)))
    for factors in iid_combos:
        case, _ = _make_case(next_id("IID"), "iid", factors, rng, test_pool=True)
        cases.append(case)

    surface_pool = [c for c in train_combos if c not in iid_combos]
    surface_combos = rng.sample(surface_pool, min(surface_size, len(surface_pool)))
    for factors in surface_combos:
        case, _ = _make_case(next_id("SURF"), "surface", factors, rng, test_pool=True)
        cases.append(case)

    comp_pool = [c for c in combos if tuple(sorted(c.items())) not in train_set]
    comp_combos = rng.sample(comp_pool, min(compositional_size, len(comp_pool)))
    for factors in comp_combos:
        case, _ = _make_case(next_id("COMP"), "compositional", factors, rng, test_pool=True)
        cases.append(case)

    used = {tuple(sorted(c.items())) for c in iid_combos + surface_combos + comp_combos}
    cf_pool = [c for c in comp_pool if tuple(sorted(c.items())) not in used]
    rng.shuffle(cf_pool)
    pairs = _pick_cf_pairs(cf_pool, rng, n_pairs=n_cf_pairs)
    qa_pairs: list[tuple[LatentCase, LatentCase]] = []
    for index, (base, perturbed, factor) in enumerate(pairs, start=1):
        pair_id = f"CF-{index:02d}"
        surface = _surface(rng, test_pool=True)
        base_case, sentences = _make_case(
            f"{pair_id}-base",
            "counterfactual",
            base,
            rng,
            test_pool=True,
            surface=surface,
        )
        # Replace only the changed factor's sentence; keep order and surface.
        new_sentences = list(sentences)
        for pos, (concept, _) in enumerate(new_sentences):
            if concept == factor:
                new_sentences[pos] = (concept, rng.choice(INDICATORS[concept][perturbed[concept]]))
                break
        perturbed_case, _ = _make_case(
            f"{pair_id}-pert",
            "counterfactual",
            perturbed,
            rng,
            test_pool=True,
            surface=surface,
            sentences=new_sentences,
        )
        base_case.pair_id = perturbed_case.pair_id = pair_id
        base_case.pair_role = "base"
        perturbed_case.pair_role = "perturbed"
        base_case.changed_factor = perturbed_case.changed_factor = factor
        cases.append(base_case)
        cases.append(perturbed_case)
        qa_pairs.append((base_case, perturbed_case))

    qa = qa_items(qa_pairs, limit=12)
    meta = {
        "seed": seed,
        "num_cases": len(cases),
        "splits": {split: sum(1 for c in cases if c.split == split) for split in sorted({c.split for c in cases})},
        "latent_values": {k: list(v) for k, v in LATENT_VALUES.items()},
        "actions": list(ACTIONS),
        "seed_case_ids": seed_ids,
        "val_case_ids": val_ids,
        "induction_case_ids": induction_ids,
        "train_case_ids": seed_ids + val_ids + induction_ids,
        "qa": qa,
        "ground_truth": "hidden decision tree over latent concepts; see src/cetra/abduction/latent_cases.py:hidden_action",
    }
    return {"meta": meta, "cases": [c.to_dict() for c in cases]}


def save_dataset(dataset: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_dataset(path: str | Path) -> tuple[list[LatentCase], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [LatentCase.from_dict(item) for item in payload["cases"]], payload["meta"]


def split_cases(cases: Iterable[LatentCase], split: str) -> list[LatentCase]:
    return [case for case in cases if case.split == split]


def cases_by_ids(cases: Iterable[LatentCase], ids: Iterable[str]) -> list[LatentCase]:
    wanted = set(ids)
    return [case for case in cases if case.case_id in wanted]
