"""Triple gate for harness edits.

An edit is accepted only when it:

1. resolves a non-negative fraction of the parent's counterexamples;
2. does not degrade held-out decision risk beyond a margin;
3. does not decrease metaproductivity.

This is the independent anchor that prevents the harness from self-confirming
by relaxing its own acceptance criterion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .diagnosis_utils import resolved_fraction
from .types import Diagnosis, Metrics


@dataclass
class GateDecision:
    accepted: bool
    delta_in: float
    delta_ho: float
    delta_mp: float
    reason: str
    gate_split: str = "val"
    parent: dict[str, Metrics] = field(default_factory=dict)
    child: dict[str, Metrics] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "delta_in": round(self.delta_in, 6),
            "delta_ho": round(self.delta_ho, 6),
            "delta_mp": round(self.delta_mp, 6),
            "reason": self.reason,
            "gate_split": self.gate_split,
            "parent": {name: metrics.to_dict() for name, metrics in self.parent.items()},
            "child": {name: metrics.to_dict() for name, metrics in self.child.items()},
        }


def accept_edit(
    parent_metrics: dict[str, Metrics],
    child_metrics: dict[str, Metrics],
    parent_errors: list[Diagnosis],
    child_errors: list[Diagnosis],
    *,
    gate_split: str = "val",
    mp_split: str | None = None,
    require_mp: bool = False,
    holdout_margin: float = 1e-9,
    mp_epsilon: float = 1e-12,
) -> GateDecision:
    delta_in = resolved_fraction(parent_errors, child_errors)
    delta_ho = child_metrics[gate_split].risk - parent_metrics[gate_split].risk
    # ``mp_split`` defaults to the historical "test" split for backwards
    # compatibility.  Real experiments should pass a held-out validation split
    # (e.g. "iid") so the gate does not select on the reported test set.
    mp_name = mp_split or "test"
    delta_mp = child_metrics[mp_name].mp - parent_metrics[mp_name].mp

    reasons: list[str] = []
    if delta_in < 0.0:
        reasons.append("counterexample resolution decreased")
    if delta_ho > holdout_margin:
        reasons.append("held-out risk degraded")
    if delta_mp < -mp_epsilon:
        reasons.append("metaproductivity decreased")
    if require_mp and delta_mp <= mp_epsilon:
        reasons.append("no metaproductivity improvement")
    strict = (delta_in > 0.0) or (delta_ho < -holdout_margin) or (delta_mp > mp_epsilon)
    if not strict:
        reasons.append("no strict improvement")

    accepted = not reasons
    return GateDecision(
        accepted=accepted,
        delta_in=delta_in,
        delta_ho=delta_ho,
        delta_mp=delta_mp,
        reason="accepted" if accepted else "; ".join(reasons),
        gate_split=gate_split,
        parent=parent_metrics,
        child=child_metrics,
    )
