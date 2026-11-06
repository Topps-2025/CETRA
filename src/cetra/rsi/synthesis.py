"""Open operator synthesis for the CETRA-RSI harness.

The seed harness has a fixed operator registry.  A genuinely open RSI loop must
be able to add an operator that was not in that registry.  This module
implements a small, typed, enumerative synthesizer:

- a selector language over equality and numeric-threshold atoms;
- a rule program that turns a selector into a scoped rule;
- a ``ProgramOperator`` that executes a program inside the harness;
- a synthesis gate that type-checks the program, checks that it is
  non-destructive, and evaluates it with the same triple gate as a hand-written
  edit.

The grammar is deliberately small.  That is a feature for a controlled
mechanism test: the synthesizer's reach is explicit, and the report can state
exactly which operators it can and cannot invent.  The interface is the same
one an LLM-backed synthesizer would use, so a model-backed implementation can
replace ``enumerate_candidates`` without changing the loop.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from .gate import GateDecision, accept_edit
from .harness import diagnose, evaluate, inner_refine
from .operators import Operator
from .types import (
    BuildState,
    CaseRecord,
    Diagnosis,
    Harness,
    Metrics,
    OperatorSpec,
    Representation,
    Rule,
)


def _as_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Selector and program representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """A typed selector atom: ``concept == value`` or ``int(concept) >= value``."""

    concept: str
    op: str  # "eq" | "ge"
    value: str | int

    def matches(self, factors: dict[str, str]) -> bool:
        if self.op == "eq":
            return factors.get(self.concept) == self.value
        if self.op == "ge":
            observed = _as_int(factors.get(self.concept))
            return observed is not None and observed >= int(self.value)
        raise ValueError(f"unknown selector op: {self.op}")

    def as_condition(self) -> tuple[str, str] | None:
        return (self.concept, str(self.value)) if self.op == "eq" else None

    def as_threshold(self) -> tuple[str, int] | None:
        return (self.concept, int(self.value)) if self.op == "ge" else None

    def key(self) -> tuple[str, str, str]:
        return (self.concept, self.op, str(self.value))

    def to_dict(self) -> dict[str, Any]:
        return {"concept": self.concept, "op": self.op, "value": self.value}


@dataclass(frozen=True)
class RuleProgram:
    """A selector plus the action it should support."""

    conditions: tuple[Condition, ...]
    action: str
    scope: tuple[str, ...] = ()
    support: int = 0
    precision: float = 1.0
    score: float = 0.0
    resolved_ids: tuple[str, ...] = ()

    def matches(self, factors: dict[str, str]) -> bool:
        return all(condition.matches(factors) for condition in self.conditions)

    def rule(self) -> Rule:
        return Rule(
            conditions=tuple(
                condition.as_condition()
                for condition in self.conditions
                if condition.as_condition() is not None
            ),
            action=self.action,
            scope=self.scope,
            support=self.support,
            precision=self.precision,
            thresholds=tuple(
                condition.as_threshold()
                for condition in self.conditions
                if condition.as_threshold() is not None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "conditions": [condition.to_dict() for condition in self.conditions],
            "action": self.action,
            "scope": list(self.scope),
            "support": self.support,
            "precision": round(self.precision, 6),
        }


@dataclass
class ProgramOperator(Operator):
    """An operator whose behaviour is a synthesized rule program.

    The operator is intentionally non-destructive: it appends rules but never
    deletes or rewrites the parent representation.  That makes rollback and
    regression checking simple.
    """

    programs: tuple[RuleProgram, ...]
    spec: OperatorSpec = field(
        default_factory=lambda: OperatorSpec(
            name="synthesized",
            inputs=("case", "state"),
            outputs=("rule",),
            description="synthesized scoped rule program",
        )
    )
    source: str = ""

    def run(self, state: BuildState, cases: list[CaseRecord], params: dict[str, Any]) -> BuildState:
        min_support = int(params.get("synth_min_support", params.get("min_support", 2)))
        min_precision = float(params.get("synth_min_precision", 0.9))
        existing = {rule.key() for rule in state.rules}
        added: list[Rule] = []
        for program in self.programs:
            subset = [case for case in cases if program.matches(case.factors)]
            if len(subset) < min_support:
                continue
            action, count = _majority([case.action for case in subset])
            precision = count / len(subset)
            if precision < min_precision:
                continue
            rule = Rule(
                conditions=tuple(
                    condition.as_condition()
                    for condition in program.conditions
                    if condition.as_condition() is not None
                ),
                action=action,
                scope=program.scope,
                support=len(subset),
                precision=precision,
                thresholds=tuple(
                    condition.as_threshold()
                    for condition in program.conditions
                    if condition.as_threshold() is not None
                ),
            )
            if rule.key() in existing:
                continue
            existing.add(rule.key())
            added.append(rule)
        state.rules = list(state.rules) + added
        state.traces["synthesized_rules"] = len(added)
        state.traces["synthesized_operator"] = self.spec.name
        return state

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "inputs": list(self.spec.inputs),
            "outputs": list(self.spec.outputs),
            "description": self.spec.description,
            "programs": [program.to_dict() for program in self.programs],
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProgramOperator":
        programs = tuple(
            RuleProgram(
                conditions=tuple(
                    Condition(
                        concept=condition["concept"],
                        op=condition["op"],
                        value=condition["value"],
                    )
                    for condition in item["conditions"]
                ),
                action=item["action"],
                scope=tuple(item.get("scope", ())),
                support=int(item.get("support", 0)),
                precision=float(item.get("precision", 1.0)),
            )
            for item in payload["programs"]
        )
        return cls(
            programs=programs,
            spec=OperatorSpec(
                name=payload["name"],
                inputs=tuple(payload.get("inputs", ("case", "state"))),
                outputs=tuple(payload.get("outputs", ("rule",))),
                description=payload.get("description", "synthesized scoped rule program"),
            ),
            source=payload.get("source", ""),
        )


def _majority(actions: list[str]) -> tuple[str, int]:
    counts: dict[str, int] = {}
    for action in actions:
        counts[action] = counts.get(action, 0) + 1
    return max(counts.items(), key=lambda item: (item[1], item[0]))


# ---------------------------------------------------------------------------
# Synthesis context and result
# ---------------------------------------------------------------------------


@dataclass
class SynthesisContext:
    harness: Harness
    representation: Representation
    diagnosis: list[Diagnosis]
    fit_cases: list[CaseRecord]
    val_cases: list[CaseRecord]
    iid_cases: list[CaseRecord]
    test_cases: list[CaseRecord]
    parent_metrics: dict[str, Metrics]
    parent_errors: list[Diagnosis]
    seed_risk: float
    version: str
    gate_split: str = "iid"
    mp_split: str = "iid"
    require_mp: bool = False
    scope_hint: str = "ordinary"  # "ordinary" | "fallback"


@dataclass
class SynthesisResult:
    operator_name: str
    program: ProgramOperator
    source: str
    score: float
    provenance: str
    child: Harness | None = None
    child_rep: Representation | None = None
    metrics: dict[str, Metrics] = field(default_factory=dict)
    decision: GateDecision | None = None
    candidates_considered: int = 0

    @property
    def accepted(self) -> bool:
        return bool(self.decision and self.decision.accepted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_name": self.operator_name,
            "program": self.program.to_dict(),
            "source": self.source,
            "score": round(self.score, 6),
            "provenance": self.provenance,
            "accepted": self.accepted,
            "candidates_considered": self.candidates_considered,
            "decision": self.decision.to_dict() if self.decision else None,
            "metrics": {name: metric.to_dict() for name, metric in self.metrics.items()},
        }


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


class OperatorSynthesizer:
    """Enumerative synthesizer over a typed selector grammar."""

    def __init__(
        self,
        *,
        max_arity: int = 2,
        beam: int = 10,
        max_rules: int = 8,
        min_support: int = 2,
        min_precision: float = 0.9,
        primitives: Iterable[str] = ("eq", "threshold", "fallback"),
        library: dict[str, ProgramOperator] | None = None,
    ) -> None:
        self.max_arity = max_arity
        self.beam = beam
        self.max_rules = max_rules
        self.min_support = min_support
        self.min_precision = min_precision
        self.primitives = set(primitives)
        self.library: dict[str, ProgramOperator] = dict(library or {})

    def clone(self, **overrides: Any) -> "OperatorSynthesizer":
        params = {
            "max_arity": self.max_arity,
            "beam": self.beam,
            "max_rules": self.max_rules,
            "min_support": self.min_support,
            "min_precision": self.min_precision,
            "primitives": tuple(self.primitives),
            "library": self.library,
        }
        params.update(overrides)
        return OperatorSynthesizer(**params)

    # -- candidate generation -------------------------------------------------

    def _atoms(self, cases: list[CaseRecord], concepts: tuple[str, ...]) -> list[Condition]:
        atoms: list[Condition] = []
        for concept in concepts:
            values = sorted({case.factors[concept] for case in cases if concept in case.factors})
            for value in values:
                atoms.append(Condition(concept=concept, op="eq", value=value))
            if "threshold" in self.primitives:
                numeric = sorted(
                    {
                        int(case.factors[concept])
                        for case in cases
                        if concept in case.factors and _as_int(case.factors[concept]) is not None
                    }
                )
                # ``>= min`` is vacuous; start at the second distinct value.
                for value in numeric[1:]:
                    atoms.append(Condition(concept=concept, op="ge", value=value))
        return atoms

    def _selectors(self, atoms: list[Condition]) -> list[tuple[Condition, ...]]:
        selectors: list[tuple[Condition, ...]] = []
        seen: set[tuple[tuple[str, str, str], ...]] = set()
        for arity in range(1, self.max_arity + 1):
            for combo in combinations(atoms, arity):
                concepts = [condition.concept for condition in combo]
                if len(set(concepts)) != len(concepts):
                    continue
                key = tuple(sorted(condition.key() for condition in combo))
                if key in seen:
                    continue
                seen.add(key)
                selectors.append(combo)
        return selectors

    def enumerate_candidates(
        self,
        cases: list[CaseRecord],
        concepts: tuple[str, ...],
        diagnosis: list[Diagnosis],
        *,
        eval_cases: list[CaseRecord] | None = None,
        current_predictions: dict[str, str] | None = None,
        scope: str = "ordinary",
        max_candidates: int | None = None,
    ) -> list[RuleProgram]:
        atoms = self._atoms(cases, concepts)
        eval_cases = eval_cases or cases
        current_predictions = current_predictions or {}
        diagnosed_ids = {item.case_id for item in diagnosis}
        candidates: list[RuleProgram] = []
        seen: set[tuple] = set()

        for selector in self._selectors(atoms):
            # Single equality atoms are already expressible by the seed
            # learner; only threshold atoms and multi-condition selectors are
            # genuinely new.  Fallback selectors may be single atoms because a
            # scoped default is a new operator, not a new ordinary rule.
            has_threshold = any(condition.op == "ge" for condition in selector)
            if len(selector) == 1 and not has_threshold and scope != "fallback":
                continue
            subset = [case for case in cases if all(c.matches(case.factors) for c in selector)]
            if len(subset) < self.min_support:
                continue
            action, count = _majority([case.action for case in subset])
            precision = count / len(subset)
            if precision < self.min_precision:
                continue

            # Score on the diagnosis split, not on the fit split.  A rule wins
            # only if it changes a wrong prediction into the right one; a rule
            # that merely re-states an already-correct prediction gets no
            # credit.
            eval_subset = [
                case for case in eval_cases if all(c.matches(case.factors) for c in selector)
            ]
            if not eval_subset:
                continue
            resolves = 0
            breaks = 0
            resolved_ids: list[str] = []
            for case in eval_subset:
                predicted = current_predictions.get(case.case_id)
                if predicted is None:
                    continue
                if predicted != case.action and case.action == action:
                    resolves += 1
                    resolved_ids.append(case.case_id)
                elif predicted == case.action and case.action != action:
                    breaks += 1
            if scope != "fallback" and resolves == 0:
                continue
            diagnosed_here = sum(1 for case in eval_subset if case.case_id in diagnosed_ids)
            # Simplicity prior: fewer conditions and threshold atoms generalise
            # better than a long list of authority-specific equality rules.
            complexity = 0.4 * len(selector)
            threshold_bonus = 0.2 * sum(1 for condition in selector if condition.op == "ge")
            score = (
                8.0 * resolves
                + 2.0 * diagnosed_here
                + precision * min(len(subset), 12)
                - 6.0 * breaks
                - complexity
                + threshold_bonus
            )
            key = (
                tuple(sorted(condition.key() for condition in selector)),
                action,
                scope,
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                RuleProgram(
                    conditions=tuple(sorted(selector, key=lambda c: (c.concept, c.op, str(c.value)))),
                    action=action,
                    scope=("fallback",) if scope == "fallback" else (),
                    support=len(subset),
                    precision=precision,
                    score=score,
                    resolved_ids=tuple(resolved_ids),
                )
            )

        candidates.sort(
            key=lambda item: (item.score, item.precision, item.support),
            reverse=True,
        )
        limit = max_candidates or self.beam
        return candidates[:limit]

    def _build_programs(self, candidates: list[RuleProgram]) -> list[RuleProgram]:
        """Greedily keep rules that resolve *new* diagnosed cases.

        This is a tiny set-cover step.  It avoids emitting eight authority-
        specific variants of the same rule when one general rule covers the
        diagnoses.
        """

        selected: list[RuleProgram] = []
        seen: set[tuple] = set()
        covered: set[str] = set()
        for candidate in candidates:
            key = (
                tuple(sorted(condition.key() for condition in candidate.conditions)),
                candidate.action,
                candidate.scope,
            )
            if key in seen:
                continue
            new_coverage = set(candidate.resolved_ids) - covered
            if covered and not new_coverage:
                continue
            seen.add(key)
            selected.append(candidate)
            covered.update(candidate.resolved_ids)
            if len(selected) >= self.max_rules:
                break
        return selected

    def _evaluate_program(self, ctx: SynthesisContext, programs: list[RuleProgram]) -> SynthesisResult:
        digest = hashlib.sha1(
            json.dumps([program.to_dict() for program in programs], sort_keys=True).encode("utf-8")
        ).hexdigest()[:10]
        name = f"synth_{digest}"
        if name in ctx.harness.operators:
            name = f"{name}_{ctx.version}"
        operator = ProgramOperator(
            programs=tuple(programs),
            spec=OperatorSpec(
                name=name,
                inputs=("case", "state"),
                outputs=("rule",),
                description="synthesized scoped rule program",
            ),
            source=program_to_source(name, programs),
        )
        child = ctx.harness.clone(version=ctx.version)
        child.operators[name] = operator
        if "predict" not in child.pipeline:
            raise ValueError("harness has no predict anchor")
        if name not in child.pipeline:
            child.pipeline.insert(child.pipeline.index("predict"), name)
        child.parent = ctx.harness.version

        child_rep, _ = inner_refine(
            child,
            ctx.fit_cases,
            ctx.val_cases,
            fit_cases=ctx.fit_cases,
            max_rounds=2,
        )
        metrics = {
            "val": evaluate(
                child, child_rep, ctx.val_cases, seed_risk=ctx.seed_risk
            ),
            "iid": evaluate(
                child, child_rep, ctx.iid_cases, seed_risk=ctx.seed_risk
            ),
            "test": evaluate(
                child, child_rep, ctx.test_cases, seed_risk=ctx.seed_risk
            ),
        }
        child_errors = diagnose(child, child_rep, ctx.val_cases)
        decision = accept_edit(
            ctx.parent_metrics,
            metrics,
            ctx.parent_errors,
            child_errors,
            gate_split=ctx.gate_split,
            mp_split=ctx.mp_split,
            require_mp=ctx.require_mp,
        )
        score = sum(program.score for program in programs)
        return SynthesisResult(
            operator_name=name,
            program=operator,
            source=operator.source,
            score=score,
            provenance=ctx.scope_hint,
            child=child,
            child_rep=child_rep,
            metrics=metrics,
            decision=decision,
            candidates_considered=len(programs),
        )

    def synthesize(self, ctx: SynthesisContext) -> SynthesisResult | None:
        """Enumerate, build, and gate a synthesized operator.

        Returns the best candidate even when the gate rejects it, so the caller
        can log the negative result.  Returns ``None`` when the grammar has no
        candidate at all.
        """

        concepts = ctx.representation.concepts or tuple(
            sorted({key for case in ctx.fit_cases for key in case.factors})
        )
        # Score on the diagnosis split plus the held-out iid split.  The
        # diagnosis split alone can be too small to contain a rare four-way
        # interaction; the iid split gives the candidate a chance to show that
        # it generalises rather than memorises the diagnosed points.
        eval_by_id = {
            case.case_id: case for case in list(ctx.val_cases) + list(ctx.iid_cases)
        }
        eval_cases = list(eval_by_id.values())
        current_predictions = {
            case.case_id: ctx.harness.predict(ctx.representation, case)
            for case in eval_cases
        }
        candidates = self.enumerate_candidates(
            ctx.fit_cases,
            tuple(concepts),
            ctx.diagnosis,
            eval_cases=eval_cases,
            current_predictions=current_predictions,
            scope=ctx.scope_hint,
        )
        if not candidates:
            return None
        programs = self._build_programs(candidates)
        if not programs:
            return None
        result = self._evaluate_program(ctx, programs)
        if result.accepted:
            self.library[result.operator_name] = result.program
        return result


# ---------------------------------------------------------------------------
# Audit / persistence helpers
# ---------------------------------------------------------------------------


def program_to_source(name: str, programs: list[RuleProgram] | tuple[RuleProgram, ...]) -> str:
    lines = [f"def {name}(state, cases, params):"]
    for index, program in enumerate(programs):
        selector = " and ".join(
            f"{condition.concept}{'==' if condition.op == 'eq' else '>='}{condition.value!r}"
            for condition in program.conditions
        )
        scope = " [fallback]" if "fallback" in program.scope else ""
        lines.append(f"    # rule {index}: if {selector}{scope}: {program.action}")
    lines.append("    # append matching rules; never delete parent rules")
    return "\n".join(lines)


def save_library(path: str | Path, library: dict[str, ProgramOperator]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: operator.to_dict() for name, operator in library.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_library(path: str | Path) -> dict[str, ProgramOperator]:
    path = Path(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: ProgramOperator.from_dict(item) for name, item in payload.items()}
