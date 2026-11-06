"""Rule representation, validation, refinement, and symbolic evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

from .cases import ACTIONS, FACTOR_VALUES, Case, ground_truth_action


@dataclass
class Rule:
    condition: dict[str, str]
    action: str
    effect: str = "success"
    scope: str = "all"
    evidence_case_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0
    precision: float | None = None
    coverage: int | None = None
    priority: float = 0.0

    def matches(self, factors: dict[str, str]) -> bool:
        return all(factors.get(key) == value for key, value in self.condition.items())

    def specificity(self) -> int:
        return len(self.condition)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        factor_values: dict[str, tuple[str, ...]] | None = None,
    ) -> "Rule":
        vocabulary = factor_values or FACTOR_VALUES
        condition = {
            str(key): str(value)
            for key, value in (payload.get("condition") or {}).items()
            if key in vocabulary and str(value) in vocabulary[key]
        }
        action = str(payload.get("action", "")).strip()
        if action not in ACTIONS:
            raise ValueError(f"invalid action: {action!r}")
        evidence = payload.get("evidence_case_ids") or []
        if not isinstance(evidence, list):
            evidence = []
        try:
            confidence = float(payload.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        try:
            priority = float(payload.get("priority", 0.0))
        except (TypeError, ValueError):
            priority = 0.0
        precision_value = payload.get("precision")
        coverage_value = payload.get("coverage")
        try:
            precision = float(precision_value) if precision_value is not None else None
        except (TypeError, ValueError):
            precision = None
        try:
            coverage = int(coverage_value) if coverage_value is not None else None
        except (TypeError, ValueError):
            coverage = None
        return cls(
            condition=condition,
            action=action,
            effect=str(payload.get("effect", "success")),
            scope=str(payload.get("scope", "all")),
            evidence_case_ids=[str(x) for x in evidence],
            confidence=max(0.0, min(1.0, confidence)),
            precision=precision,
            coverage=coverage,
            priority=priority,
        )


def _score_condition(
    condition: dict[str, str], action: str, cases: Iterable[Case]
) -> tuple[int, float]:
    covered = [case for case in cases if all(case.factors.get(k) == v for k, v in condition.items())]
    if not covered:
        return 0, 0.0
    correct = sum(1 for case in covered if case.action == action)
    return len(covered), correct / len(covered)


class RuleSet:
    def __init__(
        self,
        rules: Iterable[Rule] | None = None,
        factor_values: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.rules: list[Rule] = list(rules or [])
        self.factor_values: dict[str, tuple[str, ...]] = dict(factor_values or FACTOR_VALUES)
        self.factor_names: tuple[str, ...] = tuple(self.factor_values)

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)

    def to_dict(self) -> list[dict[str, Any]]:
        return [rule.to_dict() for rule in self.rules]

    @classmethod
    def from_dict(
        cls,
        payload: list[dict[str, Any]],
        factor_values: dict[str, tuple[str, ...]] | None = None,
    ) -> "RuleSet":
        rules: list[Rule] = []
        for item in payload:
            try:
                rules.append(Rule.from_dict(item, factor_values=factor_values))
            except ValueError:
                continue
        return cls(rules, factor_values=factor_values)

    # ------------------------------------------------------------------
    # Basic utilities
    # ------------------------------------------------------------------
    def dedupe(self) -> None:
        merged: dict[tuple[tuple[tuple[str, str], ...], str], Rule] = {}
        for rule in self.rules:
            key = (tuple(sorted(rule.condition.items())), rule.action)
            if key not in merged:
                merged[key] = rule
            else:
                existing = merged[key]
                existing.evidence_case_ids = sorted(
                    set(existing.evidence_case_ids) | set(rule.evidence_case_ids)
                )
                existing.confidence = max(existing.confidence, rule.confidence)
        self.rules = list(merged.values())

    def score(self, cases: list[Case]) -> None:
        for rule in self.rules:
            coverage, precision = _score_condition(rule.condition, rule.action, cases)
            rule.coverage = coverage
            rule.precision = precision

    def learn_priorities(self, cases: list[Case]) -> None:
        """Learn which rule should override which in a conflict.

        The base priority is precision * coverage.  For every pair of rules
        with different actions that match a common training case, the rule
        whose action agrees with the expert receives a pairwise bonus.  This
        lets a short override rule (e.g. ``permission=none``) outrank a more
        specific rule that also matches the case.
        """
        self.score(cases)
        for rule in self.rules:
            rule.priority = (rule.precision or 0.0) * max(rule.coverage or 0, 1)
        for index, first in enumerate(self.rules):
            for second in self.rules[index + 1 :]:
                if first.action == second.action:
                    continue
                common = [
                    case
                    for case in cases
                    if first.matches(case.factors) and second.matches(case.factors)
                ]
                if not common:
                    continue
                first_wins = sum(1 for case in common if case.action == first.action)
                second_wins = sum(1 for case in common if case.action == second.action)
                delta = 0.1 * (first_wins - second_wins)
                first.priority += delta
                second.priority -= delta

    # ------------------------------------------------------------------
    # Refinement: split over-general rules, merge redundant conditions
    # ------------------------------------------------------------------
    def refine(
        self, cases: list[Case], min_precision: float = 0.8, max_depth: int = 3
    ) -> dict[str, Any]:
        """Split over-general rules and prune redundant conditions.

        This is a supervised validation step: it uses the expert action on the
        training cases.  It plays the role of counterexample-guided refinement:
        a rule whose covered cases disagree with its action is split by the
        factor that best separates the disagreement.  A split can produce
        several rules, so a general rule is not simply replaced by one branch.
        """
        before = len(self.rules)
        refined: list[Rule] = []
        for rule in self.rules:
            refined.extend(self._split_rule(rule, cases, min_precision, depth=0, max_depth=max_depth))
        self.rules = refined
        self.dedupe()
        self.learn_priorities(cases)
        return {
            "rules_before": before,
            "rules_after": len(self.rules),
            "net_change": len(self.rules) - before,
        }

    @staticmethod
    def _copy_rule(rule: Rule) -> Rule:
        return Rule(
            condition=dict(rule.condition),
            action=rule.action,
            effect=rule.effect,
            scope=rule.scope,
            evidence_case_ids=list(rule.evidence_case_ids),
            confidence=rule.confidence,
            priority=rule.priority,
        )

    @staticmethod
    def _prune_conditions(rule: Rule, cases: list[Case], min_precision: float) -> None:
        changed = True
        while changed and rule.condition:
            changed = False
            current_cov, _ = _score_condition(rule.condition, rule.action, cases)
            for factor in list(rule.condition):
                candidate = dict(rule.condition)
                del candidate[factor]
                cand_cov, cand_prec = _score_condition(candidate, rule.action, cases)
                if cand_cov > current_cov and cand_prec >= min_precision:
                    rule.condition = candidate
                    changed = True
                    break

    def _split_rule(
        self,
        rule: Rule,
        cases: list[Case],
        min_precision: float,
        depth: int,
        max_depth: int,
    ) -> list[Rule]:
        working = self._copy_rule(rule)
        coverage, precision = _score_condition(working.condition, working.action, cases)
        if coverage == 0:
            return []
        if precision >= min_precision:
            self._prune_conditions(working, cases, min_precision)
            coverage, precision = _score_condition(working.condition, working.action, cases)
            if coverage > 0 and precision >= min_precision:
                working.coverage = coverage
                working.precision = precision
                return [working]
            return []
        if depth >= max_depth:
            return []

        # Pick the factor whose value-specific branches explain the most cases
        # at the required precision.  Split into one child per value, so the
        # residual cases are not silently discarded.
        best_factor: str | None = None
        best_score: tuple[int, float, int] = (-1, -1.0, -1)
        best_separation = -1.0
        for factor in self.factor_names:
            if factor in working.condition:
                continue
            explained = 0
            separation = 0.0
            total = 0
            for value in self.factor_values[factor]:
                candidate = {**working.condition, factor: value}
                cand_cov, cand_prec = _score_condition(candidate, working.action, cases)
                total += cand_cov
                separation += cand_cov * max(cand_prec - 0.5, 0.0)
                if cand_cov and cand_prec >= min_precision:
                    explained += cand_cov
            score = (explained, separation, total)
            if score > best_score:
                best_score = score
                best_factor = factor
                best_separation = separation
        if best_factor is None or best_separation <= 0:
            return []

        children: list[Rule] = []
        for value in self.factor_values[best_factor]:
            child = self._copy_rule(working)
            child.condition[best_factor] = value
            children.extend(
                self._split_rule(child, cases, min_precision, depth=depth + 1, max_depth=max_depth)
            )
        return children

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, factors: dict[str, str]) -> tuple[str | None, list[Rule], bool]:
        matched = [rule for rule in self.rules if rule.matches(factors)]
        if not matched:
            return None, [], False
        # Learned priority first; ties break on specificity, then precision,
        # coverage, and confidence.
        matched.sort(
            key=lambda r: (
                r.priority,
                r.specificity(),
                r.precision if r.precision is not None else 0.0,
                r.coverage if r.coverage is not None else 0,
                r.confidence,
            ),
            reverse=True,
        )
        best = matched[0]
        top_priority = best.priority
        top_rules = [r for r in matched if r.priority == top_priority]
        actions = {r.action for r in top_rules}
        conflict = len(actions) > 1
        if conflict:
            # Choose the action with the strongest supporting rule.
            best = max(
                top_rules,
                key=lambda r: (
                    r.specificity(),
                    (r.precision or 0.0) * (r.coverage or 0),
                    r.confidence,
                ),
            )
        return best.action, matched, conflict

    def predict_many(self, cases: Iterable[Case]) -> list[str | None]:
        return [self.predict(case.factors)[0] for case in cases]

    # ------------------------------------------------------------------
    # Symbolic evaluation
    # ------------------------------------------------------------------
    def evaluate_symbolic(
        self,
        all_combinations: list[dict[str, str]],
        cf_pairs: list[tuple[Case, Case]] | None = None,
        ground_truth_fn: Callable[[dict[str, str]], str] | None = None,
    ) -> dict[str, Any]:
        gt_fn = ground_truth_fn or ground_truth_action
        predictions = []
        conflicts = 0
        covered = 0
        for factors in all_combinations:
            action, _, conflict = self.predict(factors)
            predictions.append(action)
            if action is not None:
                covered += 1
                conflicts += int(conflict)
        correct = sum(
            1
            for factors, pred in zip(all_combinations, predictions)
            if pred is not None and pred == gt_fn(factors)
        )
        total = len(all_combinations)
        covered_correct = sum(
            1
            for factors, pred in zip(all_combinations, predictions)
            if pred is not None and pred == gt_fn(factors)
        )
        # Factor sensitivity: does the rule set react to a factor exactly when the
        # ground truth does?
        tp = fp = fn = tn = 0
        for factor in self.factor_names:
            for factors in all_combinations:
                pred = self.predict(factors)[0]
                gt = gt_fn(factors)
                pred_changes = any(
                    self.predict({**factors, factor: value})[0] != pred
                    for value in self.factor_values[factor]
                    if value != factors[factor]
                )
                gt_changes = any(
                    gt_fn({**factors, factor: value}) != gt
                    for value in self.factor_values[factor]
                    if value != factors[factor]
                )
                if pred_changes and gt_changes:
                    tp += 1
                elif pred_changes and not gt_changes:
                    fp += 1
                elif not pred_changes and gt_changes:
                    fn += 1
                else:
                    tn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        cf_consistency = None
        if cf_pairs:
            consistent = 0
            sensitivity = 0
            invariance = 0
            n_sensitivity = 0
            n_invariance = 0
            for base, perturbed in cf_pairs:
                base_pred = self.predict(base.factors)[0]
                pert_pred = self.predict(perturbed.factors)[0]
                gt_changes = base.action != perturbed.action
                pred_changes = base_pred != pert_pred
                consistent += int(gt_changes == pred_changes)
                if gt_changes:
                    n_sensitivity += 1
                    sensitivity += int(pred_changes)
                else:
                    n_invariance += 1
                    invariance += int(not pred_changes)
            cf_consistency = {
                "pair_consistency": consistent / len(cf_pairs),
                "sensitivity": sensitivity / n_sensitivity if n_sensitivity else None,
                "invariance": invariance / n_invariance if n_invariance else None,
                "n_pairs": len(cf_pairs),
            }

        return {
            "num_rules": len(self.rules),
            "num_rules_no_condition": sum(1 for r in self.rules if not r.condition),
            "avg_condition_length": (
                sum(len(r.condition) for r in self.rules) / len(self.rules) if self.rules else 0.0
            ),
            "avg_rule_precision": (
                sum(r.precision or 0.0 for r in self.rules) / len(self.rules) if self.rules else 0.0
            ),
            "avg_rule_coverage": (
                sum(r.coverage or 0 for r in self.rules) / len(self.rules) if self.rules else 0.0
            ),
            "coverage": covered / total if total else 0.0,
            "accuracy_all": correct / total if total else 0.0,
            "accuracy_on_covered": covered_correct / covered if covered else 0.0,
            "conflict_rate": conflicts / covered if covered else 0.0,
            "factor_sensitivity": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            },
            "counterfactual": cf_consistency,
        }
