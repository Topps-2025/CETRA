"""LLM/heuristic proposal of DecompIR programs.

The LLM sees only fit-split examples (raw value + action).  It proposes a
factor/extractor DSL, which is compiled by ``decomp_program.compile_program``.
It never writes labels or rules.  A heuristic proposer is the no-API fallback.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from cetra.rsi.types import CaseRecord

from .decomp_program import (
    CompileError,
    DecompositionProgram,
    FactorSpec,
    client_available,
    compile_program,
    validate_program,
)

_COMMON_FACTORS = (
    FactorSpec(
        "task_type",
        extractor="llm",
        config={
            "instruction": "the task family: speech, qa, report, coding, or other",
            "allowed_values": ["speech", "qa", "report", "coding", "other"],
        },
        description="task family",
    ),
    FactorSpec(
        "missing_slots",
        extractor="llm",
        config={
            "instruction": "whether required task slots (length, audience, format) are missing",
            "allowed_values": ["yes", "no", "unknown"],
        },
        description="slot completeness",
    ),
    FactorSpec(
        "user_feedback",
        extractor="llm",
        config={
            "instruction": "user feedback on the previous step: none, skip, revise, or other",
            "allowed_values": ["none", "skip", "revise", "other", "unknown"],
        },
        description="feedback signal",
    ),
    FactorSpec(
        "has_enough_context",
        extractor="llm",
        config={
            "instruction": "whether the current context is sufficient to act",
            "allowed_values": ["yes", "no", "unknown"],
        },
        description="context sufficiency",
    ),
    FactorSpec(
        "needs_search",
        extractor="llm",
        config={
            "instruction": "whether external search/research is still needed",
            "allowed_values": ["yes", "no", "unknown"],
        },
        description="retrieval need",
    ),
    FactorSpec(
        "output_format",
        extractor="llm",
        config={
            "instruction": "the requested output: speech, answer, report, patch, or other",
            "allowed_values": ["speech", "answer", "report", "patch", "other", "unknown"],
        },
        description="output contract",
    ),
)

_KEY_VALUE = re.compile(r"(?:^|[;|,\n\s])\s*([a-z][a-z0-9_]{0,30})\s*[=:]")


@dataclass
class LLMDecomposer:
    client: Any
    max_factors: int = 8
    max_examples: int = 20
    model_label: str = "deepseek-flash"
    last_error: str = ""

    def _raw(self, case: CaseRecord, source_field: str) -> str:
        if source_field in case.factors:
            return str(case.factors[source_field])
        if source_field == "narrative":
            return case.narrative
        return ""

    def propose(
        self, cases: list[CaseRecord], *, source_field: str = "narrative"
    ) -> DecompositionProgram | None:
        if not cases or not client_available(self.client):
            return None
        examples = [
            {
                "case_id": case.case_id,
                "raw": self._raw(case, source_field)[:1200],
                "action": case.action,
            }
            for case in cases[: self.max_examples]
        ]
        system = (
            "You design decomposition programs for agent memory. "
            "You propose factors and typed extractors, not labels. Return JSON only."
        )
        user = f"""Raw field: {source_field}

Examples (fit split only):
{json.dumps(examples, ensure_ascii=False, indent=1)}

Extractor DSL:
- key_value: config {{"key": "name"}}; parses name=value or name: value
- regex: config {{"pattern": "...", "group": 1}}
- json_path: config {{"path": "a.b"}}
- delimiter: config {{"delimiter": "|", "index": 0}}
- token: config {{"index": 0, "sep": " "}}
- identity: copies the raw value
- llm: config {{"instruction": "...", "allowed_values": [...]}}; for natural language only

Propose 3-{self.max_factors} factors that explain the action.
Use deterministic extractors whenever the raw values are structured.
Use `source` = "{source_field}" for every factor.

Common agent-memory factors to consider (not mandatory):
- task_type: speech, qa, report, coding, other
- missing_slots: yes, no
- user_feedback: none, skip, revise, other
- has_enough_context: yes, no
- needs_search: yes, no
- output_format: speech, answer, report, patch, other

Prefer one of these when it explains the labels; add a new factor only when
the common vocabulary cannot separate the actions.  A revision request is
`user_feedback=revise`, not `skip`.

Return JSON:
{{
  "rationale": "why these factors are decision-sufficient",
  "factors": [
    {{
      "name": "snake_case_name",
      "source": "{source_field}",
      "extractor": "key_value|regex|json_path|delimiter|token|identity|llm",
      "config": {{}},
      "description": "what it means",
      "required": true
    }}
  ]
}}

Rules:
1. Do not propose a factor named action, kind, actor, content, or timestamp.
2. Prefer factors that are present in every example; avoid identifiers.
3. Do not copy the action label into a factor.
4. Keep the program small; fewer factors is better if they explain the action.
"""
        feedback = ""
        for _ in range(2):
            attempt_user = user + feedback
            try:
                payload = self.client.chat_json(
                    system, attempt_user, max_tokens=1800
                )
                program = compile_program(
                    payload,
                    source=f"llm:{self.model_label}",
                    max_factors=self.max_factors,
                    default_source=source_field,
                )
                errors = validate_program(program, cases, min_coverage=0.5)
                if errors:
                    self.last_error = "; ".join(errors)[:300]
                    feedback = (
                        "\n\nPrevious attempt was rejected by the compiler:\n"
                        f"{self.last_error}\nReturn a corrected program. "
                        "A required factor must be extractable from most examples; "
                        "use an `llm` extractor with allowed_values for natural language."
                    )
                    continue
                return program
            except (CompileError, RuntimeError, ValueError, KeyError) as exc:
                self.last_error = str(exc)[:300]
                feedback = (
                    "\n\nPrevious attempt was rejected by the compiler:\n"
                    f"{self.last_error}\nReturn a corrected program."
                )
        return None

    def revise(
        self,
        current: DecompositionProgram,
        cases: list[CaseRecord],
        feedback: list[dict[str, Any]],
        *,
        source_field: str = "narrative",
        metrics: dict[str, Any] | None = None,
    ) -> DecompositionProgram | None:
        """Revise a program after SE found counterexamples.

        The feedback comes from the evaluator, not from a human prompt.  This
        is the LLM half of the co-evolution loop: propose -> evaluate -> show
        errors -> revise -> gate.
        """

        if not feedback or not client_available(self.client):
            return None
        system = (
            "You repair decomposition programs for agent memory. "
            "You see a program and the cases it still predicts wrongly. "
            "Propose a revised typed program. Return JSON only."
        )
        user = f"""Current program:
{json.dumps(current.to_dict(), ensure_ascii=False, indent=1)}

Current metrics (need = (decision gain + 0.5*compression) / cost):
{json.dumps(metrics or {}, ensure_ascii=False, indent=1)}

Counterexamples (dev split only; predicted != action):
{json.dumps(feedback[:8], ensure_ascii=False, indent=1)}

Raw field: {source_field}

Extractor DSL:
- key_value: config {{"key": "name"}}
- regex: config {{"pattern": "...", "group": 1}}
- json_path: config {{"path": "a.b"}}
- delimiter: config {{"delimiter": "|", "index": 0}}
- token: config {{"index": 0, "sep": " "}}
- identity: copies the raw value
- llm: config {{"instruction": "...", "allowed_values": [...]}}

Return the FULL revised program:
{{
  "rationale": "what was missing and why the revision fixes it",
  "factors": [
    {{
      "name": "snake_case_name",
      "source": "{source_field}",
      "extractor": "...",
      "config": {{}},
      "description": "what it means",
      "required": true
    }}
  ]
}}

Rules:
1. Keep factors that are still useful; add/replace only what the errors need.
2. A factor such as user_feedback, revision_request, or clarification_waived
   may be needed when the same context leads to replan/proceed instead of the
   default action.
3. Do not use action/kind/actor/content/timestamp as a factor.
4. 3-{self.max_factors} factors; return JSON only.
"""
        feedback = ""
        for _ in range(2):
            attempt_user = user + feedback
            try:
                payload = self.client.chat_json(
                    system, attempt_user, max_tokens=2000
                )
                revised = compile_program(
                    payload,
                    source=f"llm:{self.model_label}:revise",
                    max_factors=self.max_factors,
                    default_source=source_field,
                )
                errors = validate_program(revised, cases, min_coverage=0.5)
                if errors:
                    self.last_error = "; ".join(errors)[:300]
                    feedback = (
                        "\n\nPrevious revision was rejected by the compiler:\n"
                        f"{self.last_error}\nReturn a corrected program."
                    )
                    continue
                revised.version = current.version + 1
                revised.parent = current.program_id
                revised.metadata["edit"] = "REVISE_PROGRAM"
                return revised
            except (CompileError, RuntimeError, ValueError, KeyError) as exc:
                self.last_error = str(exc)[:300]
                feedback = (
                    "\n\nPrevious revision was rejected by the compiler:\n"
                    f"{self.last_error}\nReturn a corrected program."
                )
        return None


@dataclass
class HeuristicDecomposer:
    max_factors: int = 8

    def _raw(self, case: CaseRecord, source_field: str) -> str:
        if source_field in case.factors:
            return str(case.factors[source_field])
        if source_field == "narrative":
            return case.narrative
        return ""

    def propose_common(
        self, cases: list[CaseRecord], *, source_field: str = "narrative"
    ) -> DecompositionProgram | None:
        """A small prior over agent-memory factors (fallback hypothesis)."""

        if not cases:
            return None
        factors = [
            FactorSpec(
                name=spec.name,
                source=source_field,
                extractor=spec.extractor,
                config=dict(spec.config),
                description=spec.description,
                required=spec.required,
                cost=1.0 if spec.extractor == "llm" else 0.1,
            )
            for spec in _COMMON_FACTORS[: self.max_factors]
        ]
        return compile_program(
            {"factors": [spec.to_dict() for spec in factors]},
            source="heuristic_common",
            max_factors=self.max_factors,
            default_source=source_field,
        )

    def propose(
        self, cases: list[CaseRecord], *, source_field: str = "narrative"
    ) -> DecompositionProgram | None:
        raws = [self._raw(case, source_field) for case in cases if self._raw(case, source_field)]
        if not raws:
            return None

        keys: list[str] = []
        for raw in raws:
            for key in _KEY_VALUE.findall(raw):
                if key not in keys:
                    keys.append(key)
        factors: list[FactorSpec] = []
        for key in keys:
            if len(factors) >= self.max_factors:
                break
            if key in {"action", "kind", "actor", "content", "timestamp"}:
                continue
            factors.append(
                FactorSpec(
                    name=key,
                    source=source_field,
                    extractor="key_value",
                    config={"key": key},
                    description=f"key={key} parsed from {source_field}",
                )
            )
        if factors:
            return compile_program(
                {"factors": [spec.to_dict() for spec in factors]},
                source="heuristic",
                max_factors=self.max_factors,
                default_source=source_field,
            )

        # JSON values: top-level scalar keys.
        json_keys: list[str] = []
        for raw in raws:
            text = raw.strip()
            if not (text.startswith("{") and text.endswith("}")):
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                for key, value in payload.items():
                    if isinstance(value, (str, int, float, bool)) and key not in json_keys:
                        json_keys.append(str(key))
        for key in json_keys[: self.max_factors]:
            factors.append(
                FactorSpec(
                    name=key,
                    source=source_field,
                    extractor="json_path",
                    config={"path": key},
                    description=f"json field {key}",
                )
            )
        if factors:
            return compile_program(
                {"factors": [spec.to_dict() for spec in factors]},
                source="heuristic",
                max_factors=self.max_factors,
                default_source=source_field,
            )

        # Command-like values: first token is the tool, the rest is arguments.
        if all(re.match(r"^\S+\s+\S", raw.strip()) for raw in raws):
            return compile_program(
                {
                    "factors": [
                        FactorSpec(
                            name="tool",
                            source=source_field,
                            extractor="token",
                            config={"index": 0},
                            description="first token of the command",
                        ).to_dict(),
                        FactorSpec(
                            name="arguments",
                            source=source_field,
                            extractor="regex",
                            config={"pattern": r"^\S+\s+(.*)$", "group": 1},
                            description="arguments after the tool",
                        ).to_dict(),
                    ]
                },
                source="heuristic",
                max_factors=self.max_factors,
                default_source=source_field,
            )

        # Natural language: a small, domain-neutral factor prior.  The LLM
        # proposer is preferred; this is only the no-API fallback.
        return self.propose_common(cases, source_field=source_field)


def propose_candidates(
    cases: list[CaseRecord],
    *,
    source_field: str,
    client: Any | None = None,
    max_factors: int = 8,
    log: list[dict[str, Any]] | None = None,
) -> list[tuple[str, DecompositionProgram]]:
    """Return deduplicated proposal candidates: LLM first, heuristic fallback."""

    candidates: list[tuple[str, DecompositionProgram]] = []
    if client_available(client):
        llm_decomposer = LLMDecomposer(client, max_factors=max_factors)
        program = llm_decomposer.propose(cases, source_field=source_field)
        if program is not None:
            candidates.append(("llm", program))
        elif log is not None:
            log.append(
                {"name": "llm", "error": llm_decomposer.last_error or "no proposal"}
            )
    heuristic_decomposer = HeuristicDecomposer(max_factors=max_factors)
    heuristic = heuristic_decomposer.propose(cases, source_field=source_field)
    if heuristic is not None:
        candidates.append(("heuristic", heuristic))
    if source_field in {"narrative", "context", "content"}:
        common = heuristic_decomposer.propose_common(
            cases, source_field=source_field
        )
        if common is not None:
            candidates.append(("heuristic_common", common))
    seen = {program.program_id for _, program in candidates}
    unique: list[tuple[str, DecompositionProgram]] = []
    for name, program in candidates:
        if program.program_id in seen:
            seen.remove(program.program_id)
            unique.append((name, program))
    return unique
