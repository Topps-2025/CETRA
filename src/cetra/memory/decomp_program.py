"""DecompIR: a typed intermediate representation for memory decomposition.

The LLM does not label data directly.  It proposes a **decomposition program**
written in a small factor/extractor DSL.  The program is compiled, validated,
executed by an interpreter, scored by decision gain/need, and can be evolved by
the SE loop.  This is the difference between "prompt engineering" and an
evolvable memory operator:

- the prompt is only the proposal channel;
- the artifact is ``DecompositionProgram`` (typed, versioned, diffable);
- deterministic extractors execute the program when possible;
- an ``llm`` extractor is one backend, constrained by the same schema.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from cetra.rsi.types import CaseRecord

MISSING = "__missing__"
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
_RESERVED = {"action", "kind", "actor", "content", "timestamp", "case_id"}
_EXTRACTORS = {
    "key_value",
    "json_path",
    "regex",
    "delimiter",
    "token",
    "identity",
    "llm",
}


class CompileError(ValueError):
    """A proposal cannot be compiled into a decomposition program."""


def client_available(client: Any | None) -> bool:
    """Accept both ``client.available()`` and a boolean ``client.available``."""

    if client is None:
        return False
    value = getattr(client, "available", False)
    return bool(value() if callable(value) else value)


@dataclass(frozen=True)
class FactorSpec:
    name: str
    source: str = "narrative"
    extractor: str = "identity"
    config: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    required: bool = True
    cost: float = 0.1

    def signature(self) -> tuple:
        return (
            self.name,
            self.source,
            self.extractor,
            json.dumps(self.config, sort_keys=True, ensure_ascii=False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "extractor": self.extractor,
            "config": dict(self.config),
            "description": self.description,
            "required": self.required,
            "cost": self.cost,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FactorSpec":
        extractor = str(payload.get("extractor", "identity")).strip()
        return cls(
            name=str(payload.get("name", "")).strip(),
            source=str(payload.get("source", "")).strip(),
            extractor=extractor,
            config=dict(payload.get("config", {}) or {}),
            description=str(payload.get("description", "")),
            required=bool(payload.get("required", True)),
            cost=1.0 if extractor == "llm" else 0.1,
        )


@dataclass
class DecompositionProgram:
    program_id: str
    factors: list[FactorSpec]
    rationale: str = ""
    source: str = "heuristic"
    version: int = 1
    parent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def factor_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.factors)

    def factor(self, name: str) -> FactorSpec | None:
        return next((spec for spec in self.factors if spec.name == name), None)

    def restrict(self, names: Iterable[str], *, rationale: str = "") -> "DecompositionProgram":
        keep = set(names)
        factors = [spec for spec in self.factors if spec.name in keep]
        return DecompositionProgram(
            program_id=_program_id(factors),
            factors=factors,
            rationale=rationale or self.rationale,
            source=self.source,
            version=self.version + 1,
            parent=self.program_id,
            metadata={**self.metadata, "edit": "SELECT_FACTORS"},
        )

    def merge(self, other: "DecompositionProgram") -> "DecompositionProgram":
        by_name = {spec.name: spec for spec in self.factors}
        for spec in other.factors:
            by_name.setdefault(spec.name, spec)
        factors = list(by_name.values())
        return DecompositionProgram(
            program_id=_program_id(factors),
            factors=factors,
            rationale=f"{self.rationale} | {other.rationale}".strip(" |"),
            source=f"{self.source}+{other.source}",
            version=max(self.version, other.version) + 1,
            parent=self.program_id,
            metadata={**self.metadata, "edit": "ADD_FACTORS"},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "version": self.version,
            "parent": self.parent,
            "source": self.source,
            "rationale": self.rationale,
            "factors": [spec.to_dict() for spec in self.factors],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DecompositionProgram":
        factors = [FactorSpec.from_dict(item) for item in payload.get("factors", [])]
        return cls(
            program_id=str(payload.get("program_id") or _program_id(factors)),
            factors=factors,
            rationale=str(payload.get("rationale", "")),
            source=str(payload.get("source", "heuristic")),
            version=int(payload.get("version", 1)),
            parent=payload.get("parent"),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


def _program_id(factors: Iterable[FactorSpec]) -> str:
    payload = json.dumps(
        [spec.signature() for spec in factors], sort_keys=True, ensure_ascii=False
    )
    return "dp:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def compile_program(
    payload: dict[str, Any] | list[dict[str, Any]],
    *,
    source: str = "llm",
    max_factors: int = 8,
    default_source: str = "narrative",
    rationale: str = "",
) -> DecompositionProgram:
    """Compile and validate an LLM/heuristic proposal into DecompIR."""

    if isinstance(payload, list):
        raw_factors = payload
        meta: dict[str, Any] = {}
    elif isinstance(payload, dict):
        raw_factors = payload.get("factors", []) or []
        meta = {
            key: payload[key]
            for key in ("expected_gain", "analysis", "model")
            if key in payload
        }
        rationale = str(payload.get("rationale", rationale))
    else:
        raise CompileError("proposal must be a JSON object or a list of factors")
    if not isinstance(raw_factors, list):
        raise CompileError("'factors' must be a list")
    if not raw_factors:
        raise CompileError("proposal has no factors")
    if len(raw_factors) > max_factors:
        raise CompileError(f"too many factors: {len(raw_factors)} > {max_factors}")

    factors: list[FactorSpec] = []
    seen: set[str] = set()
    for item in raw_factors:
        if not isinstance(item, dict):
            raise CompileError("each factor must be an object")
        spec = FactorSpec.from_dict(item)
        if not spec.name or not _NAME.match(spec.name):
            raise CompileError(f"invalid factor name: {spec.name!r}")
        if spec.name in _RESERVED:
            raise CompileError(f"reserved factor name: {spec.name}")
        if spec.name in seen:
            raise CompileError(f"duplicate factor: {spec.name}")
        if spec.extractor not in _EXTRACTORS:
            raise CompileError(f"unknown extractor for {spec.name}: {spec.extractor}")
        spec = replace(spec, source=spec.source or default_source)
        _validate_config(spec)
        seen.add(spec.name)
        factors.append(spec)
    return DecompositionProgram(
        program_id=_program_id(factors),
        factors=factors,
        rationale=rationale,
        source=source,
        metadata=meta,
    )


def _validate_config(spec: FactorSpec) -> None:
    config = spec.config
    if spec.extractor == "key_value":
        if not str(config.get("key", "")).strip():
            raise CompileError(f"key_value factor {spec.name} needs config.key")
    elif spec.extractor == "json_path":
        if not str(config.get("path", "")).strip():
            raise CompileError(f"json_path factor {spec.name} needs config.path")
    elif spec.extractor == "regex":
        pattern = str(config.get("pattern", ""))
        if not pattern:
            raise CompileError(f"regex factor {spec.name} needs config.pattern")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise CompileError(f"bad regex for {spec.name}: {exc}") from exc
        try:
            group = int(config.get("group", 1))
        except (TypeError, ValueError) as exc:
            raise CompileError(
                f"regex factor {spec.name} group must be an int"
            ) from exc
        if group < 0:
            raise CompileError(f"regex factor {spec.name} group must be >= 0")
        config["group"] = group
    elif spec.extractor == "delimiter":
        if not str(config.get("delimiter", "")):
            raise CompileError(f"delimiter factor {spec.name} needs config.delimiter")
        try:
            config["index"] = int(config.get("index", 0))
        except (TypeError, ValueError) as exc:
            raise CompileError(
                f"delimiter factor {spec.name} index must be int"
            ) from exc
    elif spec.extractor == "token":
        try:
            config["index"] = int(config.get("index", 0))
        except (TypeError, ValueError) as exc:
            raise CompileError(
                f"token factor {spec.name} index must be int"
            ) from exc
    elif spec.extractor == "llm":
        if not str(config.get("instruction", "")).strip():
            raise CompileError(f"llm factor {spec.name} needs config.instruction")
        allowed = config.get("allowed_values", [])
        if allowed and not isinstance(allowed, list):
            raise CompileError(f"llm factor {spec.name} allowed_values must be a list")


@dataclass
class ExtractionReport:
    values: dict[str, dict[str, str]] = field(default_factory=dict)
    evidence: dict[str, dict[str, tuple[str, ...]]] = field(default_factory=dict)
    coverage: dict[str, float] = field(default_factory=dict)
    grounded: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    llm_calls: int = 0

    def mean_coverage(self) -> float:
        if not self.coverage:
            return 0.0
        return sum(self.coverage.values()) / len(self.coverage)

    def mean_grounded(self) -> float:
        if not self.grounded:
            return 0.0
        return sum(self.grounded.values()) / len(self.grounded)

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage": {key: round(value, 4) for key, value in self.coverage.items()},
            "grounded": {key: round(value, 4) for key, value in self.grounded.items()},
            "mean_coverage": round(self.mean_coverage(), 4),
            "mean_grounded": round(self.mean_grounded(), 4),
            "llm_calls": self.llm_calls,
            "errors": list(self.errors),
            "sample_values": {
                case_id: dict(values)
                for case_id, values in list(self.values.items())[:5]
            },
        }


def _raw_value(case: CaseRecord, source: str) -> str:
    if source in case.factors:
        return str(case.factors[source])
    if source == "narrative":
        return case.narrative
    return ""


def _extract_deterministic(spec: FactorSpec, raw: str) -> str | None:
    if not raw:
        return None
    config = spec.config
    if spec.extractor == "identity":
        return raw
    if spec.extractor == "key_value":
        key = re.escape(str(config["key"]))
        pattern = rf"(?:^|[;|,\n\s])\s*{key}\s*[=:]\s*([^;|,\n]+)"
        match = re.search(pattern, raw)
        return match.group(1).strip() if match else None
    if spec.extractor == "json_path":
        try:
            node: Any = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        for part in str(config["path"]).split("."):
            if not part:
                continue
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return None
        if isinstance(node, (str, int, float, bool)):
            return str(node)
        return None
    if spec.extractor == "regex":
        match = re.search(str(config["pattern"]), raw, re.S)
        if not match:
            return None
        group = int(config.get("group", 1))
        if group == 0 and match.lastindex:
            # Models often say group=0 while writing a capture group; the
            # intended span is almost always the first capture.
            group = 1
        return match.group(group).strip() if group <= (match.lastindex or 0) else None
    if spec.extractor == "delimiter":
        parts = raw.split(str(config["delimiter"]))
        index = int(config.get("index", 0))
        if -len(parts) <= index < len(parts):
            return parts[index].strip()
        return None
    if spec.extractor == "token":
        sep = config.get("sep")
        parts = raw.split(sep) if sep else raw.split()
        index = int(config.get("index", 0))
        if -len(parts) <= index < len(parts):
            return parts[index].strip()
        return None
    return None


def apply_program(
    program: DecompositionProgram,
    cases: list[CaseRecord],
    *,
    client: Any | None = None,
    keep_base: bool = False,
    source_override: str | None = None,
) -> tuple[list[CaseRecord], ExtractionReport]:
    """Execute DecompIR over cases; return rebuilt cases + an extraction report."""

    report = ExtractionReport()
    if not cases:
        return [], report
    per_case: dict[str, dict[str, str]] = {case.case_id: {} for case in cases}
    evidence: dict[str, dict[str, tuple[str, ...]]] = {
        case.case_id: {} for case in cases
    }

    llm_specs = [spec for spec in program.factors if spec.extractor == "llm"]
    deterministic = [spec for spec in program.factors if spec.extractor != "llm"]

    for case in cases:
        for spec in deterministic:
            source = source_override or spec.source
            raw = _raw_value(case, source)
            value = _extract_deterministic(spec, raw)
            if value is not None and value != "":
                per_case[case.case_id][spec.name] = value
                evidence[case.case_id][spec.name] = (
                    (case.case_id,) if value in raw else ()
                )

    if llm_specs:
        if not client_available(client):
            report.errors.append("llm extractor requested but no available client")
        else:
            try:
                extracted, calls = _llm_extract(
                    client, llm_specs, cases, source_override=source_override
                )
                report.llm_calls += calls
                for case_id, values in extracted.items():
                    per_case.setdefault(case_id, {}).update(values)
                    for name, value in values.items():
                        raw = _raw_value(
                            next(c for c in cases if c.case_id == case_id),
                            source_override or next(
                                s.source for s in llm_specs if s.name == name
                            ),
                        )
                        evidence.setdefault(case_id, {})[name] = (
                            (case_id,) if value and value in raw else ()
                        )
            except Exception as exc:  # API/schema failure: keep missing values
                report.errors.append(str(exc)[:240])

    rebuilt: list[CaseRecord] = []
    for case in cases:
        factors = dict(case.factors) if keep_base else {}
        for spec in program.factors:
            value = per_case.get(case.case_id, {}).get(spec.name)
            factors[spec.name] = value if value is not None else MISSING
        rebuilt.append(replace(case, factors=factors))

    for spec in program.factors:
        present = sum(
            1
            for case in cases
            if per_case.get(case.case_id, {}).get(spec.name) not in (None, "")
        )
        grounded = sum(
            1
            for case in cases
            if evidence.get(case.case_id, {}).get(spec.name)
        )
        report.coverage[spec.name] = present / len(cases)
        report.grounded[spec.name] = grounded / len(cases)
    report.values = per_case
    report.evidence = evidence
    return rebuilt, report


def apply_programs(
    programs: list[DecompositionProgram],
    cases: list[CaseRecord],
    *,
    client: Any | None = None,
    keep_base: bool = False,
) -> tuple[list[CaseRecord], list[ExtractionReport]]:
    """Apply staged programs: stage 2 can read factors produced by stage 1.

    This is the recursive/staged form of DecompIR.  The automatic decision to
    add another stage is the job of ``NeedEstimator``; the interpreter only
    guarantees that later stages see earlier factors.
    """

    current = list(cases)
    reports: list[ExtractionReport] = []
    for index, program in enumerate(programs):
        current, report = apply_program(
            program,
            current,
            client=client,
            keep_base=keep_base or index > 0,
        )
        reports.append(report)
    return current, reports


def _llm_extract(
    client: Any,
    specs: list[FactorSpec],
    cases: list[CaseRecord],
    *,
    source_override: str | None = None,
) -> tuple[dict[str, dict[str, str]], int]:
    factor_payload = []
    for spec in specs:
        factor_payload.append(
            {
                "name": spec.name,
                "instruction": spec.config.get("instruction", ""),
                "allowed_values": spec.config.get("allowed_values", []),
            }
        )
    case_payload = []
    for case in cases:
        source = source_override or specs[0].source
        raw = _raw_value(case, source)
        case_payload.append({"case_id": case.case_id, "raw": raw[:1600]})
    system = (
        "You are the execution backend of a typed decomposition program. "
        "Extract only the requested factors. Do not invent values. Return JSON only."
    )
    user = f"""Factors to extract:
{json.dumps(factor_payload, ensure_ascii=False, indent=1)}

Cases:
{json.dumps(case_payload, ensure_ascii=False, indent=1)}

Return JSON:
{{
  "values": {{
    "<case_id>": {{"<factor_name>": "<value>"}}
  }}
}}

Rules:
1. Use exactly the factor names above.
2. If a factor is absent, use "unknown".
3. If allowed_values is non-empty, choose one of them.
4. Keep values short (<= 80 chars).
"""
    last_error: Exception | None = None
    for _ in range(2):
        try:
            payload = client.chat_json(system, user, max_tokens=2600)
            raw_values = payload.get("values", {})
            if not isinstance(raw_values, dict):
                raise ValueError("'values' must be an object")
            out: dict[str, dict[str, str]] = {}
            known_cases = {case.case_id for case in cases}
            factor_names = {spec.name for spec in specs}
            for case_id, values in raw_values.items():
                if case_id not in known_cases or not isinstance(values, dict):
                    continue
                clean: dict[str, str] = {}
                for name, value in values.items():
                    if name not in factor_names:
                        continue
                    text = str(value).strip()[:80]
                    if not text:
                        continue
                    spec = next(item for item in specs if item.name == name)
                    allowed = spec.config.get("allowed_values", [])
                    if allowed and text not in allowed:
                        text = "other"
                    clean[name] = text
                if clean:
                    out[case_id] = clean
            return out, 1
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"llm extraction failed: {last_error}")


def validate_program(
    program: DecompositionProgram,
    cases: list[CaseRecord],
    *,
    min_coverage: float = 0.5,
) -> list[str]:
    """Cheap structural validation before an expensive SE evaluation."""

    errors: list[str] = []
    _, report = apply_program(program, cases)
    for spec in program.factors:
        if spec.extractor == "llm":
            continue
        coverage = report.coverage.get(spec.name, 0.0)
        # Optional factors still need enough support to be useful; a factor
        # present in <20% of cases cannot support a rule on a small trace.
        threshold = min_coverage if spec.required else max(0.2, min_coverage * 0.5)
        if coverage < threshold:
            errors.append(f"low coverage for {spec.name}: {coverage:.2f}")
    return errors
