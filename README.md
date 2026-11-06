# CETRA

CETRA is a research framework for constructing **decision-sufficient long-term memory** from raw agent experience. It does not treat a log line, a dialogue chunk, or a retrieved fact as a memory unit by default. Instead, it segments a trace at decision points, represents each decision as a schema-driven field atom, decomposes fields into typed factors, and consolidates the result into rules, chains, and executable skills.

This repository contains the core Python package and its tests. Internal research notes, datasets, paper sources, and experiment outputs are intentionally excluded, so the package can be installed and used on its own.

## Pipeline

```text
raw trace
  -> normalized events
  -> decision-point segmentation
  -> schema-driven field atoms
  -> recursive / programmatic decomposition
  -> field projection
  -> rule, chain, and skill consolidation
  -> held-out joint self-evolution
```

The main design choices are:

- **Decision-point units.** A memory unit is the smallest span that supports one decision, not an arbitrary chunk.
- **Typed decomposition.** A decomposition is a `DecompositionProgram`: a list of `FactorSpec`s with a source, an extractor, and a config. It can be compiled, validated, executed, diffed, and revised.
- **Evidence-bound fields.** Every `FieldValue` carries evidence, confidence, scope, and provenance.
- **Need-gated recursion.** A field is decomposed only when the decision-information gain justifies the added fields and depth.
- **Joint evolution.** The decomposition program and the induced rules are optimized together under held-out decision risk, with negative-transfer checks and an Occam tie-break.

## Install

Requires Python 3.10 or newer.

```bash
git clone https://github.com/Topps-2025/CETRA.git
cd CETRA
pip install -e ".[dev]"
```

Or install directly from GitHub:

```bash
pip install "cetra @ git+https://github.com/Topps-2025/CETRA.git"
```

## Quickstart

The quickstart is fully synthetic and does not require a dataset or an API key:

```bash
python examples/quickstart.py
```

It builds a short agent trace, segments it into decision points, runs the default operator workflow, and prints the resulting atoms, rules, chains, and skills.

Library usage:

```python
from cetra.memory import Event, run_events

events = [
    Event("e1", "2026-01-01T00:00:00Z", "user", "user_message",
          "Draft a launch note.", {"task_type": "launch_note"}),
    Event("e2", "2026-01-01T00:00:01Z", "agent", "agent_action",
          "Ask for the audience.",
          {"action": "ask_clarification", "info": "missing", "feedback": "none"}),
    # ...
]

result = run_events(events)
print(result.metrics["schemas"])
for atom in result.atoms:
    if atom.schema == "rule":
        print(atom.fields["conditions"].value, "->", atom.fields["action"].value)
```

`run_events` returns an `OperatorResult` with:

- `atoms`: the final `FieldAtom` list (decisions, rules, chains, and skills);
- `metrics`: per-operator metrics and final schema counts;
- `trace`: machine-readable construction records.

Pass `segment_events=False` to get the per-event ablation instead of decision-point segmentation.

## Package map

| Module | Responsibility |
|---|---|
| `cetra.memory.types` | `Event`, `Atom`, `Edge`, `MemoryGraph`, `ProgramState` |
| `cetra.memory.segment` | Boundary scoring and dynamic-programming decision-point segmentation |
| `cetra.memory.fields` | `FieldAtom`, `FieldValue`, `AtomSchema`, `SchemaRegistry` |
| `cetra.memory.need` | Decision-entropy gain and decomposition-cost `NeedEstimator` |
| `cetra.memory.decomp_program` | `FactorSpec`, `DecompositionProgram`, compile / validate / apply |
| `cetra.memory.llm_decomposer` | LLM proposal and counterexample-driven revision of decomposition programs |
| `cetra.memory.operators_generic` | Decompose, Project, Abstract, Compose, Consolidate operators |
| `cetra.memory.recompose` | Scoped rule derivation from decomposed cases |
| `cetra.memory.decomp_evolve` | Joint search over factor subsets and rules, held-out gates, negative-transfer checks |
| `cetra.memory.sources` | StackPlanner, SWE-agent, JSONL, text, and directory adapters |
| `cetra.rsi` | Rule induction, skill artifacts, gating, and self-evolving harness primitives |

## DecompIR at a glance

A decomposition program is data, not a prompt:

```python
from cetra.memory import FactorSpec, compile_program

program = compile_program({
    "factors": [
        FactorSpec("user_feedback", source="narrative", extractor="regex",
                   config={"pattern": r"user_feedback=([^;]+)"}).to_dict(),
        FactorSpec("has_context", source="narrative", extractor="key_value").to_dict(),
    ],
    "rationale": "Separate feedback from context availability.",
})
```

`compile_program` rejects reserved or duplicate names, unknown extractors, invalid configs, and proposals with too many factors. `apply_program` executes the program and reports per-factor coverage and grounding. The same artifact can be restricted, merged, versioned, and passed back into `default_workflow(decomposition_program=...)`.

## Optional LLM backend

The deterministic path works without an LLM. To enable LLM-based event parsing, decomposition proposals, or program revision:

```bash
cp .env.example .env
# edit .env and set DEEPSEEK_API_KEY
```

The client reads `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, and `DEEPSEEK_MODEL` from the environment or the repository `.env` file. Never commit `.env`.

## Tests

```bash
python -m pytest -q
```

The test suite covers segmentation, field operators, DecompIR compilation and execution, LLM revision with a fake client, joint SE gating, rule induction, skills, and trace adapters. Tests use temporary synthetic inputs; no external dataset is required.

## Status

CETRA is a research prototype (`0.1.0.dev0`). The core interfaces are usable, but APIs may change as the decomposition and evolution mechanisms are developed further. The current implementation is intended for research reproduction and integration experiments, not production deployment.
