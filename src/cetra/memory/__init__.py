"""Atomic graph memory and self-evolving memory construction.

This package is the landing point for the RP pipeline:

- ``adapter``   : raw StackPlanner logs -> events
- ``segment``   : events -> atoms (boundary detection + DP)
- ``decompose`` : atoms -> attributes
- ``recompose`` : attributes/rules -> chains and skills
- ``evolve``    : field-level and operator-level self-evolution

The package intentionally reuses ``cetra.rsi`` for rule induction and skill
artifacts, so the new pipeline can be compared with the previous one.
"""

from .types import Atom, Attribute, Edge, Event, MemoryGraph, ProgramState, Edit
from .adapter import parse_stackplanner_logs
from .segment import segment
from .decompose import deconstruct, OpenMetadataJudge
from .recompose import derive_scoped_rule
from .evolve import run_se_loop
from .llm import DeepSeekClient
from .llm_judge import LLMJudge, parse_events_with_llm
from .sources import (
    DirectorySource,
    GenericJsonSource,
    JsonlRecordSource,
    LLMTextSource,
    SQCADLifecycleSource,
    StackPlannerSource,
    TextLogSource,
    auto_source,
)
from .pipeline import TraceRun, run_source, save_runs
from .fields import AtomSchema, FieldAtom, FieldSpec, FieldValue, SchemaRegistry
from .need import NeedEstimator, NeedScore
from .decomp_program import (
    CompileError,
    DecompositionProgram,
    ExtractionReport,
    FactorSpec,
    apply_program,
    apply_programs,
    compile_program,
    validate_program,
)
from .llm_decomposer import HeuristicDecomposer, LLMDecomposer
from .operators_generic import (
    AbstractOperator,
    ComposeOperator,
    ConsolidateOperator,
    DecomposeOperator,
    OperatorResult,
    ProgramDecomposeOperator,
    ProjectOperator,
    Workflow,
    default_workflow,
    run_events,
    segment_to_field_atoms,
)
from .decomp_evolve import (
    JointEvaluation,
    ProgramScore,
    run_iterative_se,
    run_joint_se,
    score_program,
)

__all__ = [
    "Atom",
    "Attribute",
    "Edge",
    "Event",
    "MemoryGraph",
    "ProgramState",
    "Edit",
    "parse_stackplanner_logs",
    "segment",
    "deconstruct",
    "OpenMetadataJudge",
    "derive_scoped_rule",
    "run_se_loop",
    "DeepSeekClient",
    "LLMJudge",
    "parse_events_with_llm",
    "DirectorySource",
    "GenericJsonSource",
    "JsonlRecordSource",
    "LLMTextSource",
    "SQCADLifecycleSource",
    "StackPlannerSource",
    "TextLogSource",
    "auto_source",
    "TraceRun",
    "run_source",
    "save_runs",
    "AtomSchema",
    "FieldAtom",
    "FieldSpec",
    "FieldValue",
    "SchemaRegistry",
    "NeedEstimator",
    "NeedScore",
    "CompileError",
    "DecompositionProgram",
    "ExtractionReport",
    "FactorSpec",
    "apply_program",
    "apply_programs",
    "compile_program",
    "validate_program",
    "HeuristicDecomposer",
    "LLMDecomposer",
    "AbstractOperator",
    "ComposeOperator",
    "ConsolidateOperator",
    "DecomposeOperator",
    "OperatorResult",
    "ProgramDecomposeOperator",
    "ProjectOperator",
    "Workflow",
    "default_workflow",
    "run_events",
    "segment_to_field_atoms",
    "JointEvaluation",
    "ProgramScore",
    "run_iterative_se",
    "run_joint_se",
    "score_program",
]
