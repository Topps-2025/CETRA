"""CETRA: learning to decompose, abstract, and recombine agent experience.

The package mirrors the paper's method sections:

- :mod:`cetra.memory`       rule and chain representations, experience library
- :mod:`cetra.abstraction`  trajectory decomposition and rule abstraction
- :mod:`cetra.diagnosis`    paired-execution condition tests
- :mod:`cetra.composition`  chain search, compatibility, fallback paths
- :mod:`cetra.policy`       abstraction/connection policy learning
- :mod:`cetra.evaluation`   capability-transfer protocol and metrics

The current weekly prototype lives in the cross-cutting modules:

- :mod:`cetra.cases`        controlled case generator and ground-truth policy
- :mod:`cetra.rules`        rule schema, validation, split/refine, priorities
- :mod:`cetra.prompts`      decomposition / induction / prediction prompts
- :mod:`cetra.pipeline`     LLM-backed decomposer, inducer, retriever, predictor
- :mod:`cetra.llm`          DeepSeek client (stdlib, retries, JSON mode)
- :mod:`cetra.metrics`      accuracy, bootstrap CI, McNemar, counterfactuals
"""

__version__ = "0.1.0.dev0"
