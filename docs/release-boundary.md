# Release boundary

`Faster-RAE` contains the optimized RAE source, the historical optimization preset, normal training configurations, and a small first-run GPU adaptation workflow. The repository-local skill and AGENTS.md are an operational entrypoint: they help an agent install dependencies, locate assets, measure memory, and select a feasible per-GPU microbatch and accumulation count while preserving upstream global batch (default deployment: 8 GPUs).

The independent research harness owns task formulation, frozen contracts, correctness/trajectory evaluators, profiling attribution, proposal/execution/evaluation roles, candidate snapshots, and experiment ledgers. Its general optimization skills should be versioned with that harness. It can consume a Faster-RAE commit plus a configuration/preset; Faster-RAE should not depend on an unpublished harness checkout to launch training.

The interface is a source commit, environment fingerprint, training YAML, optimization environment mapping, and machine-local asset paths. Capacity reports carry `kind=capacity_smoke_only`; they are not valid research-harness speedup or numerical-equivalence verdicts. A harness integration must construct and validate its own fixed workload on the new hardware.

This change adds only the repository-side adaptation package. It does not release the full research harness, copy its frozen experimental data, or claim that its two-card historical correctness verdict automatically covers a new GPU/batch/backend.
