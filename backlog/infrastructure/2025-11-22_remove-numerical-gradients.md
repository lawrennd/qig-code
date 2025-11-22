---
id: "2025-11-22_remove-numerical-gradients"
title: "Remove numerical gradient computations from quantum game code"
status: "Proposed"
priority: "High"
created: "2025-11-22"
last_updated: "2025-11-22"
owner: "Neil D. Lawrence"
github_issue: ""
dependencies: ""
tags:
- backlog
- infrastructure
- optimisation
- numerical-stability
---

# Task: Remove numerical gradient computations from quantum game code

## Description

The current quantum inaccessible game implementation uses numerical finite-difference
approximations for several key gradients and Hessians (e.g. Fisher/BKM metric,
constraint gradients) inside `QuantumExponentialFamily` and related routines.
These numerical gradients are:

- computationally expensive (nested finite-difference loops with repeated `expm`);
- a source of potential numerical instability and noise; and
- a bottleneck for higher-level scripts such as `advanced_analysis.py` and
  `validate_qutrit_optimality.py`, which can take several minutes to run.

This task is to systematically remove numerical gradient computations from the
quantum game code and replace them with more efficient and robust alternatives,
such as:

- closed-form expressions where available;
- analytic derivatives using matrix calculus; or
- algorithmic/automatic differentiation via a suitable library.

The goal is to improve performance, numerical stability, and reliability of the
validation and analysis pipeline, especially for CI/integration use.

## Acceptance Criteria

- [ ] All finite-difference based gradient/Hessian computations in the core library
      (`qig.*` and any remaining logic in `inaccessible_game_quantum.py`) are
      identified and documented.
- [ ] For each such computation, an alternative implementation is designed
      (closed-form, analytic, or AD-based) and recorded.
- [ ] At least the main bottlenecks (Fisher/BKM metric, marginal-entropy constraint
      gradients) are re-implemented without naive finite differences.
- [ ] Existing unit tests in `test_inaccessible_game.py` continue to pass without
      relaxation of numerical tolerances.
- [ ] High-level scripts (`advanced_analysis.py`, `validate_qutrit_optimality.py`)
      run noticeably faster and no longer require multi-minute runtimes in their
      default or short modes.

## Implementation Notes

- Start by auditing:
  - `qig/exponential_family.py` (Fisher information / BKM metric and
    `marginal_entropy_constraint`);
  - any remaining gradient logic in `inaccessible_game_quantum.py`; and
  - usages of these quantities in `InaccessibleGameDynamics.flow`.
- For the BKM metric, consider:
  - exploiting known integral/KMS representations and spectral decompositions; or
  - using an automatic differentiation framework (e.g. JAX, PyTorch) on a
    re-expressed version of the log-partition function.
- For constraint gradients, derive analytic expressions for ∂C/∂θ where possible,
  using the chain rule through ρ(θ) and marginal traces.
- Preserve the public API of `QuantumExponentialFamily` and `InaccessibleGameDynamics`
  so that tests and scripts do not need major changes.
- Use the existing `QIG_SHORT` mode as a benchmark: the aim is that even full
  runs (without short mode) become significantly faster once numerical gradients
  are replaced.

## Related

- CIP: 0001 (Codebase consolidation and documentation)
- PRs: []
- Documentation:
  - `IMPLEMENTATION_SUMMARY.md`
  - `EVIDENCE_ASSESSMENT.md`

## Progress Updates

### 2025-11-22

Task created with Proposed status. Captures the need to replace expensive
finite-difference gradients in the quantum game code with more efficient and
stable alternatives, motivated by slow runs of `advanced_analysis.py` and
`validate_qutrit_optimality.py`.


