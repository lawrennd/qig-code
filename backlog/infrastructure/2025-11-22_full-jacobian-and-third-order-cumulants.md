---
id: "2025-11-22_full-jacobian-and-third-order-cumulants"
title: "Implement analytic full Jacobian and third-order Kubo–Mori cumulants"
status: "Proposed"
priority: "Medium"
created: "2025-11-22"
last_updated: "2025-11-22"
owner: "Neil D. Lawrence"
github_issue: ""
dependencies: ""
tags:
- backlog
- infrastructure
- geometry
- jacobian
---

# Task: Implement analytic full Jacobian and third-order Kubo–Mori cumulants

## Description

The current GENERIC/Jacobian analysis for the quantum inaccessible game relies on
finite-difference approximations:

- `inaccessible_game_quantum.py::compute_jacobian` computes ∂\dot{θ}/∂θ by finite
  differences of the flow.
- Third-order Kubo–Mori cumulants (∂³ψ) and related objects are explored in
  experimental scripts such as:
  - `quantum_qutrit_n3_backup.py` / `quantum_qutrit_n3.py`;
  - `verify_proof_directly.py`; and
  - validated in `test_third_cumulant_symmetry.py`.

This task is to design and implement a **clean, analytic (or AD-based) API** for:

1. The full Jacobian of the constrained dynamics in θ-space, suitable for use in
   GENERIC analysis and stability studies.
2. Third-order Kubo–Mori cumulants / derivatives of ψ and their relation to the
   Jacobian structure, in a way that is consistent with the paper’s information-
   geometric interpretation.

The goal is to replace ad-hoc finite-difference Jacobian computations with a
principled, reusable implementation that integrates cleanly with `qig.*`.

## Acceptance Criteria

- [ ] A well-defined interface (e.g. in `qig.analysis` or `qig.exponential_family`)
      for computing:
      - the full Jacobian `J(θ) = ∂\dot{θ}/∂θ` of the constrained flow; and
      - third-order Kubo–Mori cumulants or equivalent tensors needed for this
        Jacobian.
- [ ] The existing `compute_jacobian` function in `inaccessible_game_quantum.py`
      is either:
      - reimplemented to use the new analytic/AD-based primitives; or
      - replaced by a wrapper that delegates to the new API.
- [ ] `test_third_cumulant_symmetry.py` passes using the new implementation and
      no longer depends on naive finite-difference approximations.
- [ ] Any analysis functions in `advanced_analysis.py` that require Jacobians can
      call the new API instead of recomputing finite differences.
- [ ] The numerical behaviour (symmetries, Jacobi checks) matches or improves
      on the current finite-difference based results.

## Implementation Notes

- Use `quantum_qutrit_n3_backup.py`, `quantum_qutrit_n3.py`, and
  `verify_proof_directly.py` as starting points for understanding the structure
  of third-order derivatives and cumulants in the current code.
- Consider expressing third-order derivatives of ψ in terms of:
  - spectral decompositions of the effective Hamiltonian / sufficient statistics;
  - known Kubo–Mori cumulant formulae; or
  - automatic differentiation of a suitably vectorised `log_partition` function.
- The Jacobian of the constrained flow will depend on:
  - derivatives of the metric G(θ); and
  - derivatives of the constraint gradient (second-order derivatives of C).
  These should be expressed consistently with the chosen representation of G and
  C (to be aligned with the separate numerical-gradients-removal task).
- Keep the analytic/AD implementation dimension-agnostic (not just for qubits
  and qutrits) where feasible, but it is acceptable to focus first on small
  systems (2–3 sites, d=2,3) for validation.

## Related

- Backlog:
  - `2025-11-22_remove-numerical-gradients` (removal of finite-difference gradients
    for 2nd-order quantities).
- CIPs:
  - 0001 (Codebase consolidation and documentation).
- Tests / scripts:
  - `test_third_cumulant_symmetry.py`
  - `quantum_qutrit_n3_backup.py`, `quantum_qutrit_n3.py`
  - `verify_proof_directly.py`
  - `advanced_analysis.py` (GENERIC/Jacobian analysis)

## Progress Updates

### 2025-11-22

Task created with Proposed status. Separates the “full Jacobian / third-order
cumulants” work from the more general “remove numerical gradients” task, and
anchors it to existing exploratory scripts and tests.


