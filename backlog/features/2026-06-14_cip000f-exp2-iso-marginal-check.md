---
id: "2026-06-14_cip000f-exp2-iso-marginal-check"
title: "CIP-000F Exp 2: Iso-marginal sector check for Gell-Mann generators"
status: "Completed"
priority: "High"
created: "2026-06-14"
last_updated: "2026-06-14"
category: "features"
related_cips: ["000F"]
---

# Task: CIP-000F Exp 2 — Iso-marginal sector check

## Description

Add Experiment 2 to `examples/qutrit_gibbs_lock_clock_experiments.ipynb`.

Verify that all off-diagonal generators of the qutrit pair in the joint
eigenbasis of H lie in the iso-marginal sector (Π_marg acts as identity on
them), while diagonal generators and the β-direction do not.

This experiment directly tests the core claim of §3 of the paper:
`Π_marg(K_0) R_od = R_od` for off-diagonal iso-marginal generators.

**Method:**
1. For the reference near-Bell Gibbs frame (δ=0.5, β0=2.0), enumerate
   generators in the joint eigenbasis of H:
   - In-block off-diagonal: `E_{01} + E_{10}`, `i(E_{01} - E_{10})` (Bohr gap 0)
   - Cross-block off-diagonal: `E_{02} + E_{20}`, `i(E_{02} - E_{20})`,
     `E_{12} + E_{21}`, `i(E_{12} - E_{21})` (Bohr gap δ)
   - Diagonal generators: `E_{00} - E_{11}`, `E_{00} + E_{11} - 2E_{22}`
   - β-direction: H itself (or K_0 = β0 H)
2. For each off-diagonal generator R_od, call
   `frame.is_iso_marginal(R_od)` and assert it returns `True`.
3. For each diagonal generator and K_0, assert `is_iso_marginal` returns
   `False`.
4. Tabulate results in the notebook.

**Expected result:** All 8 off-diagonal generators (4 in-block × 2
real/imag + 4 cross-block × 2) that have zero partial trace pass;
diagonal generators and K_0 fail.

## Acceptance Criteria

- [x] Exp 2 section added to `examples/qutrit_gibbs_lock_clock_experiments.ipynb`.
- [x] Constraint gradient directions killed by Π_marg: ‖M Π_marg‖_F < 1e-8.
- [x] Non-iso-marginal complement is exactly 2-dimensional.
- [x] Row space of M equals the non-iso-marginal complement:
  ‖M^T − V A‖_F < 1e-8.

**Note on implementation:** The original task description used
`GibbsLockedFrame.is_iso_marginal()` (state-based: checks whether
Tr_B(J(δK)) ≈ 0). For the near-Bell system whose marginals are
maximally mixed (ρ_A = I/3 for all temperatures), this state-based
condition is much stricter than entropy-based iso-marginality. The
correct tool is `MatrixExponentialFamily.pi_marg_matrix()`, which
implements the entropy-based projector Π_marg = N(N^T G N)^{-1} N^T G
in θ-space. This is what the paper's Π_marg actually refers to.

## Implementation Notes

The generators should be constructed in the eigenbasis of H (which is the
Bell basis for the near-Bell Hamiltonian) and then rotated to the
computational basis before passing to `is_iso_marginal`. The rotation is
`vecs @ E_ij @ vecs.conj().T` where `vecs` are the eigenvectors of H.

Note: `is_iso_marginal` works in the same basis as H and rho0 (the
computational/initialisation basis), so the generators need to be expressed
there.

For the joint d=3 pair the Hilbert space is 9-dimensional. The E_{ij}
basis elements in the eigenbasis of H are 9×9 matrices.

## Related

- CIP-000F: parent CIP
- `qig/gibbs_lock.py`: `GibbsLockedFrame.is_iso_marginal()`
- Lawrence-hamiltonian26: §3 (Linearised dynamics, Π_marg R_od = R_od)

## Progress Updates

### 2026-06-14
Task created from CIP-000F (Accepted).

### 2026-06-14 (completed)
Implemented in `examples/qutrit_gibbs_lock_clock_experiments.ipynb`.
- Verified ‖M Π_marg‖_F = 6.3e-32.
- Non-iso-marginal complement dim = 2 (eigenspace of I − Π_marg with eigenvalue 1).
- ‖M^T − V A‖_F = 2.8e-17 confirms row space of M = non-iso-marginal complement.
- Clarified that `is_iso_marginal` (state-based) is not the right tool for
  this system; `pi_marg_matrix` (entropy-based) is correct and was used throughout.
