---
id: "2026-06-14_cip000f-exp1-pi-marg-computation"
title: "CIP-000F Exp 1: Π_marg computation and iso-marginal sector basis"
status: "Proposed"
priority: "High"
created: "2026-06-14"
last_updated: "2026-06-14"
category: "features"
related_cips: ["000F"]
dependencies: ["2026-06-14_pi-marg-projector-method"]
---

# Task: CIP-000F Exp 1 — Explicit Π_marg and iso-marginal sector basis

## Description

Add Experiment 1 to `examples/qutrit_gibbs_lock_clock_experiments.ipynb`.

Compute the marginal projector Π_marg(K_0) explicitly in natural-parameter
coordinates and record the structure of the off-diagonal iso-marginal sector
as a function of δ. This is the most involved experiment and depends on
`MatrixExponentialFamily.pi_marg_matrix(theta)` (completed in
`backlog/features/2026-06-14_pi-marg-projector-method.md`).

**Method:**
1. Build `MatrixExponentialFamily(n_pairs=1, d=3, pair_basis=True)`.
2. Compute θ* for the near-Bell Gibbs state by projecting
   K = -β0 · H_joint onto the su(9) basis operators:
   `theta_a = tr(F_a K) / tr(F_a F_a)`.
3. Call `exp_fam.pi_marg_matrix(theta_star)` to get Π_marg as an
   (80 × 80) matrix in natural-parameter coordinates.
4. Verify idempotency `Π_marg² ≈ Π_marg` (tolerance 1e-8).
5. Compute rank(Π_marg) and verify it equals n_params - n_sites = 78.
6. Record the dimension of the iso-marginal sector as a function of δ
   (varying δ from 0.05 to 2.0): confirm rank = 78 for generic δ > 0.
7. Identify the complement sector (image of I - Π_marg): verify it is
   spanned by the per-subsystem constraint gradient directions from
   `_marginal_entropy_gradient_per_subsystem`.
8. Push Π_marg forward to K-coordinates using the BKM metric and verify
   that the off-diagonal generators from Exp 2 lie in its image.

**Expected result:** Rank 78 for generic δ > 0. The complement sector
(rank 2) is spanned by the two per-subsystem marginal entropy gradients.
The off-diagonal generators verified in Exp 2 are fixed points of Π_marg.

## Acceptance Criteria

- [ ] Exp 1 section added to `examples/qutrit_gibbs_lock_clock_experiments.ipynb`.
- [ ] `Π_marg² ≈ Π_marg` to within 1e-8.
- [ ] `rank(Π_marg) == 78` for all δ in test grid.
- [ ] Complement sector spanned by constraint gradient directions
  (verify via SVD comparison), tolerance 1e-8.
- [ ] Off-diagonal generators from Exp 2 verified as fixed points of
  Π_marg (in natural-parameter representation).

## Implementation Notes

The θ* conversion:
```python
from qig.pair_operators import near_bell_hamiltonian, pair_basis_generators
H = near_bell_hamiltonian(d=3, delta=delta)
operators = pair_basis_generators(d=3)
K = -beta * H
theta_star = np.array([
    np.real(np.trace(F @ K)) / np.real(np.trace(F @ F))
    for F in operators
])
```

To push Π_marg forward to K-coordinates, use the BKM metric G and the
change-of-basis between K and θ (which is linear: K = -∑_a θ_a F_a, so
the Jacobian is just the operator matrix):
```python
G = exp_fam.fisher_information(theta_star)
# Π_marg in K-coordinates: Pi_K = A^{-T} Pi A^T where A_aj = tr(F_a F_j)/norm
```

This step may benefit from a helper function if the calculation is verbose;
consider adding a `pi_marg_in_K_coordinates(theta)` convenience method.

## Related

- CIP-000F: parent CIP
- `backlog/features/2026-06-14_pi-marg-projector-method.md` — prerequisite
  (completed)
- Exp 2 (same notebook): off-diagonal generators to verify
- `qig/exponential_family.py`: `MatrixExponentialFamily.pi_marg_matrix()`
- Lawrence-origin26: `eq:second-order-projector`

## Progress Updates

### 2026-06-14
Task created from CIP-000F (Accepted). Prerequisite pi_marg_matrix()
is already implemented and tested.
