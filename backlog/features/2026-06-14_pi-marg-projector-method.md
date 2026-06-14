---
id: "2026-06-14_pi-marg-projector-method"
title: "Add pi_marg_matrix() to MatrixExponentialFamily"
status: "Completed"
priority: "High"
created: "2026-06-14"
last_updated: "2026-06-14"
category: "features"
related_cips: ["000F"]
---

# Task: Add pi_marg_matrix() to MatrixExponentialFamily

## Description

CIP-000F (Exp 1) requires the explicit matrix form of the marginal projector
Π_marg in natural-parameter coordinates:

```
Π_marg(θ*) = N (N^T G N)^{-1} N^T G
```

where N is a matrix whose columns span the null space of the constraint
Jacobian M(θ*) (one row per marginal-entropy constraint, one column per
natural parameter), and G = G(θ*) is the BKM/Fisher metric.

`MatrixExponentialFamily` already exposes:
- `marginal_entropy_constraint(theta)` → `(C, grad_C)` with `grad_C` of shape
  `(n_params,)`. For a system with multiple subsystems the full Jacobian M is
  a `(n_subsystems, n_params)` matrix assembled by perturbing each subsystem's
  marginal entropy independently. Currently only the *sum* gradient is returned;
  the per-subsystem gradients need to be extracted.
- `fisher_information(theta)` → G of shape `(n_params, n_params)`.

The missing step is to (a) build the full constraint Jacobian M by computing
per-subsystem constraint gradients, (b) find its null space N via SVD, and
(c) assemble Π_marg. All three steps use existing methods plus standard
`numpy.linalg` routines.

## Acceptance Criteria

- [x] `MatrixExponentialFamily.pi_marg_matrix(theta, tol=1e-10)` added to
  `qig/exponential_family.py`.
- [x] Returns `Pi` of shape `(n_params, n_params)` satisfying:
  - `Pi @ Pi ≈ Pi` (idempotent, tolerance 1e-10)
  - `np.linalg.matrix_rank(Pi) == n_params - n_subsystems` for generic θ*
  - Columns of `(I - Pi)` equal the span of the constraint gradient directions
- [x] Per-subsystem constraint gradient helper
  `_marginal_entropy_gradient_per_subsystem(theta)` added (returns matrix of
  shape `(n_subsystems, n_params)`).
- [x] Unit tests added to `tests/test_pair_exponential_family.py` verifying
  idempotency, rank, and action on known iso-marginal directions for the d=3
  qutrit pair at the near-Bell Gibbs point.
- [x] Docstring references `eq:second-order-projector` in Lawrence-origin26.

## Implementation Notes

The full constraint Jacobian for a bipartite system is:

```python
def _marginal_entropy_gradient_per_subsystem(self, theta):
    # Returns M of shape (n_subsystems, n_params)
    # Row k is grad_C_k where C_k = marginal von Neumann entropy of subsystem k
    M = np.zeros((self.n_sites, self.n_params))
    for k in range(self.n_sites):
        # C_k = -tr(rho_k log rho_k); compute its gradient by the BKM inner
        # product against the lifted log-rho_k operator (analogous to
        # marginal_entropy_constraint_theta_only but for a single subsystem)
        _, M[k] = self._marginal_entropy_gradient_single(theta, k)
    return M
```

Then:

```python
def pi_marg_matrix(self, theta, tol=1e-10):
    M = self._marginal_entropy_gradient_per_subsystem(theta)  # (n_sub, n_params)
    G = self.fisher_information(theta)                         # (n_params, n_params)
    # Null space of M via SVD
    _, s, Vt = np.linalg.svd(M)
    null_mask = s < tol * s[0] if len(s) > 0 else np.ones(Vt.shape[0], dtype=bool)
    # ... assemble N from right singular vectors with small singular values
    N = Vt[len(s):].T  # columns of N span ker(M)
    # Projector Π = N (N^T G N)^{-1} N^T G
    NtGN = N.T @ G @ N
    Pi = N @ np.linalg.solve(NtGN, N.T @ G)
    return Pi
```

Note: the per-subsystem gradient `_marginal_entropy_gradient_single(theta, k)`
reuses the BKM kernel already computed for the Fisher metric — the BKM inner
product against the single lifted `log(rho_k)` operator is the same calculation
as `marginal_entropy_constraint_theta_only` but restricted to subsystem k.

## Related

- CIP-000F: Qutrit Gibbs-Lock Experiments (Exp 1 requires this)
- Lawrence-origin26: `eq:second-order-projector`
- `qig/exponential_family.py`: `MatrixExponentialFamily`
- `tests/test_pair_exponential_family.py`

## Progress Updates

### 2026-06-14
Task created as prerequisite for CIP-000F Exp 1 implementation.
Implemented `_marginal_entropy_gradient_single`, `_marginal_entropy_gradient_per_subsystem`,
and `pi_marg_matrix` in `qig/exponential_family.py`. All 8 unit tests pass
(63 total in `tests/test_pair_exponential_family.py`). Task completed.
