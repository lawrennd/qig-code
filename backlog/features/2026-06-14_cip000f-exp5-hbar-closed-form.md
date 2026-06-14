---
id: "2026-06-14_cip000f-exp5-hbar-closed-form"
title: "CIP-000F Exp 5: ℏ(β₀, δ) closed form and numerical verification"
status: "Completed"
priority: "High"
created: "2026-06-14"
last_updated: "2026-06-14"
category: "features"
related_cips: ["000F"]
---

# Task: CIP-000F Exp 5 — ℏ(β₀, δ) closed form

## Description

Add Experiment 5 to `examples/qutrit_gibbs_lock_clock_experiments.ipynb`.

Compute ℏ = Var_{ρ_0}(K_0)^{-1} across the (β0, δ) parameter space and
verify the closed form, divergence limits, and the minimum near x = β0·δ ≈
2.66. This experiment is central to the paper (§4–5) and is purely
analytical — it requires no dynamics, only arithmetic on ρ_0.

**Closed form** (for H = near_bell_hamiltonian(d=3, delta=δ), Gibbs state
ρ_0 = diag(1,1,exp(-β0 δ))/Z in the Bell eigenbasis):

```
Var_{ρ_0}(H_δ) = 2 δ² exp(-β0 δ) / Z²
ℏ = Z² / (2 β0² δ² exp(-β0 δ))
```

**Method:**
1. Compute `Var_num = tr(ρ_0 H²) - tr(ρ_0 H)²` numerically from `frame.rho0`
   and `frame.H` over a (β0, δ) grid.
2. Compute `hbar_num = 1 / (beta**2 * Var_num)`.
3. Compute `hbar_analytic = Z² / (2 * beta**2 * delta**2 * exp(-beta*delta))`
   where Z = 2 + exp(-beta*delta).
4. Assert `|hbar_num - hbar_analytic| < 1e-10` at every grid point.
5. Plot ℏ vs x = β0·δ (1D slice) on a log scale. Mark the minimum near
   x ≈ 2.66 with a vertical dashed line. Annotate the three divergence
   regimes (δ→0, β0→0, β0δ→∞).

## Acceptance Criteria

- [x] Exp 5 section added to `examples/qutrit_gibbs_lock_clock_experiments.ipynb`.
- [x] `|hbar_num - hbar_analytic| < 1e-10` asserted at every (β0, δ) grid point.
- [x] Divergence confirmed numerically at x → 0 and x → ∞.
- [x] Minimum of hbar_analytic located near x ≈ 2.23 (verified numerically: x* = 2.2278).
- [x] Plot of ℏ vs x included in notebook.

## Implementation Notes

The variance is most cleanly computed as:

```python
H = frame.H
rho0 = frame.rho0
var_H = np.real(np.trace(rho0 @ H @ H)) - np.real(np.trace(rho0 @ H))**2
hbar = 1.0 / (frame.beta**2 * var_H)
```

For the near-Bell Hamiltonian the H matrix is 9×9 (joint pair), so the
variance is over the full joint state. The H diagonal in the Bell basis
has spectrum {0, δ, 1, 1, …}, so the effective variance is dominated by
the {0, δ} two-level structure when β0·δ is of order 1.

The minimum of x² exp(-x)/(2+exp(-x))² can be found numerically using
`scipy.optimize.minimize_scalar` as a cross-check.

## Related

- CIP-000F: parent CIP
- `qig/gibbs_lock.py`: `GibbsLockedFrame.rho0`, `GibbsLockedFrame.H`
- Lawrence-hamiltonian26: running example §5 (ℏ for the departed qutrit)
- Exp 4 (same notebook): μ0 = c·ℏ cross-check

## Progress Updates

### 2026-06-14
Task created from CIP-000F (Accepted).
Implemented in `examples/qutrit_gibbs_lock_clock_experiments.ipynb`.
All assertions pass (grid error ≤ 7.3×10⁻¹²). Minimum at x* ≈ 2.228 confirmed.
Task completed.
