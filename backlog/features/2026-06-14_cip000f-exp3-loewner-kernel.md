---
id: "2026-06-14_cip000f-exp3-loewner-kernel"
title: "CIP-000F Exp 3: Loewner kernel two-sector structure"
status: "Proposed"
priority: "High"
created: "2026-06-14"
last_updated: "2026-06-14"
category: "features"
related_cips: ["000F"]
---

# Task: CIP-000F Exp 3 — Loewner kernel two-sector structure

## Description

Add Experiment 3 to `examples/qutrit_gibbs_lock_clock_experiments.ipynb`
(creating the notebook if it does not yet exist).

Verify the analytical Loewner weights for both off-diagonal mode types of
the departed d=3 qutrit and confirm the smooth degenerate limit δ→0. This
is the first experiment to implement because it is purely analytical, has no
prerequisites beyond existing code, and provides the Loewner weights needed
by Exp 4.

**Physical setup:** `near_bell_gibbs_frame(d=3, delta=δ, beta=β0)` with the
joint Hamiltonian H = near_bell_hamiltonian(d=3, delta=δ) whose spectrum is
{0, δ, 1, …}. At β0 the Gibbs state is ρ_0 = exp(-β0 H)/Z with two
off-diagonal mode types:

| Mode | Indices | Bohr gap | Analytical Loewner weight |
|------|---------|----------|--------------------------|
| In-block | (0,1), (1,0) | 0 | `1/Z` |
| Cross-block | (0,2), (1,2), cc | δ | `(1 - exp(-β0 δ)) / (Z β0 δ)` |

**Method:**
1. Create the notebook with a setup section: imports, physical parameters
   (reference point δ=0.5, β0=2.0), and ρ_0 table.
2. Compute `C, vals, vecs = frame.loewner_kernel()` across a (β0, δ) grid.
3. Extract the in-block entry `C[0,1]` and cross-block entry `C[0,2]`.
4. Compare to closed forms at each grid point and assert agreement to 1e-10.
5. Plot both weights as functions of x = β0·δ (fixing β0, varying δ).
   Add a dashed curve for the δ→0 limit (both converge to 1/Z). Include
   this plot in the notebook.
6. Verify `lim_{δ→0} (1-exp(-β0 δ))/(Z β0 δ) = 1/Z` numerically.

## Acceptance Criteria

- [ ] Notebook `examples/qutrit_gibbs_lock_clock_experiments.ipynb` created
  with a setup section and Exp 3 section.
- [ ] Assertion `|C_numerical - C_analytical| < 1e-10` passes for all
  (β0, δ) grid points for both in-block and cross-block entries.
- [ ] δ→0 smooth limit verified: cross-block weight converges to in-block
  weight to within 1e-10.
- [ ] Plot of Loewner weights vs x = β0·δ included in notebook.

## Implementation Notes

The Loewner kernel is accessed via `GibbsLockedFrame.loewner_kernel()` which
returns `(C, vals, vecs)` in the eigenbasis of ρ_0. For the near-Bell
Hamiltonian the eigenbasis is the generalised Bell basis, so the (0,1)
entry is the in-block weight and (0,2) is the cross-block weight.

The grid should cover β0 ∈ [0.5, 5.0] and δ ∈ [0.05, 2.0] (about 10×10
points). Use logspacing for δ to capture the δ→0 limit well.

## Related

- CIP-000F: parent CIP
- `qig/gibbs_lock.py`: `GibbsLockedFrame.loewner_kernel()`
- `qig/pair_operators.py`: `near_bell_gibbs_frame`
- Lawrence-hamiltonian26: running example in §2 (Loewner kernel)

## Progress Updates

### 2026-06-14
Task created from CIP-000F (Accepted).
