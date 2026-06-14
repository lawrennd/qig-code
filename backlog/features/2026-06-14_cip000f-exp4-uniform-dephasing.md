---
id: "2026-06-14_cip000f-exp4-uniform-dephasing"
title: "CIP-000F Exp 4: Uniform dephasing and Loewner amplitude weighting"
status: "Proposed"
priority: "High"
created: "2026-06-14"
last_updated: "2026-06-14"
category: "features"
related_cips: ["000F"]
---

# Task: CIP-000F Exp 4 — Uniform dephasing in modular-generator coordinates

## Description

Add Experiment 4 to `examples/qutrit_gibbs_lock_clock_experiments.ipynb`.

Verify that the decay rate μ0 = c·ℏ is uniform across both off-diagonal
mode types (in-block and cross-block) in modular-generator coordinates,
and that the Loewner kernel sets the amplitude weighting when pushed
forward to density-matrix coordinates.

**Method:**
1. Construct a mixed off-diagonal iso-marginal initial perturbation
   `R_od = α · (in-block generator) + γ · (cross-block generator)`
   in the eigenbasis of H, rotated back to the computational basis.
2. Choose μ0 analytically from Exp 5: `mu0 = c * hbar` with c=1.
3. Evolve using `frame.linearised_flow(R_od, mu0=mu0, t)` over a time
   grid t ∈ [0, 5/mu0].
4. Extract and plot |R_od(t)_{01}| and |R_od(t)_{02}| (the in-block and
   cross-block envelopes). Both should decay as exp(-mu0·t).
5. Apply the Loewner map: `delta_rho = frame.loewner_map(R_od)`. Compare
   initial amplitudes |δρ_{01}(0)| and |δρ_{02}(0)| to the Loewner
   weights from Exp 3.
6. Fit μ0 using `infer_mu0(times, magnitudes, frame, R_od)` and verify the
   fitted value matches the input mu0 to within 1%.
7. Cross-check: verify `mu0 = hbar` (with c=1) matches the Exp 5 value.

**Expected result:**
- Single uniform decay rate μ0 across all off-diagonal modes.
- Initial amplitude ratio |δρ_{02}(0)| / |δρ_{01}(0)| equals the ratio of
  Loewner weights from Exp 3.
- infer_mu0 fitted value matches input mu0 to within 1%.

## Acceptance Criteria

- [ ] Exp 4 section added to `examples/qutrit_gibbs_lock_clock_experiments.ipynb`.
- [ ] Uniform decay confirmed: both envelopes fit to exp(-mu0·t) with the
  same mu0 (relative error < 1%).
- [ ] Loewner amplitude ratio verified: initial amplitudes in δρ differ by
  the Loewner weights from Exp 3 (tolerance 1e-10).
- [ ] `infer_mu0` fitted rate matches input mu0 to within 1%.
- [ ] Plot of decay envelopes for in-block and cross-block modes included.

## Implementation Notes

The `infer_mu0` function in `qig/gibbs_lock.py` takes a time array and
an array of |R_od(t)_{ij}| magnitudes. Use any single off-diagonal element
(e.g. (0,2)) for the fit.

Suggested parameter values: δ=0.5, β0=2.0, c=1.0, so
`mu0 = hbar(δ=0.5, β0=2.0)` from the Exp 5 closed form. Use a normalised
R_od (divide by its Frobenius norm) so that the initial magnitude is 1.

## Related

- CIP-000F: parent CIP
- Exp 3 (same notebook): Loewner weights for amplitude comparison
- Exp 5 (same notebook): ℏ value for mu0
- `qig/gibbs_lock.py`: `GibbsLockedFrame.linearised_flow()`,
  `GibbsLockedFrame.loewner_map()`, `infer_mu0`
- Lawrence-hamiltonian26: §3, §5 (Hamiltonian clock)

## Progress Updates

### 2026-06-14
Task created from CIP-000F (Accepted).
