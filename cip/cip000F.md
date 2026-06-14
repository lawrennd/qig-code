---
id: "000F"
title: "Qutrit Gibbs-Lock Experiments: Pi_marg, Loewner Kernel, and Hamiltonian Clock"
status: "Accepted"
created: "2026-06-14"
last_updated: "2026-06-14"
compressed: false
author: "Neil D. Lawrence"
related_requirements: []
related_cips: ["000C", "000D", "000E"]
tags: ["qutrit", "gibbs-lock", "hamiltonian-clock", "loewner-kernel", "pi-marg", "experiments"]
---

# CIP-000F: Qutrit Gibbs-Lock Experiments: Pi_marg, Loewner Kernel, and Hamiltonian Clock

## Status
- [x] Proposed
- [x] Accepted
- [ ] Implemented
- [ ] Closed

## Summary

Design and implement five experiments for the d=3 qutrit system departed from
the Bell state, concretely verifying the Hamiltonian clock construction in
the companion paper (Lawrence-hamiltonian26). The experiments compute the
marginal projector Π_marg, the Loewner kernel, the iso-marginal off-diagonal
sector, the uniform dephasing rate, and the modular Fisher resolution ℏ for
the near-Bell qutrit Gibbs frame.

## Motivation

The companion paper derives a Hamiltonian clock from the inaccessible-game
dynamics near a Gibbs-locked modular generator, using an abstract treatment
of Π_marg(K_0), the Loewner map J_{ρ_0}, and the off-diagonal iso-marginal
sector. The paper states that explicit computation for small systems is needed
to test the construction. This CIP defines those experiments for the minimal
non-trivial case: a d=3 qutrit with one degenerate pair of eigenvalues.

The qutrit is the smallest system with a non-trivial degenerate block
structure and two qualitatively different off-diagonal mode types (in-block
and cross-block), making it analytically tractable and experimentally
illuminating.

## Physical Setup

The system departs from the maximally degenerate (Bell-like) state by lifting
one eigenvalue by δ. The effective Hamiltonian is

```
H_δ = diag(0, 0, δ)
K_0 = β_0 H_δ
```

giving the Gibbs state

```
ρ_0 = (1/Z) diag(1, 1, exp(-β_0 δ)),    Z = 2 + exp(-β_0 δ).
```

The spectrum has one degenerate block {λ_0 = λ_1 = 1/Z} and one separated
level {λ_2 = exp(-β_0 δ)/Z}. This produces two qualitatively different
off-diagonal mode types:

| Mode type        | Indices          | Bohr gap | Loewner weight                              |
|------------------|------------------|----------|---------------------------------------------|
| In-block (degen) | (0,1), (1,0)     | 0        | λ_0 = 1/Z                                  |
| Cross-block      | (0,2), (1,2), cc | δ        | (1 - exp(-β_0 δ)) / (Z β_0 δ)              |

The in-block Loewner weight equals ρ_0 in the degenerate limit; cross-block
weights are suppressed as x = β_0 δ grows. The two-sector structure makes all
five experiments analytically transparent while capturing the qualitative
behaviour of Gibbs-locked frames.

This setup is constructed using `near_bell_gibbs_frame(d=3, delta=δ, beta=β_0)`
from CIP-000E, which wraps `GibbsLockedFrame` (CIP-000C) around the
`near_bell_hamiltonian(d=3, delta=δ)`.

## Experiment Descriptions

### Exp 1 — Explicit Π_marg(K_0) for the departed qutrit frame

**Goal:** Compute the marginal projector Π_marg in modular-generator
coordinates and record the structure of the off-diagonal iso-marginal sector
as a function of δ.

**Method:** Construct `MatrixExponentialFamily(n_pairs=1, d=3, pair_basis=True)`
at ρ_0 = diag(1,1,exp(-β_0 δ))/Z. Compute Π_marg in natural-parameter
coordinates using the kernel-basis formula

```
Π_marg(θ*) = N (N^T G N)^{-1} N^T G
```

(equation `eq:second-order-projector` in Lawrence-origin26), where N spans
ker M(θ*) and M encodes the marginal-entropy constraint gradients. Push forward
to modular-generator K coordinates via the BKM metric. Record the dimension and
a Gell-Mann basis for the off-diagonal iso-marginal sector as a function of δ.

**Expected result:** For the bipartite structure, the iso-marginal sector
consists of operators whose partial traces vanish (correlation-only
perturbations). Its dimension should equal d⁴ - 2d² + 1 = (d²-1)² for
generic δ > 0, and grow at δ = 0 due to additional symmetry.

### Exp 2 — Iso-marginal sector check: Π_marg(K_0) R_od = R_od

**Goal:** Verify that all off-diagonal generators of the qutrit pair lie in
the iso-marginal sector and that Π_marg acts as the identity on them.

**Method:** For each off-diagonal Gell-Mann-based generator, construct R_od
and call `frame.is_iso_marginal(R_od)`. Check that the projector satisfies

```
Π_marg(K_0) R_od = R_od
```

for both in-block (indices 0,1) and cross-block (indices 0,2 and 1,2) modes.
Identify any generators that fall outside the sector (expected: diagonal
modes and δβ-directions).

**Expected result:** Off-diagonal generators with vanishing partial trace
satisfy the check; diagonal generators and the β-direction do not.

### Exp 3 — Loewner kernel and the two-sector amplitude structure

**Goal:** Verify the analytical Loewner weights for both mode types and
confirm the smooth degenerate limit.

**Method:** Compute `frame.loewner_kernel()` across a (β_0, δ) grid.
For in-block entries (0,1):

```
J_{ρ_0}(E)_{01} = λ_0 E_{01} = (1/Z) E_{01}
```

(degenerate limit of the divided difference). For cross-block entries (0,2):

```
J_{ρ_0}(E)_{02} = [(ρ_0 - ρ_2) / (K_{0,00} - K_{0,22})] E_{02}
                = [(1 - exp(-β_0 δ)) / (Z β_0 δ)] E_{02}
```

Plot both weights across the (β_0, δ) grid and verify convergence to the
above closed forms. Confirm that lim_{δ→0} [(1-exp(-β_0 δ))/(Z β_0 δ)] = 1/Z
(smooth degenerate limit, matching the in-block weight).

**Expected result:** In-block weight is constant at 1/Z for all δ.
Cross-block weight decreases monotonically in both β_0 and δ, converging
smoothly to 1/Z as δ → 0.

### Exp 4 — Uniform dephasing in modular-generator coordinates

**Goal:** Verify that the decay rate μ_0 = c ℏ is uniform across both
off-diagonal mode types in modular-generator coordinates, while the amplitude
weighting in density-matrix coordinates is set by the Loewner kernel.

**Method:** Construct an off-diagonal iso-marginal initial perturbation R_od
that mixes in-block and cross-block components. Run `frame.linearised_flow()`
and verify element-wise that

```
(d/dt R_od)_{ij} = (i β_0 Δε_{ij} - c ℏ) (R_od)_{ij}
```

with the same value of μ_0 = c ℏ for all (i,j) pairs (both in-block with
Δε=0 and cross-block with Δε=δ). Then apply the Loewner map

```
δρ = J_{ρ_0}(R_od)
```

and confirm that the decay rate in δρ_{ij}(t) remains uniform while the
initial amplitudes δρ_{ij}(0) differ by the Loewner weights from Exp 3.
Fit the decay rate using `infer_mu0` and compare to the analytical c ℏ.

**Expected result:** Single uniform decay rate μ_0 = c ℏ across all
off-diagonal modes. Initial amplitudes of δρ differ from R_od by the
mode-dependent Loewner weights but subsequent time evolution is at the
same rate.

### Exp 5 — ℏ as a function of (β_0, δ)

**Goal:** Compute ℏ = Var_{ρ_0}(K_0)^{-1} across the (β_0, δ) parameter
space and verify its limiting behaviour.

**Method:** For the departed qutrit,

```
Var_{ρ_0}(K_0) = β_0² Var_{ρ_0}(H_δ) = β_0² G_H(β_0)
```

The variance of H_δ = diag(0,0,δ) under ρ_0 = diag(1,1,exp(-β_0 δ))/Z is

```
Var_{ρ_0}(H_δ) = ⟨H_δ²⟩ - ⟨H_δ⟩²
               = (δ² exp(-β_0 δ) / Z) - (δ exp(-β_0 δ) / Z)²
               = δ² exp(-β_0 δ)(1 - exp(-β_0 δ)) / Z² + (δ² exp(-β_0 δ) / Z)(2/Z - 1/Z)
```

which simplifies to

```
Var_{ρ_0}(H_δ) = 2 δ² exp(-β_0 δ) / Z²
```

giving the closed form

```
ℏ = Z² / (2 β_0² δ² exp(-β_0 δ))
```

Compute this across a (β_0, δ) grid and verify:
- ℏ → ∞ as δ → 0 (Bell limit: no Hamiltonian direction is resolved)
- ℏ → ∞ as β_0 → 0 (infinite-temperature limit)
- ℏ → ∞ as β_0 δ → ∞ (Gibbs suppression of the separated level)
- ℏ has a minimum near x = β_0 δ ≈ 2.66 — best clock resolution

Since the Gibbs state depends on β_0 and δ only through x = β_0 δ, plot ℏ
as a function of x and confirm the minimum near x ≈ 2.66 with a broad
optimum (order-one width in x).

Cross-check μ_0 = c ℏ against `infer_mu0` fits from Exp 4 to verify the
entropy-production calibration.

**Expected result:** Closed-form ℏ verified numerically. Divergence confirmed
in all three limits (δ→0, β_0→0, β_0δ→∞). Minimum near x = β_0δ ≈ 2.66
with a broad optimum extending over an order-one range of x, confirming that
useful clock resolution does not require fine tuning of the dimensionless
spectral separation.

## Implementation

Experiments 2–5 use only existing `qig-code` machinery:

- `near_bell_gibbs_frame(d=3, delta=δ, beta=β_0)` — CIP-000E
- `GibbsLockedFrame.loewner_kernel()` — CIP-000C
- `GibbsLockedFrame.is_iso_marginal()` — CIP-000C
- `GibbsLockedFrame.linearised_flow()` — CIP-000C
- `infer_mu0()` — CIP-000C
- `MatrixExponentialFamily(n_pairs=1, d=3, pair_basis=True)` — existing

**Prerequisite for Exp 1 (Π_marg projector):**

Exp 1 requires the explicit matrix form of Π_marg in natural-parameter
coordinates:

```
Π_marg(θ*) = N (N^T G N)^{-1} N^T G
```

where N spans ker M(θ*) (null space of the constraint Jacobian) and G is the
BKM/Fisher metric. `MatrixExponentialFamily` currently exposes the *sum*
constraint gradient `marginal_entropy_constraint(theta)` but not the full
per-subsystem Jacobian M or the assembled projector. A new method
`pi_marg_matrix(theta)` must be added to `MatrixExponentialFamily` before
Exp 1 can be implemented.

This is tracked as a separate backlog task:
`backlog/features/2026-06-14_pi-marg-projector-method.md`

The experiments are implemented as a notebook
`examples/qutrit_gibbs_lock_clock_experiments.ipynb` that validates the
theory in Lawrence-hamiltonian26. Experiments 3, 5, 2, 4 can be implemented
immediately; Exp 1 depends on the backlog task above.

## Backward Compatibility

New notebook only; no API changes.

## Testing Strategy

The notebook itself serves as an integration test. Key assertions:
- Loewner weights match the analytical closed forms to within 1e-10
- Uniform decay rate μ_0 verified across all off-diagonal mode types
- ℏ matches the closed form Z²/(2 β_0² δ² exp(-β_0 δ)) to within 1e-10
- Π_marg acts as identity on all iso-marginal off-diagonal generators

## Implementation Status
- [x] **Prerequisite:** `MatrixExponentialFamily.pi_marg_matrix(theta)` — see
  `backlog/features/2026-06-14_pi-marg-projector-method.md`
- [x] Implement Exp 3: Loewner kernel two-sector structure and δ→0 limit
- [x] Implement Exp 5: ℏ(β_0, δ) closed form and numerical cross-check
- [ ] Implement Exp 2: Iso-marginal sector check for all Gell-Mann generators
- [x] Implement Exp 4: Uniform dephasing and Loewner amplitude weighting
- [ ] Implement Exp 1: Π_marg computation and iso-marginal sector basis
  (depends on prerequisite above)
- [ ] Notebook `examples/qutrit_gibbs_lock_clock_experiments.ipynb`

## References

- Lawrence-hamiltonian26 — companion paper defining the Hamiltonian clock
- Lawrence-origin26 — origin paper with Π_marg formula (`eq:second-order-projector`)
- `qig/gibbs_lock.py` — `GibbsLockedFrame`, `infer_mu0` (CIP-000C)
- `qig/pair_operators.py` — `near_bell_gibbs_frame`, `near_bell_hamiltonian` (CIP-000E)
- `qig/exponential_family.py` — `MatrixExponentialFamily`
- `cip/cip000C.md` — Gibbs-lock API
- `cip/cip000D.md` — Hamiltonian paper end-to-end companion
- `cip/cip000E.md` — near-Bell testbed constructors
- `backlog/features/2026-06-14_pi-marg-projector-method.md` — prerequisite for Exp 1
