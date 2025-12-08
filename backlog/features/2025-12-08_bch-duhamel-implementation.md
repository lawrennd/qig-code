---
id: "2025-12-08_bch-duhamel-implementation"
title: "Implement BCH-based Duhamel derivatives for Lie-closed bases"
status: "proposed"
priority: "high"
created: "2025-12-08"
updated: "2025-12-08"
tags:
  - duhamel
  - bch
  - lie-algebra
  - theoretical-validation
related_cips:
  - cip0009
---

# Task: Implement BCH-based Duhamel derivatives for Lie-closed bases

## Description

The current Duhamel derivative implementation (`qig/duhamel.py`) uses **numerical integration** (trapezoid rule with 50 points) to compute Kubo-Mori / Duhamel derivatives:

```
∂ρ/∂θ_a = ∫₀¹ ρ^{1-s} (F_a - ⟨F_a⟩) ρ^s ds
```

However, according to the paper (lines 815, 1150, 1273), when the operators {F_a} form a Lie algebra and the adjoint action stays in a finite-dimensional span, the **Duhamel integral can be evaluated analytically via the adjoint/BCH structure**, i.e. as a matrix function of the adjoint action:

- The Heisenberg-evolved operators remain in span{F_b}, so the Duhamel kernel
  \[
  X \mapsto \int_0^1 \rho^{1-s} X \rho^s\,\mathrm{d}s
  \]
  becomes a linear operator \(K_\rho = f(\mathrm{ad}_H)\) on the Lie algebra, with
  \(f(z) = (e^z - 1)/z\).
- This means we should be able to **avoid explicit numerical quadrature over s**, replacing it with a closed-form operator function (e.g. via spectral decomposition of H or an explicit adjoint/BCH representation).

The current backlog/test wording *over-interpreted* this into a much stronger claim:

> "with Lie closure, the BCH identity should hold
> \[
> \sum_a \eta_a \partial_a \rho = -i[H_\mathrm{eff}, \rho], \quad H_\mathrm{eff} = \sum_a \eta_a F_a
> \]
> for the \(\eta\) extracted from the antisymmetric GENERIC sector."

This equality is **not guaranteed** by Lie closure alone: the left-hand side lives in the image of the Kubo–Mori / Duhamel kernel \(K_\rho\), whereas the right-hand side is a pure commutator direction. Reconciling (or explicitly refuting) this strong identity is part of the task.

## Problem

**Empirical finding**: Testing this identity with the current implementation shows **~14x relative error**:

```python
# Test: sum_a eta_a * drho_dtheta_a  vs  -i[H_eff, rho]
||LHS||:       9.6087e-03
||RHS||:       6.9485e-04
||difference||: 9.6337e-03
Relative error: 1.3864e+01  # Should be ~1e-12 if BCH holds!
```

This suggests two issues:

1. The numerical Duhamel integration is **not** yet exploiting Lie closure to use an analytical (spectral/BCH) kernel instead of explicit quadrature.
2. More importantly, the **strong BCH identity being tested is likely too strong**: even with an exact Kubo–Mori kernel the map
   \(\eta \mapsto \sum_a \eta_a \partial_a \rho\) need not coincide with a bare commutator \(-i[H_\mathrm{eff},\rho]\) unless the Duhamel kernel is explicitly inverted in how \(\eta\) is defined.

## Possible Explanations

1. **Implementation gap**: The current code doesn't implement a BCH/spectral Duhamel kernel (it uses direct s-quadrature).
2. **Theory limitation / over-claim**: The paper/backlog currently state or suggest a stronger identity \(\sum_a \eta_a \partial_a \rho = -i[H_\mathrm{eff},\rho]\) that may **not** hold in general, even with an exact Kubo–Mori kernel.
3. **Numerical issue**: Integration accuracy might obscure finer structure (though 50-point trapezoid should give ~5e-5 error, far smaller than the ~O(1) discrepancy observed).
4. **Basis issue**: The pair basis may not satisfy the required Lie-closure properties in the expected way (or the antisymmetric GENERIC sector may not map to \(\eta\) in the way the strong identity assumes).

## Investigation Needed

### Step 1: Verify Lie closure empirically
Test whether the adjoint action stays in the span:
```python
# For Lie-closed basis, check: e^{-sH} F_a e^{sH} ∈ span{F_b}
# Compute for various s ∈ [0,1] and check linear independence
```

### Step 2: Review paper derivation
- Re-read lines 800-850, 1240-1280 carefully
- Check Appendix on Lie structure
- Look for any unstated assumptions about parameter regime, basis choice, etc.

### Step 3: Implement BCH/spectral Duhamel kernel
If Lie closure is confirmed, implement analytical evaluation:
```python
def duhamel_derivative_spectral(H, F_centered):
    """
    Use the spectral / adjoint representation of H to evaluate the
    Duhamel integral analytically:

        ∂ρ = ∫_0^1 e^{(1-s)H} F_centered e^{sH} ds
            = D exp_H[F_centered]
            = f(ad_H)[F_centered]

    where f(z) = (e^z - 1)/z is applied as a matrix function in a
    finite-dimensional representation (e.g. in the eigenbasis of H).

    This avoids explicit numerical quadrature over s while retaining
    the full Kubo–Mori structure.
    """
    # Implement via eigen-decomposition of H, or via explicit adjoint/BCH
    # in the Lie-algebra basis when convenient.
```

### Step 4: Cross-validate
Compare the spectral/BCH-based result with:

1. High-precision numerical integration (n_points=500) to establish that the kernel has been implemented correctly; and
2. The commutator \(-i[H_\mathrm{eff},\rho]\) used in CIP-0009, to determine **precisely how** the antisymmetric GENERIC sector maps into Hamiltonian evolution once the Kubo–Mori kernel is taken into account.

## Impact on CIP-0009

CIP-0009 (Hamiltonian extraction from antisymmetric flow) has been **implemented** with the following status:

✅ **Completed**:
- Extraction formula `A @ θ = f @ η` implemented
- Structural properties verified (H_eff is Hermitian, traceless)
- Internal consistency checked

❌ **Blocked by this issue**:
- Cannot verify `∑_a (A@θ)_a ∂_a ρ = -i[H_eff, ρ]` to paper-claimed accuracy
- Tests verify structural properties but not full dynamical equivalence

## Acceptance Criteria

- [ ] Implement a **BCH/spectral Duhamel kernel** (e.g. eigenbasis formula or explicit adjoint/BCH) that:
  - [ ] Matches high-precision numerical quadrature to ~1e-10 on representative examples, and
  - [ ] Avoids explicit s-quadrature in small finite-dimensional cases (qutrit pairs, etc.).
- [ ] Analyse the **strong BCH identity** currently assumed:
  - [ ] Determine whether there exist natural choices of \(\eta\) (possibly incorporating the inverse of the Kubo–Mori kernel) for which
        \(\sum_a \eta_a \partial_a \rho = -i[H_\mathrm{eff},\rho]\) holds, or
  - [ ] Clearly document why this equality does **not** hold in general and what the correct relationship is between the antisymmetric GENERIC sector, the Duhamel kernel, and \(H_\mathrm{eff}\).
- [ ] Update CIP-0009 verification to use the **corrected formulation**, i.e. tests that separately:
  - [ ] Verify structural Hamiltonian properties (already done), and
  - [ ] Verify the most accurate identity that actually holds, given the Kubo–Mori kernel structure.
- [ ] Add tests that:
  - [ ] Validate the BCH/spectral Duhamel implementation itself, and
  - [ ] Guard against reintroducing the over-strong identity without explicit justification.
- [ ] Update documentation (paper + docs) explaining:
  - [ ] How Lie closure enables analytic evaluation of the Duhamel kernel via BCH,
  - [ ] That the Kubo–Mori structure is encoded in a nontrivial operator function \(K_\rho = f(\mathrm{ad}_H)\), and
  - [ ] What is (and is not) true about identities of the form \(\sum_a \eta_a \partial_a \rho = -i[H_\mathrm{eff},\rho]\).

## References

**Paper** (`~/lawrennd/the-inaccessible-game-orgin/the-inaccessible-game-origin.tex`):
- Lines 815: BCH identities for Lie-closed bases
- Lines 1150: "evaluated using Baker–Campbell–Hausdorff identities rather than explicit integration"
- Lines 1273: "relies on Lie closure... allowing operator-ordered integral to be evaluated via BCH formulas"

**Code**:
- `qig/duhamel.py`: Current numerical implementation
- `qig/generic.py`: `verify_antisymmetric_flow_equals_commutator()` - reveals the discrepancy
- `tests/test_generic_hamiltonian.py`: Tests documenting the issue

**Related**:
- CIP-0009: Explicit Hamiltonian extraction (completed modulo this issue)
- CIP-0006: GENERIC decomposition (assumes Duhamel is correct)

## Priority Justification

**High priority** because:
1. Affects validity of core theoretical claim in the paper
2. Blocks full verification of CIP-0009
3. May indicate fundamental issue with Duhamel implementation used throughout codebase
4. If paper is correct, we're leaving ~10-100x performance on the table by not using BCH

## Progress Updates

### 2025-12-08
- Issue discovered during CIP-0009 implementation
- Empirical tests show ~14x relative error in BCH identity
- Current implementation uses numerical integration, not BCH formulas
- Backlog task created to track investigation
