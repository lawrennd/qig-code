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

The current Duhamel derivative implementation (`qig/duhamel.py`) uses **numerical integration** (trapezoid rule with 50 points) to compute Kubo-Mori derivatives:

```
∂ρ/∂θ_a = ∫₀¹ ρ^{1-s} (F_a - ⟨F_a⟩) ρ^s ds
```

However, according to the paper (lines 815, 1150, 1273):

> "when the operators {F_a} form a Lie algebra... derivatives can be evaluated using Baker–Campbell–Hausdorff identities **rather than explicit integration**"

> "The derivation relies on Lie closure: the adjoint action ρ^{-s} F_a ρ^s remains in the span of {F_b}, allowing the operator-ordered integral to be evaluated via BCH formulas, producing the commutator form."

The paper claims that with Lie closure, the BCH identity should hold:

```
∑_a η_a ∂_a ρ = -i[H_eff, ρ]    where H_eff = ∑_a η_a F_a
```

## Problem

**Empirical finding**: Testing this identity with the current implementation shows **~14x relative error**:

```python
# Test: sum_a eta_a * drho_dtheta_a  vs  -i[H_eff, rho]
||LHS||:       9.6087e-03
||RHS||:       6.9485e-04
||difference||: 9.6337e-03
Relative error: 1.3864e+01  # Should be ~1e-12 if BCH holds!
```

This suggests the numerical Duhamel integration is **not** exploiting Lie closure to reduce to the analytical BCH form.

## Possible Explanations

1. **Implementation gap**: The current code doesn't implement the BCH-based simplification
2. **Theory limitation**: The paper's claim may have unstated conditions or be approximate
3. **Numerical issue**: Integration accuracy insufficient (though 50-point trapezoid should give ~5e-5 error)
4. **Basis issue**: The pair basis may not satisfy the required Lie closure properties in the expected way

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

### Step 3: Implement BCH-based version
If Lie closure is confirmed, implement analytical evaluation:
```python
def duhamel_derivative_bch(rho, H, F_centered, f_abc, operators):
    """
    Use BCH formulas to evaluate Duhamel integral analytically
    for Lie-closed operator bases.
    
    The adjoint action Ad_{e^{-sH}}(F_a) = sum_b M_ab(s) F_b
    can be computed using BCH series, avoiding numerical integration.
    """
    # Use structure constants to compute evolution in Lie algebra
    # Return analytical result
```

### Step 4: Cross-validate
Compare BCH-based result with numerical integration at high precision (n_points=500) to determine ground truth.

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

- [ ] Understand root cause of BCH identity failure
- [ ] Either:
  - Implement BCH-based Duhamel that satisfies identity to ~1e-10, OR
  - Document why paper's claim doesn't hold and what the correct relationship is
- [ ] Update CIP-0009 verification to use corrected formulation
- [ ] Add tests verifying BCH identity with proper implementation
- [ ] Update documentation explaining Lie closure benefits and limitations

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
