---
id: "2025-11-22_analytic-jacobian-implementation"
title: "Implement Analytic Jacobian for Quantum Dynamics"
status: "In Progress"
priority: "High"
created: "2025-11-22"
last_updated: "2025-11-22"
owner: Neil D. Lawrence
github_issue: ""
dependencies: "2025-11-22_remove-numerical-gradients (partial - BKM metric and marginal entropy gradient done)"
tags:
- backlog
- infrastructure
- quantum
- jacobian
---

# Task: Implement Analytic Jacobian for Quantum Dynamics

## Description

Implement the analytic Jacobian M = ∂F/∂θ for the quantum information game dynamics, following the exact derivation in the paper's Appendix.

The Jacobian is given by (eq. 842):

**M = -G - (∇G)[θ] + ν ∇²C + a(∇ν)ᵀ**

where:
- G = BKM metric (Fisher information) - **already implemented and validated ✅**
- (∇G)[θ] = third cumulant tensor contracted with θ
- ∇²C = constraint Hessian (second derivative of ∑ᵢ hᵢ)
- ν = Lagrange multiplier = (aᵀGθ)/(aᵀa)
- ∇ν = gradient of Lagrange multiplier (eq. 835-836)
- a = ∇C = constraint gradient - **already implemented and validated ✅**

## Motivation

The current implementation uses finite differences for the Jacobian, which is:
- Slow (requires O(n) flow evaluations)
- Numerically unstable (sensitive to step size)
- Inaccurate for stiff systems

An analytic Jacobian will:
- Speed up dynamics integration
- Improve numerical stability
- Enable better analysis of GENERIC structure
- Provide exact decomposition into symmetric/antisymmetric parts

## Acceptance Criteria

- [ ] Third cumulant (∇G)[θ] implemented and validated
- [ ] Constraint Hessian ∇²C implemented and validated
- [ ] Lagrange multiplier gradient ∇ν implemented and validated
- [ ] Full Jacobian M assembled and validated
- [ ] All components match finite differences to < 10⁻⁵ relative error
- [ ] GENERIC degeneracies verified: Sa ≈ 0, A∇H ≈ 0
- [ ] Tests pass for: diagonal, single qubit, two qubit cases
- [ ] All quantum derivative principles applied at each step

## Related

- CIP: None
- Paper: the-inaccessible-game.tex Appendix (eq. 821-846)
- Dependencies: BKM metric (validated), marginal entropy gradient (validated)

## Implementation Notes

### Step 1: Third Cumulant (∇G)[θ]

**Goal**: Compute ∂G_ab/∂θ_c for all a,b,c

**Approach**: Differentiate the spectral BKM formula using perturbation theory
- Eigenvalue derivatives: ∂λ_i/∂θ = ⟨i|∂ρ/∂θ|i⟩ (Hellmann-Feynman)
- Eigenvector derivatives: ∂|i⟩/∂θ = ∑_{j≠i} (⟨j|∂ρ/∂θ|i⟩)/(λ_i - λ_j) |j⟩
- Apply product rule to spectral BKM sum

**Tests**:
- Diagonal case: Should match classical third cumulant
- Single qubit: Compare with finite differences
- Symmetry: ∂G_ab/∂θ_c = ∂G_ba/∂θ_c
- Total symmetry: ∇³ψ is symmetric in all three indices

**Files to create**:
- `test_third_cumulant.py`: Validation tests
- Add method to `QuantumExponentialFamily`: `third_cumulant(theta)`

### Step 2: Constraint Hessian ∇²C

**Goal**: Compute ∂²C/∂θ_a∂θ_b where C = ∑ᵢ hᵢ

**Approach**: Use Daleckii-Krein formula for ∂(log ρᵢ)/∂θ
- For each marginal i:
  - Compute ∂ρᵢ/∂θ_a (partial trace of ∂ρ/∂θ_a)
  - Compute ∂²ρᵢ/∂θ_a∂θ_b (partial trace of second derivative)
  - Compute ∂(log ρᵢ)/∂θ_b using Daleckii-Krein
  - Combine: ∂²hᵢ/∂θ_a∂θ_b = -Tr(∂²ρᵢ/∂θ_a∂θ_b (I + log ρᵢ)) - Tr(∂ρᵢ/∂θ_a ∂(log ρᵢ)/∂θ_b)
- Sum over all marginals

**Daleckii-Krein formula** (in eigenbasis of A):
```
[∂ log A/∂x]_ij = {
    (∂A/∂x)_ij * (log λ_i - log λ_j)/(λ_i - λ_j)  if i ≠ j
    (∂A/∂x)_ii / λ_i                                if i = j
}
```

**Tests**:
- Symmetry: ∇²C must be symmetric
- Finite differences of ∂C/∂θ (already validated)
- Diagonal case validation

**Files to modify**:
- `test_constraint_hessian.py`: New test file
- Add method to `QuantumExponentialFamily`: `constraint_hessian(theta)`

### Step 3: Lagrange Multiplier Gradient ∇ν

**Goal**: Compute ∂ν/∂θ_j for all j

**Formula** (eq. 835-836):
```
∂ν/∂θ_j = (1/||a||²) [
    aᵀG e_j                      # G applied to basis vector
  + aᵀ(∇G)[θ] e_j                # Third cumulant term
  + (∇a)_jᵀ Gθ                   # Constraint Hessian times Gθ
  - 2ν aᵀ(∇a)_j                  # Normalization correction
]
```

**Tests**:
- Finite differences of ν(θ)
- Check on diagonal case
- Verify formula structure

**Files to modify**:
- `test_lagrange_multiplier_gradient.py`: New test file
- Add method to `QuantumExponentialFamily`: `lagrange_multiplier_gradient(theta)`

### Step 4: Assemble and Validate Jacobian

**Goal**: M = -G - (∇G)[θ] + ν ∇²C + a(∇ν)ᵀ

**Tests**:
- Compare M with finite-difference Jacobian
- Check GENERIC properties:
  - S = ½(M + Mᵀ) is symmetric
  - A = ½(M - Mᵀ) is antisymmetric
  - Sa ≈ 0 (first degeneracy)
  - A∇H ≈ 0 (second degeneracy)
- Test on diagonal, single qubit, two qubit cases
- Verify eigenvalue structure

**Files to modify**:
- `test_jacobian_analytic.py`: Update with new implementation
- Add method to `QuantumExponentialFamily`: `jacobian(theta)`

## Quantum Derivative Principles

At every step, verify:
1. ✅ Check operator commutation
2. ✅ Verify operator ordering (ABC ≠ CBA for non-commuting)
3. ✅ Distinguish quantum vs classical (no classical shortcuts)
4. ✅ Respect Hilbert space structure (tensor products, partial traces)
5. ✅ Question each derivative step (derive from first principles)

## Progress Updates

### 2025-11-22 - Step 1 Complete ✅
- Task created
- Detailed implementation plan written
- **Step 1 (Third cumulant) COMPLETE**:
  - Implemented `third_cumulant_contraction()` using perturbation theory
  - All tests passing with excellent precision:
    - Diagonal (qutrit): rel_err = 1.97e-09
    - Single qubit: rel_err = 1.03e-08
    - Two qubits: rel_err = 1.50e-09
    - Two qutrits: rel_err = 1.98e-09
  - Symmetry verified: ∂G/∂θ_c is symmetric for all c

### 2025-11-22 - Step 2 In Progress (Debugging) 🔧
- **Step 2 (Constraint Hessian) - Issue identified**:
  - Implemented `constraint_hessian()` with Daleckii-Krein formula
  - Tests show errors:
    - Diagonal case: 8.4% relative error
    - Single qubit: 0.3% error on off-diagonal elements
  
- **Root cause identified**: ∂²ρ/∂θ_a∂θ_b formula has ~10% error
  
- **Diagnostic results**:
  ```
  For single qubit with θ = [0.1, 0.0, 0.0]:
  ∂²ρ/∂θ_X∂θ_Y analytic vs finite-diff:
  Max error: 0.050 (10% of magnitude)
  ```
  
- **Problem**: When computing ∂²ρ/∂θ_a∂θ_b from ∂ρ/∂θ_a = ρ(F_a - ⟨F_a⟩I), 
  the product rule application is incomplete.
  
  Current formula:
  ```
  ∂²ρ/∂θ_a∂θ_b = ∂ρ/∂θ_b (F_a - ⟨F_a⟩I) - ρ Cov(F_b, F_a) I
  ```
  
  This is missing terms! Need to carefully apply product rule to:
  ```
  ∂/∂θ_b [ρ(F_a - ⟨F_a⟩I)]
  ```
  
  Should be:
  ```
  = (∂ρ/∂θ_b)(F_a - ⟨F_a⟩I) + ρ(∂F_a/∂θ_b - ∂⟨F_a⟩/∂θ_b I)
  ```
  
  But F_a is constant (doesn't depend on θ), so ∂F_a/∂θ_b = 0.
  However, ⟨F_a⟩ = Tr(ρ F_a) DOES depend on θ_b!
  
  So:
  ```
  = (∂ρ/∂θ_b)(F_a - ⟨F_a⟩I) - ρ (∂⟨F_a⟩/∂θ_b) I
  ```
  
  And ∂⟨F_a⟩/∂θ_b = Tr((∂ρ/∂θ_b) F_a) = Cov(F_b, F_a) = G_ba
  
  Wait... that's what I have! So the formula looks correct in principle.
  
- **Next steps**:
  - Re-examine the formula derivation from first principles
  - Check if there's an issue with how I'm computing Cov(F_b, F_a)
  - Verify the partial trace is being applied correctly
  - Consider whether the issue is in the Daleckii-Krein application instead
  - Test on even simpler case (pure state?) to isolate the issue

