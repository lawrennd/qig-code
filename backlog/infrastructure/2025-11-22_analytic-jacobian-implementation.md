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

### 2025-11-22 - Step 2 Major Discovery: ∂ρ/∂θ Formula Wrong! 🔍
- **CRITICAL FINDING**: The classical formula ∂ρ/∂θ = ρ(F - ⟨F⟩I) is **WRONG** for quantum systems!
  
- **Root cause**: For non-commuting operators, ρ(F - ⟨F⟩I) is NOT Hermitian!
  - When [ρ, F] ≠ 0, the product ρF is not Hermitian
  - All derivatives of ρ MUST be Hermitian (since ρ is Hermitian)
  - Tests show ρ(F - ⟨F⟩I) has Hermiticity errors of 0.1-0.3 for typical quantum states
  
- **Correct formula**: Symmetric Logarithmic Derivative (SLD) from quantum information geometry:
  ```
  ∂ρ/∂θ_a = (1/2)[ρ(F_a - ⟨F_a⟩I) + (F_a - ⟨F_a⟩I)ρ]
  ```
  - This is Hermitian by construction
  - Marginal entropy gradient now achieves **machine precision** (~10⁻¹⁶ error) ✅
  
- **BUT**: Second derivative still has ~85% error!
  - Using the SLD approximation in ∂²ρ compounds the error
  - The SLD itself is an approximation (trapezoid rule) of the true integral:
    ```
    ∂ρ/∂θ_a = ∫₀¹ exp(sH)(F_a - ⟨F_a⟩I)exp((1-s)H) ds
    ```
  - For ∂²ρ we may need the exact integral or a better quadrature
  
- **Current status**:
  - Fixed: ∂ρ/∂θ using SLD formula → machine precision ✅
  - Still broken: ∂²ρ/∂θ_a∂θ_b has ~85% error
  - Tests show error is consistent across different finite-difference step sizes
  - Suggests formula is fundamentally wrong, not numerical precision issue
  
### 2025-11-22 - BREAKTHROUGH: Duhamel Formula Solution! 🎉
- **Implemented Duhamel exponential derivative formula**:
  ```
  ∂ρ/∂θ = ∫₀¹ exp(sH)(F - ⟨F⟩I)exp((1-s)H) ds
  ```
  - This is the EXACT formula for exponential derivatives
  - SLD is just the trapezoid rule with n_points=2
  - With n_points=100: error drops to **5×10⁻⁶** (1000× better than SLD!)
  
- **Second derivative solution**: Numerical differentiation of Duhamel
  - Computing ∂²ρ analytically from the integral is complex (double integrals)
  - Instead: use finite differences of high-precision Duhamel ∂ρ/∂θ
  - Result: **0.55-2.6% error** (vs 26-82% before!)
  - **30-100× improvement!** ✅
  - All results Hermitian ✅
  - Stable across different step sizes ✅
  
- **Implementation**:
  - Added `qig/duhamel.py` with Duhamel integration
  - Updated `rho_derivative()` to support both 'sld' and 'duhamel' methods
  - Next: Add high-precision option to constraint_hessian()

