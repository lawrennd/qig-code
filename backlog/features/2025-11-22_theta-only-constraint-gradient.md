---
id: "2025-11-22_theta-only-constraint-gradient"
title: "Implement θ-only constraint gradient using BKM inner products"
status: "Proposed"
priority: "High"
created: "2025-11-22"
last_updated: "2025-11-22"
owner: "TBD"
github_issue: ""
dependencies: "None (pure optimization)"
tags:
- backlog
- optimization
- performance
- constraint-gradient
---

# Task: θ-only Constraint Gradient

## Description

Rewrite `marginal_entropy_constraint()` to compute the gradient ∇C using **θ-only formulas** via BKM inner products, eliminating expensive ∂ρ/∂θ materialization.

### Current Approach (Slow)
```python
# For each parameter a:
drho_a = self.rho_derivative(theta, a, method='duhamel')  # Expensive!
for i in range(n_sites):
    drho_i_a = partial_trace(drho_a, self.dims, keep=i)
    grad_C[a] += -Tr(drho_i_a @ log(rho_i))
```

**Cost**: O(n_params × n_sites × Duhamel)  
**Problem**: Materializes drho for every parameter (15-80 Duhamel integrations!)

### New Approach (Fast θ-only)
For the constraint C = ∑ᵢ hᵢ where hᵢ = -Tr(ρᵢ log ρᵢ):

```
∂C/∂θ_a = ⟨F̃_a, B⟩_BKM
```

where:
- F̃_a = F_a - ⟨F_a⟩I (centered operator)
- B = ∑ᵢ Bᵢ (lifted test operator)
- Bᵢ = (log ρᵢ + Iᵢ) ⊗ I_rest (lift marginal log to full space)
- ⟨·,·⟩_BKM is the BKM inner product already computed in `fisher_information()`

**Cost**: O(n_sites × eigen)  
**Speedup**: ~100× faster (no drho materialization!)

### Motivation

1. **Speed**: Current implementation is dominated by n_params Duhamel calls
   - Qubit pair (15 params): 15 Duhamel integrations
   - Qutrit pair (80 params): 80 Duhamel integrations!
   
2. **Accuracy**: BKM inner product uses exact eigendecomposition
   - No Duhamel integration errors
   - No SLD approximation errors
   - Machine precision (~10⁻¹⁴)

3. **Memory**: Reuses eigendecomposition from Fisher metric G
   - Compute ρ = U diag(p) U† once
   - Use same BKM kernel k(pᵢ, pⱼ) for all gradients
   - No storage of O(n_params) drho matrices

4. **Conceptual Clarity**: "Constraint gradient = BKM inner product with lifted log"
   - Matches theoretical derivation in exponential family theory
   - Same kernel as Fisher metric (Legendre duality manifest)

## Acceptance Criteria

### Implementation
- [ ] Implement `_lift_to_full_space(op_i, site_i, dims)` helper function
  - Takes operator on subsystem i
  - Returns operator on full Hilbert space via ⊗I_rest
  - Adjoint of `partial_trace()`

- [ ] Implement `marginal_entropy_constraint_theta_only(theta)`
  - Compute ρ = exp(K)/Z and eigendecompose once
  - Build BKM kernel k(pᵢ, pⱼ) once
  - Build lifted B = ∑ᵢ (log ρᵢ + Iᵢ) ⊗ I_rest
  - Compute gradient via BKM inner products
  - Return (C, grad_C) matching current API

- [ ] Make θ-only method the **default** in `marginal_entropy_constraint()`
  - Add optional `method` parameter: 'theta_only' (default), 'materialize_drho' (legacy)
  - Keep old implementation for verification only

### Performance
- [ ] Achieve **≥50× speedup** for 15-parameter systems
- [ ] Achieve **≥100× speedup** for 80-parameter systems
- [ ] Memory usage: O(D²) regardless of n_params

### Accuracy
- [ ] Match current Duhamel method to **≤10⁻¹⁰ absolute error**
- [ ] Maintain machine precision (~10⁻¹⁴) for constraint value C

### Testing
- [ ] Unit test: `test_lift_to_full_space()`
  - Verify adjoint property: Tr(op_i A) = Tr(lift(op_i) full_A) after partial trace
  - Test for qubits, qutrits, mixed dimensions

- [ ] Validation test: `test_theta_only_matches_duhamel()`
  - Compare θ-only vs current Duhamel method
  - Tolerance: 1e-10 absolute error
  - Test at random θ for single/two pairs

- [ ] Performance benchmark: `test_theta_only_speedup()`
  - Measure time for both methods on 80-parameter qutrit pair
  - Assert speedup ≥ 50×

- [ ] Integration test: Update existing tests to use θ-only by default
  - `test_pair_numerical_validation.py::TestConstraintGradient`
  - Verify no regressions in accuracy

## Implementation Notes

### Lifted Operator Construction

For a d-dimensional operator Aᵢ on subsystem i, the lift to full space is:

```python
def _lift_to_full_space(op_i, site_i, dims):
    """
    Lift operator on subsystem i to full Hilbert space.
    
    Result: I₀ ⊗ ... ⊗ I_{i-1} ⊗ op_i ⊗ I_{i+1} ⊗ ... ⊗ I_{n-1}
    """
    result = None
    for j, d_j in enumerate(dims):
        if j == site_i:
            current = op_i
        else:
            current = np.eye(d_j, dtype=complex)
        result = current if result is None else np.kron(result, current)
    return result
```

This is the **adjoint** of `partial_trace()`:
- `partial_trace` maps full → marginal (trace out j≠i)
- `_lift_to_full_space` maps marginal → full (tensor with I on j≠i)

### BKM Kernel Reuse

Extract the BKM kernel computation as a helper method:

```python
def _bkm_kernel(self, rho):
    """Compute BKM kernel k(pᵢ, pⱼ) from eigenvalues of ρ."""
    p, U = eigh(rho)
    p = np.clip(p.real, 1e-14, None)
    
    p_i = p[:, None]
    p_j = p[None, :]
    diff = p_i - p_j
    log_diff = np.log(p_i) - np.log(p_j)
    
    k = np.zeros_like(diff)
    off = np.abs(diff) > 1e-14
    k[off] = diff[off] / log_diff[off]
    k[np.diag_indices(len(p))] = p
    
    return k, p, U
```

### Gradient Computation

```python
# Get eigendecomposition and kernel
k, p, U = self._bkm_kernel(rho)

# Build lifted test operator B
B_full = np.zeros((self.D, self.D), dtype=complex)
for i in range(self.n_sites):
    rho_i = partial_trace(rho, self.dims, keep=i)
    log_rho_i = logm_safe(rho_i)  # Use eigh internally
    B_i = log_rho_i + np.eye(self.dims[i])
    B_full += self._lift_to_full_space(B_i, i, self.dims)

# Transform to eigenbasis
B_tilde = U.conj().T @ B_full @ U

# Compute centered operators in eigenbasis
grad_C = np.zeros(self.n_params)
I_full = np.eye(self.D, dtype=complex)
for a, F_a in enumerate(self.operators):
    mean_Fa = np.trace(rho @ F_a).real
    F_tilde = F_a - mean_Fa * I_full
    F_tilde_eigen = U.conj().T @ F_tilde @ U
    
    # BKM inner product: -∑ᵢⱼ k[i,j] F_tilde[i,j] conj(B_tilde[i,j])
    grad_C[a] = -np.real(np.sum(k * (F_tilde_eigen * np.conj(B_tilde))))

return C, grad_C
```

## Related

- CIP: None (pure optimization, no API changes)
- Depends on: Existing `fisher_information()` BKM implementation
- Blocks: `2025-11-22_theta-only-constraint-hessian` (Hessian optimization)

## Progress Updates

### 2025-11-22

Task created based on user-provided mathematical derivation showing:
- Current approach materializes drho unnecessarily (O(n_params) Duhamel calls)
- θ-only approach via BKM inner products is exact and ~100× faster
- Indentation bug in `__init__` fixed before starting this work
