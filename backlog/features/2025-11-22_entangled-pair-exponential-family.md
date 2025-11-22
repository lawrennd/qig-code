---
id: 2025-11-22_entangled-pair-exponential-family
title: Implement exponential family with entangled pairs and interaction terms
status: Proposed
priority: High
created: 2025-11-22
last_updated: 2025-11-22
owner: TBD
tags: [quantum, entanglement, exponential-family, pairs]
dependencies: []
---

# Task: Implement exponential family with entangled pairs and interaction terms

## Description

The current `QuantumExponentialFamily` implementation uses **local operators only** (e.g., σ_x⊗I, I⊗σ_y), which can ONLY represent separable states. This makes it fundamentally inadequate for the quantum inaccessible game, which:

1. **Starts at maximally entangled pairs** (Bell states for qubits, qutrit equivalents)
2. **Requires entanglement to evolve** away from the origin
3. **Studies information flow** between entangled subsystems

The implementation must be updated to support:
- Systems composed of **entangled pairs** as the fundamental unit
- Operators that act on the **pair subspace** (su(4) for qubit pairs, su(9) for qutrit pairs)
- Evolution dynamics starting from maximally entangled states
- Larger systems with multiple pairs
- Scalable to n pairs with arbitrary local dimension d

## Motivation

### Current Limitations

The existing implementation with local operators {F_a = σ_i ⊗ I} produces states of the form:
```
ρ ∝ exp(K₁⊗I + I⊗K₂) = exp(K₁) ⊗ exp(K₂)  [if commuting]
```
This gives **ONLY separable states**, meaning:
- No entanglement: I = C - H = 0 always
- Cannot represent Bell states or any entangled configuration
- The constraint C = ∑h_i equals total entropy H
- Dynamics are effectively classical (Legendre duality)

### Required for Quantum Game

The quantum inaccessible game requires:
1. **Origin**: Product of maximally entangled pairs
   - Qubit pair: |Φ⟩ = (|00⟩ + |11⟩)/√2
   - Qutrit pair: |Φ⟩ = (|00⟩ + |11⟩ + |22⟩)/√3
   - Properties: S(ρ) = 0 (pure), S(ρ_A) = S(ρ_B) = log(d) (maximally mixed locally)

2. **Dynamics**: Evolution away from origin
   - Breaking entanglement while conserving marginal entropies
   - Exploring I > 0 regions 

3. **Scalability**: Multiple pairs
   - 2 qubit pairs (16D Hilbert space)
   - 2 qutrit pairs (81D Hilbert space)
   - General: n pairs, each of dimension d

## Acceptance Criteria

### 1. Pair-Based Operator Basis

**Implement operators for pair subspaces:**
- [ ] For qubit pair (d=2): 15 su(4) generators (traceless Hermitian)
  - 12 off-diagonal: symmetric and antisymmetric
  - 3 diagonal: traceless combinations
- [ ] For qutrit pair (d=3): 8 su(9) generators  
- [ ] For general d: d²-1 su(d²) generators

**Validation:**
```python
exp_family = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
assert exp_family.n_params == 15  # Full su(4)
assert exp_family.D == 4  # 2×2 Hilbert space
```

### 2. Maximally Entangled Initial States

**Implement methods to:**
- [ ] Generate maximally entangled pair states
  ```python
  rho_bell = exp_family.maximally_entangled_pair(d=2)
  # Should give |Φ⟩⟨Φ| where |Φ⟩ = (∑_j |jj⟩)/√d
  ```
- [ ] Compute parameters θ* near maximally entangled state
  - Pure state is at θ → ∞, need regularization
  - Use entropy-time parametrization approach
- [ ] Verify properties:
  - S(ρ) ≈ 0 (globally pure)
  - S(ρ_A) = S(ρ_B) = log(d) (locally maximally mixed)
  - I = 2log(d) (maximum mutual information)

### 3. Entanglement Metrics

**Add methods to track entanglement:**
- [ ] Mutual information: `I(θ) = ∑h_i - H`
- [ ] Von Neumann entropy: `H(θ) = -Tr(ρ log ρ)`
- [ ] Purity: `Tr(ρ²)`
- [ ] Entanglement detection: verify I > 0 is achievable

**Validation:**
```python
# Test that entanglement can be created
theta = np.random.randn(15) * 0.5
rho = exp_family.rho_from_theta(theta)
I = exp_family.mutual_information(theta)
assert I > 0.01, "Should be able to create entangled states"
```

### 4. Multiple Pairs

**Support systems of n pairs:**
- [ ] `QuantumExponentialFamily(n_pairs=2, d=2)` → 30 parameters
- [ ] Direct sum structure: operators act independently on each pair
- [ ] Marginal entropies: one per subsystem (2n subsystems for n pairs)
- [ ] Constraint: C = ∑_{i=1}^{2n} h_i

**Example:**
```python
# 2 Bell pairs
exp_family = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
assert exp_family.n_params == 30  # 15 operators per pair
assert exp_family.D == 16  # 4×4 Hilbert space
assert len(exp_family.dims) == 4  # 4 subsystems
```

### 5. Evolution from Origin

**Implement and test dynamics:**
- [ ] Verify F(θ*) at origin (maximally entangled)
  - Is F = 0 at origin? (equilibrium)
  - Or F ≠ 0? (dynamics exist)
- [ ] Check structural identity Gθ = -∇C with entanglement
  - Should NOT hold if C ≠ H
- [ ] Compute trajectories θ(t)
- [ ] Track I(t), H(t), C(t) along trajectories

### 6. Backward Compatibility

**Ensure existing tests still work:**
- [ ] Single qubit tests (when n_pairs=None, use local basis)
- [ ] BKM metric tests
- [ ] Third cumulant tests
- [ ] Jacobian tests

**Add mode flag:**
```python
# Old behavior (local operators)
exp_family_local = QuantumExponentialFamily(n_sites=2, d=2)

# New behavior (pair operators)
exp_family_pairs = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
```

## Implementation Notes

### Architecture Changes

**Files to modify:**
1. `qig/exponential_family.py`:
   - Add `pair_basis` parameter to constructor
   - Implement su(4), su(9), su(d²) generator functions
   - Add `maximally_entangled_pair()` method
   - Add `mutual_information()` method
   - Update `dims` handling for pairs

2. New file `qig/pair_operators.py`:
   - `gell_mann_generators(d: int)` → d²-1 traceless Hermitian matrices
   - `multi_pair_basis(n_pairs: int, d: int)` → direct sum structure
   - `bell_state(d: int)` → maximally entangled state

3. New tests `test_pair_exponential_family.py`:
   - Test su(4) generators properties
   - Test maximally entangled states
   - Test entanglement creation
   - Test dynamics from origin
   - Test Gθ = -∇C with entanglement

### Technical Considerations

**1. Constraint-enforced sparsity (CRITICAL INSIGHT)**

The marginal entropy constraint C = ∑h_i = const is **highly restrictive** and likely enforces sparse structure:

- **Starting point**: Maximally entangled pairs → locally maximally mixed marginals
- **Constraint locks marginals**: e.g. perhaps h_i ≈ log(d) must be preserved
- **Limited entanglement spreading**: Cannot create arbitrary multi-party entanglement while maintaining constant marginals
- **Sparse interactions**: Pairs remain relatively independent; full 2n-body entanglement is likely inaccessible

**Implications for implementation:**
- May NOT need full Hilbert space exploration
- Operators acting independently on each pair (direct sum structure) might be sufficient
- Cross-pair interactions, if needed, likely sparse (nearest-neighbor or pairwise only)
- Effective dynamics occur in low-dimensional manifold within large Hilbert space
- If this is true then we should see the sparsity in the gradients. 

**Validation strategy:**
- Track accessible subspace dimension during dynamics
- Monitor rank and support of states
- Check if cross-pair entanglement ever forms
- Verify computational cost scales linearly with n (not exponentially)

**Implementation approach (following tensor product structure):**
1. **Start at origin**: Product of maximally entangled pairs defines natural coordinates
2. **Use direct sum operators**: F_a^(i) = I⊗...⊗F_a⊗...⊗I (acts only on pair i)
3. **Derivatives decompose naturally**: ∂ρ/∂θ_a involves only pair i → block structure is automatic
4. **Verify sparsity in gradients**: G should be block-diagonal, M should have clear block structure
5. **Tensor products make computation easy**: Don't need to compute full exponentials, use Kronecker structure

**2. Entropy-time parametrization**
- Pure states (θ → ∞) are singular in exponential family
- Either 
  - need to start "close to" origin, not exactly at it
    - Use regularized states: ρ_ε = (1-ε)|Φ⟩⟨Φ| + ε I/d²
  - Or condiser the reparameterised time that removes the singularity??

**3. Operator basis choice (tensor product structure)**

**Direct sum structure (the natural choice):**
```
For n pairs, each with su(d²) generators {F_α^(i)}, α=1,...,d²-1:

Operators: F_α^(i) = I ⊗ ... ⊗ F_α ⊗ ... ⊗ I  (F_α at position i)
           └────────┘         └─┘         └────┘
           pairs 1..i-1     pair i      pairs i+1..n

Total: n(d²-1) parameters
```

**Why this works:**
- ρ = exp(∑_i ∑_α θ_α^(i) F_α^(i)) / Z
- Density matrix factorizes: ρ ∝ ρ^(1) ⊗ ρ^(2) ⊗ ... ⊗ ρ^(n) if pairs independent
- Derivatives respect structure: ∂ρ/∂θ_α^(i) = (∂ρ^(i)/∂θ_α) ⊗ (other pairs unchanged)
- Fisher metric G is block-diagonal: G_αβ^(ij) = 0 if i≠j
- Constraint gradient ∇C decomposes by pair
- **Sparsity is automatic from tensor product structure**

**Implementation using Kronecker products:**
```python
# For pair i, operator α
F_alpha_i = np.eye(d**2) ⊗ ... ⊗ F_alpha ⊗ ... ⊗ np.eye(d**2)

# Better: Store only the non-identity factor and use Kronecker rules
# Derivatives can be computed pair-by-pair
```

**Expected computational pattern:**
- Each pair contributes (d²-1)×(d²-1) block to Fisher metric
- Total G is block-diagonal with n blocks
- Constraint Hessian ∇²C similarly structured
- Jacobian M inherits block structure
- **Cost scales as O(n·d⁴), not O(d^(4n))**

**4. Computational scaling**

**With sparse structure (Option A):**
- Qubit pair: 4×4 = 16D, 15 parameters → trivial
- Qutrit pair: 9×9 = 81D, 80 parameters → manageable
- 2 qubit pairs: 16×16 = 256D, **30 parameters** (NOT 255!) → very feasible
- 3 qubit pairs: 64×64 = 4096D, **45 parameters** → still tractable
- n qubit pairs: 4ⁿ×4ⁿ Hilbert space, **15n parameters** → linear scaling!

**Without sparse structure (full tensor):**
- 2 qubit pairs: 256D Hilbert, (256²-1) = 65535 parameters → intractable
- Game would be computationally infeasible

**Key hypothesis**: Constraint enforces sparsity → linear scaling is achievable

**5. Numerical stability**
- BKM metric for highly entangled states can be ill-conditioned
- Use higher precision (float64) throughout
- Regularize small eigenvalues in Fisher information
- Test Duhamel integral convergence for entangled states
- Monitor condition number of G during dynamics

## Related

- Paper sections:
  - Section 1.2: Origin structure (maximally entangled pairs)
  - Section 6: Qutrit optimality
  - Lemma 3.1: Characterization of locally maximally entangled states
  - Appendix: GENERIC decomposition with entanglement

- Related backlog items:
  - 2025-11-22_analytic-jacobian-implementation.md (completed, but needs re-testing with entanglement)

- CIPs:
  - Consider creating CIP for major refactoring of exponential family architecture

## Progress Updates

### 2025-11-22
- Task created with Proposed status
- Identified current limitation: local operators only produce separable states
- Determined acceptance criteria and implementation strategy
- Priority: High (blocks proper testing of quantum game dynamics)

