---
id: 2025-11-22_entangled-pair-exponential-family
title: Implement exponential family with entangled pairs and interaction terms
status: Completed
priority: High
created: 2025-11-22
last_updated: 2025-11-22
owner: Neil D. Lawrence
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

### 1. Pair-Based Operator Basis ✅ COMPLETED

**Implement operators for pair subspaces:**
- [x] For qubit pair (d=2): 15 su(4) generators (traceless Hermitian)
  - 12 off-diagonal: symmetric and antisymmetric
  - 3 diagonal: traceless combinations
- [x] For qutrit pair (d=3): 80 su(9) generators  
- [x] For general d: d²-1 su(d²) generators

**Validation:**
```python
exp_family = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
assert exp_family.n_params == 15  # Full su(4)
assert exp_family.D == 4  # 2×2 Hilbert space
```
✅ All tests pass (test_pair_exponential_family.py)

### 2. Maximally Entangled Initial States ✅ COMPLETED

**Implement methods to:**
- [x] Generate maximally entangled pair states
  ```python
  rho_bell = bell_state_density_matrix(d=2)
  # Gives |Φ⟩⟨Φ| where |Φ⟩ = (∑_j |jj⟩)/√d
  ```
- [x] Placeholder for parameters θ* near maximally entangled state
  - `get_bell_state_parameters(epsilon)` method added
  - Note: Full inverse exponential family map deferred
- [x] Verify properties:
  - S(ρ) = 0 (globally pure) ✅
  - S(ρ_A) = S(ρ_B) = log(d) (locally maximally mixed) ✅
  - I = 2log(d) (maximum mutual information) ✅

### 3. Entanglement Metrics ✅ COMPLETED

**Add methods to track entanglement:**
- [x] Mutual information: `I(θ) = ∑h_i - H`
- [x] Von Neumann entropy: `H(θ) = -Tr(ρ log ρ)`
- [x] Purity: `Tr(ρ²)`
- [x] Entanglement detection: verify I > 0 is achievable

**Validation:**
```python
# Test that entanglement can be created
theta = np.random.randn(15) * 0.5
rho = exp_family.rho_from_theta(theta)
I = exp_family.mutual_information(theta)
assert I > 0.01, "Should be able to create entangled states"
```
✅ Tests confirm I > 0 for pair basis, I ≈ 0 for local basis

### 4. Multiple Pairs ✅ COMPLETED

**Support systems of n pairs:**
- [x] `QuantumExponentialFamily(n_pairs=2, d=2)` → 30 parameters
- [x] Direct sum structure: operators act independently on each pair
- [x] Marginal entropies: one per subsystem (2n subsystems for n pairs)
- [x] Constraint: C = ∑_{i=1}^{2n} h_i

**Example:**
```python
# 2 Bell pairs
exp_family = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
assert exp_family.n_params == 30  # 15 operators per pair
assert exp_family.D == 16  # 4×4 Hilbert space
assert len(exp_family.dims) == 4  # 4 subsystems
```
✅ All assertions pass

### 5. Evolution from Origin ✅ COMPLETED

**Implement and test dynamics:**
- [x] Verify F(θ) for entangled states
  - F ≠ 0: **Genuine dynamics exist** ✅
  - ||F|| ≈ 0.38 for random entangled state
- [x] Check structural identity Gθ = -∇C with entanglement
  - **BROKEN** as expected: ||Gθ + a||/||a|| ≈ 1.52 ✅
  - C ≠ H: mutual information I ≈ 0.32
- [x] Verified ∇C ≠ ∇H (relative difference ~110%)
- [ ] Compute trajectories θ(t) - DEFERRED (integration not yet implemented)
- [ ] Track I(t), H(t), C(t) along trajectories - DEFERRED

### 6. Backward Compatibility ✅ COMPLETED

**Ensure existing tests still work:**
- [x] Single qubit tests (when n_pairs=None, use local basis)
- [x] BKM metric tests
- [x] Third cumulant tests
- [x] Jacobian tests (corrected for pair basis)

**Add mode flag:**
```python
# Old behavior (local operators)
exp_family_local = QuantumExponentialFamily(n_sites=2, d=2, pair_basis=False)

# New behavior (pair operators)
exp_family_pairs = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
```
✅ Both modes working, all tests passing

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
- **Test block-diagonal structure of G**: Verify G_{(k,i),(k',j)} ≈ 0 for k≠k'
- Track accessible subspace dimension during dynamics
- Monitor rank and support of states  
- Check if cross-pair entanglement ever forms (should remain zero with direct sum ops)
- **Verify constraint coupling**: Show that ν and ∇C create the only inter-pair coordination
- Verify computational cost scales linearly with n (not exponentially)
- Test that G_k blocks can be computed independently and assembled

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
- Total G is block-diagonal with n blocks: G = G₁ ⊕ G₂ ⊕ ... ⊕ Gₙ
- Cross-pair elements vanish: G_{(k,i),(k',j)} = 0 for k≠k'
- Constraint Hessian ∇²C similarly structured
- Jacobian M inherits block structure from G
- **Cost scales as O(n·d⁴), not O(d^(4n))**

**Critical insight: Coupling only through constraint**
- Geometry (G) is block-diagonal → no intrinsic pair-pair coupling
- ALL interaction comes from constraint term ν∇C in dynamics θ̇ = -Gθ + ν∇C
- Lagrange multiplier ν = (a^T Gθ)/||a||² couples all pairs through global entropy budget
- Constraint gradient a = ∇(∑h_i) spans all pairs
- Physical: pairs coordinate through shared marginal entropy constraint, not through geometry
- Implementation: Can compute G_k independently, coupling enters only in constraint projection

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

### 2025-11-22 (Creation)
- Task created with Proposed status
- Identified current limitation: local operators only produce separable states
- Determined acceptance criteria and implementation strategy
- Priority: High (blocks proper testing of quantum game dynamics)

### 2025-11-22 (Implementation)
- **Implemented** `qig/pair_operators.py`:
  - `gell_mann_generators(d)`: General su(d) generators
  - `pair_basis_generators(d)`: su(d²) for pair Hilbert space
  - `bell_state(d)`, `bell_state_density_matrix(d)`: Maximally entangled states
  - `multi_pair_basis(n_pairs, d)`: Direct sum structure for n pairs
  - `product_of_bell_states(n_pairs, d)`: Origin state
  
- **Extended** `QuantumExponentialFamily`:
  - Added `pair_basis=True` mode with `n_pairs` parameter
  - New methods: `von_neumann_entropy()`, `mutual_information()`, `purity()`
  - Placeholder `get_bell_state_parameters(epsilon)` for near-Bell states
  - Updated `marginal_entropy_constraint()` to accept `method='duhamel'`
  - **Corrected** `jacobian()` to use full formula (not simplified version)
  
- **Created comprehensive test suites** (26 tests total, all passing):
  - `test_pair_exponential_family.py` (16 tests):
    - Initialization (1-2 pairs, qubits and qutrits)
    - Operator properties (Hermitian, traceless)
    - Density matrix properties
    - Entanglement metrics (I > 0 for pairs, I ≈ 0 for local)
    - **Block-diagonal G**: cross-pair elements ~10⁻¹⁶
    - Scaling validation
    
  - `test_pair_numerical_validation.py` (10 tests):
    - ∂ρ/∂θ validation: error ~1-3×10⁻⁵
    - Fisher metric G validation: error ~4×10⁻⁴
    - Block structure: cross-pair ~10⁻³ (FD) vs ~10⁻¹⁶ (analytic)
    - Constraint gradient ∇C validation: error ~9×10⁻⁶
    - Constraint Hessian ∇²C validation: error ~6×10⁻⁴
    - **Jacobian M validation: error ~1.3×10⁻⁵**
    - Dynamics verification: F ≠ 0, Gθ ≠ -a

### 2025-11-22 (Validation & Discovery)
- **Confirmed entanglement capability**:
  - Local operators: I ≈ 0 always (C = H)
  - Pair operators: I > 0 achievable (C ≠ H)
  - Example: I ≈ 0.32 for random entangled state
  
- **Validated structural identity breaking**:
  - Local operators: Gθ = -a (identity holds)
  - Pair operators: ||Gθ + a||/||a|| ≈ 1.52 (identity BROKEN)
  - Lagrange multiplier: ν ≈ -0.50 (not constant at -1)
  
- **Confirmed genuine dynamics**:
  - Local operators: F = 0 everywhere on manifold
  - Pair operators: ||F|| ≈ 0.38 (non-zero dynamics)
  - Gradient distinction: ||∇C - ∇H||/||∇H|| ≈ 110%
  
- **Validated Fisher metric block structure**:
  - Analytic: cross-pair elements ~10⁻¹⁶ (machine precision)
  - Finite differences: cross-pair ~10⁻³ (numerical limitation)
  - Confirms constraint coupling only through ν∇C term

### 2025-11-22 (Completion)
- Task status updated to **Completed**
- All acceptance criteria met except trajectory integration (deferred)
- Jacobian corrected and fully validated for entangled systems
- Ready for quantum inaccessible game dynamics exploration

