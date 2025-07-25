---
title: "Jaynes Natural Parameter Framework Implementation"
id: "2025-07-25_jaynes-natural-parameters"
status: "proposed"
priority: "high"
created: "2025-07-25"
updated: "2025-07-25"
owner: "Neil"
dependencies: ["CIP-0005"]
category: "features"
---

# Task: Jaynes Natural Parameter Framework Implementation

## Description

Implement the Jaynes/MaxEnt natural parameter framework that expresses all MEPP dynamics in canonical coordinates θ_j = log p_j - ψ. This provides the foundation for consistent MaxEnt-based analysis and separates hard (exact) from soft (relaxing) constraints.

## Acceptance Criteria

### 1. Hard Constraint Projector
- [ ] Implement `_projector_on_correlations()` to build P_hard matrix
- [ ] Separate single-site populations (hard constraints) from correlations
- [ ] Store projector P_soft = P_hard for reuse throughout simulation
- [ ] Handle both qubit (d=2) and qutrit (d=3) systems correctly

### 2. Natural Parameter Computation
- [ ] Implement `natural_params()` method for canonical coordinates
- [ ] Compute θ_j = log p_j - ψ with proper normalization ψ = log Σ exp(θ_j)
- [ ] Project to soft subspace: θ_soft = P_soft @ θ_full
- [ ] Handle numerical stability for small probabilities

### 3. Framework Integration
- [ ] Add natural parameter tracking to main simulation loop
- [ ] Store θ_soft history alongside entropy and Fisher data
- [ ] Provide conversion utilities between density matrix and natural parameters
- [ ] Enable visualization of θ_soft evolution

### 4. Consistency Validation
- [ ] Verify parameter normalization: Σ p_j = 1
- [ ] Check projector properties: P_soft @ P_soft.T = I
- [ ] Validate conversion: rho → θ_soft → rho roundtrip accuracy
- [ ] Test scaling with system size

## Implementation Notes

### Core Algorithm
```python
def _projector_on_correlations(self):
    """Build matrix that projects away single-site populations (hard constraints)."""
    D = self.d**self.n_qubits
    P = np.eye(D-1)         # start in prob-simplex basis (skip trace)
    hard_rows = []
    for i in range(self.n_qubits):
        for a in range(self.d-1):   # only d-1 independent probs/site
            hard_rows.append(self._index_of_basis_proj(i, a))
    keep = [j for j in range(D-1) if j not in hard_rows]
    return P[keep, :]       # shape (D-1-2N, D-1)

def natural_params(self, rho):
    """Compute canonical parameters θ_j = log p_j - ψ in soft coordinates."""
    p = np.diag(rho).real
    p = p / p.sum()
    theta_full = np.log(p) - np.log(p).mean()         # subtract ψ
    theta_soft = self.P_soft @ theta_full             # drop hard components
    return theta_soft
```

### Basis Projector Indexing
```python
def _index_of_basis_proj(self, site_i, state_a):
    """Map (site, state) to basis projector index in prob-simplex."""
    # Implementation depends on chosen basis ordering
    # For computational basis: |00...0⟩, |00...1⟩, ..., |11...1⟩
    # Skip first element (trace constraint) and last state per site
    pass
```

### Storage and Visualization
- Add `self.theta_soft_history = []` to track natural parameters
- Create plotting utilities for θ_soft vs time
- Show evolution in natural parameter space alongside probability space

## Physics Insights

### What We Learn
1. **MaxEnt Foundation**: All dynamics expressed in Lagrange multiplier coordinates
2. **Constraint Separation**: Clear distinction between fixed marginals vs correlations
3. **Natural Coordinates**: Probability manifold geometry becomes explicit
4. **Preparation for SEA**: Canonical parameters are what SEA actually evolves

### Expected Results
- Natural parameters provide smoother evolution curves than probabilities
- Hard constraints become zero-eigenvalue directions in parameter space
- Soft constraints show continuous relaxation in θ_soft coordinates
- Connection to Fisher information becomes transparent

## Testing Strategy

### Unit Tests
- Verify projector construction for different system sizes
- Test natural parameter computation accuracy
- Check numerical stability for edge cases (small probabilities)

### Integration Tests
- Validate with known Bell states and product states
- Compare θ_soft evolution with probability evolution
- Test consistency across qubit/qutrit systems

### Physics Validation
- Verify MaxEnt interpretation of natural parameters
- Check that hard constraints remain exactly satisfied
- Validate connection to Fisher information matrix

## Mathematical Framework

### Key Relationships
- **Probabilities**: p_j = exp(θ_j) / Z where Z = Σ exp(θ_j)
- **Partition Function**: ψ = log Z normalizes the distribution
- **Hard Constraints**: Fixed single-site marginals eliminate degrees of freedom
- **Soft Constraints**: Correlation parameters free to evolve under SEA

### Coordinate Systems
- **Probability Space**: p_j ∈ [0,1] with Σ p_j = 1
- **Natural Parameter Space**: θ_j ∈ ℝ with exponential map to probabilities
- **Soft Parameter Space**: θ_soft ∈ ℝ^(D-1-2N) excluding hard constraints

## Related

- CIP: 0005 (MEPP Thermalization)
- Theory: Jaynes MaxEnt principle, Fisher information geometry
- Files: `mepp.py`, `mepp_demo.ipynb`
- Enables: SEA verification, charge classification, particle interpretation

## Progress Updates

### 2025-07-25
Task created with Proposed status. Jaynes natural parameter framework designed with clear MaxEnt foundation. 