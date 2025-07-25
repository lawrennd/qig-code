---
title: "Automatic Charge Classification (Hard/Soft Constraints)"
id: "2025-07-25_charge-classification"
status: "proposed"
priority: "high"
created: "2025-07-25"
updated: "2025-07-25"
owner: "Neil"
dependencies: ["CIP-0005", "2025-07-25_jaynes-natural-parameters"]
category: "features"
---

# Task: Automatic Charge Classification (Hard/Soft Constraints)

## Description

Implement automatic classification of hard charges (λ=0, exactly conserved) and soft charges (λ<λ_cut, quasi-conserved) based on Fisher eigenvalue analysis. This directly connects Fisher matrix sloppiness to emergent Gauss laws and conservation principles.

## Acceptance Criteria

### 1. Charge Detection Algorithm
- [ ] Implement `classify_charges()` method with configurable λ_cut threshold
- [ ] Identify hard charges: Fisher eigenvalues λ ≈ 0 (within numerical precision)
- [ ] Identify soft charges: eigenvalues λ < λ_cut (user-defined cutoff)
- [ ] Return charge indices, counts, and associated constraint directions

### 2. Constraint Interpretation
- [ ] Map Fisher eigenvectors to linear combinations of basis projectors
- [ ] Identify which MaxEnt observables correspond to each charge
- [ ] Print human-readable descriptions of conserved quantities
- [ ] Show connection between charge and physical symmetries

### 3. Dynamic Monitoring
- [ ] Track charge classification throughout simulation evolution
- [ ] Detect when soft charges become hard (λ→0) or disappear (λ grows)
- [ ] Log charge creation/annihilation events with timing
- [ ] Correlate charge dynamics with entropy plateaus

### 4. Gauss Law Connection
- [ ] Identify spatial patterns in charge density for embedded systems
- [ ] Compute divergence ∇·J for charge flow visualization
- [ ] Demonstrate conservation laws in discrete lattice setting
- [ ] Show emergence of gauge-like symmetries

## Implementation Notes

### Core Classification Algorithm
```python
def classify_charges(self, fisher_eigenvalues, fisher_eigenvectors, lambda_cut=1e-6):
    """Classify hard (λ=0) and soft (λ<λ_cut) charges automatically."""
    hard_charges = np.where(fisher_eigenvalues < 1e-12)[0]  # Exactly conserved
    soft_charges = np.where((fisher_eigenvalues >= 1e-12) & 
                           (fisher_eigenvalues < lambda_cut))[0]  # Quasi-conserved
    
    charge_info = {
        'hard_charges': hard_charges.tolist(),
        'soft_charges': soft_charges.tolist(),
        'hard_count': len(hard_charges),
        'soft_count': len(soft_charges),
        'lambda_cut': lambda_cut,
        'hard_vectors': fisher_eigenvectors[:, hard_charges],
        'soft_vectors': fisher_eigenvectors[:, soft_charges],
        'hard_eigenvals': fisher_eigenvalues[hard_charges],
        'soft_eigenvals': fisher_eigenvalues[soft_charges]
    }
    return charge_info

def interpret_charge_vector(self, charge_vector):
    """Interpret Fisher eigenvector as linear combination of basis projectors."""
    # Map charge_vector components to basis projector coefficients
    basis_weights = self.P_soft.T @ charge_vector  # Project back to full basis
    
    # Identify dominant basis elements
    significant_indices = np.where(np.abs(basis_weights) > 0.1)[0]
    
    interpretation = {
        'dominant_basis_elements': significant_indices.tolist(),
        'coefficients': basis_weights[significant_indices].tolist(),
        'description': self._describe_charge_pattern(significant_indices, basis_weights)
    }
    return interpretation
```

### Charge Evolution Tracking
```python
def track_charge_evolution(self):
    """Monitor how charge classification changes during simulation."""
    charge_history = []
    
    for step, (evals, evecs) in enumerate(zip(self.fisher_eigenval_history, 
                                             self.fisher_eigenvec_history)):
        charge_info = self.classify_charges(evals, evecs)
        charge_info['step'] = step
        charge_history.append(charge_info)
    
    # Detect charge transitions
    transitions = self._detect_charge_transitions(charge_history)
    
    return charge_history, transitions

def _detect_charge_transitions(self, charge_history):
    """Identify when charges appear, disappear, or change type."""
    transitions = []
    
    for i in range(1, len(charge_history)):
        prev_hard = set(charge_history[i-1]['hard_charges'])
        curr_hard = set(charge_history[i]['hard_charges'])
        
        # New hard charges
        new_hard = curr_hard - prev_hard
        if new_hard:
            transitions.append({
                'step': i,
                'type': 'soft_to_hard',
                'charges': list(new_hard)
            })
        
        # Lost hard charges  
        lost_hard = prev_hard - curr_hard
        if lost_hard:
            transitions.append({
                'step': i,
                'type': 'hard_to_soft',
                'charges': list(lost_hard)
            })
    
    return transitions
```

### Gauss Law Implementation
```python
def compute_charge_density(self, charge_vector, spatial_positions):
    """Map charge vector to spatial charge density ρ(x)."""
    # Bin basis projectors by spatial location
    charge_density = np.zeros(len(spatial_positions))
    
    for i, pos in enumerate(spatial_positions):
        # Sum charge vector components for projectors at position i
        local_projectors = self._get_projectors_at_site(i)
        charge_density[i] = np.sum(charge_vector[local_projectors])
    
    return charge_density

def compute_charge_divergence(self, charge_density, topology='ring'):
    """Compute ∇·J using finite differences."""
    if topology == 'ring':
        # Periodic boundary conditions
        div_J = np.roll(charge_density, -1) - 2*charge_density + np.roll(charge_density, 1)
    elif topology == '1d_chain':
        # Open boundary conditions
        div_J = np.zeros_like(charge_density)
        div_J[1:-1] = charge_density[2:] - 2*charge_density[1:-1] + charge_density[:-2]
    
    return div_J
```

## Physics Insights

### What We Learn
1. **Conservation Hierarchy**: Clear separation of exact vs approximate conservation
2. **Emergent Gauge Theory**: Soft charges become local gauge symmetries
3. **Critical Phenomena**: Charge transitions mark thermalization phase changes
4. **Gauss Law**: Spatial charge patterns follow discrete conservation laws

### Expected Results
- Stage-1: Many soft charges as correlations build
- Stage-2: Charges disappear as λ values grow during thermalization
- Plateaus: Persistent soft charges create quasi-equilibrium states
- Spatial embedding: Charge conservation becomes local Gauss law

## Testing Strategy

### Unit Tests
- Verify charge classification with known eigenvalue patterns
- Test threshold sensitivity for λ_cut parameter
- Check conservation of total charge during evolution

### Physics Validation
- Compare with analytical conservation laws for known systems
- Verify Gauss law compliance in spatial configurations
- Check charge interpretation consistency

### Integration Tests
- Validate charge tracking across different system sizes
- Test robustness to numerical errors in eigenvalue computation
- Cross-check with plateau detection and entropy analysis

## Visualization Strategy

### Charge Evolution Plots
- Time series of hard/soft charge counts
- Eigenvalue spectrum with charge classification coloring
- Charge transition event timeline

### Spatial Charge Maps
- Charge density ρ(x) heatmaps for embedded systems
- Divergence ∇·J visualization showing conservation
- Animation of charge flow during evolution

## Related

- CIP: 0005 (MEPP Thermalization)
- Theory: Conservation laws, gauge theory, Gauss law
- Files: `mepp.py`, `mepp_demo.ipynb`
- Dependencies: Jaynes natural parameters, Fisher eigenanalysis

## Progress Updates

### 2025-07-25
Task created with Proposed status. Charge classification theory established with clear connection to conservation laws and gauge symmetries. 