---
title: "Fisher Matrix Eigenspectrum Analysis for MEPP Hierarchy"
id: "2025-07-24_fisher-matrix-analysis"
status: "proposed"
priority: "high"
created: "2025-07-25"
updated: "2025-07-25"
owner: "Neil"
dependencies: ["CIP-0005"]
category: "features"
---

# Task: Fisher Matrix Eigenspectrum Analysis for MEPP Hierarchy

## Description

Implement block-level Fisher matrix analysis to track the sloppy/fast hierarchy and identify the slowest mode λ_min that creates long entropy plateaus. This provides insight into the information geometry underlying the thermalization process.

## Acceptance Criteria

### 1. Fisher Matrix Computation
- [ ] Implement `fisher_eigenspectrum()` method for computing eigenvalues
- [ ] Build full Fisher matrix on diagonal subspace: G = diag(1/p) - 1
- [ ] Project onto correlation space excluding single-site marginals
- [ ] Return sorted eigenvalues for hierarchy analysis

### 2. Eigenvalue Tracking
- [ ] Store Fisher eigenvalues at every evolution step
- [ ] Track λ_min(t) - the slowest mode governing plateau lifetime
- [ ] Create log-histogram of eigenvalue distribution
- [ ] Identify band gap closing during thermalization

### 3. Visualization
- [ ] Plot λ_min^(-1) as plateau length indicator
- [ ] Create spectrum waterfall: heat-map of eigenvalues vs step number
- [ ] Overlay with entropy curves for correlation analysis
- [ ] Show sloppy/fast mode separation clearly

### 4. Integration with MEPP
- [ ] Add Fisher analysis to main simulation loop
- [ ] Store results in simulator history arrays
- [ ] Minimal performance impact on existing simulations
- [ ] Optional activation via parameter flag

## Implementation Notes

### Core Algorithm
```python
def fisher_eigenspectrum(self, rho):
    """Compute Fisher matrix eigenvalues for sloppy/fast hierarchy analysis."""
    diag = np.diag(rho).real
    G_full = np.diag(1./diag) - 1.0
    P = self._projector_on_correlations()   # shape: (D-1-2N, D-1)
    G_corr = P @ G_full @ P.T
    evals = np.linalg.eigvalsh(G_corr)
    return np.sort(evals)

def _projector_on_correlations(self):
    """Build projector that excludes single-site marginals."""
    # Implementation details depend on qubit/qutrit structure
    # Remove rows/cols corresponding to single-site constraints
    pass
```

### Storage Strategy
- Track `lambda_min_history` for slowest mode evolution
- Store `lambda_histograms` for full spectrum at key steps
- Add Fisher analysis to existing plotting framework

### Performance Considerations
- Eigenvalue computation scales as O(D^3) where D = d^n
- Use sparse methods for larger systems if needed
- Cache projector matrices to avoid recomputation

## Physics Insights

### What We Learn
1. *Hierarchy Structure*: Clear separation of sloppy/fast modes
2. *Plateau Mechanism*: λ_min^(-1) predicts plateau duration
3. *Critical Points*: Band gap closing marks phase transitions
4. *Information Geometry*: Fisher metric reveals natural coordinates

### Expected Results
- Initial fast thermalization: large eigenvalue gaps
- Plateau phase: λ_min approaches zero
- Final approach: slow Fisher mode dominates

## Testing Strategy

### Unit Tests
- Verify Fisher matrix positive definite
- Check eigenvalue ordering and reality
- Test projector orthogonality

### Integration Tests
- Compare with known analytical cases
- Verify consistency with entropy evolution
- Test scaling with system size

### Validation
- Cross-check with independent hierarchy measures
- Verify plateau prediction accuracy
- Compare with theoretical expectations

## Related

- CIP: 0005 (MEPP Thermalization)
- Theory: Information geometry and thermalization
- Files: `mepp.py`, `mepp_demo.ipynb`

## Progress Updates

### 2025-07-25
Task created with Proposed status. Core Fisher matrix theory established, ready for implementation. 