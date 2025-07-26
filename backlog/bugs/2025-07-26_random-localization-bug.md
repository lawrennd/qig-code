---
title: "Fix Random Localization Bug in MEPP Isolation Stage"
id: "2025-07-26_random-localization-bug"
status: "ready"
priority: "high"
created: "2025-07-26"
updated: "2025-07-26"
owner: "Neil"
dependencies: ["2025-07-25_fisher-matrix-analysis"]
category: "bugs"
related_cip: "CIP-0005"
---

# Bug: Fix Random Localization Bug in MEPP Isolation Stage

## Description

The current MEPP implementation shows random localization behavior during Stage B (isolation) instead of following deterministic Steepest Entropy Ascent (SEA) dynamics. This violates the Maximum Entropy Production Principle and prevents proper convergence to natural equilibrium states.

## Bug Analysis

### Symptoms
- Localization appears to move about randomly during isolation stage
- Entropy evolution is erratic and noisy instead of smooth
- No clear convergence to equilibrium states
- Multiple runs with same parameters show chaotic variations

### Root Cause
Complete randomness in gate selection without MEPP/SEA guidance:

```python
# In _generate_pauli_string() for isolation stage:
pauli_type = random.choice(['x', 'y', 'z'])  # ❌ RANDOM - No MEPP guidance

# In apply_gate_block():
for _ in range(block_size):
    gate, qubits, alpha = self._generate_random_gate(sigma_alpha, stage)  # ❌ ALL RANDOM
```

### Expected Behavior
Evolution should follow Steepest Entropy Ascent: ∂τρ = i[log ρ*, ρ] with systematic direction toward maximum entropy production, not random exploration.

## Acceptance Criteria

### 1. Fisher Matrix Foundation
- [ ] Implement `compute_fisher_matrix(rho)` method
- [ ] Implement `compute_entropy_gradient(rho)` method  
- [ ] Add Fisher eigenvalue computation and analysis
- [ ] Test Fisher matrix correctness against known cases

### 2. SEA-Guided Gate Selection
- [ ] Replace random Pauli selection with gradient-based selection
- [ ] Replace random qubit selection with gradient-magnitude-based selection
- [ ] Implement `select_sea_guided_gate()` method based on ∇S direction
- [ ] Test gate selection produces systematic evolution

### 3. Integration and Validation
- [ ] Modify `apply_gate_block()` to use SEA guidance for isolation stage
- [ ] Add configuration flag: `use_sea_guidance=True` (default True)
- [ ] Maintain backward compatibility with `use_sea_guidance=False` for comparison
- [ ] Ensure dephasing stage remains unchanged (gradual dephasing model is correct)

### 4. Comparison Testing
- [ ] Create test comparing old random vs new deterministic behavior
- [ ] Verify entropy evolution shows smooth monotonic increase
- [ ] Verify Fisher λ_min decreases systematically during isolation
- [ ] Verify localization converges to stable equilibrium configuration

## Implementation Notes

### Technical Approach
- Use Fisher information matrix G_ij to guide gate selection
- Select Pauli operators aligned with steepest entropy ascent direction
- Target qubits where gradient magnitude is largest, not randomly
- Maintain controlled stochasticity through α coefficients only

### Physics Validation
- Evolution should match MEPP theoretical predictions
- Clear systematic trends with deterministic convergence behavior
- Natural transition toward Stage 3 plateau regime
- Consistent behavior across multiple runs with same parameters

## Testing Strategy

### Before/After Comparison
```python
def test_random_vs_sea_localization():
    # Same initial state, both methods
    entropy_random = simulate_with_random_gates()
    entropy_sea = simulate_with_sea_guided_gates()
    # Compare smoothness, convergence, consistency
```

### Quantitative Metrics
- Entropy evolution smoothness (reduced fluctuations)
- Fisher eigenvalue systematic evolution
- Localization pattern consistency across runs
- Convergence rate to equilibrium states

## Dependencies

- **Fisher Matrix Analysis**: Provides gradient computation for SEA guidance
- **CIP-0005**: Base MEPP framework that contains the bug
- **Existing SEA Generator**: May need correction/enhancement

## Impact Assessment

### Current Impact
- **CRITICAL**: Violates core MEPP theoretical framework
- **BLOCKS**: Proper CIP-0007 Stage 3 implementation (depends on correct Stage 2)
- **AFFECTS**: All isolation stage simulations and AISTATS results

### Post-Fix Benefits
- Deterministic, physically meaningful evolution
- Smooth entropy curves suitable for publication
- Proper foundation for Stage 3 plateau detection
- Validation of MEPP theoretical predictions

## References

- Beretta (2020): SEA dynamics and Fisher metric formalism
- CIP-0005: Two-stage MEPP framework (contains bug)
- CIP-0007: Stage 3 implementation (blocked by this bug)
- Backlog task: 2025-07-25_fisher-matrix-analysis (prerequisite) 