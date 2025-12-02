---
id: "2025-12-02_symbolic-qubit-su4-pair"
title: "Symbolic Computation for Qubit Pairs (su(4) basis)"
status: ready
priority: high
created: "2025-12-02"
owner: ""
dependencies:
  - "CIP-0007 Phase 4 (qutrits complete)"
---

# Task: Symbolic Computation for Qubit Pairs (su(4) basis)

## Description

Implement symbolic computation for qubit pairs (d=2) using the su(4) Lie algebra,
following the same pattern established for qutrits (d=3) in `su9_pair.py`.

This is simpler than the qutrit case:
- **15 parameters** (vs 80 for qutrits)
- **2×2 reduced density matrix** → eigenvalues always from quadratic formula
- No block structure optimization needed—inherently simple

## Acceptance Criteria

- [ ] Create `qig/symbolic/su4_pair.py` following `su9_pair.py` pattern
- [ ] Implement su(4) generators (15 matrices, 4×4)
- [ ] Implement su(4) structure constants
- [ ] Implement symbolic density matrix, partial trace
- [ ] Implement exact marginal entropies (2×2 eigenvalues trivial)
- [ ] Implement constraint gradient, Lagrange multiplier, ∇ν
- [ ] Implement antisymmetric part A
- [ ] Validate: ratio to numerical ≈ 1.0 for constraint gradient
- [ ] Validate: A ≠ 0 for su(4) pair basis
- [ ] Export results to `qig/symbolic/results/symbolic_expressions_qubit.py`
- [ ] Add to Sphinx documentation

## Implementation Notes

### Incremental Approach (lessons from qutrit implementation)

**Phase 1: Infrastructure (~30 min)**
1. Create `su4_pair.py` with su(4) generators
2. Quick test: verify generator properties (Hermitian, traceless, Tr(F_a F_b)=2δ_ab)
3. Structure constants with commutation relation test

**Phase 2: Density Matrix & Entropy (~1 hour)**
1. Symbolic density matrix (4×4)
2. Partial trace (4→2, simpler than 9→3)
3. Exact entropy: 2×2 eigenvalues are λ± = (1 ± √(1-4det))/2
4. Quick test: compare to numerical at one point

**Phase 3: Constraint Geometry (~1 hour)**
1. Marginal entropies h₁, h₂
2. Constraint gradient a = ∇(h₁+h₂)
3. Quick test: ratio to numerical ≈ 1.0
4. Verify: structural identity broken (Gθ ≠ -a)

**Phase 4: Antisymmetric Part (~1 hour)**
1. Lagrange multiplier ν
2. Gradient ∇ν
3. Antisymmetric part A = (1/2)[a⊗(∇ν)ᵀ - (∇ν)⊗aᵀ]
4. Validate: A ≠ 0

### Key Timing Estimates (from qutrit experience)

| Operation | Qutrits (80 params) | Qubits (15 params) |
|-----------|---------------------|---------------------|
| Generators | instant | instant |
| Density matrix | ~0.2s | ~0.05s (smaller) |
| Partial trace | ~0.2s | ~0.05s |
| Eigenvalues | ~0.04s | instant (2×2) |
| Differentiate | ~0.08s | ~0.02s (fewer params) |

### Testing Strategy

1. **After each function**: Quick sanity check (one component, one θ value)
2. **Profile before running**: Estimate time, use timeouts
3. **Cache everything**: First run slow, subsequent instant
4. **Validate incrementally**: Don't wait for full pipeline

## Related

- **CIP**: 0007
- **Depends on**: qutrit implementation complete
- **Documentation**: `docs/source/theory/symbolic_computation.rst`

## Progress Updates

### 2025-12-02
Task created. Qutrit implementation (su9_pair.py) complete and validated.
Ready to begin qubit implementation following same pattern.

