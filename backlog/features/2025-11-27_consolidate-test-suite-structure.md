---
author: "Neil D. Lawrence"
created: "2025-11-27"
id: "2025-11-27_consolidate-test-suite-structure"
last_updated: "2025-11-27"
status: proposed
priority: medium
tags:
- cip-0004
- testing
- refactoring
- code-organization
title: "Consolidate Test Suite to Mirror qig Module Structure"
---

# Task: Consolidate Test Suite to Mirror qig Module Structure

## Description

Reorganize the 19 test files into a more logical structure that mirrors the `qig` package organization. Currently, test files have unclear division of responsibility with significant overlap and redundancy. The goal is to consolidate 19 files → 10 files while maintaining all 131 tests.

## Background

After completing CIP-0004 Phase 2 (tolerance framework conversion), analysis revealed that test file organization doesn't reflect the logical decomposition of the `qig` codebase:

- **15 test files** target `qig.exponential_family` but test different aspects (BKM, constraints, Jacobian, etc.)
- **Significant redundancy**: `test_quantum_qutrit.py` duplicates tests in `test_inaccessible_game.py`
- **Poor discoverability**: "Where's the Fisher metric test?" requires searching multiple files
- **No clear mapping**: Test structure doesn't document codebase architecture

### qig Module Structure

```
qig/
├── core.py                  # State utilities, entropy, partial traces
├── exponential_family.py    # QuantumExponentialFamily + all derivatives
├── pair_operators.py        # su(d²) generators for entangled pairs
├── duhamel.py              # High-precision quantum derivatives
└── dynamics.py             # InaccessibleGameDynamics
```

## Current Test File Analysis

| Test File | # Tests | Tests Functionality | QIG Module(s) | Status |
|-----------|---------|---------------------|---------------|---------|
| **CORE UTILITIES** |
| `test_inaccessible_game.py` | 40 | Everything: state utils, operators, exponential family, dynamics, GENERIC | `core.py`, `exponential_family.py`, `dynamics.py` | **Should split** |
| `test_quantum_qutrit.py` | 5 | Qutrit-specific validation | `core.py`, `exponential_family.py` | **Redundant** |
| **EXPONENTIAL FAMILY - PAIR BASIS** |
| `test_pair_exponential_family.py` | 21 | Pair basis initialization, operators, entanglement | `exponential_family.py`, `pair_operators.py` | Keep + merge |
| `test_pair_numerical_validation.py` | 10 | Numerical accuracy of pair basis derivatives | `exponential_family.py` | Merge into pair tests |
| **FISHER INFORMATION (BKM METRIC)** |
| `test_commuting_bkm.py` | 7 | BKM validation for diagonal/commuting operators | `exponential_family.fisher_information()` | **Consolidate** |
| `test_non_commuting_bkm.py` | 3 | BKM diagnostic for non-commuting cases | `exponential_family.fisher_information()` | **Consolidate** |
| `test_nondiagonal_commuting_bkm.py` | 6 | BKM for rotated/partially commuting operators | `exponential_family.fisher_information()` | **Consolidate** |
| `test_bkm_integral.py` | 0 | Diagnostic script (not pytest) | N/A | Keep as diagnostic |
| `test_bkm_spectral_variants.py` | 0 | Diagnostic script (not pytest) | N/A | Keep as diagnostic |
| **CONSTRAINT DERIVATIVES** |
| `test_marginal_entropy_gradient.py` | 2 | ∇C = gradient of marginal entropy constraint | `exponential_family.marginal_entropy_constraint()` | **Consolidate** |
| `test_constraint_hessian.py` | 4 | ∇²C = Hessian of constraint | `exponential_family.constraint_hessian()` | **Consolidate** |
| `test_constraint_hessian_duhamel.py` | 2 | High-precision Duhamel Hessian | `exponential_family.constraint_hessian()` | **Consolidate** |
| `test_lagrange_multiplier_gradient.py` | 4 | ∇ν = gradient of Lagrange multiplier | `exponential_family.lagrange_multiplier_gradient()` | **Consolidate** |
| `test_theta_only_constraint.py` | 21 | θ-only optimization method (performance) | `exponential_family` (internal helpers) | Keep - perf focus |
| **JACOBIAN & THIRD CUMULANT** |
| `test_jacobian.py` | 5 | Full Jacobian M = ∂F/∂θ | `exponential_family.jacobian()` | **Consolidate** |
| `test_jacobian_analytic.py` | 2 | Analytic Jacobian computation | `exponential_family.jacobian()` | **Consolidate** |
| `test_third_cumulant.py` | 4 | Third cumulant tensor (∇G)[θ] | `exponential_family.third_cumulant_contraction()` | **Consolidate** |
| `test_third_cumulant_symmetry.py` | 1 | Symmetry of third cumulant | `exponential_family.third_cumulant_contraction()` | **Consolidate** |
| **UTILITY** |
| `test_notebook.py` | 0 | Notebook execution test | N/A | Keep as utility |

**Summary**: 19 files, 131 tests, targeting primarily `exponential_family.py`

## Proposed Consolidated Structure

### **New Structure: 10 files (down from 19)**

```
tests/
├── Core Library Tests (qig.core, qig.exponential_family basics)
│   ├── test_core_utilities.py          [NEW] Split from test_inaccessible_game
│   │   • State utilities: partial_trace, von_neumann_entropy, create_lme_state
│   │   • Operator basis construction: Pauli, Gell-Mann
│   │   • GENERIC decomposition
│   │   • ~15 tests
│   │
│   ├── test_exponential_family.py      [NEW] Split from test_inaccessible_game
│   │   • QuantumExponentialFamily initialization (local basis)
│   │   • rho_from_theta(), log_partition()
│   │   • Basic exponential family properties
│   │   • ~12 tests from test_inaccessible_game.py
│   │   • ABSORB: 5 tests from test_quantum_qutrit.py
│   │   • ~17 tests total
│   │
│   └── test_pair_exponential_family.py [KEEP + MERGE]
│       • All pair basis tests (21 existing tests)
│       • ABSORB: 10 tests from test_pair_numerical_validation.py
│       • ~31 tests total
│
├── Fisher Information / BKM Metric
│   └── test_fisher_metric.py           [NEW - CONSOLIDATE 3 FILES]
│       • MERGE: test_commuting_bkm.py (7 tests)
│       • MERGE: test_non_commuting_bkm.py (3 tests)
│       • MERGE: test_nondiagonal_commuting_bkm.py (6 tests)
│       • ~16 tests total
│       • Sections: diagonal → rotated → non-commuting
│
├── Constraint Derivatives
│   └── test_constraint_derivatives.py  [NEW - CONSOLIDATE 4 FILES]
│       • MERGE: test_marginal_entropy_gradient.py (2 tests)
│       • MERGE: test_constraint_hessian.py (4 tests)
│       • MERGE: test_constraint_hessian_duhamel.py (2 tests)
│       • MERGE: test_lagrange_multiplier_gradient.py (4 tests)
│       • ~12 tests total
│
├── Higher-Order Derivatives
│   └── test_higher_derivatives.py      [NEW - CONSOLIDATE 4 FILES]
│       • MERGE: test_jacobian.py (5 tests)
│       • MERGE: test_jacobian_analytic.py (2 tests)
│       • MERGE: test_third_cumulant.py (4 tests)
│       • MERGE: test_third_cumulant_symmetry.py (1 test)
│       • ~12 tests total
│
├── Dynamics & Integration
│   └── test_dynamics.py                [NEW] Split from test_inaccessible_game
│       • InaccessibleGameDynamics class
│       • Constraint preservation, entropy increase
│       • Integration tests
│       • ~13 tests
│
├── Performance & Optimization
│   └── test_theta_only_constraint.py   [KEEP]
│       • θ-only optimization method (21 tests)
│       • Performance benchmarks
│
└── Utilities
    ├── test_notebook.py                [KEEP]
    ├── test_bkm_integral.py            [KEEP - diagnostic]
    └── test_bkm_spectral_variants.py   [KEEP - diagnostic]
```

## Consolidation Mapping

| Current Files | → | New File | Tests | Action |
|--------------|---|----------|-------|--------|
| `test_inaccessible_game.py` (40) | → | Split into 3 files | 15+12+13 | **SPLIT** |
| `test_quantum_qutrit.py` (5) | → | `test_exponential_family.py` | +5 | **MERGE** |
| `test_pair_exponential_family.py` (21)<br>`test_pair_numerical_validation.py` (10) | → | `test_pair_exponential_family.py` | 31 | **MERGE** |
| `test_commuting_bkm.py` (7)<br>`test_non_commuting_bkm.py` (3)<br>`test_nondiagonal_commuting_bkm.py` (6) | → | `test_fisher_metric.py` | 16 | **MERGE 3→1** |
| `test_marginal_entropy_gradient.py` (2)<br>`test_constraint_hessian.py` (4)<br>`test_constraint_hessian_duhamel.py` (2)<br>`test_lagrange_multiplier_gradient.py` (4) | → | `test_constraint_derivatives.py` | 12 | **MERGE 4→1** |
| `test_jacobian.py` (5)<br>`test_jacobian_analytic.py` (2)<br>`test_third_cumulant.py` (4)<br>`test_third_cumulant_symmetry.py` (1) | → | `test_higher_derivatives.py` | 12 | **MERGE 4→1** |
| `test_theta_only_constraint.py` (21) | → | Keep unchanged | 21 | **KEEP** |
| `test_notebook.py`, diagnostics (0) | → | Keep unchanged | 0 | **KEEP** |

**Result**: 19 → 10 files, 131 tests maintained

## Benefits

1. **Mirrors qig structure**: Tests organized by which qig module they validate
2. **Clear responsibility**: Each file tests one cohesive set of functionality
3. **Better discoverability**: "Where's Fisher metric test?" → `test_fisher_metric.py`
4. **Self-documenting**: File structure documents codebase architecture
5. **Reduced redundancy**: Eliminates duplicate qutrit tests
6. **Easier maintenance**: Related tests together = easier updates when code changes
7. **Better onboarding**: New developers can understand test coverage at a glance

## Implementation Plan

### Phase 1: Create Consolidated Files (Safest Approach)
- [ ] Create `test_fisher_metric.py` - merge 3 BKM files
  - Copy all test classes from commuting, non-commuting, nondiagonal files
  - Organize into sections with clear headers
  - Ensure all imports are consolidated
- [ ] Create `test_constraint_derivatives.py` - merge 4 constraint files
  - Organize by: gradient (∇C) → Hessian (∇²C) → Lagrange (∇ν)
  - Keep Duhamel tests in dedicated section
- [ ] Create `test_higher_derivatives.py` - merge 4 Jacobian/cumulant files
  - Section 1: Jacobian tests
  - Section 2: Third cumulant tests
- [ ] Expand `test_pair_exponential_family.py` - absorb numerical validation
  - Add 10 tests from `test_pair_numerical_validation.py`
  - Maintain clear organization

### Phase 2: Split Large File
- [ ] Create `test_core_utilities.py` - extract from `test_inaccessible_game.py`
  - TestQuantumStateUtilities class
  - TestOperatorBases class  
  - TestGENERICDecomposition class
  - ~15 tests total
- [ ] Create `test_exponential_family.py` - extract + absorb
  - TestQuantumExponentialFamily class from test_inaccessible_game
  - Absorb 5 tests from test_quantum_qutrit.py
  - ~17 tests total
- [ ] Create `test_dynamics.py` - extract dynamics tests
  - TestConstrainedDynamics class
  - TestIntegration class
  - TestMathematicalProperties class
  - ~13 tests total

### Phase 3: Verification & Cleanup
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Verify test count: should be 131 tests
- [ ] Check no tests were lost: compare test names
- [ ] Remove old files only after verification
- [ ] Update `TESTING.md` with new structure
- [ ] Update `README_VALIDATION.md` test documentation
- [ ] Commit with clear message documenting consolidation

### Phase 4: Documentation Updates
- [ ] Update CIP-0004 Implementation Status
- [ ] Document new test file → qig module mapping
- [ ] Add section headers with module references in each test file
- [ ] Update any references in other documentation

## Success Criteria

- [ ] All 131 tests still pass
- [ ] Test suite runs in same time (or faster)
- [ ] No test coverage lost
- [ ] Files reduced from 19 → 10
- [ ] Each file clearly maps to qig module(s)
- [ ] Documentation updated to reflect new structure
- [ ] Git history preserves that tests were moved, not changed

## Acceptance Criteria

1. **Completeness**: All 131 existing tests are present in new structure
2. **Organization**: Each new file tests one cohesive functionality area
3. **Clarity**: File names clearly indicate what they test
4. **Documentation**: Each file has header documenting which qig module(s) it tests
5. **No regressions**: All tests pass with same tolerance framework
6. **Maintainability**: Related tests are now co-located for easier updates

## Related

- **CIP-0004**: Test suite rewrite with rigorous tolerances
  - This task completes Phase 2: Integration test consolidation
  - Dependencies: Requires Phase 2 tolerance conversions (completed)
- **CIP-0001**: Code consolidation and module structure
  - Test structure now mirrors the consolidated qig module structure

## Notes

- Keep diagnostic scripts (`test_bkm_integral.py`, `test_bkm_spectral_variants.py`) - they're useful for development
- `test_theta_only_constraint.py` stays separate - it's performance-focused
- Use git mv to preserve history when moving tests
- Add clear section headers in consolidated files for navigation
- Maintain all tolerance framework integration from CIP-0004

## Priority Justification

**Medium Priority**:
- Not blocking current work (all tests pass)
- Significant improvement to maintainability
- Makes codebase more accessible to new contributors
- Natural next step after CIP-0004 Phase 2 completion
- Should be done before CIP-0004 Phase 3 (final documentation)

