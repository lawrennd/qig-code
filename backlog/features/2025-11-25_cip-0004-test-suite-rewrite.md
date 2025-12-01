---
author: "Neil D. Lawrence"
created: "2025-11-25"
id: "2025-11-25_cip-0004-test-suite-rewrite"
last_updated: "2025-11-25"
status: in_progress
tags:
- cip-implementation
- testing
- tolerances
- quality-assurance
- quantum-validation
title: "Implement CIP-0004: Comprehensive Test Suite Rewrite with Rigorous Tolerances"
---

# Task: Implement CIP-0004: Comprehensive Test Suite Rewrite with Rigorous Tolerances

## Description

Implement CIP-0004 to systematically rewrite the quantum inaccessible game test suite with scientifically derived tolerances and improved test organization. The current test suite has 19 files and 168 individual tests with inconsistent tolerances that don't reflect the mathematical precision of quantum algorithms.

## Background

The current test suite shows:
- 19 test files with unclear organization
- Inconsistent tolerances (1e-14 to 1e-4)
- 1 failed test, 85 passed (98.8% pass rate)
- 5777 warnings (many deprecation warnings)
- Poor separation between unit, integration, and validation tests

## Goals

- **Scientifically derived tolerances** based on quantum algorithm precision analysis
- **Complete test coverage preservation** while improving reliability
- **Hierarchical test structure** (unit → integration → validation)
- **Mathematical justification** for all tolerance thresholds
- **Better debugging and maintenance** capabilities

## Implementation Plan

### Phase 1: Tolerance Analysis and Design (Week 1-2) ✅ COMPLETED

#### 1. Mathematical Precision Analysis
- [x] Review all quantum algorithms for error sources:
  - Matrix exponentiation in exponential families
  - Partial trace operations
  - Fisher information metric computation
  - Jacobian and constraint calculations
  - ODE integration for dynamics
- [x] Calculate theoretical error bounds for each operation
- [x] Document floating-point error accumulation patterns

#### 2. Current Test Inventory
- [x] Catalog all 19 test files and their coverage areas
- [x] Map tests to mathematical requirements
- [x] Identify redundant or overlapping tests
- [x] Document current tolerance usage patterns

#### 3. Tolerance Framework Design
- [x] Define tolerance categories with mathematical justification
- [x] Create utility functions for consistent tolerance application
- [x] Design statistical validation methods for tolerance setting
- [x] Document absolute vs relative tolerance strategy

### Phase 2: Core Test Rewrite (Week 3-6) 🔄 IN PROGRESS

#### 1. Unit Test Restructuring
- [ ] Rewrite core mathematical tests with justified tolerances
- [ ] Implement tolerance validation utilities
- [ ] Add comprehensive error documentation

#### 2. Integration Test Consolidation
- [ ] Merge overlapping integration tests
- [ ] Standardize integration test tolerances
- [ ] Improve test isolation and reproducibility

#### 3. Validation Test Enhancement
- [ ] Rewrite physical/mathematical validation tests
- [ ] Implement statistical significance testing
- [ ] Add tolerance sensitivity analysis

### Phase 3: Quality Assurance and Documentation (Week 7-8) 📋 PLANNED

#### 1. Test Suite Validation
- [ ] Run comprehensive test suite against all tolerance levels
- [ ] Validate that all original coverage is maintained
- [ ] Performance benchmarking of new test suite

#### 2. Documentation and Guidelines
- [ ] Create tolerance selection guidelines
- [ ] Document mathematical basis for each tolerance category
- [ ] Update testing documentation with new structure

#### 3. Migration Support
- [ ] Backward compatibility testing
- [ ] Migration guide for developers
- [ ] Training materials for tolerance rationale

## Success Criteria

- **Test Coverage**: 100% preservation of existing test coverage
- **Pass Rate**: Maintain or improve current 98.8% pass rate
- **Tolerance Justification**: Every tolerance backed by mathematical analysis
- **Maintainability**: Improved debugging and test organization
- **Performance**: No significant degradation in test execution time

## Dependencies

- **CIP-0001**: Test suite must validate consolidated code structure
- **CIP-0002**: Tests must work for both local (I=0) and pair (I>0) operators
- **CIP-0003**: Tolerance framework must validate analytical fixes

## Related Files

- `cip/cip0004.md` - Detailed CIP specification
- `tests/` - Current test suite (19 files, 168 tests)
- `qig/` - Core quantum game modules being tested

## Progress Updates

### 2025-11-25 (Phase 1 Completed ✅)
**Phase 1: Tolerance Analysis and Design - COMPLETED**

✅ **Mathematical Precision Analysis**:
- Analyzed all core quantum operations (exponentiation, eigenvalues, Fisher metric, partial traces, ODE integration)
- Derived 6 scientifically justified tolerance categories (A-F) based on error propagation analysis
- Documented absolute vs relative tolerance strategy
- Created comprehensive precision analysis document (`docs/cip0004_precision_analysis.md`)

✅ **Current Test Inventory**:
- Catalogued all 19 test files (168 individual tests)
- Analyzed coverage areas and tolerance usage patterns
- Identified inconsistencies and quality issues
- Documented test organization problems and improvement opportunities
- Created detailed inventory (`docs/cip0004_test_inventory.md`)

✅ **Tolerance Framework Design**:
- Implemented complete tolerance framework (`tests/tolerance_framework.py`)
- Created utility functions for quantum-appropriate assertions
- Built automatic tolerance selection based on operation type
- Added validation utilities for tolerance appropriateness
- Included backward compatibility support for gradual migration

**Key Achievements:**
- Scientifically derived tolerances replace arbitrary thresholds
- Comprehensive test suite analysis completed
- Ready-to-use tolerance framework implemented
- Mathematical justification for all tolerance bounds

**Current Status (after Phase 1 + initial Phase 2 work):**
- Test suite: 19 files, 168 tests
- Core tolerance framework: Implemented and in active use
- Documentation: Precision analysis and test inventory completed

✅ **COMPLETED: Rewritten test_pair_exponential_family.py**
- Applied new tolerance framework to entire test file (23 tests)
- Replaced manual tolerances with `quantum_assert_close()` calls
- Added scientifically justified tolerance categories
- Improved error messages with tolerance context
- All tests pass with proper quantum-appropriate bounds

🔍 **INVESTIGATION NEEDED: test_pair_numerical_validation.py**
- Rewritten with scientific tolerance framework and shared FD helpers
- **5 tests failing** – suspected bugs in analytical implementations (Fisher metric, constraint Hessian/gradient, Jacobian, multi-pair ρ derivatives)
- Finite difference vs analytical comparisons exceed Category D tolerances by orders of magnitude in key components
- **CRITICAL**: These failures indicate potential bugs in core quantum algorithms
- Tests left failing to highlight issues requiring investigation/fixes (no tolerance loosening)

✅ **IN PROGRESS: test_inaccessible_game.py refactor**
- Integrated `tests/tolerance_framework.py` into core state/operator tests
- Replaced hand-written tolerance checks with `quantum_assert_*` helpers at original numeric bounds
- Switched from deprecated `log_partition()` to `psi()` for Fisher FD checks
- Centralised Fisher FD logic through `tests/fd_helpers.py`
- All 45 tests still passing; deprecation warnings removed for this file

✅ **IN PROGRESS: Finite-difference helper consolidation**
- Created `tests/fd_helpers.py` with shared FD implementations:
  - `finite_difference_rho_derivative`, `finite_difference_fisher`
  - `finite_difference_constraint_gradient`, `finite_difference_constraint_hessian`
  - `finite_difference_jacobian`
- Wired both `test_inaccessible_game.py` and `test_pair_numerical_validation.py` to use shared helpers
- Ensured entangled vs local regimes are tested in separate files but use the same FD machinery

✅ **COMPLETED: Test cleanup and FD helper consolidation**
- Removed redundant backup files (`_old`, `_new` variants)
- Reduced test file count from 22 to 19 files
- Refactored `test_non_commuting_bkm.py` and `test_commuting_bkm.py` to use shared `fd_helpers`
- Fixed ~2340 deprecation warnings (log_partition → psi)
- Now: **173 tests total, 165 passing, 8 failing, 5 warnings**

**Current Test Status (2025-11-26):**
- Total tests: 173 (down from 209 after removing duplicates)
- Pass rate: 95.4% (165/173)
- Warnings: 5 (down from ~2340)
- Test execution time: ~10 minutes

**Known Failures (8 tests):**
1. 5 tests in `test_pair_numerical_validation.py` - analytical bugs (Fisher, Hessian, Jacobian, ρ derivatives)
2. 1 test in `test_inaccessible_game.py` - constrained maxent dynamics
3. 1 test in `test_jacobian_analytic.py` - degenerate local basis case
4. 1 test in `test_non_commuting_bkm.py` - non-commuting BKM metric issue

**Next:**
- Apply tolerance framework to remaining constraint tests (4 files)
- Apply tolerance framework to Jacobian tests (2 files)
- Consider consolidating BKM tests (currently 7 files with similar coverage)
