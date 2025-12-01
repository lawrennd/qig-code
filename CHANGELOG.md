# Changelog

All notable changes to the `qig` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - GENERIC Decomposition Framework (CIP-0006)

Complete implementation of GENERIC decomposition for Lie-algebraic bases, enabling systematic extraction of effective Hamiltonian and diffusion operators from constrained quantum dynamics.

#### New Modules

- **`qig/structure_constants.py`**: Lie algebra structure constant computation and verification
  - `compute_structure_constants()`: Calculate f_abc from operator basis
  - `verify_antisymmetry()`, `verify_jacobi_identity()`: Validation functions
  - Caching mechanism for efficiency
  
- **`qig/generic.py`**: Core GENERIC decomposition components
  - `effective_hamiltonian_coefficients()`: Extract η from antisymmetric flow (Method A: linear solver)
  - `effective_hamiltonian_coefficients_lstsq()`: Extract η via least-squares (Method B)
  - `effective_hamiltonian_operator()`: Construct H_eff = Σ η_a F_a
  - `kubo_mori_derivatives()`: Compute ∂ρ/∂θ using Duhamel formula
  - `diffusion_operator()`: Construct D[ρ] from symmetric flow
  - `milburn_approximation()`: Near-equilibrium approximation
  - Cross-validation functions for both components

- **`qig/generic_decomposition.py`**: High-level orchestration interface
  - `GenericDecomposition` class: Complete 12-step procedure
    - `compute_all()`: Execute all steps systematically
    - `_compute_diagnostics()`: Comprehensive validation checks
    - `print_summary()`: Human-readable results
  - `run_generic_decomposition()`: Convenience function

- **`qig/validation.py`**: Robust validation framework
  - `ValidationReport` class for test result collection
  - Matrix comparison utilities with tolerance hierarchies
  - Property checks (Hermiticity, symmetry, tracelessness)
  - Finite difference validation
  - Physical constraint verification

- **`qig/reference_data.py`**: Reference data for testing
  - Known SU(2) and SU(3) structure constants
  - Verification functions for Lie algebra properties

#### Extended Modules

- **`qig/exponential_family.py`**:
  - `symmetric_part()`: Compute S = ½(M + M^T) from flow Jacobian
  - `antisymmetric_part()`: Compute A = ½(M - M^T) from flow Jacobian
  - `verify_degeneracy_conditions()`: Check GENERIC degeneracy conditions

- **`qig/dynamics.py`**:
  - `GenericDynamics` class (extends `InaccessibleGameDynamics`):
    - `compute_generic_decomposition()`: Full GENERIC analysis at any point
    - `integrate_reversible()`: Hamiltonian-only dynamics
    - `integrate_irreversible()`: Dissipative-only dynamics
    - `integrate_with_monitoring()`: Full dynamics with GENERIC structure tracking
    - Tracks H_eff, entropy production, S/A norms, cumulative entropy

#### Examples

- `examples/generic_decomposition_demo.py`: Basic decomposition with eigenvalue analysis
- `examples/generic_decomposition_complete.py`: Complete workflow with visualizations

#### Documentation

- **API Reference**:
  - `docs/source/api/structure_constants.rst`: Structure constant API
  - `docs/source/api/validation.rst`: Validation framework API
  - `docs/source/api/reference_data.rst`: Reference data API
  - `docs/source/api/generic.rst`: GENERIC components API
  - `docs/source/api/generic_decomposition.rst`: High-level interface API

- **User Guide**:
  - `docs/source/user_guide/generic_decomposition.rst`: Comprehensive user guide
    - Quick start and step-by-step tutorial
    - Advanced usage patterns
    - Performance considerations
    - Troubleshooting guide

- **Theory**:
  - `docs/source/theory/generic_structure.rst`: Enhanced with Duhamel integral discussion
    - When Duhamel integrals are needed
    - Lie closure cancellation for scalar derivatives
    - No cancellation for matrix-valued derivatives

- **Development**:
  - `docs/source/development/validation_framework.rst`: Validation infrastructure guide

#### Tests

Comprehensive test suite with 115 tests across 7 test files:

- `tests/test_validation.py` (32 tests): Validation framework
- `tests/test_structure_constants.py` (21 tests): Lie algebra structure
- `tests/test_generic_decomposition.py` (12 tests): S/A decomposition
- `tests/test_generic_hamiltonian.py` (15 tests): Hamiltonian extraction
- `tests/test_diffusion_operator.py` (16 tests): Diffusion operator
- `tests/test_generic_dynamics.py` (16 tests): Dynamics integration
- `tests/test_generic_decomposition_interface.py` (15 tests): High-level interface
- `tests/test_generic_decomposition_integration.py` (20 tests): End-to-end integration

All tests pass with documented tolerances.

#### Key Features

1. **Multi-Method Cross-Validation**: Multiple independent methods for critical computations
2. **Robust Testing**: Tolerance hierarchies and comprehensive property checks
3. **Performance Options**: Trade-off between speed (SLD method) and accuracy (Duhamel method)
4. **Comprehensive Diagnostics**: Automatic validation of all mathematical properties
5. **User-Friendly Interface**: High-level orchestration with detailed reporting
6. **Complete Documentation**: Theory, API reference, and practical user guide

#### Performance

- Structure constants: Computed once and cached
- Jacobian computation: ~1s for 15-parameter 2-qubit system
- Full decomposition (without D[ρ]): ~2-5s
- With diffusion operator: ~30-60s (expensive Kubo-Mori derivatives)

#### Backward Compatibility

This release is fully backward compatible:
- No changes to existing API
- New modules are independent
- New methods on existing classes are optional
- All existing tests continue to pass

### Implementation Details

**CIP-0006 Phases Completed:**
- Phase 0: Validation Infrastructure ✅
- Phase 1: Lie Algebra Structure ✅
- Phase 2: GENERIC Decomposition ✅
- Phase 3: Effective Hamiltonian Extraction ✅
- Phase 4: Diffusion Operator ✅
- Phase 5: Complete Dynamics ✅
- Phase 6: High-Level Interface ✅
- Phase 7: Testing and Validation ✅
- Phase 8: Documentation ✅

**Total Lines Added**: ~3,700+ lines of production code and tests

**Commits**:
- Validation framework and reference data
- Structure constants with Jacobi verification
- Symmetric/antisymmetric decomposition
- Effective Hamiltonian extraction (dual methods)
- Diffusion operator with Kubo-Mori derivatives
- GenericDynamics class with monitoring
- High-level GenericDecomposition interface
- Integration test suite
- Complete documentation

### Documentation Enhancements

- Conceptual clarification: When Duhamel integrals are needed
- Lie closure cancellation explained
- Parameter space vs density matrix space distinction
- Performance guidelines and optimization tips
- Troubleshooting section for common issues

---

## [Previous Versions]

*Previous changelog entries to be added as package evolves.*

