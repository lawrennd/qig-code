Validation Framework
====================

Overview
--------

The validation framework provides comprehensive testing utilities for verifying the correctness of GENERIC decomposition computations. This infrastructure is used throughout all implementation phases to ensure numerical accuracy and mathematical consistency.

Components
----------

1. Validation Utilities (``qig/validation.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**ValidationReport Class**

Collects validation checks throughout a computation:

- Provides summary statistics and detailed reporting
- Identifies which checks passed and which failed
- Can print verbose or failure-only reports

**Matrix Property Checks**

- ``check_hermitian()``: Verify M = M† (tol ~ 10⁻¹²)
- ``check_symmetric()``: Verify M = Mᵀ (tol ~ 10⁻¹⁴)
- ``check_antisymmetric()``: Verify M = -Mᵀ (tol ~ 10⁻¹⁴)
- ``check_traceless()``: Verify Tr(M) = 0 (tol ~ 10⁻¹⁰)

**Comparison Utilities**

- ``compare_matrices()``: Smart matrix comparison with multiple error norms
- Handles edge cases: NaN, Inf, near-zero values
- Reports Frobenius norm and relative error

**Cross-Validation Tools**

- ``finite_difference_jacobian()``: Independent Jacobian computation
- Used to validate analytically computed Jacobians
- Central differences for accuracy

**Physical Constraint Checks**

- ``check_constraint_tangency()``: Verify M @ ∇C ≈ 0
- ``check_entropy_monotonicity()``: Verify θᵀMθ ≤ 0
- ``check_positive_semidefinite()``: Verify all eigenvalues ≥ 0

2. Reference Data (``qig/reference_data.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**SU(2) Structure Constants**

- 3 generators (Pauli matrices)
- Completely antisymmetric: f_abc = ε_ijk
- Verified: Antisymmetry 0.00e+00, Jacobi 0.00e+00

**SU(3) Structure Constants**

- 8 generators (Gell-Mann matrices)
- Reference: Gell-Mann (1962), Particle Data Group
- Verified: Antisymmetry 0.00e+00, Jacobi 1.11e-16

**Verification Functions**

- ``verify_structure_constant_properties()``: Check antisymmetry and Jacobi identity
- ``get_reference_structure_constants()``: Access by algebra name

3. Test Suite (``tests/test_validation.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**32 comprehensive tests covering:**

- ValidationReport functionality (4 tests)
- Matrix comparison (5 tests)
- Matrix properties (7 tests)
- Finite differences (2 tests)
- Physical constraints (5 tests)
- Reference data (9 tests)

All tests passing ✓ (0.49s runtime)

Usage Examples
--------------

Basic Validation Report
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from qig.validation import ValidationReport, check_hermitian

    # Create report
    report = ValidationReport("Hamiltonian Validation")

    # Check properties
    passed, error = check_hermitian(H_eff, tol=1e-12)
    report.add_check("Hermiticity", passed, error, 1e-12)

    passed, error = check_traceless(H_eff, tol=1e-10)
    report.add_check("Traceless", passed, error, 1e-10)

    # Print results
    report.print_summary()

    # Check if all passed
    if not report.all_passed():
        failures = report.get_failures()
        for fail in failures:
            print(f"FAILED: {fail}")

Cross-Validation
~~~~~~~~~~~~~~~~

.. code-block:: python

    from qig.validation import compare_matrices, finite_difference_jacobian

    # Compute Jacobian analytically
    J_analytic = exp_fam.jacobian(theta)

    # Compute Jacobian numerically
    flow = lambda th: exp_fam.flow_vector(th)
    J_numeric = finite_difference_jacobian(flow, theta, eps=1e-5)

    # Compare
    passed, error, msg = compare_matrices(
        J_analytic, J_numeric, tol=1e-5,
        name="Jacobian cross-validation"
    )
    print(f"Cross-validation: {error:.2e} (tol: 1e-5)")

Using Reference Data
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from qig.reference_data import get_reference_structure_constants

    # Load reference structure constants
    f_ref = get_reference_structure_constants("su3")

    # Compare computed vs reference
    passed, error, msg = compare_matrices(
        f_computed, f_ref, tol=1e-10,
        name="Structure constants"
    )

Tolerance Hierarchy
-------------------

Different quantities have different expected tolerances:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Check
     - Tolerance
     - Reason
   * - Symmetry (S = Sᵀ)
     - ~10⁻¹⁴
     - Machine precision
   * - Hermiticity (H = H†)
     - ~10⁻¹²
     - Direct computation
   * - Tracelessness
     - ~10⁻¹⁰
     - Numerical accumulation
   * - Structure constants
     - ~10⁻¹⁰
     - Reference comparison
   * - Jacobi identity
     - ~10⁻⁸
     - Triple sum
   * - Commutator matching
     - ~10⁻⁶
     - Transformation error
   * - Finite difference
     - ~10⁻⁵
     - FD approximation

Phase 0 Completion Status
--------------------------

✅ **All Phase 0 Completion Criteria Met:**

- ☑ All validation utilities implemented and tested (32 tests passing)
- ☑ ValidationReport class produces readable output
- ☑ Reference data for SU(2) verified (antisymmetry 0.00e+00, Jacobi 0.00e+00)
- ☑ Reference data for SU(3) verified (antisymmetry 0.00e+00, Jacobi 1.11e-16)
- ☑ All comparison functions handle edge cases (NaN, Inf tested)
- ☑ Documentation complete (this document)

**Gate to Phase 1: OPEN** ✓

The validation infrastructure is ready for use in all subsequent phases of the GENERIC decomposition implementation.

References
----------

1. Gell-Mann, M. (1962). "Symmetries of Baryons and Mesons". *Physical Review*.
2. Particle Data Group. *Review of Particle Physics* (structure constants tables).
3. CIP-0006: GENERIC Decomposition for Lie-Algebraic Bases (implementation plan).

API Reference
-------------

See :doc:`../api/index` for detailed API documentation of validation utilities.

