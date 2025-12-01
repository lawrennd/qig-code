"""
Tests for validation utilities and reference data.

This module tests the Phase 0 validation infrastructure to ensure
it's ready for use in all subsequent phases.
"""

import pytest
import numpy as np
from qig.validation import (
    ValidationReport, ValidationCheck,
    compare_matrices, check_hermitian, check_symmetric, check_antisymmetric,
    check_traceless, finite_difference_jacobian,
    check_constraint_tangency, check_entropy_monotonicity,
    check_positive_semidefinite
)
from qig.reference_data import (
    get_su2_structure_constants, get_su3_structure_constants,
    get_reference_structure_constants, verify_structure_constant_properties
)


class TestValidationReport:
    """Test ValidationReport class."""
    
    def test_empty_report(self):
        """Test empty validation report."""
        report = ValidationReport("Test")
        assert report.all_passed()
        assert len(report.get_failures()) == 0
        assert len(report.get_passes()) == 0
    
    def test_add_passing_check(self):
        """Test adding a passing check."""
        report = ValidationReport("Test")
        report.add_check("Test check", True, 1e-10, 1e-8)
        assert report.all_passed()
        assert len(report.get_passes()) == 1
        assert len(report.get_failures()) == 0
    
    def test_add_failing_check(self):
        """Test adding a failing check."""
        report = ValidationReport("Test")
        report.add_check("Test check", False, 1e-6, 1e-8)
        assert not report.all_passed()
        assert len(report.get_passes()) == 0
        assert len(report.get_failures()) == 1
    
    def test_mixed_checks(self):
        """Test report with both passing and failing checks."""
        report = ValidationReport("Test")
        report.add_check("Pass 1", True, 1e-10, 1e-8)
        report.add_check("Fail 1", False, 1e-6, 1e-8)
        report.add_check("Pass 2", True, 1e-12, 1e-10)
        
        assert not report.all_passed()
        assert len(report.get_passes()) == 2
        assert len(report.get_failures()) == 1


class TestMatrixComparison:
    """Test matrix comparison utilities."""
    
    def test_identical_matrices(self):
        """Test comparison of identical matrices."""
        A = np.random.rand(5, 5)
        passed, error, msg = compare_matrices(A, A, 1e-14, "Identity")
        assert passed
        assert error < 1e-14
    
    def test_small_difference(self):
        """Test matrices with small differences."""
        A = np.eye(3)
        B = A + 1e-10 * np.random.rand(3, 3)
        passed, error, msg = compare_matrices(A, B, 1e-8, "Small diff")
        assert passed
        assert error > 0
    
    def test_large_difference(self):
        """Test matrices with large differences."""
        A = np.eye(3)
        B = 2 * A
        passed, error, msg = compare_matrices(A, B, 1e-8, "Large diff")
        assert not passed
        assert error > 1e-8
    
    def test_shape_mismatch(self):
        """Test matrices with different shapes."""
        A = np.eye(3)
        B = np.eye(4)
        passed, error, msg = compare_matrices(A, B, 1e-8, "Shape mismatch")
        assert not passed
        assert "Shape mismatch" in msg
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        A = np.eye(3)
        B = A.copy()
        B[0, 0] = np.nan
        passed, error, msg = compare_matrices(A, B, 1e-8, "NaN test")
        assert not passed
        assert "NaN" in msg


class TestMatrixProperties:
    """Test matrix property checks."""
    
    def test_hermitian_real(self):
        """Test Hermitian check on real symmetric matrix."""
        A = np.random.rand(4, 4)
        A = A + A.T  # Make symmetric (hence Hermitian)
        passed, error = check_hermitian(A)
        assert passed
        assert error < 1e-14
    
    def test_hermitian_complex(self):
        """Test Hermitian check on complex matrix."""
        A = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
        A = A + A.conj().T  # Make Hermitian
        passed, error = check_hermitian(A)
        assert passed
        assert error < 1e-14
    
    def test_non_hermitian(self):
        """Test Hermitian check on non-Hermitian matrix."""
        A = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
        passed, error = check_hermitian(A)
        assert not passed
        assert error > 1e-12
    
    def test_symmetric(self):
        """Test symmetry check."""
        A = np.random.rand(4, 4)
        A = 0.5 * (A + A.T)
        passed, error = check_symmetric(A)
        assert passed
        assert error < 1e-14
    
    def test_antisymmetric(self):
        """Test antisymmetry check."""
        A = np.random.rand(4, 4)
        A = 0.5 * (A - A.T)
        passed, error = check_antisymmetric(A)
        assert passed
        assert error < 1e-14
    
    def test_traceless(self):
        """Test traceless check."""
        # Create traceless matrix
        A = np.random.rand(4, 4)
        A = A - np.eye(4) * np.trace(A) / 4
        passed, error = check_traceless(A)
        assert passed
        assert error < 1e-14
    
    def test_non_traceless(self):
        """Test traceless check on matrix with trace."""
        A = np.eye(5)
        passed, error = check_traceless(A)
        assert not passed
        assert np.abs(error - 5.0) < 1e-10


class TestFiniteDifferences:
    """Test finite difference Jacobian."""
    
    def test_linear_function(self):
        """Test Jacobian of linear function."""
        # f(x) = A @ x, so Jacobian = A
        A = np.random.rand(3, 4)
        func = lambda x: A @ x
        x = np.random.rand(4)
        
        J_fd = finite_difference_jacobian(func, x, eps=1e-6)
        
        assert J_fd.shape == (3, 4)
        assert np.allclose(J_fd, A, atol=1e-4)
    
    def test_quadratic_function(self):
        """Test Jacobian of quadratic function."""
        # f(x) = x^T A x / 2, so ∇f = A @ x (for symmetric A)
        A = np.random.rand(5, 5)
        A = 0.5 * (A + A.T)
        
        func = lambda x: 0.5 * x @ A @ x
        x = np.random.rand(5)
        
        # Jacobian is ∇f = A @ x
        J_fd = finite_difference_jacobian(lambda x: np.array([func(x)]), x, eps=1e-6)
        J_analytic = A @ x
        
        assert np.allclose(J_fd.flatten(), J_analytic, atol=1e-4)


class TestPhysicalConstraints:
    """Test physical constraint checks."""
    
    def test_constraint_tangency_satisfied(self):
        """Test tangency when M @ a = 0."""
        M = np.random.rand(5, 5)
        a = np.random.rand(5)
        
        # Make M orthogonal to a
        M = M - np.outer(M @ a, a) / (a @ a)
        
        passed, error = check_constraint_tangency(M, a, tol=1e-8)
        assert passed
        assert error < 1e-8
    
    def test_constraint_tangency_violated(self):
        """Test tangency when M @ a != 0."""
        M = np.eye(5)
        a = np.ones(5)
        
        passed, error = check_constraint_tangency(M, a, tol=1e-8)
        assert not passed
        assert error > 1e-8
    
    def test_entropy_monotonicity_decreasing(self):
        """Test entropy monotonicity when θ^T M θ < 0."""
        # M negative definite
        A = np.random.rand(4, 4)
        M = -A @ A.T
        theta = np.random.rand(4)
        
        passed, value = check_entropy_monotonicity(M, theta, tol=1e-12)
        assert passed
        assert value < 0
    
    def test_entropy_monotonicity_increasing(self):
        """Test entropy monotonicity violation when θ^T M θ > 0."""
        # M positive definite
        A = np.random.rand(4, 4)
        M = A @ A.T
        theta = np.random.rand(4)
        
        passed, value = check_entropy_monotonicity(M, theta, tol=1e-12)
        assert not passed
        assert value > 0
    
    def test_positive_semidefinite(self):
        """Test positive semidefiniteness check."""
        # Create positive semidefinite matrix
        A = np.random.rand(5, 3)
        M = A @ A.T
        
        passed, min_eig = check_positive_semidefinite(M)
        assert passed
        assert min_eig >= -1e-14


class TestReferenceData:
    """Test reference structure constants."""
    
    def test_su2_shape(self):
        """Test SU(2) structure constants have correct shape."""
        f = get_su2_structure_constants()
        assert f.shape == (3, 3, 3)
    
    def test_su3_shape(self):
        """Test SU(3) structure constants have correct shape."""
        f = get_su3_structure_constants()
        assert f.shape == (8, 8, 8)
    
    def test_su2_antisymmetry(self):
        """Test SU(2) structure constants are antisymmetric."""
        f = get_su2_structure_constants()
        # f_abc = -f_bac
        assert np.allclose(f, -np.transpose(f, (1, 0, 2)))
    
    def test_su3_antisymmetry(self):
        """Test SU(3) structure constants are antisymmetric."""
        f = get_su3_structure_constants()
        # f_abc = -f_bac
        assert np.allclose(f, -np.transpose(f, (1, 0, 2)))
    
    def test_su2_specific_values(self):
        """Test specific SU(2) structure constant values."""
        f = get_su2_structure_constants()
        # f_123 = 1 (and cyclic permutations)
        assert np.abs(f[0, 1, 2] - 1.0) < 1e-14
        assert np.abs(f[1, 2, 0] - 1.0) < 1e-14
        assert np.abs(f[2, 0, 1] - 1.0) < 1e-14
        # f_213 = -1 (and cyclic permutations)
        assert np.abs(f[1, 0, 2] + 1.0) < 1e-14
    
    def test_su2_verification(self):
        """Test SU(2) structure constants pass verification."""
        f = get_su2_structure_constants()
        results = verify_structure_constant_properties(f, "SU(2)")
        assert results['passed']
        assert results['antisymmetry_error'] < 1e-10
        assert results['jacobi_error'] < 1e-8
    
    def test_su3_verification(self):
        """Test SU(3) structure constants pass verification."""
        f = get_su3_structure_constants()
        results = verify_structure_constant_properties(f, "SU(3)")
        assert results['passed']
        assert results['antisymmetry_error'] < 1e-10
        assert results['jacobi_error'] < 1e-8
    
    def test_get_reference_by_name(self):
        """Test getting reference data by algebra name."""
        f_su2 = get_reference_structure_constants("su2")
        assert f_su2.shape == (3, 3, 3)
        
        f_su3 = get_reference_structure_constants("su3")
        assert f_su3.shape == (8, 8, 8)
    
    def test_unknown_algebra_raises(self):
        """Test that unknown algebra type raises error."""
        with pytest.raises(ValueError, match="Unknown algebra type"):
            get_reference_structure_constants("su5")


# Phase 0 Gate: All these tests must pass
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

