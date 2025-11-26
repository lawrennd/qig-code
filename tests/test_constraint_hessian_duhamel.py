"""
Test constraint Hessian with high-precision Duhamel method.

This should achieve < 1% error using the numerical differentiation
of Duhamel ∂ρ/∂θ for computing ∂²ρ/∂θ_a∂θ_b.
"""

import numpy as np
import pytest

from qig.exponential_family import QuantumExponentialFamily
from tests.fd_helpers import finite_difference_constraint_hessian
from tests.tolerance_framework import quantum_assert_close, quantum_assert_symmetric


class TestConstraintHessianDuhamel:
    """Test constraint Hessian with high-precision Duhamel method."""
    
    def test_single_qubit_duhamel(self):
        """Test constraint Hessian with Duhamel method on single qubit."""
        exp_family = QuantumExponentialFamily(n_sites=1, d=2)
        theta = np.array([0.3, 0.5, 0.2])  # X, Y, Z
        
        # Ground truth via finite differences
        hessian_fd = finite_difference_constraint_hessian(exp_family, theta, eps=1e-7)
        
        # Check FD symmetry
        quantum_assert_symmetric(hessian_fd, 'constraint_hessian',
                                err_msg="FD Hessian not symmetric")
        
        # Test Duhamel method with high precision (n=100 points)
        hessian_duhamel = exp_family.constraint_hessian(
            theta, method='duhamel', n_points=100, eps=1e-7
        )
        
        # Check Duhamel symmetry
        quantum_assert_symmetric(hessian_duhamel, 'constraint_hessian',
                                err_msg="Duhamel Hessian not symmetric",
)
        
        # Compare Duhamel vs FD (should be < 1% error with n=100)
        # Duhamel uses finite-point quadrature, so use Category E_coarse tolerances
        quantum_assert_close(hessian_duhamel, hessian_fd, 'duhamel_integration',
                           err_msg="Duhamel method: analytic vs FD mismatch")


    def test_diagonal_case_duhamel(self):
        """Test on diagonal operators (classical case)."""
        exp_family = QuantumExponentialFamily(n_sites=1, d=3)
        
        # Use only diagonal operators (λ3 and λ8 from Gell-Mann)
        theta = np.zeros(exp_family.n_params)
        theta[2] = 0.5  # λ3 (diagonal)
        theta[7] = 0.3  # λ8 (diagonal)
        
        # Ground truth via finite differences
        hessian_fd = finite_difference_constraint_hessian(exp_family, theta, eps=1e-7)
        
        # Check FD symmetry
        quantum_assert_symmetric(hessian_fd, 'constraint_hessian',
                                err_msg="Diagonal case: FD Hessian not symmetric")
        
        # Test Duhamel method
        hessian_duhamel = exp_family.constraint_hessian(theta, method='duhamel', n_points=100)
        
        # Check Duhamel symmetry
        quantum_assert_symmetric(hessian_duhamel, 'constraint_hessian',
                                err_msg="Diagonal case: Duhamel Hessian not symmetric",
)
        
        # Compare Duhamel vs FD (should achieve < 1% error)
        # Duhamel uses finite-point quadrature, so use Category E_coarse tolerances
        quantum_assert_close(hessian_duhamel, hessian_fd, 'duhamel_integration',
                           err_msg="Diagonal case: Duhamel vs FD mismatch")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

