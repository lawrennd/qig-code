"""
Analytic Jacobian computation for quantum inaccessible game dynamics.

The flow is:
    F(θ) = -Π_∥(θ) G(θ) θ

where:
    Π_∥(θ) = I - a(θ)a(θ)^T / ||a(θ)||²
    G(θ) = BKM Fisher information matrix
    a(θ) = ∇C(θ) = gradient of marginal entropy constraint

The Jacobian is:
    M_ij = ∂F_i/∂θ_j

This requires:
1. ∂G/∂θ (third-order Kubo-Mori cumulant)
2. ∂a/∂θ (Hessian of constraint)
3. ∂Π_∥/∂θ (from ∂a/∂θ)

QUANTUM DERIVATIVE CHECKLIST:
✅ Check operator commutation
✅ Verify operator ordering
✅ Distinguish quantum vs classical
✅ Respect Hilbert space structure
✅ Question each derivative step
"""

import numpy as np
from scipy.linalg import eigh
import pytest

from qig.exponential_family import QuantumExponentialFamily
from qig.dynamics import InaccessibleGameDynamics
from tests.fd_helpers import finite_difference_constraint_hessian, finite_difference_jacobian
from tests.tolerance_framework import quantum_assert_close


# 

class TestJacobianAnalytic:
    """Test analytic Jacobian computation."""
    
    @pytest.mark.parametrize("n_sites,d", [
        (2, 2),  # Two qubits
    ])
    def test_constraint_hessian(self, n_sites, d):
        """Test that constraint Hessian is computed correctly."""
        exp_family = QuantumExponentialFamily(n_sites, d)
        
        np.random.seed(42)
        theta = np.random.randn(exp_family.n_params) * 0.2
        
        # Analytic Hessian (using QIG implementation)
        H_analytic = exp_family.constraint_hessian(theta)
        
        # Finite-difference Hessian using shared helper
        H_fd = finite_difference_constraint_hessian(exp_family, theta, eps=1e-6)
        
        # Compare using tolerance framework (Category D: analytical derivatives)
        quantum_assert_close(H_analytic, H_fd, 'constraint_hessian',
                           err_msg=f"Constraint Hessian test ({n_sites} sites, d={d})")
    
    @pytest.mark.parametrize("n_pairs,d", [
        (1, 3),  # One entangled qutrit pair (where Jacobian actually matters)
    ])
    def test_jacobian_vs_finite_difference(self, n_pairs, d):
        """Test that analytic Jacobian matches finite differences."""
        exp_family = QuantumExponentialFamily(n_pairs=n_pairs, d=d, pair_basis=True)
        dynamics = InaccessibleGameDynamics(exp_family)
        
        np.random.seed(24)
        theta = np.random.randn(exp_family.n_params)  
        
        # Analytic Jacobian (using QIG implementation)
        M_analytic = exp_family.jacobian(theta)
        
        # Finite-difference Jacobian using shared helper
        M_fd = finite_difference_jacobian(exp_family, theta, eps=1e-6)
        
        # Compare using tolerance framework (Category D: analytical derivatives)
        # Note: This test is known to fail for entangled systems - analytical Jacobian has bugs
        quantum_assert_close(M_analytic, M_fd, 'jacobian',
                           err_msg=f"Jacobian test ({n_pairs} pair(s), d={d}) - known analytical bug")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

