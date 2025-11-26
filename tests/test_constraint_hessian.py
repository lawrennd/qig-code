"""
Test constraint Hessian ∇²C computation.

The constraint is C(θ) = ∑ᵢ hᵢ(θ) where hᵢ = -Tr(ρᵢ log ρᵢ).

The Hessian is ∇²C with elements:
    ∂²C/∂θ_a∂θ_b = ∑ᵢ ∂²hᵢ/∂θ_a∂θ_b

For each marginal entropy hᵢ:
    ∂²hᵢ/∂θ_a∂θ_b = -Tr(∂²ρᵢ/∂θ_a∂θ_b (I + log ρᵢ))
                      -Tr(∂ρᵢ/∂θ_a ∂(log ρᵢ)/∂θ_b)

where ∂(log ρᵢ)/∂θ_b is computed using the Daleckii-Krein formula.

QUANTUM DERIVATIVE CHECKLIST:
✅ Check operator commutation: Marginal operators may not commute
✅ Verify operator ordering: Careful with matrix products
✅ Distinguish quantum vs classical: Uses quantum marginal entropies
✅ Respect Hilbert space structure: Partial traces for marginals
✅ Question each derivative step: Daleckii-Krein for log derivative
"""

import numpy as np
import pytest

from qig.exponential_family import QuantumExponentialFamily
from tests.fd_helpers import finite_difference_constraint_hessian
from tests.tolerance_framework import quantum_assert_close, quantum_assert_symmetric


class TestConstraintHessian:
    """Test constraint Hessian computation."""
    
    def test_diagonal_case(self):
        """
        Test on diagonal operators where everything is classical.
        """
        # Create simple diagonal family (qutrit with diagonal operators)
        n_sites = 1
        d = 3
        exp_family = QuantumExponentialFamily(n_sites, d)
        
        # Use only diagonal operators (λ3 and λ8 from Gell-Mann)
        theta = np.zeros(exp_family.n_params)
        theta[2] = 0.5  # λ3 (diagonal)
        theta[7] = 0.3  # λ8 (diagonal)
        
        # Compute via finite differences (ground truth)
        hessian_fd = finite_difference_constraint_hessian(exp_family, theta, eps=1e-7)
        
        # Check FD symmetry using tolerance framework
        quantum_assert_symmetric(hessian_fd, 'constraint_hessian',
                                err_msg="FD Hessian not symmetric")
        
        # Compute analytic version
        hessian_analytic = exp_family.constraint_hessian(theta)
        
        # Check analytic symmetry (should be exact)
        quantum_assert_symmetric(hessian_analytic, 'constraint_hessian',
                                err_msg="Analytic Hessian not symmetric",
)
        
        # Compare analytical vs FD (Category D: analytical derivatives)
        quantum_assert_close(hessian_analytic, hessian_fd, 'constraint_hessian',
                           err_msg="Diagonal case: analytic vs FD mismatch")
    
    def test_single_qubit(self):
        """
        Test on single qubit (simplest non-commuting case).
        """
        exp_family = QuantumExponentialFamily(n_sites=1, d=2)
        theta = np.array([0.3, 0.5, 0.2])  # X, Y, Z
        
        # Compute via finite differences
        hessian_fd = finite_difference_constraint_hessian(exp_family, theta, eps=1e-7)
        
        # Check FD symmetry
        quantum_assert_symmetric(hessian_fd, 'constraint_hessian',
                                err_msg="FD Hessian not symmetric")
        
        # Compute analytic version
        hessian_analytic = exp_family.constraint_hessian(theta)
        
        # Check analytic symmetry (should be exact)
        quantum_assert_symmetric(hessian_analytic, 'constraint_hessian',
                                err_msg="Analytic Hessian not symmetric",
)
        
        # Compare analytical vs FD (Category D: analytical derivatives)
        quantum_assert_close(hessian_analytic, hessian_fd, 'constraint_hessian',
                           err_msg="Single qubit: analytic vs FD mismatch")
    
    def test_symmetry(self):
        """
        Test that ∇²C is symmetric (as it must be for any Hessian).
        """
        exp_family = QuantumExponentialFamily(n_sites=1, d=2)
        theta = np.array([0.3, 0.5, 0.2])
        
        hessian_fd = finite_difference_constraint_hessian(exp_family, theta, eps=1e-7)
        
        # Check symmetry using tolerance framework
        quantum_assert_symmetric(hessian_fd, 'constraint_hessian',
                                err_msg="Hessian must be symmetric")
    
    @pytest.mark.parametrize("n_sites,d", [
        (2, 2),  # Two qubits
        (2, 3),  # Two qutrits
    ])
    def test_multiple_systems(self, n_sites, d):
        """Test on multi-site systems."""
        exp_family = QuantumExponentialFamily(n_sites, d)
        
        np.random.seed(42)
        theta = np.random.randn(exp_family.n_params) * 0.2
        
        # Compute via finite differences
        hessian_fd = finite_difference_constraint_hessian(exp_family, theta, eps=1e-6)
        
        # Check FD symmetry
        quantum_assert_symmetric(hessian_fd, 'constraint_hessian',
                                err_msg=f"{n_sites} sites d={d}: FD Hessian not symmetric")
        
        # Compute analytic version
        hessian_analytic = exp_family.constraint_hessian(theta)
        
        # Check analytic symmetry (should be exact)
        quantum_assert_symmetric(hessian_analytic, 'constraint_hessian',
                                err_msg=f"{n_sites} sites d={d}: Analytic Hessian not symmetric",
)
        
        # Compare analytical vs FD (Category D: analytical derivatives)
        quantum_assert_close(hessian_analytic, hessian_fd, 'constraint_hessian',
                           err_msg=f"{n_sites} sites d={d}: analytic vs FD mismatch")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

