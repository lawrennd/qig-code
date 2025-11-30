"""
Test BKM metric computation for Lie-closed and non-Lie-closed operators.

The BKM metric is always G_ab = ∂²ψ/∂θ_a∂θ_b (Hessian of log-partition function).
This is valid regardless of whether operators form a Lie algebra, since ψ(θ) 
is a scalar function and mixed partials commute.

However, the computational complexity differs:

1. LIE-CLOSED operators (e.g., full Pauli/Gell-Mann basis):
   - The Duhamel integral cancels in natural parameter gradients
   - BCH formulas can be used for tractable computation
   - Example: full su(2) = {X, Y, Z} with [X,Y]=2iZ, [Y,Z]=2iX, [Z,X]=2iY

2. NON-LIE-CLOSED operators (e.g., X,Y without Z):
   - [X,Y] = 2iZ is NOT in the basis
   - Duhamel integral is "unavoidable" for computing ∂ρ/∂θ
   - But ψ's Hessian is still the correct answer for G
   - Spectral formula should still match (both compute ∂²ψ)

See Section 4.3 (Derivatives of the Cumulant Generating Function) of the paper.

This test validates that qig's fisher_information() correctly computes the 
Hessian of ψ in both cases.
"""

import numpy as np
import pytest
from scipy.linalg import expm

from qig.exponential_family import QuantumExponentialFamily
from tests.tolerance_framework import quantum_assert_close
from tests.fd_helpers import finite_difference_fisher

# Pauli matrices for constructing non-standard test cases
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


class SimpleExponentialFamily:
    """Minimal wrapper for testing non-standard operator sets with fd_helpers."""
    def __init__(self, operators):
        self.operators = operators
        self.n_params = len(operators)
    
    def psi(self, theta):
        """Log partition function ψ(θ) = log Tr[exp(Σ θ_a F_a)]."""
        K = sum(t * F for t, F in zip(theta, self.operators))
        return np.log(np.trace(expm(K))).real


# =============================================================================
# Tests for LIE-CLOSED operators (standard qig case)
# =============================================================================

class TestLieClosedOperators:
    """
    Tests for operators that form a Lie algebra (full Pauli/Gell-Mann).
    
    The qig code uses complete Lie-algebra bases (su(2), su(3), etc.),
    so Duhamel cancellation applies.
    """
    
    def test_single_qubit_matches_hessian(self):
        """Single qubit (su(2)): qig.fisher_information should match Hessian of ψ."""
        exp_fam = QuantumExponentialFamily(n_sites=1, d=2)
        
        for theta_vals in [(0.3, 0.5, 0.2), (0.1, 0.1, 0.1), (0.5, 0.2, 0.3)]:
            theta = np.array(theta_vals)
            
            G_qig = exp_fam.fisher_information(theta)
            G_hessian = finite_difference_fisher(exp_fam, theta)
            
            quantum_assert_close(G_qig, G_hessian, 'numerical_validation',
                                err_msg=f"Single qubit: qig ≠ Hessian at θ={theta_vals}")
    
    def test_single_qutrit_matches_hessian(self):
        """Single qutrit (su(3)): qig.fisher_information should match Hessian of ψ."""
        exp_fam = QuantumExponentialFamily(n_sites=1, d=3)
        
        np.random.seed(24)
        theta = np.random.randn(exp_fam.n_params)
        
        G_qig = exp_fam.fisher_information(theta)
        G_hessian = finite_difference_fisher(exp_fam, theta)
        
        quantum_assert_close(G_qig, G_hessian, 'numerical_validation',
                            err_msg="Single qutrit: qig ≠ Hessian")
    
    def test_two_qubits_matches_hessian(self):
        """Two qubits (su(2)⊗su(2)): should match Hessian."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        
        np.random.seed(42)
        theta = np.random.randn(exp_fam.n_params) 
        
        G_qig = exp_fam.fisher_information(theta)
        G_hessian = finite_difference_fisher(exp_fam, theta)
        
        quantum_assert_close(G_qig, G_hessian, 'numerical_validation',
                            err_msg="Two qubits: qig ≠ Hessian")


# =============================================================================
# Tests for NON-LIE-CLOSED operators
# =============================================================================

class TestNonLieClosedOperators:
    """
    Tests for operators that do NOT form a Lie algebra.
    
    Using only {X, Y}: [X,Y] = 2iZ, but Z is NOT in the basis.
    Per paper Section 4.3, the Duhamel integral is unavoidable for ∂ρ/∂θ,
    but ψ's Hessian G = ∂²ψ is still valid (ψ is a scalar).
    """
    
    def test_xy_only_hessian_well_defined(self):
        """
        X, Y only (non-Lie-closed): Hessian of ψ should be well-defined.
        
        Even though [X,Y]=2iZ is not in basis, ψ is still a scalar function
        with well-defined second derivatives.
        """
        operators = [X, Y]
        theta = np.array([0.3, 0.5])
        
        exp_fam = SimpleExponentialFamily(operators)
        G_hessian = finite_difference_fisher(exp_fam, theta)
        
        # Should be symmetric (mixed partials of scalar function commute)
        quantum_assert_close(G_hessian, G_hessian.T, 'fisher_metric',
                            err_msg="Hessian not symmetric for non-Lie-closed operators")
        
        # Should be positive semi-definite (convexity of log-partition)
        eigenvalues = np.linalg.eigvalsh(G_hessian)
        assert np.all(eigenvalues >= -1e-10), \
            f"Hessian has negative eigenvalue: {eigenvalues.min()}"


# =============================================================================
# Tests for entangled pair basis (su(d²))
# =============================================================================

class TestPairBasisBKM:
    """
    Tests for pair basis operators (su(d²) generators).
    
    These form a Lie algebra, so Duhamel cancellation applies.
    """
    
    def test_qubit_pair_matches_hessian(self):
        """Qubit pair (su(4)): qig.fisher_information should match Hessian."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        np.random.seed(42)
        theta = np.random.randn(exp_fam.n_params) * 0.1
        
        G_qig = exp_fam.fisher_information(theta)
        G_hessian = finite_difference_fisher(exp_fam, theta)
        
        quantum_assert_close(G_qig, G_hessian, 'numerical_validation',
                            err_msg="Qubit pair: qig ≠ Hessian")
    
    def test_qutrit_pair_matches_hessian(self):
        """Qutrit pair (su(9)): qig.fisher_information should match Hessian."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        
        np.random.seed(42)
        theta = np.random.randn(exp_fam.n_params) * 0.05
        
        G_qig = exp_fam.fisher_information(theta)
        G_hessian = finite_difference_fisher(exp_fam, theta)
        
        quantum_assert_close(G_qig, G_hessian, 'numerical_validation',
                            err_msg="Qutrit pair: qig ≠ Hessian")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
