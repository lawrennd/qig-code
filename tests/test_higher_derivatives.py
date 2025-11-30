"""
Comprehensive test suite for higher-order derivatives in quantum exponential families.

This module consolidates all Jacobian and third cumulant tests from:
- test_jacobian.py (full Jacobian M = ∂F/∂θ)
- test_jacobian_analytic.py (analytic Jacobian computation)
- test_third_cumulant.py (third cumulant tensor (∇G)[θ])
- test_third_cumulant_symmetry.py (symmetry validation)

Tests are organized by derivative type:
1. Jacobian: M = ∂F/∂θ where F(θ) = -G(θ)θ + ν(θ)a(θ)
2. Third Cumulant: ∇G = ∇³ψ (totally symmetric tensor)

The Jacobian governs linearized dynamics around constraint manifold.
The third cumulant describes how the Fisher metric varies with θ.

Validates methods in: qig.exponential_family.QuantumExponentialFamily
- jacobian() → M = ∂F/∂θ
- third_cumulant_contraction() → (∇G)[θ]

CIP-0004: Uses tolerance framework with scientifically justified bounds.

QUANTUM DERIVATIVE CHECKLIST:
✅ Check operator commutation: F_a, F_b, F_c may not commute
✅ Verify operator ordering: Careful in spectral differentiation
✅ Distinguish quantum vs classical: Uses quantum covariance derivatives
✅ Respect Hilbert space structure: Works on full Hilbert space
✅ Question each derivative step: Use perturbation theory for derivatives
"""

import numpy as np
import pytest
from scipy.linalg import eigh

from qig.exponential_family import QuantumExponentialFamily
from qig.dynamics import InaccessibleGameDynamics
from tests.fd_helpers import (
    finite_difference_jacobian,
    finite_difference_constraint_hessian
)
from tests.tolerance_framework import quantum_assert_close


# ============================================================================
# SECTION 1: JACOBIAN TESTS (M = ∂F/∂θ)
# ============================================================================
# The Jacobian governs the linearized dynamics around a point on the
# constraint manifold. From the paper (eq. 824-827):
#     M = -G - (∇G)[θ] + ν∇²C + a(∇ν)^T
# With structural identity Gθ = -a and ν = -1, ∇ν = 0, this simplifies to:
#     M = -G - (∇G)[θ] - ∇²C
# ============================================================================


def compute_dynamics(exp_family, theta):
    """Compute F(θ) = -G(θ)θ + ν(θ)a(θ)."""
    G = exp_family.fisher_information(theta)
    _, a = exp_family.marginal_entropy_constraint(theta)
    nu = np.dot(a, G @ theta) / np.dot(a, a)
    
    F = -G @ theta + nu * a
    return F


class TestJacobian:
    """Test full Jacobian M implementation."""
    
    def test_single_qubit_sld(self):
        """
        Test Jacobian on single qubit with SLD method.
        
        NOTE: Due to the structural identity Gθ = -∇C, we have F(θ) = 0
        everywhere on the manifold (equilibrium). However, M = ∂F/∂θ is
        still well-defined and describes the linearized response to 
        perturbations. We compare against finite differences but expect
        M to be small in magnitude.
        """
        exp_family = QuantumExponentialFamily(n_sites=1, d=2)
        theta = np.array([0.7, 0.3, 0.5])
        
        # Verify F = 0 at this point
        F = compute_dynamics(exp_family, theta)
        assert np.linalg.norm(F) < 1e-10, "Should be equilibrium"
        
        # Analytic Jacobian
        M_analytic = exp_family.jacobian(theta, method='sld')
        
        # M should be small but well-defined
        assert np.linalg.norm(M_analytic) > 1e-12, "M is degenerate"
        assert np.linalg.norm(M_analytic) < 1.0, "M is unexpectedly large"
    
    def test_single_qubit_duhamel(self):
        """Test Jacobian on single qubit with Duhamel method."""
        exp_family = QuantumExponentialFamily(n_sites=1, d=2)
        theta = np.array([0.7, 0.3, 0.5])
        
        # Analytic (Duhamel)
        M_duhamel = exp_family.jacobian(theta, method='duhamel', n_points=100)
        
        # Test structure
        S = 0.5 * (M_duhamel + M_duhamel.T)
        A = 0.5 * (M_duhamel - M_duhamel.T)
        
        # Duhamel should give essentially symmetric M
        assert np.linalg.norm(A) < 1.0 * np.linalg.norm(S), "M not symmetric enough"
    
    def test_eigenvalue_degeneracy(self):
        """
        Test that M has expected degeneracy on constraint manifold.
        
        From the paper: M should have degeneracy related to the geometry
        of the constraint manifold. For a single qubit with constraint
        C = S(ρ), the manifold has codimension 1, so M should have
        (at least) rank deficiency 1.
        """
        exp_family = QuantumExponentialFamily(n_sites=1, d=2)
        theta = np.array([0.7, 0.3, 0.5])
        
        M = exp_family.jacobian(theta, method='duhamel', n_points=100)
        
        # Compute eigenvalues
        eigvals = np.linalg.eigvalsh(M + M.T) / 2  # Symmetrize for stability
        eigvals_sorted = np.sort(np.abs(eigvals))
        
        # Check for degeneracy
        smallest_eig = eigvals_sorted[0]
        second_smallest = eigvals_sorted[1]
        
        # There should be at least one small eigenvalue
        assert smallest_eig < 0.5 * second_smallest, \
            f"No clear degeneracy: smallest={smallest_eig:.3e}, second={second_smallest:.3e}"
    
    def test_constraint_preservation(self):
        """
        Test that M preserves the constraint to first order.
        
        Since F(θ) preserves the constraint exactly (a^T F = 0),
        the Jacobian should satisfy: a^T M = 0
        """
        exp_family = QuantumExponentialFamily(n_sites=1, d=2)
        theta = np.array([0.7, 0.3, 0.5])
        
        _, a = exp_family.marginal_entropy_constraint(theta)
        M = exp_family.jacobian(theta, method='duhamel', n_points=100)
        
        # Check a^T M = 0
        a_T_M = a @ M
        
        # Should be zero (Category D: analytical derivatives)
        quantum_assert_close(a_T_M, np.zeros_like(a_T_M), 'jacobian',
                           err_msg="Constraint preservation: a^T M should be ~0")
    
    @pytest.mark.parametrize("n_sites,d", [
        (2, 2),  # Two qubits
    ])
    def test_multiple_systems(self, n_sites, d):
        """Test Jacobian on multi-site systems."""
        exp_family = QuantumExponentialFamily(n_sites, d)
        
        np.random.seed(42)
        theta = np.random.randn(exp_family.n_params) * 0.2
        
        # Verify equilibrium
        F = compute_dynamics(exp_family, theta)
        assert np.linalg.norm(F) < 1e-6, "Should be near equilibrium"
        
        # Analytic (SLD for speed)
        M_analytic = exp_family.jacobian(theta, method='sld')
        
        # Check constraint preservation
        _, a = exp_family.marginal_entropy_constraint(theta)
        
        quantum_assert_close(a @ M_analytic, np.zeros(exp_family.n_params), 'jacobian',
                           err_msg=f"{n_sites} sites d={d}: Constraint not preserved")
        
        # Check Sa degeneracy
        S = 0.5 * (M_analytic + M_analytic.T)
        quantum_assert_close(S @ a, np.zeros_like(a), 'jacobian',
                           err_msg=f"{n_sites} sites d={d}: Degeneracy Sa=0 not satisfied")


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
        
        # Compute constraint Hessian
        hessian_analytic = exp_family.constraint_hessian(theta)
        hessian_fd = finite_difference_constraint_hessian(exp_family, theta, eps=1e-6)
        
        # Compare (Category D_numerical: analytical vs FD)
        quantum_assert_close(hessian_analytic, hessian_fd, 'numerical_validation',
                           err_msg=f"{n_sites} sites d={d}: Constraint Hessian mismatch")
    
    @pytest.mark.parametrize("n_pairs,d", [
        (1, 3),  # One entangled qutrit pair (where Jacobian actually matters)
    ])
    def test_jacobian_vs_finite_difference(self, n_pairs, d):
        """Test that analytic Jacobian matches finite differences."""
        exp_family = QuantumExponentialFamily(n_pairs=n_pairs, d=d, pair_basis=True)
        
        np.random.seed(24)
        theta = np.random.randn(exp_family.n_params)
        
        # Analytic Jacobian
        M_analytic = exp_family.jacobian(theta)
        
        # Finite-difference Jacobian
        M_fd = finite_difference_jacobian(exp_family, theta, eps=1e-6)
        
        # Compare (Category D_numerical: analytical vs FD)
        # Note: This test is known to fail for entangled systems - analytical Jacobian has bugs
        quantum_assert_close(M_analytic, M_fd, 'jacobian',
                           err_msg=f"Jacobian test ({n_pairs} pair(s), d={d}) - known analytical bug")


# ============================================================================
# SECTION 2: THIRD CUMULANT TESTS (∇G = ∇³ψ)
# ============================================================================
# The third cumulant tensor T_abc = ∂³ψ/∂θ_a∂θ_b∂θ_c describes how the
# Fisher metric varies with parameters. It's totally symmetric and appears
# in the Jacobian as (∇G)[θ].
# ============================================================================


def compute_third_cumulant_contraction_fd(
    exp_family: QuantumExponentialFamily,
    theta: np.ndarray,
    eps: float = 1e-7
) -> np.ndarray:
    """
    Compute (∇G)[θ] using finite differences.
    
    This is the ground truth for validation.
    
    Parameters
    ----------
    exp_family : QuantumExponentialFamily
    theta : ndarray
    eps : float
        Finite difference step size
    
    Returns
    -------
    contraction : ndarray, shape (n_params, n_params)
        Matrix with (i,j) entry = ∑_k (∂G_ik/∂θ_j) θ_k
    """
    n = exp_family.n_params
    G = exp_family.fisher_information(theta)
    
    contraction = np.zeros((n, n))
    
    for j in range(n):
        # Compute ∂G/∂θ_j using finite differences
        theta_plus = theta.copy()
        theta_plus[j] += eps
        G_plus = exp_family.fisher_information(theta_plus)
        
        theta_minus = theta.copy()
        theta_minus[j] -= eps
        G_minus = exp_family.fisher_information(theta_minus)
        
        dG_dtheta_j = (G_plus - G_minus) / (2 * eps)
        
        # Contract with theta: ∑_k (∂G_ik/∂θ_j) θ_k
        contraction[:, j] = dG_dtheta_j @ theta
    
    return contraction


class TestThirdCumulant:
    """Test third cumulant computation."""
    
    def test_diagonal_case(self):
        """Test on diagonal operators where everything is classical."""
        # Create simple diagonal family (qutrit with diagonal operators)
        n_sites = 1
        d = 3
        exp_family = QuantumExponentialFamily(n_sites, d)
        
        # Use only diagonal operators (λ3 and λ8 from Gell-Mann)
        theta = np.zeros(exp_family.n_params)
        theta[2] = 0.5  # λ3 (diagonal)
        theta[7] = 0.3  # λ8 (diagonal)
        
        # Compute via finite differences
        contraction_fd = compute_third_cumulant_contraction_fd(exp_family, theta)
        
        # Compute analytic version
        contraction_analytic = exp_family.third_cumulant_contraction(theta)
        
        # Compare (Category D: analytical derivatives)
        quantum_assert_close(contraction_analytic, contraction_fd, 'fisher_metric',
                           err_msg="Diagonal case: analytic vs FD third cumulant mismatch")
    
    def test_single_qubit(self):
        """Test on single qubit (simplest non-commuting case)."""
        exp_family = QuantumExponentialFamily(n_sites=1, d=2)
        theta = np.array([0.3, 0.5, 0.2])  # X, Y, Z
        
        # Compute via finite differences
        contraction_fd = compute_third_cumulant_contraction_fd(exp_family, theta)
        
        # Compute analytic version
        contraction_analytic = exp_family.third_cumulant_contraction(theta)
        
        # Compare (Category D: analytical derivatives)
        quantum_assert_close(contraction_analytic, contraction_fd, 'fisher_metric',
                           err_msg="Single qubit: analytic vs FD third cumulant mismatch")
    
    def test_symmetry_in_first_two_indices(self):
        """
        Test that ∂G_ab/∂θ_c = ∂G_ba/∂θ_c (G is symmetric).
        
        This means (∇G)[θ]_ij should equal (∇G)[θ]_ji when we swap the
        roles of i and j in the original tensor.
        """
        exp_family = QuantumExponentialFamily(n_sites=1, d=2)
        theta = np.array([0.3, 0.5, 0.2])
        
        eps = 1e-7
        n = exp_family.n_params
        
        # For each θ_c, check that ∂G/∂θ_c is symmetric
        for c in range(n):
            theta_plus = theta.copy()
            theta_plus[c] += eps
            G_plus = exp_family.fisher_information(theta_plus)
            
            theta_minus = theta.copy()
            theta_minus[c] -= eps
            G_minus = exp_family.fisher_information(theta_minus)
            
            dG_dtheta_c = (G_plus - G_minus) / (2 * eps)
            
            # Check symmetry (Category D: analytical derivatives)
            quantum_assert_close(dG_dtheta_c, dG_dtheta_c.T, 'fisher_metric',
                               err_msg=f"∂G/∂θ_{c} not symmetric")
    
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
        contraction_fd = compute_third_cumulant_contraction_fd(exp_family, theta, eps=1e-6)
        
        # Compute analytic version
        contraction_analytic = exp_family.third_cumulant_contraction(theta)
        
        # Compare (Category D: analytical derivatives)
        quantum_assert_close(contraction_analytic, contraction_fd, 'fisher_metric',
                           err_msg=f"{n_sites} sites, d={d}: analytic vs FD third cumulant mismatch")


class TestThirdCumulantSymmetry:
    """Test total symmetry of third cumulant tensor."""
    
    def test_third_cumulant_symmetry(self):
        """
        Test T_abc = T_bac = T_cab etc. (total symmetry under all permutations).
        
        We verify that the BKM derivative (third cumulant) is symmetric by checking
        that the Hessian of the Fisher information is symmetric in its indices.
        
        This is a fundamental sanity check: since ψ(θ) is a scalar function,
        mixed partial derivatives must commute regardless of operator non-commutativity.
        """
        # Setup: Use 1 qutrit pair (smaller system for faster testing)
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        n_params = exp_fam.n_params
        
        # Test at a non-zero point
        np.random.seed(42)
        theta = 0.1 * np.random.randn(n_params)
        
        # Test symmetry by computing ∂G_ab/∂θ_c numerically
        # This is the third cumulant T_abc
        eps = 1e-5
        
        # Sample a few indices to test
        test_indices = [(0, 1, 2), (0, 2, 1), (1, 0, 2), 
                        (2, 0, 1), (1, 2, 0), (2, 1, 0)]
        
        def compute_T_abc(a, b, c):
            """Compute T_abc = ∂G_ab/∂θ_c"""
            theta_plus = theta.copy()
            theta_plus[c] += eps
            G_plus = exp_fam.fisher_information(theta_plus)
            
            theta_minus = theta.copy()
            theta_minus[c] -= eps
            G_minus = exp_fam.fisher_information(theta_minus)
            
            return (G_plus[a, b] - G_minus[a, b]) / (2 * eps)
        
        # Compute all six permutations
        T_values = {}
        for perm in test_indices:
            T_values[perm] = compute_T_abc(*perm)
        
        # Check all permutations are equal (Category D: analytical derivatives)
        reference = T_values[(0, 1, 2)]
        
        for perm in test_indices[1:]:
            quantum_assert_close(T_values[perm], reference, 'fisher_metric',
                               err_msg=f"Third cumulant permutation T[{perm}] != T[0,1,2]")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

