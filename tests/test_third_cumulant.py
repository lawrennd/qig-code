"""
Test third cumulant (∇G)[θ] computation.

Following paper's Appendix (eq. 824-826):
    ∂/∂θ_j (-Gθ)_i = -G_ij - ∑_k (∂G_ik/∂θ_j) θ_k

In matrix form: -G - (∇G)[θ]

where (∇G)[θ] is the matrix with (i,j) entry: ∑_k (∂G_ik/∂θ_j) θ_k

The tensor ∇G = ∇³ψ is the third cumulant tensor (totally symmetric).

QUANTUM DERIVATIVE CHECKLIST:
✅ Check operator commutation: F_a, F_b, F_c may not commute
✅ Verify operator ordering: Careful in spectral differentiation
✅ Distinguish quantum vs classical: Uses quantum covariance derivatives
✅ Respect Hilbert space structure: Works on full Hilbert space
✅ Question each derivative step: Use perturbation theory for eigenvalue/eigenvector derivatives
"""

import numpy as np
import pytest
from scipy.linalg import eigh

from qig.exponential_family import QuantumExponentialFamily


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
        """
        Test on diagonal operators where everything is classical.
        
        For diagonal operators, the BKM metric reduces to classical Fisher information,
        and the third cumulant should match classical third cumulants.
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
        contraction_fd = compute_third_cumulant_contraction_fd(exp_family, theta)
        
        print(f"\nDiagonal case:")
        print(f"Finite difference contraction shape: {contraction_fd.shape}")
        print(f"Finite difference contraction norm: {np.linalg.norm(contraction_fd):.6e}")
        
        # Compute analytic version
        contraction_analytic = exp_family.third_cumulant_contraction(theta)
        
        # Compare
        diff = contraction_analytic - contraction_fd
        max_err = np.max(np.abs(diff))
        rel_err = max_err / (np.max(np.abs(contraction_fd)) + 1e-10)
        
        print(f"Analytic contraction norm: {np.linalg.norm(contraction_analytic):.6e}")
        print(f"Max absolute error: {max_err:.6e}")
        print(f"Relative error: {rel_err:.6e}")
        
        assert rel_err < 1e-3, f"Diagonal case failed: rel_err={rel_err:.3e}"
    
    def test_single_qubit(self):
        """
        Test on single qubit (simplest non-commuting case).
        """
        exp_family = QuantumExponentialFamily(n_sites=1, d=2)
        theta = np.array([0.3, 0.5, 0.2])  # X, Y, Z
        
        # Compute via finite differences
        contraction_fd = compute_third_cumulant_contraction_fd(exp_family, theta)
        
        print(f"\nSingle qubit:")
        print(f"Finite difference contraction shape: {contraction_fd.shape}")
        print(f"Finite difference contraction norm: {np.linalg.norm(contraction_fd):.6e}")
        
        # Compute analytic version
        contraction_analytic = exp_family.third_cumulant_contraction(theta)
        
        # Compare
        diff = contraction_analytic - contraction_fd
        max_err = np.max(np.abs(diff))
        rel_err = max_err / (np.max(np.abs(contraction_fd)) + 1e-10)
        
        print(f"Analytic contraction norm: {np.linalg.norm(contraction_analytic):.6e}")
        print(f"Max absolute error: {max_err:.6e}")
        print(f"Relative error: {rel_err:.6e}")
        
        assert rel_err < 1e-3, f"Single qubit failed: rel_err={rel_err:.3e}"
    
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
            
            # Check symmetry
            diff = dG_dtheta_c - dG_dtheta_c.T
            max_err = np.max(np.abs(diff))
            
            assert max_err < 1e-6, (
                f"∂G/∂θ_{c} not symmetric: max_err={max_err:.3e}"
            )
        
        print(f"\nSymmetry test: All ∂G/∂θ_c are symmetric ✓")
    
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
        
        print(f"\n{n_sites} sites, d={d}:")
        print(f"Finite difference contraction shape: {contraction_fd.shape}")
        print(f"Finite difference contraction norm: {np.linalg.norm(contraction_fd):.6e}")
        
        # Compute analytic version
        contraction_analytic = exp_family.third_cumulant_contraction(theta)
        
        # Compare
        diff = contraction_analytic - contraction_fd
        max_err = np.max(np.abs(diff))
        rel_err = max_err / (np.max(np.abs(contraction_fd)) + 1e-10)
        
        print(f"Analytic contraction norm: {np.linalg.norm(contraction_analytic):.6e}")
        print(f"Max absolute error: {max_err:.6e}")
        print(f"Relative error: {rel_err:.6e}")
        
        assert rel_err < 1e-3, f"{n_sites} sites, d={d} failed: rel_err={rel_err:.3e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

