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
from scipy.linalg import eigh

from qig.exponential_family import QuantumExponentialFamily


def compute_constraint_hessian_fd(
    exp_family: QuantumExponentialFamily,
    theta: np.ndarray,
    eps: float = 1e-7
) -> np.ndarray:
    """
    Compute ∇²C using finite differences of ∇C.
    
    This is the ground truth for validation.
    
    Parameters
    ----------
    exp_family : QuantumExponentialFamily
    theta : ndarray
    eps : float
        Finite difference step size
    
    Returns
    -------
    hessian : ndarray, shape (n_params, n_params)
        Constraint Hessian ∇²C
    """
    n = exp_family.n_params
    
    # Get gradient at theta
    _, grad_C = exp_family.marginal_entropy_constraint(theta)
    
    hessian = np.zeros((n, n))
    
    for j in range(n):
        # Compute ∂(∇C)/∂θ_j using finite differences
        theta_plus = theta.copy()
        theta_plus[j] += eps
        _, grad_C_plus = exp_family.marginal_entropy_constraint(theta_plus)
        
        theta_minus = theta.copy()
        theta_minus[j] -= eps
        _, grad_C_minus = exp_family.marginal_entropy_constraint(theta_minus)
        
        hessian[:, j] = (grad_C_plus - grad_C_minus) / (2 * eps)
    
    return hessian


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
        hessian_fd = compute_constraint_hessian_fd(exp_family, theta)
        
        print(f"\nDiagonal case:")
        print(f"Finite difference Hessian shape: {hessian_fd.shape}")
        print(f"Finite difference Hessian norm: {np.linalg.norm(hessian_fd):.6e}")
        
        # Check symmetry
        symmetry_err = np.max(np.abs(hessian_fd - hessian_fd.T))
        print(f"Symmetry error (FD): {symmetry_err:.6e}")
        assert symmetry_err < 1e-6, f"Hessian not symmetric: {symmetry_err:.3e}"
        
        # Compute analytic version
        hessian_analytic = exp_family.constraint_hessian(theta)
        
        # Check analytic symmetry
        symmetry_err_analytic = np.max(np.abs(hessian_analytic - hessian_analytic.T))
        print(f"Symmetry error (analytic): {symmetry_err_analytic:.6e}")
        assert symmetry_err_analytic < 1e-10, f"Analytic Hessian not symmetric: {symmetry_err_analytic:.3e}"
        
        # Compare
        diff = hessian_analytic - hessian_fd
        max_err = np.max(np.abs(diff))
        rel_err = max_err / (np.max(np.abs(hessian_fd)) + 1e-10)
        
        print(f"Analytic Hessian norm: {np.linalg.norm(hessian_analytic):.6e}")
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
        hessian_fd = compute_constraint_hessian_fd(exp_family, theta)
        
        print(f"\nSingle qubit:")
        print(f"Finite difference Hessian shape: {hessian_fd.shape}")
        print(f"Finite difference Hessian norm: {np.linalg.norm(hessian_fd):.6e}")
        
        # Check symmetry
        symmetry_err = np.max(np.abs(hessian_fd - hessian_fd.T))
        print(f"Symmetry error (FD): {symmetry_err:.6e}")
        assert symmetry_err < 1e-6, f"Hessian not symmetric: {symmetry_err:.3e}"
        
        # Compute analytic version
        hessian_analytic = exp_family.constraint_hessian(theta)
        
        # Check analytic symmetry
        symmetry_err_analytic = np.max(np.abs(hessian_analytic - hessian_analytic.T))
        print(f"Symmetry error (analytic): {symmetry_err_analytic:.6e}")
        assert symmetry_err_analytic < 1e-10, f"Analytic Hessian not symmetric: {symmetry_err_analytic:.3e}"
        
        # Compare
        diff = hessian_analytic - hessian_fd
        max_err = np.max(np.abs(diff))
        rel_err = max_err / (np.max(np.abs(hessian_fd)) + 1e-10)
        
        print(f"Analytic Hessian norm: {np.linalg.norm(hessian_analytic):.6e}")
        print(f"Max absolute error: {max_err:.6e}")
        print(f"Relative error: {rel_err:.6e}")
        
        assert rel_err < 1e-3, f"Single qubit failed: rel_err={rel_err:.3e}"
    
    def test_symmetry(self):
        """
        Test that ∇²C is symmetric (as it must be for any Hessian).
        """
        exp_family = QuantumExponentialFamily(n_sites=1, d=2)
        theta = np.array([0.3, 0.5, 0.2])
        
        hessian_fd = compute_constraint_hessian_fd(exp_family, theta, eps=1e-7)
        
        # Check symmetry
        symmetry_err = np.max(np.abs(hessian_fd - hessian_fd.T))
        
        print(f"\nSymmetry test:")
        print(f"Max |H - Hᵀ|: {symmetry_err:.6e}")
        
        assert symmetry_err < 1e-6, f"Hessian not symmetric: {symmetry_err:.3e}"
    
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
        hessian_fd = compute_constraint_hessian_fd(exp_family, theta, eps=1e-6)
        
        print(f"\n{n_sites} sites, d={d}:")
        print(f"Finite difference Hessian shape: {hessian_fd.shape}")
        print(f"Finite difference Hessian norm: {np.linalg.norm(hessian_fd):.6e}")
        
        # Check symmetry
        symmetry_err = np.max(np.abs(hessian_fd - hessian_fd.T))
        print(f"Symmetry error (FD): {symmetry_err:.6e}")
        assert symmetry_err < 1e-5, f"Hessian not symmetric: {symmetry_err:.3e}"
        
        # Compute analytic version
        hessian_analytic = exp_family.constraint_hessian(theta)
        
        # Check analytic symmetry
        symmetry_err_analytic = np.max(np.abs(hessian_analytic - hessian_analytic.T))
        print(f"Symmetry error (analytic): {symmetry_err_analytic:.6e}")
        assert symmetry_err_analytic < 1e-10, f"Analytic Hessian not symmetric: {symmetry_err_analytic:.3e}"
        
        # Compare
        diff = hessian_analytic - hessian_fd
        max_err = np.max(np.abs(diff))
        rel_err = max_err / (np.max(np.abs(hessian_fd)) + 1e-10)
        
        print(f"Analytic Hessian norm: {np.linalg.norm(hessian_analytic):.6e}")
        print(f"Max absolute error: {max_err:.6e}")
        print(f"Relative error: {rel_err:.6e}")
        
        assert rel_err < 1e-3, f"{n_sites} sites, d={d} failed: rel_err={rel_err:.3e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

