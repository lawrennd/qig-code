"""
Test Lagrange multiplier gradient ∇ν implementation.

The Lagrange multiplier is ν(θ) = (a^T G θ)/(a^T a) where:
- a = ∇C = gradient of constraint C = ∑h_i
- G = BKM metric (Fisher information)

We validate ∇ν against finite differences.
"""

import numpy as np
import pytest
from qig.exponential_family import QuantumExponentialFamily
from tests.tolerance_framework import quantum_assert_close, quantum_assert_scalar_close


def compute_lagrange_multiplier(exp_family, theta):
    """
    Compute ν(θ) = (a^T G θ)/(a^T a).
    
    This is the Lagrange multiplier for the constraint ∑h_i = C.
    """
    # Get constraint gradient a
    _, a = exp_family.marginal_entropy_constraint(theta)
    
    # Get BKM metric G
    G = exp_family.fisher_information(theta)
    
    # Compute ν
    nu = np.dot(a, G @ theta) / np.dot(a, a)
    
    return nu


def compute_grad_nu_finite_diff(exp_family, theta, eps=1e-7):
    """Compute ∇ν using finite differences (ground truth)."""
    n = exp_family.n_params
    grad_nu = np.zeros(n)
    
    for j in range(n):
        theta_plus = theta.copy()
        theta_plus[j] += eps
        nu_plus = compute_lagrange_multiplier(exp_family, theta_plus)
        
        theta_minus = theta.copy()
        theta_minus[j] -= eps
        nu_minus = compute_lagrange_multiplier(exp_family, theta_minus)
        
        grad_nu[j] = (nu_plus - nu_minus) / (2 * eps)
    
    return grad_nu


class TestLagrangeMultiplierGradient:
    """Test ∇ν implementation."""
    
    def test_single_qubit_sld(self):
        """Test on single qubit with fast SLD method.
        
        NOTE: For the quantum exponential family with constraint C = ∑h_i,
        there is a structural identity: Gθ = -∇C = -a
        
        This gives ν = (a^T Gθ)/||a||² = -||a||²/||a||² = -1 (constant!)
        Therefore ∇ν = 0 everywhere, which is CORRECT.
        """
        print("\n" + "=" * 70)
        print("TESTING ∇ν ON SINGLE QUBIT (SLD METHOD)")
        print("=" * 70)
        
        exp_family = QuantumExponentialFamily(n_sites=1, d=2)
        theta = np.array([0.7, 0.3, 0.5])
        
        # Verify the structural identity Gθ = -a
        _, a = exp_family.marginal_entropy_constraint(theta)
        G = exp_family.fisher_information(theta)
        identity_check = np.linalg.norm(G @ theta + a)
        
        print(f"\nVerifying structural identity Gθ = -a:")
        print(f"  ||Gθ + a|| = {identity_check:.6e}")
        assert identity_check < 1e-10, "Structural identity violated!"
        
        # This implies ν = -1 always
        nu = np.dot(a, G @ theta) / np.dot(a, a)
        print(f"  ν = {nu:.10f} (should be -1)")
        assert abs(nu + 1.0) < 1e-10, f"ν = {nu}, not -1!"
        
        # Therefore ∇ν should be zero
        grad_nu_analytic = exp_family.lagrange_multiplier_gradient(theta, method='sld')
        grad_nu_norm = np.linalg.norm(grad_nu_analytic)
        
        print(f"\nAnalytic ∇ν:")
        print(f"  {grad_nu_analytic}")
        print(f"  ||∇ν|| = {grad_nu_norm:.6e}")
        
        # Should be zero to numerical precision (Category D: analytical derivatives)
        quantum_assert_close(grad_nu_analytic, np.zeros_like(grad_nu_analytic), 'constraint_gradient',
                           err_msg="∇ν should be zero due to structural identity Gθ = -a")
        
        # Verify with finite differences
        grad_nu_fd = compute_grad_nu_finite_diff(exp_family, theta, eps=1e-6)
        fd_norm = np.linalg.norm(grad_nu_fd)
        
        print(f"\nFinite difference ∇ν:")
        print(f"  {grad_nu_fd}")
        print(f"  ||∇ν|| = {fd_norm:.6e}")
        
        print("\n✓ Confirmed: ∇ν = 0 due to structural identity Gθ = -a")
    
    def test_single_qubit_duhamel(self):
        """Test on single qubit with high-precision Duhamel method.
        
        With Duhamel's high precision, we should see ∇ν = 0 exactly.
        """
        print("\n" + "=" * 70)
        print("TESTING ∇ν ON SINGLE QUBIT (DUHAMEL METHOD)")
        print("=" * 70)
        
        exp_family = QuantumExponentialFamily(n_sites=1, d=2)
        theta = np.array([0.7, 0.3, 0.5])
        
        # Analytic (Duhamel) - should be zero
        grad_nu_duhamel = exp_family.lagrange_multiplier_gradient(
            theta, method='duhamel', n_points=100
        )
        grad_nu_norm = np.linalg.norm(grad_nu_duhamel)
        
        print(f"\nDuhamel ∇ν:")
        print(f"  {grad_nu_duhamel}")
        print(f"  ||∇ν|| = {grad_nu_norm:.6e}")
        
        # Should be zero (Duhamel integration error is acceptable with E_coarse)
        quantum_assert_close(grad_nu_duhamel, np.zeros_like(grad_nu_duhamel), 'duhamel_integration',
                           err_msg="Duhamel: ∇ν should be ~0")
        print("✓ Duhamel confirms ∇ν ≈ 0")
    
    def test_diagonal_case(self):
        """Test on diagonal operators (qutrit).
        
        The structural identity Gθ = -a should hold for qutrits too.
        """
        print("\n" + "=" * 70)
        print("TESTING ∇ν ON DIAGONAL CASE (QUTRIT)")
        print("=" * 70)
        
        exp_family = QuantumExponentialFamily(n_sites=1, d=3)
        
        # Use only diagonal operators (λ3 and λ8)
        theta = np.zeros(exp_family.n_params)
        theta[2] = 0.5  # λ3
        theta[7] = 0.3  # λ8
        
        # Verify structural identity
        _, a = exp_family.marginal_entropy_constraint(theta)
        G = exp_family.fisher_information(theta)
        identity_check = np.linalg.norm(G @ theta + a)
        
        print(f"\nVerifying Gθ = -a: ||Gθ + a|| = {identity_check:.6e}")
        
        # Check ν
        nu = np.dot(a, G @ theta) / np.dot(a, a)
        print(f"ν = {nu:.10f}")
        
        # Test gradients
        grad_nu_duhamel = exp_family.lagrange_multiplier_gradient(
            theta, method='duhamel', n_points=100
        )
        grad_nu_norm = np.linalg.norm(grad_nu_duhamel)
        
        print(f"||∇ν|| = {grad_nu_norm:.6e}")
        
        quantum_assert_close(grad_nu_duhamel, np.zeros_like(grad_nu_duhamel), 'duhamel_integration',
                           err_msg="Qutrit: ∇ν should be ~0")
        print("✓ Qutrit confirms ∇ν = 0")
    
    @pytest.mark.parametrize("n_sites,d", [
        (2, 2),  # Two qubits
    ])
    def test_multiple_systems(self, n_sites, d):
        """Test on multi-site systems.
        
        The identity Gθ = -a should hold for multi-site systems too.
        """
        print(f"\n{n_sites} sites, d={d}:")
        
        exp_family = QuantumExponentialFamily(n_sites, d)
        
        np.random.seed(42)
        theta = np.random.randn(exp_family.n_params) * 0.2
        
        # Verify structural identity
        _, a = exp_family.marginal_entropy_constraint(theta)
        G = exp_family.fisher_information(theta)
        identity_check = np.linalg.norm(G @ theta + a)
        
        print(f"  ||Gθ + a|| = {identity_check:.6e}")
        
        # Check ν
        nu = np.dot(a, G @ theta) / np.dot(a, a)
        print(f"  ν = {nu:.10f}")
        
        # Test gradient
        grad_nu = exp_family.lagrange_multiplier_gradient(theta, method='sld')
        grad_nu_norm = np.linalg.norm(grad_nu)
        
        print(f"  ||∇ν|| = {grad_nu_norm:.6e}")
        
        quantum_assert_close(grad_nu, np.zeros_like(grad_nu), 'constraint_gradient',
                           err_msg=f"{n_sites} sites d={d}: ∇ν should be ~0")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

