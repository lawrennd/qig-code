"""
Comprehensive test suite for constraint derivatives in quantum exponential families.

This module consolidates all constraint-related derivative tests from:
- test_marginal_entropy_gradient.py (gradient ∇C)
- test_constraint_hessian.py (Hessian ∇²C)
- test_constraint_hessian_duhamel.py (high-precision Duhamel Hessian)
- test_lagrange_multiplier_gradient.py (Lagrange multiplier gradient ∇ν)

Tests are organized by derivative order:
1. First-order: ∇C (marginal entropy constraint gradient)
2. Second-order: ∇²C (constraint Hessian)
3. Lagrange multiplier: ν and ∇ν

The constraint is C(θ) = ∑ᵢ hᵢ(θ) where hᵢ = -Tr(ρᵢ log ρᵢ).

Validates methods in: qig.exponential_family.QuantumExponentialFamily
- marginal_entropy_constraint() → (C, ∇C)
- constraint_hessian() → ∇²C  
- lagrange_multiplier_gradient() → ∇ν

CIP-0004: Uses tolerance framework with scientifically justified bounds.

QUANTUM DERIVATIVE CHECKLIST:
✅ Check operator commutation: Marginal operators may not commute
✅ Verify operator ordering: Careful with matrix products
✅ Distinguish quantum vs classical: Uses quantum marginal entropies
✅ Respect Hilbert space structure: Partial traces for marginals
✅ Question each derivative step: Daleckii-Krein for log derivative
"""

import numpy as np
import pytest
from scipy.linalg import expm, logm

from qig.core import partial_trace, von_neumann_entropy, marginal_entropies
from qig.exponential_family import QuantumExponentialFamily
from tests.fd_helpers import (
    finite_difference_constraint_gradient,
    finite_difference_constraint_hessian
)
from tests.tolerance_framework import (
    quantum_assert_close,
    quantum_assert_scalar_close,
    quantum_assert_symmetric
)


# ============================================================================
# SECTION 1: FIRST-ORDER DERIVATIVES (∇C)
# ============================================================================
# Gradient of marginal entropy constraint: ∂C/∂θ where C = ∑ᵢ hᵢ
# ============================================================================


class TestMarginalEntropyGradient:
    """Test marginal entropy constraint gradient ∇C."""
    
    @pytest.mark.parametrize("n_sites,d", [
        (2, 2),  # Two qubits
        (2, 3),  # Two qutrits
        (3, 2),  # Three qubits
    ])
    def test_analytic_vs_finite_difference(self, n_sites, d):
        """Test that analytic gradient matches finite differences."""
        exp_family = QuantumExponentialFamily(n_sites, d)
        
        # Test at multiple parameter points
        np.random.seed(42)
        for trial in range(3):
            theta = np.random.randn(exp_family.n_params) * 0.3
            
            # Implementation gradient
            C_impl, grad_impl = exp_family.marginal_entropy_constraint(theta)
            
            # Finite-difference gradient
            grad_fd = finite_difference_constraint_gradient(exp_family, theta, eps=1e-6)
            
            # Compare gradients (Category D_numerical: analytical vs FD)
            quantum_assert_close(grad_impl, grad_fd, 'numerical_validation',
                               err_msg=f"{n_sites} sites d={d} trial {trial}: Gradients mismatch")
    
    def test_gradient_chain_rule(self):
        """Verify gradient using numerical differentiation of C(θ)."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        theta = np.array([0.1, 0.2, -0.1, 0.3, -0.2, 0.15])
        
        C_impl, grad_impl = exp_family.marginal_entropy_constraint(theta)
        grad_numerical = finite_difference_constraint_gradient(exp_family, theta, eps=1e-6)
        
        # Compare (Category D_numerical: analytical vs FD)
        quantum_assert_close(grad_impl, grad_numerical, 'numerical_validation',
                           err_msg="Chain rule gradient doesn't match numerical")


# ============================================================================
# SECTION 2: SECOND-ORDER DERIVATIVES (∇²C)
# ============================================================================
# Hessian of marginal entropy constraint: ∂²C/∂θₐ∂θᵦ
# Tests both standard analytical method and high-precision Duhamel method
# ============================================================================


class TestConstraintHessian:
    """Test constraint Hessian ∇²C computation."""
    
    def test_diagonal_case(self):
        """Test on diagonal operators where everything is classical."""
        n_sites = 1
        d = 3
        exp_family = QuantumExponentialFamily(n_sites, d)
        
        # Use only diagonal operators (λ3 and λ8 from Gell-Mann)
        theta = np.zeros(exp_family.n_params)
        theta[2] = 0.5  # λ3 (diagonal)
        theta[7] = 0.3  # λ8 (diagonal)
        
        # Compute via finite differences (ground truth)
        hessian_fd = finite_difference_constraint_hessian(exp_family, theta, eps=1e-7)
        
        # Check FD symmetry
        quantum_assert_symmetric(hessian_fd, 'constraint_hessian',
                                err_msg="FD Hessian not symmetric")
        
        # Compute analytic version
        hessian_analytic = exp_family.constraint_hessian(theta)
        
        # Check analytic symmetry
        quantum_assert_symmetric(hessian_analytic, 'constraint_hessian',
                                err_msg="Analytic Hessian not symmetric")
        
        # Compare analytical vs FD (Category D: analytical derivatives)
        quantum_assert_close(hessian_analytic, hessian_fd, 'constraint_hessian',
                           err_msg="Diagonal case: analytic vs FD mismatch")
    
    def test_single_qubit(self):
        """Test on single qubit (simplest non-commuting case)."""
        exp_family = QuantumExponentialFamily(n_sites=1, d=2)
        theta = np.array([0.3, 0.5, 0.2])  # X, Y, Z
        
        # Compute via finite differences
        hessian_fd = finite_difference_constraint_hessian(exp_family, theta, eps=1e-7)
        
        # Check FD symmetry
        quantum_assert_symmetric(hessian_fd, 'constraint_hessian',
                                err_msg="FD Hessian not symmetric")
        
        # Compute analytic version
        hessian_analytic = exp_family.constraint_hessian(theta)
        
        # Check analytic symmetry
        quantum_assert_symmetric(hessian_analytic, 'constraint_hessian',
                                err_msg="Analytic Hessian not symmetric")
        
        # Compare analytical vs FD (Category D: analytical derivatives)
        quantum_assert_close(hessian_analytic, hessian_fd, 'constraint_hessian',
                           err_msg="Single qubit: analytic vs FD mismatch")
    
    def test_symmetry(self):
        """Test that ∇²C is symmetric (as it must be for any Hessian)."""
        exp_family = QuantumExponentialFamily(n_sites=1, d=2)
        theta = np.array([0.3, 0.5, 0.2])
        
        hessian_fd = finite_difference_constraint_hessian(exp_family, theta, eps=1e-7)
        
        # Check symmetry
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
        
        # Check analytic symmetry
        quantum_assert_symmetric(hessian_analytic, 'constraint_hessian',
                                err_msg=f"{n_sites} sites d={d}: Analytic Hessian not symmetric")
        
        # Compare analytical vs FD (Category D: analytical derivatives)
        quantum_assert_close(hessian_analytic, hessian_fd, 'constraint_hessian',
                           err_msg=f"{n_sites} sites d={d}: analytic vs FD mismatch")


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
                                err_msg="Duhamel Hessian not symmetric")
        
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
                                err_msg="Diagonal case: Duhamel Hessian not symmetric")
        
        # Compare Duhamel vs FD (should achieve < 1% error)
        quantum_assert_close(hessian_duhamel, hessian_fd, 'duhamel_integration',
                           err_msg="Diagonal case: Duhamel vs FD mismatch")


# ============================================================================
# SECTION 3: LAGRANGE MULTIPLIER GRADIENT (∇ν)
# ============================================================================
# Gradient of Lagrange multiplier: ν(θ) = (a^T G θ)/(a^T a)
# For systems with structural identity Gθ = -a, ν = -1 and ∇ν = 0
# ============================================================================


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
        """
        Test on single qubit with fast SLD method.
        
        NOTE: For the quantum exponential family with constraint C = ∑h_i,
        there is a structural identity: Gθ = -∇C = -a
        
        This gives ν = (a^T Gθ)/||a||² = -||a||²/||a||² = -1 (constant!)
        Therefore ∇ν = 0 everywhere, which is CORRECT.
        """
        exp_family = QuantumExponentialFamily(n_sites=1, d=2)
        theta = np.array([0.7, 0.3, 0.5])
        
        # Verify the structural identity Gθ = -a
        _, a = exp_family.marginal_entropy_constraint(theta)
        G = exp_family.fisher_information(theta)
        identity_check = np.linalg.norm(G @ theta + a)
        
        assert identity_check < 1e-10, "Structural identity violated!"
        
        # This implies ν = -1 always
        nu = np.dot(a, G @ theta) / np.dot(a, a)
        assert abs(nu + 1.0) < 1e-10, f"ν = {nu}, not -1!"
        
        # Therefore ∇ν should be zero
        grad_nu_analytic = exp_family.lagrange_multiplier_gradient(theta, method='sld')
        
        # Should be zero (Category D: analytical derivatives)
        quantum_assert_close(grad_nu_analytic, np.zeros_like(grad_nu_analytic), 'constraint_gradient',
                           err_msg="∇ν should be zero due to structural identity Gθ = -a")
        
        # Verify with finite differences
        grad_nu_fd = compute_grad_nu_finite_diff(exp_family, theta, eps=1e-6)
        assert np.linalg.norm(grad_nu_fd) < 1e-6, "FD confirms ∇ν ≈ 0"
    
    def test_single_qubit_duhamel(self):
        """Test on single qubit with high-precision Duhamel method."""
        exp_family = QuantumExponentialFamily(n_sites=1, d=2)
        theta = np.array([0.7, 0.3, 0.5])
        
        # Analytic (Duhamel) - should be zero
        grad_nu_duhamel = exp_family.lagrange_multiplier_gradient(
            theta, method='duhamel', n_points=100
        )
        
        # Should be zero (Duhamel integration error acceptable with E_coarse)
        quantum_assert_close(grad_nu_duhamel, np.zeros_like(grad_nu_duhamel), 'duhamel_integration',
                           err_msg="Duhamel: ∇ν should be ~0")
    
    def test_diagonal_case(self):
        """Test on diagonal operators (qutrit)."""
        exp_family = QuantumExponentialFamily(n_sites=1, d=3)
        
        # Use only diagonal operators (λ3 and λ8)
        theta = np.zeros(exp_family.n_params)
        theta[2] = 0.5  # λ3
        theta[7] = 0.3  # λ8
        
        # Verify structural identity
        _, a = exp_family.marginal_entropy_constraint(theta)
        G = exp_family.fisher_information(theta)
        identity_check = np.linalg.norm(G @ theta + a)
        assert identity_check < 1e-10, "Structural identity should hold"
        
        # Test gradients
        grad_nu_duhamel = exp_family.lagrange_multiplier_gradient(
            theta, method='duhamel', n_points=100
        )
        
        quantum_assert_close(grad_nu_duhamel, np.zeros_like(grad_nu_duhamel), 'duhamel_integration',
                           err_msg="Qutrit: ∇ν should be ~0")
    
    @pytest.mark.parametrize("n_sites,d", [
        (2, 2),  # Two qubits
    ])
    def test_multiple_systems(self, n_sites, d):
        """Test on multi-site systems."""
        exp_family = QuantumExponentialFamily(n_sites, d)
        
        np.random.seed(42)
        theta = np.random.randn(exp_family.n_params) * 0.2
        
        # Verify structural identity
        _, a = exp_family.marginal_entropy_constraint(theta)
        G = exp_family.fisher_information(theta)
        identity_check = np.linalg.norm(G @ theta + a)
        assert identity_check < 1e-10, "Structural identity should hold"
        
        # Check ν
        nu = np.dot(a, G @ theta) / np.dot(a, a)
        assert abs(nu + 1.0) < 1e-10, "ν should be -1"
        
        # Test gradient
        grad_nu = exp_family.lagrange_multiplier_gradient(theta, method='sld')
        
        quantum_assert_close(grad_nu, np.zeros_like(grad_nu), 'constraint_gradient',
                           err_msg=f"{n_sites} sites d={d}: ∇ν should be ~0")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

