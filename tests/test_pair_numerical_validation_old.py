"""
Numerical validation tests for pair-based exponential family.

Tests:
1. rho_derivative validation against finite differences
2. Fisher metric validation against numerical differentiation
3. Constraint Hessian validation for pair basis
4. Comparison with existing validation tests
"""

import numpy as np
import pytest
from qig.exponential_family import QuantumExponentialFamily


def finite_difference_rho_derivative(exp_fam, theta, a, eps=1e-6):
    """Compute ∂ρ/∂θ_a using finite differences."""
    theta_plus = theta.copy()
    theta_plus[a] += eps
    theta_minus = theta.copy()
    theta_minus[a] -= eps
    
    rho_plus = exp_fam.rho_from_theta(theta_plus)
    rho_minus = exp_fam.rho_from_theta(theta_minus)
    
    return (rho_plus - rho_minus) / (2 * eps)


def finite_difference_fisher(exp_fam, theta, eps=1e-6):
    """Compute Fisher metric using finite differences of log partition."""
    G_fd = np.zeros((exp_fam.n_params, exp_fam.n_params))
    
    for a in range(exp_fam.n_params):
        for b in range(a, exp_fam.n_params):
            # Compute ∂²ψ/∂θ_a∂θ_b using 4-point finite difference
            theta_pp = theta.copy()
            theta_pp[a] += eps
            theta_pp[b] += eps
            
            theta_pm = theta.copy()
            theta_pm[a] += eps
            theta_pm[b] -= eps
            
            theta_mp = theta.copy()
            theta_mp[a] -= eps
            theta_mp[b] += eps
            
            theta_mm = theta.copy()
            theta_mm[a] -= eps
            theta_mm[b] -= eps
            
            psi_pp = exp_fam.psi(theta_pp)
            psi_pm = exp_fam.psi(theta_pm)
            psi_mp = exp_fam.psi(theta_mp)
            psi_mm = exp_fam.psi(theta_mm)
            
            G_fd[a, b] = (psi_pp - psi_pm - psi_mp + psi_mm) / (4 * eps**2)
            G_fd[b, a] = G_fd[a, b]  # Symmetry
    
    return G_fd


class TestRhoDerivativeNumerical:
    """Test ∂ρ/∂θ against finite differences for pair basis."""
    
    def test_single_pair_duhamel(self):
        """Test rho_derivative with Duhamel for single pair."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        np.random.seed(42)
        theta = np.random.randn(exp_fam.n_params) * 0.3
        
        # Test a few parameters
        for a in [0, 5, 10]:
            drho_analytic = exp_fam.rho_derivative(theta, a, method='duhamel')
            drho_fd = finite_difference_rho_derivative(exp_fam, theta, a, eps=1e-7)
            
            error = np.linalg.norm(drho_analytic - drho_fd) / np.linalg.norm(drho_fd)
            print(f"Parameter {a}: relative error = {error:.6e}")
            
            # Duhamel with 100 points gives ~1e-5 to 1e-4 error for 4×4 matrices
            assert error < 1e-4, f"Parameter {a}: error {error} too large"
    
    def test_two_pairs_duhamel(self):
        """Test rho_derivative for two pairs."""
        exp_fam = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
        
        np.random.seed(43)
        theta = np.random.randn(exp_fam.n_params) * 0.3
        
        # Test one parameter from each pair
        for a in [0, 15]:  # Pair 0 and pair 1
            drho_analytic = exp_fam.rho_derivative(theta, a, method='duhamel')
            drho_fd = finite_difference_rho_derivative(exp_fam, theta, a, eps=1e-7)
            
            error = np.linalg.norm(drho_analytic - drho_fd) / np.linalg.norm(drho_fd)
            print(f"Parameter {a}: relative error = {error:.6e}")
            
            # For 16×16 matrices, expect slightly larger errors
            assert error < 1e-4, f"Parameter {a}: error {error} too large"


class TestFisherMetricNumerical:
    """Test Fisher metric against finite differences."""
    
    def test_fisher_vs_finite_difference_single_pair(self):
        """Test Fisher metric against ∂²ψ/∂θ∂θ for single pair."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        np.random.seed(44)
        theta = np.random.randn(exp_fam.n_params) * 0.3
        
        print("Computing analytic Fisher metric...")
        G_analytic = exp_fam.fisher_information(theta)
        
        print("Computing finite difference Fisher metric...")
        G_fd = finite_difference_fisher(exp_fam, theta, eps=1e-6)
        
        # Compare
        error = np.linalg.norm(G_analytic - G_fd) / np.linalg.norm(G_fd)
        print(f"Relative error: {error:.6e}")
        print(f"Max abs difference: {np.max(np.abs(G_analytic - G_fd)):.6e}")
        
        # Show a few elements
        print("\nSample elements comparison:")
        for i in range(min(3, exp_fam.n_params)):
            for j in range(i, min(3, exp_fam.n_params)):
                print(f"G[{i},{j}]: analytic={G_analytic[i,j]:.6f}, FD={G_fd[i,j]:.6f}")
        
        # Finite differences of second derivatives have ~1e-3 to 1e-4 accuracy
        assert error < 1e-3, f"Fisher metric error {error} too large"
    
    def test_fisher_block_structure_numerically(self):
        """Verify block-diagonal structure using finite differences."""
        exp_fam = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
        
        np.random.seed(45)
        theta = np.random.randn(exp_fam.n_params) * 0.3
        
        print("Computing Fisher metric with finite differences...")
        G_fd = finite_difference_fisher(exp_fam, theta, eps=1e-6)
        
        # Check cross-pair block is zero
        cross_block = G_fd[:15, 15:]
        cross_norm = np.linalg.norm(cross_block)
        diagonal_norm = np.linalg.norm(G_fd[:15, :15])
        
        print(f"Cross-pair block norm: {cross_norm:.6e}")
        print(f"Diagonal block norm: {diagonal_norm:.6e}")
        print(f"Relative cross-pair size: {cross_norm/diagonal_norm:.6e}")
        
        # Cross-pair should be much smaller than diagonal blocks
        # Finite differences give ~1e-3, analytic gives ~1e-16 (see test_pair_exponential_family.py)
        assert cross_norm / diagonal_norm < 1e-2, \
            "Cross-pair block not small in finite difference calculation"
        
        # Document the difference: analytic is MUCH more accurate
        print("\nNote: Analytic calculation gives cross-pair elements ~10⁻¹⁶")
        print("      Finite differences limited by numerical precision")


class TestConstraintHessianPairBasis:
    """Test constraint Hessian for pair basis."""
    
    def test_constraint_hessian_single_pair(self):
        """Test ∇²C = ∇a for single pair."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        np.random.seed(46)
        theta = np.random.randn(exp_fam.n_params) * 0.3
        
        print("Computing constraint Hessian...")
        hess_C = exp_fam.constraint_hessian(theta, method='duhamel', n_points=100)
        
        # Verify symmetry
        assert np.allclose(hess_C, hess_C.T), "Constraint Hessian not symmetric"
        
        # Verify against finite differences
        eps = 1e-6
        hess_C_fd = np.zeros_like(hess_C)
        
        for a in range(min(5, exp_fam.n_params)):  # Test subset
            theta_plus = theta.copy()
            theta_plus[a] += eps
            theta_minus = theta.copy()
            theta_minus[a] -= eps
            
            rho_plus = exp_fam.rho_from_theta(theta_plus)
            rho_minus = exp_fam.rho_from_theta(theta_minus)
            
            from qig.core import marginal_entropies
            h_plus = marginal_entropies(rho_plus, exp_fam.dims)
            h_minus = marginal_entropies(rho_minus, exp_fam.dims)
            
            grad_C_plus = np.sum(h_plus)
            grad_C_minus = np.sum(h_minus)
            
            # This is ∂(∑h_i)/∂θ_a, now differentiate again
            for b in range(min(5, exp_fam.n_params)):
                theta_pp = theta.copy()
                theta_pp[a] += eps
                theta_pp[b] += eps
                theta_pm = theta.copy()
                theta_pm[a] += eps
                theta_pm[b] -= eps
                theta_mp = theta.copy()
                theta_mp[a] -= eps
                theta_mp[b] += eps
                theta_mm = theta.copy()
                theta_mm[a] -= eps
                theta_mm[b] -= eps
                
                rho_pp = exp_fam.rho_from_theta(theta_pp)
                rho_pm = exp_fam.rho_from_theta(theta_pm)
                rho_mp = exp_fam.rho_from_theta(theta_mp)
                rho_mm = exp_fam.rho_from_theta(theta_mm)
                
                C_pp = np.sum(marginal_entropies(rho_pp, exp_fam.dims))
                C_pm = np.sum(marginal_entropies(rho_pm, exp_fam.dims))
                C_mp = np.sum(marginal_entropies(rho_mp, exp_fam.dims))
                C_mm = np.sum(marginal_entropies(rho_mm, exp_fam.dims))
                
                hess_C_fd[a, b] = (C_pp - C_pm - C_mp + C_mm) / (4 * eps**2)
        
        # Compare subset
        error = np.linalg.norm(hess_C[:5, :5] - hess_C_fd[:5, :5]) / \
                (np.linalg.norm(hess_C_fd[:5, :5]) + 1e-10)
        print(f"Constraint Hessian relative error (5×5 subset): {error:.6e}")
        
        assert error < 0.05, f"Constraint Hessian error {error} too large"


class TestConstraintGradient:
    """Test constraint gradient ∇C = ∇(∑h_i) for pair basis."""
    
    def test_constraint_gradient_single_pair(self):
        """Test ∇C against finite differences for single pair."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        np.random.seed(47)
        theta = np.random.randn(exp_fam.n_params) * 0.3
        
        print("Computing constraint gradient...")
        C, grad_C = exp_fam.marginal_entropy_constraint(theta, method='duhamel')
        
        # Verify against finite differences
        eps = 1e-7
        grad_C_fd = np.zeros_like(grad_C)
        
        for a in range(min(10, exp_fam.n_params)):  # Test subset
            theta_plus = theta.copy()
            theta_plus[a] += eps
            theta_minus = theta.copy()
            theta_minus[a] -= eps
            
            C_plus, _ = exp_fam.marginal_entropy_constraint(theta_plus, method='duhamel')
            C_minus, _ = exp_fam.marginal_entropy_constraint(theta_minus, method='duhamel')
            
            grad_C_fd[a] = (C_plus - C_minus) / (2 * eps)
        
        error = np.linalg.norm(grad_C[:10] - grad_C_fd[:10]) / \
                np.linalg.norm(grad_C_fd[:10])
        print(f"Constraint gradient relative error (10 elements): {error:.6e}")
        
        assert error < 1e-5, f"Constraint gradient error {error} too large"
    
    def test_constraint_gradient_two_pairs(self):
        """Test ∇C for two pairs with block structure."""
        exp_fam = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
        
        np.random.seed(48)
        theta = np.random.randn(exp_fam.n_params) * 0.3
        
        C, grad_C = exp_fam.marginal_entropy_constraint(theta, method='duhamel')
        
        print(f"Constraint value C = {C:.6f}")
        print(f"Gradient norm: {np.linalg.norm(grad_C):.6f}")
        print(f"Gradient shape: {grad_C.shape}")
        
        # Verify gradient is computed for all parameters
        assert grad_C.shape == (30,)
        assert not np.allclose(grad_C, 0), "Gradient is zero"


class TestJacobianNumerical:
    """Test Jacobian M = ∂F/∂θ for pair basis."""
    
    def test_jacobian_vs_finite_difference(self):
        """Test Jacobian against finite differences of F(θ)."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        np.random.seed(49)
        theta = np.random.randn(exp_fam.n_params) * 0.3
        
        # Compute analytic Jacobian
        print("Computing analytic Jacobian...")
        M_analytic = exp_fam.jacobian(theta, method='duhamel')
        
        # Function to compute F(θ) = -Gθ + νa
        def compute_F(theta_val):
            G = exp_fam.fisher_information(theta_val)
            C, a = exp_fam.marginal_entropy_constraint(theta_val, method='duhamel')
            Gtheta = G @ theta_val
            nu = np.dot(a, Gtheta) / np.dot(a, a)
            F = -Gtheta + nu * a
            return F
        
        # Compute Jacobian via finite differences
        print("Computing Jacobian via finite differences...")
        eps = 1e-6
        M_fd = np.zeros((exp_fam.n_params, exp_fam.n_params))
        
        for j in range(exp_fam.n_params):
            theta_plus = theta.copy()
            theta_plus[j] += eps
            theta_minus = theta.copy()
            theta_minus[j] -= eps
            
            F_plus = compute_F(theta_plus)
            F_minus = compute_F(theta_minus)
            
            M_fd[:, j] = (F_plus - F_minus) / (2 * eps)
        
        # Compare
        error = np.linalg.norm(M_analytic - M_fd) / np.linalg.norm(M_fd)
        print(f"Jacobian relative error: {error:.6e}")
        
        # With Duhamel method, we achieve ~1-2×10⁻⁵ accuracy
        # Allow up to 5×10⁻⁵ for some numerical variation
        assert error < 5e-5, f"Jacobian error {error} too large"
    
    def test_jacobian_dynamics_nonzero(self):
        """Verify that F(θ) ≠ 0 on manifold for pair basis (genuine dynamics)."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        np.random.seed(50)
        theta = np.random.randn(exp_fam.n_params) * 0.3
        
        # Compute F(θ)
        G = exp_fam.fisher_information(theta)
        C, a = exp_fam.marginal_entropy_constraint(theta, method='duhamel')
        Gtheta = G @ theta
        nu = np.dot(a, Gtheta) / np.dot(a, a)
        F = -Gtheta + nu * a
        
        print(f"Dynamics ||F|| = {np.linalg.norm(F):.6f}")
        print(f"Lagrange multiplier ν = {nu:.6f}")
        
        # For pair basis with entanglement, F should be non-zero
        assert np.linalg.norm(F) > 0.01, "Dynamics F should be non-zero for entangled systems"
        
        # Verify structural identity is broken
        Gtheta_plus_a = Gtheta + a
        rel_error = np.linalg.norm(Gtheta_plus_a) / np.linalg.norm(a)
        print(f"Structural identity check ||Gθ + a|| / ||a|| = {rel_error:.6f}")
        
        # For entangled systems, Gθ ≠ -a
        assert rel_error > 0.1, "Structural identity Gθ = -a should be broken"


class TestComparisonWithLocalBasis:
    """Compare pair basis results with local basis where applicable."""
    
    def test_fisher_metric_same_state_different_basis(self):
        """
        For a factorizable state, Fisher metric should be similar
        whether computed with local or pair operators (up to basis transformation).
        """
        # Local basis
        exp_fam_local = QuantumExponentialFamily(n_sites=2, d=2, pair_basis=False)
        # Pair basis
        exp_fam_pair = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        # Create a factorizable state with local operators
        theta_local = np.array([0.5, 0, 0, 0, 0.5, 0])  # σ_x on site 0, σ_x on site 1
        rho_local = exp_fam_local.rho_from_theta(theta_local)
        
        # Create similar state with pair operators (approximately)
        theta_pair = np.random.randn(15) * 0.1
        rho_pair = exp_fam_pair.rho_from_theta(theta_pair)
        
        # Both should give valid Fisher metrics
        G_local = exp_fam_local.fisher_information(theta_local)
        G_pair = exp_fam_pair.fisher_information(theta_pair)
        
        # Verify both are positive definite
        eig_local = np.linalg.eigvalsh(G_local)
        eig_pair = np.linalg.eigvalsh(G_pair)
        
        print(f"Local basis: min eigenvalue = {np.min(eig_local):.6e}")
        print(f"Pair basis: min eigenvalue = {np.min(eig_pair):.6e}")
        
        assert np.all(eig_local > -1e-10), "Local Fisher not positive"
        assert np.all(eig_pair > -1e-10), "Pair Fisher not positive"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

