"""
Test suite for quantum inaccessible game dynamics.

Tests verify:
1. Constrained dynamics (constraint preservation, entropy increase)
2. GENERIC decomposition
3. Numerical stability and edge cases
4. Integration tests

Run with: pytest test_dynamics.py -v

For quick tests (skip slow ones):
    pytest test_dynamics.py -v -m "not slow"

CIP-0004: Uses tolerance framework with scientifically justified bounds.
"""

import numpy as np
import pytest

from qig.core import (
    partial_trace,
    von_neumann_entropy,
    create_lme_state,
    marginal_entropies,
    generic_decomposition,
)
from qig.exponential_family import QuantumExponentialFamily
from qig.dynamics import InaccessibleGameDynamics


# ============================================================================
# Test: Constrained Dynamics
# ============================================================================

class TestConstrainedDynamics:
    """Test constrained maximum entropy production dynamics."""
    
    def test_initialisation(self):
        """Test dynamics initialisation."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        dynamics = InaccessibleGameDynamics(exp_family)
        assert dynamics.time_mode == 'affine'
    
    def test_time_mode_setting(self):
        """Test time mode switching."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        dynamics = InaccessibleGameDynamics(exp_family)
        
        dynamics.set_time_mode('entropy')
        assert dynamics.time_mode == 'entropy'
        
        dynamics.set_time_mode('real')
        assert dynamics.time_mode == 'real'
    
    def test_flow_tangent_to_constraint(self):
        """Flow should be tangent to constraint manifold: a^T · θ̇ = 0."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        dynamics = InaccessibleGameDynamics(exp_family)
        
        theta = np.random.randn(exp_family.n_params) * 0.01  # Smaller for speed
        theta_dot = dynamics.flow(0.0, theta)
        
        _, a = exp_family.marginal_entropy_constraint(theta)
        
        tangency = np.dot(a, theta_dot)
        assert np.abs(tangency) < 1e-5, "Flow should be tangent to constraint manifold"
    
    def test_integration_preserves_constraint(self):
        """Integration should preserve marginal entropy constraint."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        dynamics = InaccessibleGameDynamics(exp_family)
        
        theta_0 = np.random.randn(exp_family.n_params) * 0.01
        # Ultra-short integration for speed
        solution = dynamics.integrate(theta_0, (0, 0.05), n_points=3)
        
        assert solution['success'], "Integration should succeed"
        
        # Check constraint preservation
        constraint_violations = np.abs(solution['constraint'] - solution['constraint'][0])
        max_violation = np.max(constraint_violations)
        
        assert max_violation < 5e-2, f"Constraint violation too large: {max_violation}"
    
    def test_entropy_monotonic_increase(self):
        """Joint entropy should increase monotonically."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        dynamics = InaccessibleGameDynamics(exp_family)
        
        theta_0 = np.random.randn(exp_family.n_params) * 0.01
        # Ultra-short integration
        solution = dynamics.integrate(theta_0, (0, 0.05), n_points=3)
        
        H = solution['H']
        dH = np.diff(H)
        
        # Allow small numerical violations
        negative_count = np.sum(dH < -1e-4)
        assert negative_count == 0, f"Entropy decreased at {negative_count} points"
    
    @pytest.mark.slow
    def test_constrained_maxent_dynamics(self):
        """Test constrained maximum entropy dynamics using stable gradient descent."""
        # Use entangled pairs where the constrained optimization actually has work to do
        exp_family = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)  # 1 entangled pair
        dynamics = InaccessibleGameDynamics(exp_family)

        np.random.seed(24)
        theta_0 = np.random.randn(exp_family.n_params)

        # Run with sufficient iterations to check convergence behavior
        solution = dynamics.solve_constrained_maxent(theta_0, n_steps=100000, dt=1e-5,
                                                    convergence_tol=1e-6, project=True, project_every=200,
                                                    use_entropy_time=True)

        # Debug: Check final flow norms
        print(f"Final flow norms (last 10): {solution['flow_norms'][-10:]}")
        print(f"Min flow norm: {np.min(solution['flow_norms'])}")
        print(f"Max flow norm: {np.max(solution['flow_norms'])}")

        # Debug: Check constraint values
        print(f"Constraint values (last 10): {solution['constraint_values'][-10:]}")
        print(f"Initial C: {solution['C_init']:.6f}")
        print(f"Final constraint violation: {np.abs(solution['constraint_values'][-1] - solution['C_init']):.2e}")

        # Check that joint entropy increased (second law of thermodynamics)
        # Use the fast exponential family method
        H_init = exp_family.von_neumann_entropy(theta_0)
        H_final = exp_family.von_neumann_entropy(solution['trajectory'][-1])

        entropy_increase = H_final - H_init
        print(f"Joint entropy increase: {entropy_increase:.6f}")

        # Should increase (second law)
        assert entropy_increase > 0.01, f"Entropy should increase: ΔH = {entropy_increase}"

        # Check constraint preservation 
        # For long integration with periodic projection, max violation occurs just before projection
        # With project_every=200 and dt=1e-5, expect ~200 steps of drift before correction
        constraint_violation = np.max(np.abs(solution['constraint_values'] - solution['C_init']))
        print(f"Max constraint violation: {constraint_violation:.2e}")
        
        # Use tolerance framework: constraint_preservation is Category E (atol=1e-7 single-step)
        # But for 200 steps between projections, expect accumulated drift
        # Check against tolerance framework's dynamics category
        from tests.tolerance_framework import quantum_assert_scalar_close
        quantum_assert_scalar_close(
            constraint_violation, 0.0, 'constraint_preservation',
            err_msg=f"Constraint not preserved: violation = {constraint_violation}"
        )

        # Check entropy monotonicity - should increase throughout
        # Sample entropy at various points during trajectory
        n_check_points = min(10, len(solution['trajectory']))
        check_indices = np.linspace(0, len(solution['trajectory'])-1, n_check_points, dtype=int)

        entropies = []
        for idx in check_indices:
            theta_check = solution['trajectory'][idx]
            H_check = exp_family.von_neumann_entropy(theta_check)
            entropies.append(H_check)

        # Check monotonic increase
        for i in range(1, len(entropies)):
            assert entropies[i] >= entropies[i-1] - 1e-10, \
                f"Entropy decreased: H[{check_indices[i-1]}]={entropies[i-1]:.6f}, H[{check_indices[i]}]={entropies[i]:.6f}"

        # Should eventually converge to tight tolerance
        assert solution['converged'], f"Dynamics should converge to 1e-6 tolerance. Min flow norm: {np.min(solution['flow_norms']):.6e}"


# ============================================================================
# Test: GENERIC Decomposition
# ============================================================================

class TestGENERICDecomposition:
    """Test GENERIC decomposition analysis."""
    
    def test_generic_decomposition_symmetric_antisymmetric(self):
        """Test that S is symmetric and A is antisymmetric."""
        M = np.random.randn(5, 5)
        S, A = generic_decomposition(M)
        
        assert np.allclose(S, S.T, atol=1e-10), "S should be symmetric"
        assert np.allclose(A, -A.T, atol=1e-10), "A should be antisymmetric"
    
    def test_generic_decomposition_reconstruction(self):
        """Test that M = S + A."""
        M = np.random.randn(5, 5)
        S, A = generic_decomposition(M)
        
        M_reconstructed = S + A
        assert np.allclose(M, M_reconstructed, atol=1e-10), "M should equal S + A"
    
    def test_jacobian_shape(self):
        """Jacobian should have correct shape."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        dynamics = InaccessibleGameDynamics(exp_family)
        theta = np.random.randn(exp_family.n_params) * 0.1
        
        M = dynamics.exp_family.jacobian(theta)
        
        expected_shape = (exp_family.n_params, exp_family.n_params)
        assert M.shape == expected_shape, f"Jacobian shape should be {expected_shape}"


# ============================================================================
# Test: Numerical Stability
# ============================================================================

class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_zero_parameters(self):
        """System should handle zero natural parameters (maximally mixed)."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        theta_0 = np.zeros(exp_family.n_params)
        rho = exp_family.rho_from_theta(theta_0)
        
        # Should be maximally mixed: I/D
        expected = np.eye(exp_family.D) / exp_family.D
        assert np.allclose(rho, expected, atol=1e-8), "Zero parameters should give maximally mixed"
    
    def test_small_parameters(self):
        """System should handle very small parameters."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        theta_0 = np.random.randn(exp_family.n_params) * 1e-8
        
        dynamics = InaccessibleGameDynamics(exp_family)
        solution = dynamics.integrate(theta_0, (0, 0.1), n_points=10)
        
        assert solution['success'], "Integration with small parameters should succeed"
    
    def test_different_seeds_reproducible(self):
        """Same seed should give reproducible results."""
        np.random.seed(42)
        exp_family1 = QuantumExponentialFamily(n_sites=2, d=2)
        theta1 = np.random.randn(exp_family1.n_params) * 0.1
        
        np.random.seed(42)
        exp_family2 = QuantumExponentialFamily(n_sites=2, d=2)
        theta2 = np.random.randn(exp_family2.n_params) * 0.1
        
        assert np.allclose(theta1, theta2), "Same seed should give same parameters"


# ============================================================================
# Test: Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests using qig library."""
    
    @pytest.mark.slow
    def test_full_validation_two_qubits(self):
        """Full validation pipeline for 2 qubits (1 entangled pair)."""
        # Create system with pair basis (1 pair = 2 sites)
        exp_family = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        dynamics = InaccessibleGameDynamics(exp_family)
        
        # Random initial state
        theta_0 = np.random.randn(exp_family.n_params) * 0.1
        
        # Very short integration for testing speed
        solution = dynamics.integrate(theta_0, (0, 0.1), n_points=3)
        
        # Verify integration succeeded
        assert solution['success'], "Integration should succeed"
        
        # Verify constraint preservation
        constraint_violations = np.abs(solution['constraint'] - solution['constraint'][0])
        max_violation = np.max(constraint_violations)
        assert max_violation < 5e-2, f"Constraint violation too large: {max_violation}"
        
        # Verify entropy increase
        delta_H = solution['H'][-1] - solution['H'][0]
        assert delta_H >= -1e-6, f"Entropy should not decrease: ΔH = {delta_H}"
    
    @pytest.mark.slow
    def test_full_validation_two_qutrits(self):
        """Full validation pipeline for 2 qutrits (1 entangled pair)."""
        # Create system with pair basis (1 pair = 2 sites)
        exp_family = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        dynamics = InaccessibleGameDynamics(exp_family)
        
        # Random initial state
        theta_0 = np.random.randn(exp_family.n_params) * 0.1
        
        # Very short integration for testing speed
        solution = dynamics.integrate(theta_0, (0, 0.1), n_points=3)
        
        # Verify integration succeeded
        assert solution['success'], "Integration should succeed"
        
        # Verify constraint preservation
        constraint_violations = np.abs(solution['constraint'] - solution['constraint'][0])
        max_violation = np.max(constraint_violations)
        assert max_violation < 5e-2, f"Constraint violation too large: {max_violation}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

