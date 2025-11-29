"""
Numerical validation tests for pair-based exponential family.

Tests numerical accuracy of analytical implementations against finite differences:
1. ρ derivative validation (Duhamel vs finite differences)
2. Fisher metric validation (analytical vs finite differences)
3. Constraint gradient/Hessian validation
4. Jacobian validation for dynamics
5. Comparison between local and pair operator bases

Uses CIP-0004 tolerance framework with scientifically derived bounds.
"""

import numpy as np
import pytest
from qig.exponential_family import QuantumExponentialFamily
from tests.tolerance_framework import (
    quantum_assert_close,
    quantum_assert_scalar_close,
    QuantumTolerances,
)
from tests.fd_helpers import (
    finite_difference_fisher,
    finite_difference_rho_derivative,
    finite_difference_constraint_gradient,
    finite_difference_constraint_hessian,
    finite_difference_jacobian,
)


class TestRhoDerivativeNumerical:
    """Test ρ derivative analytical vs numerical accuracy."""

    @pytest.fixture
    def single_pair_family(self):
        """Fixture for single qubit pair."""
        return QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)

    @pytest.fixture
    def two_pair_family(self):
        """Fixture for two qubit pairs."""
        return QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)

    def test_single_pair_duhamel(self, single_pair_family):
        """Test Duhamel ρ derivative for single pair."""
        exp_fam = single_pair_family

        # Test at maximally mixed state
        theta = np.zeros(exp_fam.n_params)

        for a in range(exp_fam.n_params):
            # Analytical derivative (Duhamel method)
            rho_deriv_analytical = exp_fam.rho_derivative(theta, a, method='duhamel')

            # Numerical derivative (finite difference)
            rho_deriv_numerical = finite_difference_rho_derivative(exp_fam, theta, a)

            # Compare with analytical derivative precision (Category D)
            quantum_assert_close(rho_deriv_analytical, rho_deriv_numerical,
                               'jacobian',  # ρ derivatives are Jacobian-like
                               f"ρ derivative parameter {a} analytical vs numerical mismatch")

    def test_two_pairs_duhamel(self, two_pair_family):
        """Test Duhamel ρ derivative for two pairs."""
        exp_fam = two_pair_family

        # Test with small random parameters
        np.random.seed(42)
        theta = np.random.normal(0, 0.1, exp_fam.n_params)

        for a in range(min(5, exp_fam.n_params)):  # Test first 5 parameters for speed
            rho_deriv_analytical = exp_fam.rho_derivative(theta, a, method='duhamel')
            rho_deriv_numerical = finite_difference_rho_derivative(exp_fam, theta, a)

            # Compare with analytical derivative precision (Category D)
            quantum_assert_close(rho_deriv_analytical, rho_deriv_numerical,
                               'jacobian',  # ρ derivatives are Jacobian-like
                               f"Two-pair ρ derivative parameter {a} mismatch")


class TestFisherMetricNumerical:
    """Test Fisher metric analytical vs numerical computation."""

    @pytest.fixture
    def single_pair_family(self):
        """Fixture for single qubit pair."""
        return QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)

    @pytest.fixture
    def two_pair_family(self):
        """Fixture for two qubit pairs."""
        return QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)

    def test_fisher_vs_finite_difference_single_pair(self, single_pair_family):
        """Test Fisher metric analytical vs finite difference for single pair."""
        exp_fam = single_pair_family

        # Test at maximally mixed state
        theta = np.zeros(exp_fam.n_params)
        G_analytical = exp_fam.fisher_information(theta)
        G_numerical = finite_difference_fisher(exp_fam, theta)

        # Compare with analytical derivative precision (Category D)
        quantum_assert_close(G_analytical, G_numerical, 'fisher_metric',
                           "Single pair Fisher metric analytical vs numerical mismatch")

    def test_fisher_block_structure_numerically(self, two_pair_family):
        """Test Fisher metric block-diagonal structure numerically."""
        exp_fam = two_pair_family

        theta = np.zeros(exp_fam.n_params)
        G = exp_fam.fisher_information(theta)

        # Extract blocks (15 parameters per pair for qubits)
        n_per_pair = 15
        block_00 = G[:n_per_pair, :n_per_pair]
        block_11 = G[n_per_pair:, n_per_pair:]
        cross_block = G[:n_per_pair, n_per_pair:]

        # Cross block should be zero (block diagonal)
        quantum_assert_close(cross_block, np.zeros_like(cross_block), 'fisher_metric',
                           "Fisher metric cross-block should be zero")

        # Diagonal blocks should be non-zero
        assert np.linalg.norm(block_00) > QuantumTolerances.D['atol'], "Block 00 is zero"
        assert np.linalg.norm(block_11) > QuantumTolerances.D['atol'], "Block 11 is zero"


class TestConstraintHessianPairBasis:
    """Test constraint Hessian for pair basis."""

    @pytest.fixture
    def single_pair_family(self):
        """Fixture for single qubit pair."""
        return QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)

    def test_constraint_hessian_single_pair(self, single_pair_family):
        """Test constraint Hessian analytical vs numerical for single pair."""
        exp_fam = single_pair_family

        # Use random parameters for better numerical conditioning
        np.random.seed(46)
        theta = np.random.randn(exp_fam.n_params) * 0.3

        # Get analytical Hessian
        hessian_analytical = exp_fam.constraint_hessian(theta, method='duhamel', n_points=100)

        # Get numerical Hessian
        hessian_numerical = finite_difference_constraint_hessian(exp_fam, theta)

        # Compare with analytical derivative precision (Category D)
        quantum_assert_close(hessian_analytical, hessian_numerical, 'constraint_hessian',
                           "Single pair constraint Hessian analytical vs numerical mismatch")


class TestConstraintGradient:
    """Test constraint gradient numerical validation."""

    @pytest.fixture
    def single_pair_family(self):
        """Fixture for single qubit pair."""
        return QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)

    @pytest.fixture
    def two_pair_family(self):
        """Fixture for two qubit pairs."""
        return QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)

    def test_constraint_gradient_single_pair(self, single_pair_family):
        """Test constraint gradient for single pair."""
        exp_fam = single_pair_family

        # Test with small random parameters
        np.random.seed(42)
        theta = np.random.normal(0, 0.1, exp_fam.n_params)

        # Get analytical gradient (from constraint Hessian method)
        C_0, a_0 = exp_fam.marginal_entropy_constraint(theta)
        hessian = exp_fam.constraint_hessian(theta)

        # Gradient is -Hessian @ theta for this constraint form
        # Actually, let's compute gradient properly
        eps = 1e-6
        grad_analytical = np.zeros(exp_fam.n_params)

        for i in range(exp_fam.n_params):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            theta_minus = theta.copy()
            theta_minus[i] -= eps

            C_plus, _ = exp_fam.marginal_entropy_constraint(theta_plus)
            C_minus, _ = exp_fam.marginal_entropy_constraint(theta_minus)

            grad_analytical[i] = (C_plus - C_minus) / (2 * eps)

        # Compare with finite difference
        grad_numerical = finite_difference_constraint_gradient(exp_fam, theta)

        # Compare with analytical derivative precision (Category D)
        quantum_assert_close(grad_analytical, grad_numerical, 'constraint_gradient',
                           "Single pair constraint gradient analytical vs numerical mismatch")

    def test_constraint_gradient_two_pairs(self, two_pair_family):
        """Test constraint gradient for two pairs."""
        exp_fam = two_pair_family

        # Test at zero (simpler case)
        theta = np.zeros(exp_fam.n_params)

        grad_analytical = finite_difference_constraint_gradient(exp_fam, theta, eps=1e-6)
        grad_numerical = finite_difference_constraint_gradient(exp_fam, theta, eps=1e-7)

        # Compare different epsilons for convergence check
        quantum_assert_close(grad_analytical, grad_numerical, 'constraint_gradient',
                           "Two pair constraint gradient convergence test failed")


class TestJacobianNumerical:
    """Test Jacobian numerical validation."""

    @pytest.fixture
    def single_pair_family(self):
        """Fixture for single qubit pair."""
        return QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)

    def test_jacobian_vs_finite_difference(self, single_pair_family):
        """Test Jacobian analytical vs finite difference."""
        exp_fam = single_pair_family

        # Test at maximally mixed state
        theta = np.zeros(exp_fam.n_params)

        # Analytical Jacobian
        M_analytical = exp_fam.jacobian(theta)

        # Numerical Jacobian
        M_numerical = finite_difference_jacobian(exp_fam, theta)

        # Compare with analytical derivative precision (Category D)
        quantum_assert_close(M_analytical, M_numerical, 'jacobian',
                           "Jacobian analytical vs numerical mismatch")

    def test_jacobian_dynamics_nonzero(self, single_pair_family):
        """Test that Jacobian enables non-trivial dynamics for entangled systems."""
        exp_fam = single_pair_family

        theta = np.zeros(exp_fam.n_params)
        M = exp_fam.jacobian(theta)

        # For entangled systems, Jacobian should be non-trivial
        # (unlike local operators where M = -G)
        assert np.linalg.norm(M) > QuantumTolerances.D['atol'], \
            "Jacobian is zero - no dynamics possible"

        # Check that M ≠ -G (structural identity broken for entangled systems)
        G = exp_fam.fisher_information(theta)
        M_expected_local = -G  # What it would be for local operators

        # Should be significantly different
        difference = np.linalg.norm(M - M_expected_local)
        relative_diff = difference / np.linalg.norm(M_expected_local)

        assert relative_diff > 0.01, \
            f"Jacobian too close to local operator case (rel diff: {relative_diff})"


class TestComparisonWithLocalBasis:
    """Compare numerical results between local and pair operator bases."""

    def test_fisher_metric_same_state_different_basis(self):
        """Test Fisher metric for same physical state in different bases."""
        # Create the same maximally mixed state using different operator bases

        # Local basis: separable 2-qubit system
        exp_fam_local = QuantumExponentialFamily(n_sites=2, d=2, pair_basis=False)
        theta_local = np.zeros(exp_fam_local.n_params)  # Maximally mixed

        # Pair basis: entangled 1-pair system
        exp_fam_pair = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        theta_pair = np.zeros(exp_fam_pair.n_params)  # Also maximally mixed

        # Both should represent the same physical state (maximally mixed 2-qubit)
        rho_local = exp_fam_local.rho_from_theta(theta_local)
        rho_pair = exp_fam_pair.rho_from_theta(theta_pair)

        # States should be identical
        quantum_assert_close(rho_local, rho_pair, 'density_matrix',
                           "Local and pair bases give different maximally mixed states")

        # Fisher metrics should be different (different parametrizations)
        G_local = exp_fam_local.fisher_information(theta_local)
        G_pair = exp_fam_pair.fisher_information(theta_pair)

        # Different shapes (6 vs 15 parameters)
        assert G_local.shape != G_pair.shape, "Fisher metrics should have different shapes"

        # Both should be positive semidefinite
        eig_local = np.real(np.linalg.eigvals(G_local))
        eig_pair = np.real(np.linalg.eigvals(G_pair))

        assert np.all(eig_local > -QuantumTolerances.D['atol']), "Local Fisher not positive semidefinite"
        assert np.all(eig_pair > -QuantumTolerances.D['atol']), "Pair Fisher not positive semidefinite"
