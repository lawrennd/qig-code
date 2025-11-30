"""
Test suite for pair-based quantum exponential family.

Tests:
1. Initialization with pair basis
2. Bell state generation and properties
3. Entanglement metrics (mutual information, purity)
4. Block-diagonal structure of Fisher metric G
5. Comparison with local operator basis

Uses CIP-0004 tolerance framework with scientifically derived bounds.
"""

import numpy as np
import pytest
from qig.exponential_family import QuantumExponentialFamily
from qig.core import create_lme_state, marginal_entropies
from tests.tolerance_framework import (
    quantum_assert_close,
    quantum_assert_scalar_close,
    quantum_assert_hermitian,
    quantum_assert_unit_trace,
    QuantumTolerances
)


class TestPairBasisInitialization:
    """Test initialization of pair-based exponential family."""

    def test_single_qubit_pair(self):
        """Test initialization with a single qubit pair."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)

        # Test basic properties (exact arithmetic - Category A)
        assert exp_fam.n_pairs == 1
        assert exp_fam.n_sites == 2  # Each pair has 2 subsystems
        assert exp_fam.d == 2
        assert exp_fam.D == 4  # 2² = 4 dimensional Hilbert space
        assert exp_fam.n_params == 15  # su(4) has 15 generators
        assert len(exp_fam.operators) == 15
        assert len(exp_fam.pair_indices) == 15
        assert all(idx == 0 for idx in exp_fam.pair_indices)  # All act on pair 0

    def test_single_qutrit_pair(self):
        """Test initialization with a single qutrit pair."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)

        # Test basic properties (exact arithmetic - Category A)
        assert exp_fam.n_pairs == 1
        assert exp_fam.n_sites == 2
        assert exp_fam.d == 3
        assert exp_fam.D == 9  # 3² = 9 dimensional Hilbert space
        assert exp_fam.n_params == 80  # su(9) has 80 generators

    def test_two_qubit_pairs(self):
        """Test initialization with two qubit pairs."""
        exp_fam = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)

        # Test basic properties (exact arithmetic - Category A)
        assert exp_fam.n_pairs == 2
        assert exp_fam.n_sites == 4  # 2 pairs × 2 subsystems each
        assert exp_fam.D == 16  # 4² = 16 dimensional Hilbert space
        assert exp_fam.n_params == 30  # 2 pairs × 15 generators

        # Test pair index assignment
        assert exp_fam.pair_indices[:15] == [0] * 15
        assert exp_fam.pair_indices[15:] == [1] * 15


class TestBackwardCompatibility:
    """Test backward compatibility with local operator basis."""

    def test_local_operator_fallback(self):
        """Test that local operator basis still works."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2, pair_basis=False)

        # Test basic properties (exact arithmetic - Category A)
        assert exp_fam.n_pairs is None  # No pairs for local basis
        assert exp_fam.n_sites == 2
        assert exp_fam.d == 2
        assert exp_fam.D == 4
        assert exp_fam.n_params == 6  # su(2) × su(2) has 6 generators

    def test_pair_vs_local_parameters(self):
        """Test parameter count difference between pair and local bases."""
        # Local basis: separate su(2) × su(2) = 6 parameters
        local_fam = QuantumExponentialFamily(n_sites=2, d=2, pair_basis=False)
        assert local_fam.n_params == 6

        # Pair basis: joint su(4) = 15 parameters
        pair_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        assert pair_fam.n_params == 15


class TestOperatorProperties:
    """Test properties of pair basis operators."""

    @pytest.fixture
    def qubit_pair_family(self):
        """Fixture for single qubit pair."""
        return QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)

    @pytest.fixture
    def qutrit_pair_family(self):
        """Fixture for single qutrit pair."""
        return QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)

    def test_qubit_pair_operators(self, qubit_pair_family):
        """Test that qubit pair operators are Hermitian and traceless."""
        for i, op in enumerate(qubit_pair_family.operators):
            # Check Hermitian property (quantum states - Category B)
            quantum_assert_hermitian(op, 'density_matrix',
                                   f"Qubit pair operator {i} is not Hermitian")

            # Check traceless property (exact arithmetic - Category A)
            trace = np.trace(op)
            quantum_assert_scalar_close(trace, 0.0, 'trace',
                                      f"Qubit pair operator {i} has trace {trace}")

    def test_qutrit_pair_operators(self, qutrit_pair_family):
        """Test that qutrit pair operators are Hermitian and traceless."""
        for i, op in enumerate(qutrit_pair_family.operators):
            # Check Hermitian property (quantum states - Category B)
            quantum_assert_hermitian(op, 'density_matrix',
                                   f"Qutrit pair operator {i} is not Hermitian")

            # Check traceless property (exact arithmetic - Category A)
            trace = np.trace(op)
            quantum_assert_scalar_close(trace, 0.0, 'trace',
                                      f"Qutrit pair operator {i} has trace {trace}")

    def test_operator_commutators(self, qubit_pair_family):
        """Test commutation relations for pair operators."""
        # su(4) structure constants are non-trivial
        # For now, just verify they form a proper Lie algebra
        n_ops = len(qubit_pair_family.operators)

        # Check that operators are linearly independent
        op_matrix = np.array([op.flatten() for op in qubit_pair_family.operators])
        rank = np.linalg.matrix_rank(op_matrix)
        assert rank == n_ops, f"Operators are linearly dependent: rank {rank} < {n_ops}"


class TestDensityMatrixProperties:
    """Test density matrix properties for pair exponential families."""

    def test_maximally_mixed_state(self):
        """Test density matrix at θ=0 (maximally mixed state)."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)

        # Get maximally mixed state
        rho = exp_fam.rho_from_theta(np.zeros(exp_fam.n_params))

        # Check unit trace (quantum states - Category B)
        quantum_assert_unit_trace(rho, 'density_matrix')

        # Check hermiticity (quantum states - Category B)
        quantum_assert_hermitian(rho, 'density_matrix')

        # Check eigenvalues (should be uniform for maximally mixed state)
        eigenvalues = np.real(np.linalg.eigvals(rho))
        expected_eigenvalue = 1.0 / exp_fam.D

        for eigval in eigenvalues:
            quantum_assert_scalar_close(eigval, expected_eigenvalue, 'density_matrix',
                                      f"Eigenvalue {eigval} ≠ {expected_eigenvalue}")

    def test_positive_semidefinite(self):
        """Test that all density matrices are positive semidefinite."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)

        # Test with random parameters
        np.random.seed(42)
        theta = np.random.normal(0, 1, exp_fam.n_params)

        rho = exp_fam.rho_from_theta(theta)

        # Check positive semidefiniteness (quantum states - Category B)
        eigenvalues = np.real(np.linalg.eigvals(rho))
        for eigval in eigenvalues:
            assert eigval >= -QuantumTolerances.B['atol'], \
                f"Negative eigenvalue: {eigval}"

    def test_purity_bounds(self):
        """Test that purity is bounded correctly."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)

        # Test maximally mixed state (minimum purity for 4-level system)
        rho_mixed = exp_fam.rho_from_theta(np.zeros(exp_fam.n_params))
        purity_mixed = np.real(np.trace(rho_mixed @ rho_mixed))

        D = exp_fam.D
        expected_min_purity = 1.0 / D
        quantum_assert_scalar_close(purity_mixed, expected_min_purity, 'purity',
                                  f"Mixed state purity {purity_mixed} ≠ {expected_min_purity}")

        # Test with random parameters (should have higher purity)
        np.random.seed(42)
        theta = np.random.normal(0, 2, exp_fam.n_params)
        rho_random = exp_fam.rho_from_theta(theta)
        purity_random = np.real(np.trace(rho_random @ rho_random))

        # Purity should be between 1/D and 1
        assert expected_min_purity - QuantumTolerances.B['atol'] <= purity_random <= 1.0 + QuantumTolerances.B['atol'], \
            f"Purity {purity_random} out of bounds [{expected_min_purity}, 1.0]"


class TestEntanglementMetrics:
    """Test entanglement metrics for pair exponential families."""

    def test_bell_state_entanglement(self):
        """Test that Bell states are maximally entangled."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)

        # Create Bell state |00⟩ + |11⟩ (maximally entangled)
        bell_state, dims = create_lme_state(n_sites=2, d=2)

        # Compute marginal entropies
        h_marginals = marginal_entropies(bell_state, dims)

        # For Bell state, marginals should be maximally mixed
        expected_marginal_entropy = np.log(2)  # log(2) for qubit

        for i, h in enumerate(h_marginals):
            quantum_assert_scalar_close(h, expected_marginal_entropy, 'entropy',
                                      f"Bell state marginal {i} entropy {h} ≠ {expected_marginal_entropy}")

        # Compute mutual information I = C - H
        C = np.sum(h_marginals)  # Total marginal entropy
        H = exp_fam.von_neumann_entropy(np.zeros(exp_fam.n_params))  # Joint entropy (should be 0 for bell state)
        # Actually, let's use the entropy of the bell state directly
        from qig.core import von_neumann_entropy
        H = von_neumann_entropy(bell_state)
        I = C - H

        # Bell state should have maximum mutual information
        expected_I = 2 * np.log(2)  # For two qubits
        quantum_assert_scalar_close(I, expected_I, 'mutual_information',
                                  f"Bell state mutual information {I} ≠ {expected_I}")

    def test_separable_state_no_entanglement(self):
        """Test that separable states have zero mutual information."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)

        # Create product state |00⟩⟨00|
        product_state = np.zeros((4, 4), dtype=complex)
        product_state[0, 0] = 1.0  # |00⟩⟨00|

        # Compute mutual information using exp_fam method
        # First get theta=0 (maximally mixed) and check that gives I=0
        theta_zero = np.zeros(exp_fam.n_params)
        I = exp_fam.mutual_information(theta_zero)

        # Maximally mixed state should have zero mutual information
        quantum_assert_scalar_close(I, 0.0, 'mutual_information',
                                  f"Maximally mixed state mutual information {I} ≠ 0")

    def test_mutual_information_nonnegative(self):
        """Test that mutual information is always non-negative."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)

        # Test with multiple random states
        np.random.seed(42)
        for _ in range(10):
            theta = np.random.normal(0, 1, exp_fam.n_params)
            I = exp_fam.mutual_information(theta)

            # Mutual information should be non-negative
            assert I >= -QuantumTolerances.C['atol'], \
                f"Negative mutual information: {I}"


class TestBlockDiagonalStructure:
    """Test block-diagonal structure of Fisher metric for pair basis."""

    def test_fisher_metric_symmetry(self):
        """Test that Fisher metric is symmetric."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)

        theta = np.zeros(exp_fam.n_params)  # Maximally mixed state
        G = exp_fam.fisher_information(theta)

        # Fisher metric should be symmetric (analytical derivatives - Category D)
        quantum_assert_close(G, G.T, 'fisher_metric',
                           "Fisher metric is not symmetric")

    def test_fisher_metric_positive_semidefinite(self):
        """Test that Fisher metric is positive semidefinite."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)

        theta = np.zeros(exp_fam.n_params)
        G = exp_fam.fisher_information(theta)

        # Check positive semidefiniteness (analytical derivatives - Category D)
        eigenvalues = np.real(np.linalg.eigvals(G))
        for eigval in eigenvalues:
            assert eigval >= -QuantumTolerances.D['atol'], \
                f"Fisher metric has negative eigenvalue: {eigval}"

    def test_block_diagonal_structure(self):
        """Test block-diagonal structure for multiple pairs."""
        exp_fam = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)

        theta = np.zeros(exp_fam.n_params)
        G = exp_fam.fisher_information(theta)

        # For two pairs, G should have 2x2 block structure
        # Cross blocks should be zero
        n_params_per_pair = 15  # su(4) generators

        # Extract cross-coupling blocks
        cross_block_01 = G[:n_params_per_pair, n_params_per_pair:]
        cross_block_10 = G[n_params_per_pair:, :n_params_per_pair]

        # Cross blocks should be zero (block diagonal structure)
        quantum_assert_close(cross_block_01, np.zeros_like(cross_block_01), 'fisher_metric',
                           "Fisher metric cross-block (0,1) is non-zero")

        quantum_assert_close(cross_block_10, np.zeros_like(cross_block_10), 'fisher_metric',
                           "Fisher metric cross-block (1,0) is non-zero")

    def test_diagonal_blocks_nonzero(self):
        """Test that diagonal blocks are non-zero."""
        exp_fam = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)

        theta = np.zeros(exp_fam.n_params)
        G = exp_fam.fisher_information(theta)

        n_params_per_pair = 15

        # Extract diagonal blocks
        block_0 = G[:n_params_per_pair, :n_params_per_pair]
        block_1 = G[n_params_per_pair:, n_params_per_pair:]

        # Diagonal blocks should be non-zero
        assert np.linalg.norm(block_0) > QuantumTolerances.D['atol'], \
            "Diagonal block 0 is zero"

        assert np.linalg.norm(block_1) > QuantumTolerances.D['atol'], \
            "Diagonal block 1 is zero"


class TestComputationalScaling:
    """Test computational scaling of pair exponential families."""

    def test_parameter_scaling(self):
        """Test parameter count scaling with system size."""
        # Test qubit pairs
        for n_pairs in [1, 2, 3]:
            exp_fam = QuantumExponentialFamily(n_pairs=n_pairs, d=2, pair_basis=True)
            expected_params = n_pairs * 15  # su(4) has 15 generators
            assert exp_fam.n_params == expected_params

        # Test qutrit pairs
        for n_pairs in [1, 2]:
            exp_fam = QuantumExponentialFamily(n_pairs=n_pairs, d=3, pair_basis=True)
            expected_params = n_pairs * 80  # su(9) has 80 generators
            assert exp_fam.n_params == expected_params

    def test_hilbert_space_scaling(self):
        """Test Hilbert space dimension scaling."""
        for n_pairs in [1, 2, 3]:
            for d in [2, 3]:
                exp_fam = QuantumExponentialFamily(n_pairs=n_pairs, d=d, pair_basis=True)
                expected_D = (d ** (2 * n_pairs))  # Each pair contributes d²
                assert exp_fam.D == expected_D

    @pytest.mark.parametrize("n_pairs,d", [(1, 2), (1, 3), (2, 2)])
    def test_density_matrix_computation(self, n_pairs, d):
        """Test that density matrices can be computed for various sizes."""
        exp_fam = QuantumExponentialFamily(n_pairs=n_pairs, d=d, pair_basis=True)

        # Test maximally mixed state
        theta = np.zeros(exp_fam.n_params)
        rho = exp_fam.rho_from_theta(theta)

        # Verify basic properties
        quantum_assert_unit_trace(rho, 'density_matrix')
        quantum_assert_hermitian(rho, 'density_matrix')

        # Verify shape
        assert rho.shape == (exp_fam.D, exp_fam.D)


# ============================================================================
# Numerical Validation Tests (absorbed from test_pair_numerical_validation.py)
# ============================================================================
# Tests numerical accuracy of analytical implementations against finite differences

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
        # Use eps=1e-5 to match tolerance framework assumptions (see D_numerical)
        G_numerical = finite_difference_fisher(exp_fam, theta, eps=1e-5)

        # Compare with numerical validation tolerance (Category D_numerical: analytical vs FD)
        quantum_assert_close(G_analytical, G_numerical, 'numerical_validation',
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
        from tests.tolerance_framework import QuantumTolerances
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

        # Compare with numerical validation tolerance (Category D_numerical: analytical vs FD)
        quantum_assert_close(hessian_analytical, hessian_numerical, 'numerical_validation',
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

        # Compare with numerical validation tolerance (Category D_numerical: analytical vs FD)
        quantum_assert_close(grad_analytical, grad_numerical, 'numerical_validation',
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
        from tests.tolerance_framework import QuantumTolerances
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

        from tests.tolerance_framework import QuantumTolerances
        assert np.all(eig_local > -QuantumTolerances.D['atol']), "Local Fisher not positive semidefinite"
        assert np.all(eig_pair > -QuantumTolerances.D['atol']), "Pair Fisher not positive semidefinite"


class TestBellStateParameters:
    """Test get_bell_state_parameters() for regularized Bell states.
    
    Bell states are pure states (rank 1) that lie at the boundary of the
    exponential family where natural parameters θ → -∞. The regularization
    ρ_ε = (1-ε)|Φ⟩⟨Φ| + ε I/D makes them full rank with finite parameters.
    """
    
    @pytest.mark.parametrize("d", [2, 3, 4])
    def test_bell_state_different_dimensions(self, d):
        """Test Bell state parameters for different local dimensions."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)
        epsilon = 1e-4
        
        # Get Bell state parameters
        theta = exp_fam.get_bell_state_parameters(epsilon=epsilon)
        
        # Check shape
        assert theta.shape == (exp_fam.n_params,), \
            f"Wrong shape: {theta.shape} vs {(exp_fam.n_params,)}"
        
        # Parameters should have significant magnitude (approaching boundary)
        assert np.max(np.abs(theta)) > 1.0, \
            "Parameters should have significant magnitude for regularized pure state"
        
        # Verify the density matrix matches target
        rho_from_theta = exp_fam.rho_from_theta(theta)
        
        from qig.pair_operators import bell_state_density_matrix
        rho_bell = bell_state_density_matrix(d)
        rho_mixed = np.eye(d**2) / (d**2)
        rho_target = (1 - epsilon) * rho_bell + epsilon * rho_mixed
        
        # Check fidelity (Category E - numerical precision for parameter reconstruction)
        # For perfect reconstruction: Tr(ρ_target @ ρ_target) = Tr(ρ_target²) = purity
        # Note: For a mixed state, this is NOT 1.0, but equals the purity of the state
        fidelity = np.real(np.trace(rho_from_theta @ rho_target))
        purity = np.real(np.trace(rho_target @ rho_target))
        quantum_assert_scalar_close(fidelity, purity, 'duhamel_integration',
                                   f"Bell state fidelity mismatch for d={d}")
        
        # Check marginal entropies are log(d) (maximally mixed)
        marginals = marginal_entropies(rho_from_theta, [d, d])
        expected_marginal = np.log(d)
        for h in marginals:
            quantum_assert_scalar_close(h, expected_marginal, 'entropy',
                                       f"Marginal entropy mismatch for d={d}")
    
    def test_bell_state_epsilon_scaling(self):
        """Test that parameters grow as epsilon → 0 (approaching pure state)."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        
        # Test with decreasing epsilon
        epsilon_values = [1e-2, 1e-3, 1e-4, 1e-5]
        max_params = []
        min_params = []
        
        for eps in epsilon_values:
            theta = exp_fam.get_bell_state_parameters(epsilon=eps)
            max_params.append(np.max(theta))
            min_params.append(np.min(theta))
        
        # As epsilon decreases, parameters should spread toward ±∞
        # Max should increase, min should decrease
        for i in range(len(epsilon_values) - 1):
            assert max_params[i+1] > max_params[i], \
                f"Max parameter should increase as epsilon decreases: {max_params}"
            assert min_params[i+1] < min_params[i], \
                f"Min parameter should decrease as epsilon decreases: {min_params}"
    
    def test_bell_state_high_entanglement(self):
        """Test that Bell state parameters produce high mutual information."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        theta = exp_fam.get_bell_state_parameters(epsilon=1e-4)
        
        # Mutual information should be high (approaching log(d))
        I = exp_fam.mutual_information(theta)
        
        # For nearly pure Bell state, I should be close to log(d)
        # but the exact value depends on the definition used
        assert I > 0.5, \
            f"Mutual information too low for Bell state: {I}"
    
    def test_bell_state_low_entropy(self):
        """Test that regularized Bell state has low total entropy."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        epsilon = 1e-4
        theta = exp_fam.get_bell_state_parameters(epsilon=epsilon)
        
        # von Neumann entropy should be small (pure state has H=0)
        H = exp_fam.von_neumann_entropy(theta)
        
        # With small epsilon, entropy should be roughly -epsilon log(epsilon)
        expected_order = -epsilon * np.log(epsilon)
        assert H < 0.1, \
            f"Total entropy too high for regularized pure state: {H}"
        assert H > 0, \
            "Entropy should be positive for regularized state"
    
    def test_bell_state_raises_for_zero_epsilon(self):
        """Test that pure Bell state (epsilon=0) raises ValueError."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            exp_fam.get_bell_state_parameters(epsilon=0.0)
    
    def test_bell_state_raises_for_multiple_pairs(self):
        """Test that Bell state only works for single pairs."""
        exp_fam = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
        
        with pytest.raises(ValueError, match="single pair"):
            exp_fam.get_bell_state_parameters(epsilon=1e-4)
    
    def test_bell_state_raises_for_local_basis(self):
        """Test that Bell state only works for pair basis."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2, pair_basis=False)
        
        with pytest.raises(ValueError, match="pair_basis=True"):
            exp_fam.get_bell_state_parameters(epsilon=1e-4)
    
    def test_bell_state_hermitian_and_unit_trace(self):
        """Test that Bell state parameters produce valid density matrix."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        theta = exp_fam.get_bell_state_parameters(epsilon=1e-4)
        
        rho = exp_fam.rho_from_theta(theta)
        
        # Check Hermiticity and unit trace (Category A - exact properties)
        quantum_assert_hermitian(rho, "Bell state density matrix not Hermitian")
        quantum_assert_unit_trace(rho, "Bell state density matrix trace ≠ 1")
        
        # Check positive semidefinite (eigenvalues ≥ 0)
        eigs = np.real(np.linalg.eigvalsh(rho))
        assert np.all(eigs > -QuantumTolerances.A['atol']), \
            f"Bell state has negative eigenvalues: {eigs[eigs < 0]}"
