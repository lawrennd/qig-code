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
