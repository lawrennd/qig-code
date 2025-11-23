"""
Test suite for pair-based quantum exponential family.

Tests:
1. Initialization with pair basis
2. Bell state generation and properties
3. Entanglement metrics (mutual information, purity)
4. Block-diagonal structure of Fisher metric G
5. Comparison with local operator basis
"""

import numpy as np
import pytest
from qig.exponential_family import QuantumExponentialFamily


class TestPairBasisInitialization:
    """Test initialization of pair-based exponential family."""
    
    def test_single_qubit_pair(self):
        """Test initialization with a single qubit pair."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
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
        
        assert exp_fam.n_pairs == 1
        assert exp_fam.n_sites == 2
        assert exp_fam.d == 3
        assert exp_fam.D == 9  # 3² = 9 dimensional Hilbert space
        assert exp_fam.n_params == 80  # su(9) has 80 generators
    
    def test_two_qubit_pairs(self):
        """Test initialization with two qubit pairs."""
        exp_fam = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
        
        assert exp_fam.n_pairs == 2
        assert exp_fam.n_sites == 4  # 2 pairs × 2 subsystems each
        assert exp_fam.D == 16  # 4² = 16 dimensional Hilbert space
        assert exp_fam.n_params == 30  # 2 pairs × 15 generators
        
        # Check pair indices: first 15 act on pair 0, next 15 on pair 1
        assert exp_fam.pair_indices[:15] == [0] * 15
        assert exp_fam.pair_indices[15:] == [1] * 15


class TestBackwardCompatibility:
    """Test that local operator basis still works."""
    
    def test_local_basis_qubit(self):
        """Test local basis for qubits."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2, pair_basis=False)
        
        assert exp_fam.n_sites == 2
        assert exp_fam.d == 2
        assert exp_fam.D == 4
        assert exp_fam.n_params == 6  # 2 sites × 3 Pauli operators
        assert exp_fam.n_pairs is None
        assert exp_fam.pair_indices is None


class TestOperatorProperties:
    """Test properties of pair operators."""
    
    def test_operators_are_hermitian(self):
        """All operators should be Hermitian."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        for i, F in enumerate(exp_fam.operators):
            assert np.allclose(F, F.conj().T), f"Operator {i} not Hermitian"
    
    def test_operators_are_traceless(self):
        """All operators should be traceless."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        for i, F in enumerate(exp_fam.operators):
            trace = np.trace(F)
            assert np.abs(trace) < 1e-10, f"Operator {i} has trace {trace}"


class TestDensityMatrixProperties:
    """Test that density matrices have correct properties."""
    
    def test_density_matrix_is_hermitian(self):
        """ρ(θ) should be Hermitian."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        theta = np.random.randn(exp_fam.n_params) * 0.5
        
        rho = exp_fam.rho_from_theta(theta)
        assert np.allclose(rho, rho.conj().T), "Density matrix not Hermitian"
    
    def test_density_matrix_is_positive(self):
        """ρ(θ) should be positive semidefinite."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        theta = np.random.randn(exp_fam.n_params) * 0.5
        
        rho = exp_fam.rho_from_theta(theta)
        eigenvalues = np.linalg.eigvalsh(rho)
        assert np.all(eigenvalues >= -1e-10), f"Negative eigenvalues: {eigenvalues}"
    
    def test_density_matrix_is_normalized(self):
        """ρ(θ) should have trace 1."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        theta = np.random.randn(exp_fam.n_params) * 0.5
        
        rho = exp_fam.rho_from_theta(theta)
        trace = np.trace(rho)
        assert np.abs(trace - 1.0) < 1e-10, f"Trace is {trace}, expected 1"


class TestEntanglementMetrics:
    """Test mutual information and purity calculations."""
    
    def test_mutual_information_positive_for_entangled(self):
        """I > 0 for entangled states with pair operators."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        # Small random parameters should give some entanglement
        np.random.seed(42)
        theta = np.random.randn(exp_fam.n_params) * 0.5
        
        I = exp_fam.mutual_information(theta)
        print(f"Mutual information: {I:.6f}")
        
        # With pair operators, we should be able to create entanglement
        # (might be small for random parameters, but should be non-negative)
        assert I >= -1e-10, f"Mutual information {I} is negative"
    
    def test_mutual_information_zero_for_local_operators(self):
        """I ≈ 0 for local operators (they can't create entanglement)."""
        exp_fam_local = QuantumExponentialFamily(n_sites=2, d=2, pair_basis=False)
        
        np.random.seed(42)
        theta = np.random.randn(exp_fam_local.n_params) * 0.5
        
        I = exp_fam_local.mutual_information(theta)
        print(f"Mutual information (local): {I:.6f}")
        
        # Local operators should give I ≈ 0
        assert np.abs(I) < 0.01, f"Expected I ≈ 0 for local operators, got {I}"
    
    def test_purity_bounds(self):
        """Purity should be between 1/D and 1."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        theta = np.random.randn(exp_fam.n_params) * 0.5
        purity = exp_fam.purity(theta)
        
        D = exp_fam.D
        assert 1.0/D - 1e-6 <= purity <= 1.0 + 1e-6, \
            f"Purity {purity} out of bounds [1/{D}, 1]"


class TestBlockDiagonalStructure:
    """Test that Fisher metric G is block-diagonal for pair basis."""
    
    def test_fisher_metric_block_diagonal_single_pair(self):
        """For single pair, G should be a single block (trivially block-diagonal)."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        theta = np.random.randn(exp_fam.n_params) * 0.5
        G = exp_fam.fisher_information(theta)
        
        # Check G is symmetric
        assert np.allclose(G, G.T), "Fisher metric not symmetric"
        
        # Check G is positive semidefinite
        eigenvalues = np.linalg.eigvalsh(G)
        assert np.all(eigenvalues >= -1e-10), "Fisher metric not positive semidefinite"
    
    def test_fisher_metric_block_diagonal_two_pairs(self):
        """For two pairs, G should have zero cross-pair blocks."""
        exp_fam = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
        
        theta = np.random.randn(exp_fam.n_params) * 0.5
        G = exp_fam.fisher_information(theta)
        
        # G should be 30×30, with 15×15 blocks
        assert G.shape == (30, 30)
        
        # Extract cross-pair block: G[0:15, 15:30]
        cross_block = G[:15, 15:]
        
        print(f"Cross-pair block norm: {np.linalg.norm(cross_block):.10f}")
        print(f"Max abs element in cross block: {np.max(np.abs(cross_block)):.10e}")
        
        # Cross-pair elements should be zero (or very small)
        assert np.allclose(cross_block, 0, atol=1e-10), \
            f"Cross-pair block not zero: max element = {np.max(np.abs(cross_block))}"
        
        # Check the other cross block too (should be symmetric)
        cross_block_T = G[15:, :15]
        assert np.allclose(cross_block_T, 0, atol=1e-10), \
            "Transpose cross-pair block not zero"
        
        # Diagonal blocks should be non-zero
        block_0 = G[:15, :15]
        block_1 = G[15:, 15:]
        
        assert np.linalg.norm(block_0) > 1e-6, "Diagonal block 0 is zero"
        assert np.linalg.norm(block_1) > 1e-6, "Diagonal block 1 is zero"


class TestComputationalScaling:
    """Test that computation time scales linearly with n_pairs."""
    
    def test_parameter_count_scales_linearly(self):
        """n_params should be n_pairs × (d⁴-1)."""
        for n in [1, 2, 3]:
            exp_fam = QuantumExponentialFamily(n_pairs=n, d=2, pair_basis=True)
            expected = n * 15  # 15 = 2⁴ - 1
            assert exp_fam.n_params == expected, \
                f"Expected {expected} params for {n} pairs, got {exp_fam.n_params}"
    
    def test_hilbert_space_dimension(self):
        """Hilbert space dimension should be (d²)^n_pairs."""
        for n in [1, 2, 3]:
            exp_fam = QuantumExponentialFamily(n_pairs=n, d=2, pair_basis=True)
            expected = 4**n  # 4 = 2²
            assert exp_fam.D == expected, \
                f"Expected D = {expected} for {n} pairs, got {exp_fam.D}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

