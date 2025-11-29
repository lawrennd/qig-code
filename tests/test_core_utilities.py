"""
Test suite for quantum core utilities.

Tests verify:
1. Quantum state utilities (entropy, partial trace, LME states)
2. Operator basis construction (Pauli, Gell-Mann)
3. GENERIC decomposition

Validates: qig.core module functions

Run with: pytest test_core_utilities.py -v
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
from qig.exponential_family import (
    pauli_basis,
    gell_mann_matrices,
    create_operator_basis,
)
from qig.dynamics import InaccessibleGameDynamics
from qig.exponential_family import QuantumExponentialFamily
from tests.tolerance_framework import (
    quantum_assert_close,
    quantum_assert_scalar_close,
    quantum_assert_hermitian,
    quantum_assert_unit_trace,
)


# ============================================================================
# Test: Quantum State Utilities
# ============================================================================

class TestQuantumStateUtilities:
    """Test basic quantum state operations."""
    
    def test_von_neumann_entropy_pure_state(self):
        """Pure state should have zero entropy."""
        psi = np.array([1, 0], dtype=complex)
        rho = np.outer(psi, psi.conj())
        S = von_neumann_entropy(rho)
        quantum_assert_scalar_close(
            S,
            0.0,
            "entropy",
            "Pure state entropy should be ~0",
            atol=1e-10,
            rtol=0.0,
        )
    
    def test_von_neumann_entropy_maximally_mixed(self):
        """Maximally mixed state should have entropy log(d)."""
        d = 3
        rho = np.eye(d) / d
        S = von_neumann_entropy(rho)
        expected = np.log(d)
        quantum_assert_scalar_close(
            S,
            expected,
            "entropy",
            f"Maximally mixed entropy should be log({d})",
            atol=1e-10,
            rtol=0.0,
        )
    
    def test_von_neumann_entropy_bounds(self):
        """Entropy should satisfy 0 ≤ S ≤ log(d)."""
        d = 4
        # Random density matrix
        A = np.random.randn(d, d) + 1j * np.random.randn(d, d)
        rho = A @ A.conj().T
        rho = rho / np.trace(rho)
        
        S = von_neumann_entropy(rho)
        assert 0 <= S <= np.log(d) + 1e-8, "Entropy out of bounds"
    
    def test_partial_trace_two_qubits(self):
        """Test partial trace for Bell state."""
        # Bell state: |Φ+⟩ = (|00⟩ + |11⟩)/√2
        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho = np.outer(psi, psi.conj())
        
        # Trace out second qubit
        rho_1 = partial_trace(rho, dims=[2, 2], keep=0)
        
        # Should be maximally mixed: I/2
        expected = np.eye(2) / 2
        quantum_assert_close(
            rho_1,
            expected,
            "density_matrix",
            "Bell state marginal should be I/2",
            atol=1e-10,
            rtol=0.0,
        )
    
    def test_partial_trace_preserves_trace(self):
        """Partial trace should preserve trace = 1."""
        d1, d2 = 2, 3
        D = d1 * d2
        A = np.random.randn(D, D) + 1j * np.random.randn(D, D)
        rho = A @ A.conj().T
        rho = rho / np.trace(rho)
        
        rho_1 = partial_trace(rho, dims=[d1, d2], keep=0)
        rho_2 = partial_trace(rho, dims=[d1, d2], keep=1)
        
        quantum_assert_unit_trace(
            rho_1,
            "density_matrix",
            "Partial trace should preserve unit trace (subsystem 1)",
        )
        quantum_assert_unit_trace(
            rho_2,
            "density_matrix",
            "Partial trace should preserve unit trace (subsystem 2)",
        )
    
    def test_create_lme_state_two_qubits(self):
        """LME state for 2 qubits should be Bell state."""
        rho, dims = create_lme_state(n_sites=2, d=2)
        
        # Should be pure
        quantum_assert_scalar_close(
            np.trace(rho @ rho),
            1.0,
            "purity",
            "LME state should be pure",
            atol=1e-10,
            rtol=0.0,
        )
        
        # Marginals should be maximally mixed
        h = marginal_entropies(rho, dims)
        expected = np.log(2)
        quantum_assert_close(
            h,
            np.array([expected, expected]),
            "marginal_entropy",
            "Marginals should be maximally mixed",
            atol=1e-10,
            rtol=0.0,
        )
    
    def test_create_lme_state_three_qutrits(self):
        """LME state for 3 qutrits should have correct marginal entropies."""
        rho, dims = create_lme_state(n_sites=3, d=3)
        
        # Should be pure
        purity = np.trace(rho @ rho).real
        quantum_assert_scalar_close(
            purity,
            1.0,
            "purity",
            "LME state should be pure",
            atol=1e-10,
            rtol=0.0,
        )
        
        # Check marginal entropy sum (one site will be pure for odd n)
        h = marginal_entropies(rho, dims)
        # Two sites paired: 2*log(3), one site pure: 0
        expected_sum = 2 * np.log(3)
        quantum_assert_scalar_close(
            h.sum(),
            expected_sum,
            "marginal_entropy",
            "Marginal entropy sum incorrect",
            atol=1e-8,
            rtol=0.0,
        )


# ============================================================================
# Test: Operator Bases
# ============================================================================

class TestOperatorBases:
    """Test Pauli and Gell-Mann operator construction."""
    
    def test_pauli_basis_hermitian(self):
        """Pauli operators should be Hermitian."""
        ops = pauli_basis(site=0, n_sites=2)
        for op in ops:
            quantum_assert_hermitian(
                op,
                "density_matrix",
                "Pauli operators should be Hermitian",
            )
    
    def test_pauli_basis_traceless(self):
        """Pauli operators should be traceless."""
        ops = pauli_basis(site=0, n_sites=2)
        for op in ops:
            quantum_assert_scalar_close(
                np.trace(op),
                0.0,
                "trace",
                "Pauli operators should be traceless",
                atol=1e-10,
                rtol=0.0,
            )
    
    def test_pauli_commutation_relations(self):
        """Check [σ_x, σ_y] = 2iσ_z at single site."""
        ops = pauli_basis(site=0, n_sites=1)
        X, Y, Z = ops
        
        # [X, Y] = 2iZ
        commutator = X @ Y - Y @ X
        expected = 2j * Z
        quantum_assert_close(
            commutator,
            expected,
            "commutator",
            "Pauli commutation relation failed",
            atol=1e-10,
            rtol=0.0,
        )
    
    def test_gell_mann_hermitian(self):
        """Gell-Mann matrices should be Hermitian."""
        gm = gell_mann_matrices()
        for G in gm:
            quantum_assert_hermitian(
                G,
                "density_matrix",
                "Gell-Mann matrices should be Hermitian",
            )
    
    def test_gell_mann_traceless(self):
        """Gell-Mann matrices should be traceless."""
        gm = gell_mann_matrices()
        for G in gm:
            quantum_assert_scalar_close(
                np.trace(G),
                0.0,
                "trace",
                "Gell-Mann matrices should be traceless",
                atol=1e-10,
                rtol=0.0,
            )
    
    def test_operator_basis_count(self):
        """Operator basis should have correct number of elements."""
        # Qubits: 3 operators per site
        ops_qubits, _ = create_operator_basis(n_sites=2, d=2)
        assert len(ops_qubits) == 6, "2 qubits should have 6 operators"
        
        # Qutrits: 8 operators per site
        ops_qutrits, _ = create_operator_basis(n_sites=2, d=3)
        assert len(ops_qutrits) == 16, "2 qutrits should have 16 operators"


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
# Parametrised Tests
# ============================================================================

@pytest.mark.parametrize("n_sites,d", [(2, 2), (2, 3), (3, 2)])
def test_various_systems(n_sites, d):
    """Test framework works for various system sizes."""
    exp_family = QuantumExponentialFamily(n_sites, d)
    assert exp_family.D == d ** n_sites
    assert exp_family.n_params == n_sites * (d**2 - 1)


@pytest.mark.parametrize("d", [2, 3, 4])
def test_entropy_bounds_various_dimensions(d):
    """Test entropy bounds for various dimensions."""
    rho = np.eye(d) / d  # Maximally mixed
    S = von_neumann_entropy(rho)
    expected = np.log(d)
    assert np.abs(S - expected) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

