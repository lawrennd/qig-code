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
    loewner_kernel,
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
# Test: Loewner Kernel
# ============================================================================

class TestLoewnerKernel:
    """
    Test the Loewner (Kubo-Mori / BKM) divided-difference kernel.

    Mathematical properties verified:
    - Output shapes and eigenvalue clipping
    - Eigenvectors are unitary
    - Kernel matrix is symmetric
    - Diagonal entries equal eigenvalues (L'Hopital limit)
    - Off-diagonal entries satisfy the divided-difference formula
    - LME limit: all entries equal 1/d for rho = I/d
    - Near-degenerate eigenvalues fall back to arithmetic mean
    - Kernel matrix is positive semidefinite
    - Loewner map J_rho0(X) applies correctly
    - Kernel is finite for near-pure states
    """

    def test_output_shapes_qubit(self):
        """Return shapes are (D,D), (D,), (D,D) for a qubit."""
        rho = np.eye(2, dtype=complex) / 2
        C, vals, vecs = loewner_kernel(rho)
        assert C.shape == (2, 2)
        assert vals.shape == (2,)
        assert vecs.shape == (2, 2)

    def test_output_shapes_qutrit(self):
        """Return shapes are correct for a qutrit."""
        rho = np.eye(3, dtype=complex) / 3
        C, vals, vecs = loewner_kernel(rho)
        assert C.shape == (3, 3)
        assert vals.shape == (3,)
        assert vecs.shape == (3, 3)

    def test_eigenvalues_clipped_positive(self):
        """Eigenvalues should be clipped to >= 1e-14."""
        psi = np.array([1, 0], dtype=complex)
        rho = np.outer(psi, psi.conj())
        C, vals, vecs = loewner_kernel(rho)
        assert np.all(vals >= 1e-14), "All eigenvalues must be >= 1e-14"

    def test_vecs_unitary(self):
        """Eigenvectors should form a unitary matrix."""
        rho = np.diag([0.3, 0.7]).astype(complex)
        C, vals, vecs = loewner_kernel(rho)
        VtV = vecs.conj().T @ vecs
        quantum_assert_close(
            VtV,
            np.eye(2),
            "density_matrix",
            "Eigenvectors should be unitary",
            atol=1e-12,
            rtol=0.0,
        )

    def test_vecs_unitary_random_qutrit(self):
        """Eigenvectors are unitary for a random mixed qutrit state."""
        np.random.seed(0)
        A = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
        rho = A @ A.conj().T
        rho /= np.trace(rho)
        C, vals, vecs = loewner_kernel(rho)
        VtV = vecs.conj().T @ vecs
        quantum_assert_close(
            VtV,
            np.eye(3),
            "density_matrix",
            "Eigenvectors should be unitary for qutrit",
            atol=1e-12,
            rtol=0.0,
        )

    def test_kernel_symmetric(self):
        """C is symmetric: c(lambda_i, lambda_j) = c(lambda_j, lambda_i)."""
        np.random.seed(42)
        A = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
        rho = A @ A.conj().T
        rho /= np.trace(rho)
        C, vals, vecs = loewner_kernel(rho)
        assert np.allclose(C, C.T, atol=1e-12), "Kernel matrix should be symmetric"

    def test_diagonal_equals_eigenvalues(self):
        """Diagonal C[i,i] equals lambda_i (the L'Hopital limit p -> p)."""
        rho = np.diag([0.2, 0.8]).astype(complex)
        C, vals, vecs = loewner_kernel(rho)
        np.testing.assert_allclose(
            np.diag(C),
            vals,
            atol=1e-12,
            err_msg="Diagonal of C should equal eigenvalues",
        )

    def test_off_diagonal_divided_difference(self):
        """Off-diagonal C[i,j] = (lambda_i - lambda_j) / (log lambda_i - log lambda_j)."""
        rho = np.diag([0.3, 0.7]).astype(complex)
        C, vals, vecs = loewner_kernel(rho)
        lam0, lam1 = vals[0], vals[1]
        expected_01 = (lam0 - lam1) / (np.log(lam0) - np.log(lam1))
        quantum_assert_scalar_close(
            C[0, 1],
            expected_01,
            "kernel",
            "Off-diagonal entry should satisfy divided-difference formula",
            atol=1e-12,
            rtol=0.0,
        )

    def test_lme_limit_maximally_mixed_qubit(self):
        """For rho = I/2, all kernel entries equal 1/2."""
        d = 2
        rho = np.eye(d, dtype=complex) / d
        C, vals, vecs = loewner_kernel(rho)
        expected = np.ones((d, d)) / d
        quantum_assert_close(
            C,
            expected,
            "kernel",
            "LME limit for qubit: all kernel entries should be 1/d",
            atol=1e-12,
            rtol=0.0,
        )

    def test_lme_limit_maximally_mixed_qutrit(self):
        """For rho = I/3, all kernel entries equal 1/3."""
        d = 3
        rho = np.eye(d, dtype=complex) / d
        C, vals, vecs = loewner_kernel(rho)
        expected = np.ones((d, d)) / d
        quantum_assert_close(
            C,
            expected,
            "kernel",
            "LME limit for qutrit: all kernel entries should be 1/d",
            atol=1e-12,
            rtol=0.0,
        )

    def test_near_degenerate_uses_arithmetic_mean(self):
        """Near-degenerate eigenvalues below tol use arithmetic mean."""
        eps = 1e-9
        lam = 0.5
        rho = np.diag([lam - eps, lam + eps]).astype(complex)
        # With tol=1e-7 the difference 2*eps=2e-9 is treated as degenerate
        C, vals, vecs = loewner_kernel(rho, tol=1e-7)
        # Arithmetic mean of (lam-eps) and (lam+eps) is lam
        quantum_assert_scalar_close(
            C[0, 1],
            lam,
            "kernel",
            "Near-degenerate off-diagonal should use arithmetic mean",
            atol=1e-7,
            rtol=0.0,
        )

    def test_kernel_entries_positive(self):
        """
        All C[i,j] entries are positive.

        The logarithmic mean of two positive numbers is positive, so every
        entry of the kernel matrix (both on-diagonal and off-diagonal) is
        strictly positive for a full-rank density matrix.
        """
        np.random.seed(7)
        A = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
        rho = A @ A.conj().T
        rho /= np.trace(rho)
        C, vals, vecs = loewner_kernel(rho)
        assert np.all(C >= -1e-14), (
            f"All kernel entries should be positive; min = {C.min():.3e}"
        )

    def test_loewner_map_diagonal_state(self):
        """
        J_rho0(X) for diagonal rho is element-wise multiplication C * X
        (since vecs = I for diagonal rho).
        """
        rho = np.diag([0.3, 0.7]).astype(complex)
        C, vals, vecs = loewner_kernel(rho)

        X = np.array([[1.0, 0.5 + 0.2j], [0.5 - 0.2j, -1.0]], dtype=complex)

        # Apply Loewner map: rotate to eigenbasis, multiply by C, rotate back
        X_eig = vecs.conj().T @ X @ vecs
        JX = vecs @ (C * X_eig) @ vecs.conj().T

        # For diagonal rho eigh returns identity (up to sign), so JX ≈ C * X
        expected = C * X
        quantum_assert_close(
            JX,
            expected,
            "density_matrix",
            "Loewner map for diagonal rho should equal C * X element-wise",
            atol=1e-12,
            rtol=0.0,
        )

    def test_kernel_finite_near_pure_state(self):
        """Kernel is finite and non-negative for a near-pure state."""
        psi = np.array([np.cos(0.1), np.sin(0.1)], dtype=complex)
        rho = np.outer(psi, psi.conj())
        C, vals, vecs = loewner_kernel(rho)
        assert np.all(np.isfinite(C)), "Kernel must be finite for near-pure state"
        assert np.all(C >= -1e-12), "Kernel entries must be non-negative"

    def test_kernel_entries_bounded_by_max_eigenvalue(self):
        """Every kernel entry c(lambda_i, lambda_j) <= max(lambda_i, lambda_j)."""
        rho = np.diag([0.1, 0.3, 0.6]).astype(complex)
        C, vals, vecs = loewner_kernel(rho)
        # The logarithmic mean satisfies min(x,y) <= L(x,y) <= max(x,y)
        for i in range(len(vals)):
            for j in range(len(vals)):
                assert C[i, j] <= max(vals[i], vals[j]) + 1e-12, (
                    f"Kernel entry C[{i},{j}]={C[i,j]:.4f} exceeds "
                    f"max(λ_{i},λ_{j})={max(vals[i],vals[j]):.4f}"
                )
                assert C[i, j] >= min(vals[i], vals[j]) - 1e-12, (
                    f"Kernel entry C[{i},{j}]={C[i,j]:.4f} is below "
                    f"min(λ_{i},λ_{j})={min(vals[i],vals[j]):.4f}"
                )


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

