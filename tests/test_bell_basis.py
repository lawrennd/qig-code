"""
Tests for CIP-000E: Generalised Bell Basis and Near-Bell Testbed Constructors.

Covers:
- generalised_bell_basis: unitarity, orthonormality, maximally-mixed marginals
- near_bell_hamiltonian: Hermiticity, correct spectrum in Bell basis
- near_bell_gibbs_frame: successful construction, Gibbs-lock residual
"""

import numpy as np
import pytest
from qig.pair_operators import (
    generalised_bell_basis,
    near_bell_hamiltonian,
    near_bell_gibbs_frame,
)
from qig.core import partial_trace


# ---------------------------------------------------------------------------
# generalised_bell_basis
# ---------------------------------------------------------------------------


class TestGeneralisedBellBasis:
    """Tests for the generalised Bell basis constructor."""

    @pytest.mark.parametrize("d", [2, 3])
    def test_shape(self, d):
        U = generalised_bell_basis(d)
        assert U.shape == (d * d, d * d)

    @pytest.mark.parametrize("d", [2, 3])
    def test_unitary(self, d):
        U = generalised_bell_basis(d)
        D = d * d
        assert np.allclose(U.conj().T @ U, np.eye(D), atol=1e-10)
        assert np.allclose(U @ U.conj().T, np.eye(D), atol=1e-10)

    @pytest.mark.parametrize("d", [2, 3])
    def test_columns_normalised(self, d):
        U = generalised_bell_basis(d)
        for col in range(d * d):
            v = U[:, col]
            assert np.isclose(np.dot(v.conj(), v), 1.0, atol=1e-10)

    @pytest.mark.parametrize("d", [2, 3])
    def test_maximally_mixed_marginals(self, d):
        """Every Bell state has ρ_A = ρ_B = I/d."""
        U = generalised_bell_basis(d)
        for col in range(d * d):
            v = U[:, col]
            rho = np.outer(v, v.conj())
            rho_A = partial_trace(rho, keep=0, dims=[d, d])
            rho_B = partial_trace(rho, keep=1, dims=[d, d])
            assert np.allclose(rho_A, np.eye(d) / d, atol=1e-10), (
                f"Column {col}: ρ_A ≠ I/d"
            )
            assert np.allclose(rho_B, np.eye(d) / d, atol=1e-10), (
                f"Column {col}: ρ_B ≠ I/d"
            )

    def test_qubit_ground_state_matches_bell_state(self):
        """Column 0 should be the standard Bell state |Φ₀₀⟩ = (|00⟩+|11⟩)/√2."""
        from qig.pair_operators import bell_state
        U = generalised_bell_basis(d=2)
        v = U[:, 0]
        expected = bell_state(d=2, k=0)
        # Allow global phase
        assert np.isclose(np.abs(np.dot(v.conj(), expected)), 1.0, atol=1e-10), (
            f"Column 0 = {v} does not match standard Bell state {expected}"
        )


# ---------------------------------------------------------------------------
# near_bell_hamiltonian
# ---------------------------------------------------------------------------


class TestNearBellHamiltonian:
    """Tests for the near-Bell Hamiltonian constructor."""

    @pytest.mark.parametrize("d", [2, 3])
    def test_shape(self, d):
        H = near_bell_hamiltonian(d)
        assert H.shape == (d * d, d * d)

    @pytest.mark.parametrize("d", [2, 3])
    def test_hermitian(self, d):
        H = near_bell_hamiltonian(d)
        assert np.allclose(H, H.conj().T, atol=1e-10)

    @pytest.mark.parametrize("d", [2, 3])
    @pytest.mark.parametrize("delta", [0.05, 0.1, 0.5])
    def test_spectrum_in_bell_basis(self, d, delta):
        """Eigenvalues in Bell-basis should be 0, delta, then 1s."""
        H = near_bell_hamiltonian(d, delta=delta)
        U = generalised_bell_basis(d)
        H_bell = U.conj().T @ H @ U
        diag = np.diag(H_bell).real
        expected = np.ones(d * d)
        expected[0] = 0.0
        expected[1] = delta
        assert np.allclose(np.sort(diag), np.sort(expected), atol=1e-10), (
            f"Spectrum mismatch: got {np.sort(diag)}, expected {np.sort(expected)}"
        )

    @pytest.mark.parametrize("d", [2, 3])
    def test_off_diagonal_in_bell_basis_small(self, d):
        """H should be diagonal in the Bell basis (off-diag terms < 1e-10)."""
        H = near_bell_hamiltonian(d)
        U = generalised_bell_basis(d)
        H_bell = U.conj().T @ H @ U
        off_diag = H_bell - np.diag(np.diag(H_bell))
        assert np.allclose(off_diag, 0, atol=1e-10)


# ---------------------------------------------------------------------------
# near_bell_gibbs_frame
# ---------------------------------------------------------------------------


class TestNearBellGibbsFrame:
    """Tests for the near_bell_gibbs_frame convenience constructor."""

    def test_returns_frame(self):
        from qig.gibbs_lock import GibbsLockedFrame
        frame = near_bell_gibbs_frame(d=2)
        assert isinstance(frame, GibbsLockedFrame)

    @pytest.mark.parametrize("d", [2, 3])
    def test_gibbs_lock_residual(self, d):
        """[K₀, H] should be ≈ 0 (trivially satisfied since K₀ = βH)."""
        frame = near_bell_gibbs_frame(d=d, delta=0.1, beta=5.0)
        residual = frame.gibbs_lock_residual()
        assert residual < 1e-10, f"Gibbs-lock residual = {residual}"

    @pytest.mark.parametrize("d", [2, 3])
    def test_rho0_is_lme(self, d):
        """Gibbs state should have maximally mixed marginals (LME)."""
        frame = near_bell_gibbs_frame(d=d, delta=0.1, beta=5.0)
        rho0 = frame.rho0
        rho_A = partial_trace(rho0, keep=0, dims=[d, d])
        rho_B = partial_trace(rho0, keep=1, dims=[d, d])
        assert np.allclose(rho_A, np.eye(d) / d, atol=1e-6), (
            f"d={d}: ρ_A ≠ I/d  (max dev {np.abs(rho_A - np.eye(d)/d).max():.2e})"
        )
        assert np.allclose(rho_B, np.eye(d) / d, atol=1e-6), (
            f"d={d}: ρ_B ≠ I/d  (max dev {np.abs(rho_B - np.eye(d)/d).max():.2e})"
        )

    @pytest.mark.parametrize("d,beta", [(2, 0.1), (2, 5.0), (3, 2.0)])
    def test_rho0_unit_trace(self, d, beta):
        frame = near_bell_gibbs_frame(d=d, beta=beta)
        assert np.isclose(np.trace(frame.rho0).real, 1.0, atol=1e-10)

    @pytest.mark.parametrize("d,beta", [(2, 0.1), (2, 5.0), (3, 2.0)])
    def test_rho0_positive_semidefinite(self, d, beta):
        frame = near_bell_gibbs_frame(d=d, beta=beta)
        eigvals = np.linalg.eigvalsh(frame.rho0)
        assert np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min()}"

    def test_high_beta_approaches_bell_state(self):
        """At very low temperature, the Gibbs state should be close to |Φ₀₀⟩⟨Φ₀₀|."""
        frame = near_bell_gibbs_frame(d=2, delta=0.1, beta=50.0)
        rho0 = frame.rho0
        U = generalised_bell_basis(d=2)
        phi00 = U[:, 0]
        rho_bell = np.outer(phi00, phi00.conj())
        overlap = np.trace(rho0 @ rho_bell).real
        assert overlap > 0.99, f"Low-T overlap with |Φ₀₀⟩ = {overlap:.4f}"
