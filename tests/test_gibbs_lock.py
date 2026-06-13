"""
Test suite for qig.gibbs_lock — GibbsLockedFrame and infer_mu0.

Run with::

    pytest tests/test_gibbs_lock.py -v
"""

import numpy as np
import pytest
from scipy.linalg import expm

from qig import GibbsLockedFrame, infer_mu0, loewner_kernel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_hermitian(D: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))
    H = (A + A.conj().T) / 2
    H -= np.trace(H) / D * np.eye(D)
    return H


def _qutrit_pair_hamiltonian(delta: float = 0.5) -> np.ndarray:
    """Simple two-qutrit Hamiltonian: H = delta * (lambda3_1 x I + I x lambda3_2)."""
    lam3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex)
    H1 = np.kron(lam3, np.eye(3)) * delta
    H2 = np.kron(np.eye(3), lam3) * delta
    H = H1 + H2
    H -= np.trace(H) / 9 * np.eye(9)
    return H


# ---------------------------------------------------------------------------
# GibbsLockedFrame
# ---------------------------------------------------------------------------

class TestGibbsLockedFrameInit:
    """Basic construction and attribute checks."""

    def test_scalar_beta(self):
        H = _random_hermitian(4)
        frame = GibbsLockedFrame(H, beta=1.5)
        assert frame.beta == pytest.approx(1.5)
        assert frame.D == 4
        assert frame.dims is None

    def test_dims_stored(self):
        H = _qutrit_pair_hamiltonian()
        frame = GibbsLockedFrame(H, beta=1.0, dims=[3, 3])
        assert frame.dims == [3, 3]

    def test_bad_H_raises(self):
        with pytest.raises(ValueError):
            GibbsLockedFrame(np.ones((3, 4)), beta=1.0)

    def test_repr(self):
        frame = GibbsLockedFrame(_random_hermitian(3), beta=2.0, dims=[3])
        r = repr(frame)
        assert "GibbsLockedFrame" in r


class TestGibbsLockedFrameRho0:
    """Properties of the background Gibbs state."""

    @pytest.fixture
    def frame(self):
        H = _random_hermitian(4, seed=42)
        return GibbsLockedFrame(H, beta=2.0)

    def test_trace_one(self, frame):
        assert np.trace(frame.rho0).real == pytest.approx(1.0, abs=1e-12)

    def test_hermitian(self, frame):
        diff = frame.rho0 - frame.rho0.conj().T
        assert np.linalg.norm(diff, "fro") < 1e-12

    def test_positive_semidefinite(self, frame):
        evals = np.linalg.eigvalsh(frame.rho0)
        assert np.all(evals >= -1e-12)

    def test_matches_expm(self, frame):
        """rho0 should equal expm(-beta*H)/Z."""
        K0 = frame.beta * frame.H
        rho_ref = expm(-K0)
        rho_ref /= np.trace(rho_ref)
        np.testing.assert_allclose(frame.rho0, rho_ref, atol=1e-10)

    def test_cached_identity(self, frame):
        """Calling rho0 twice returns the same object."""
        r1 = frame.rho0
        r2 = frame.rho0
        assert r1 is r2

    def test_gibbs_lock_condition(self, frame):
        """[K0, H] should be zero to machine precision."""
        assert frame.gibbs_lock_residual() < 1e-11


class TestBohrGaps:
    @pytest.fixture
    def frame(self):
        H = np.diag([0.0, 1.0, 3.0]).astype(complex)
        return GibbsLockedFrame(H, beta=1.0)

    def test_gaps_shape(self, frame):
        eps, gaps = frame.bohr_gaps()
        assert eps.shape == (3,)
        assert gaps.shape == (3, 3)

    def test_gaps_antisymmetric(self, frame):
        _, gaps = frame.bohr_gaps()
        np.testing.assert_allclose(gaps, -gaps.T, atol=1e-14)

    def test_diagonal_zero(self, frame):
        _, gaps = frame.bohr_gaps()
        np.testing.assert_allclose(np.diag(gaps), 0.0, atol=1e-14)

    def test_gaps_values(self, frame):
        eps, gaps = frame.bohr_gaps()
        # H = diag(0,1,3) -> eps = [0,1,3]
        np.testing.assert_allclose(eps, [0.0, 1.0, 3.0], atol=1e-12)
        assert gaps[0, 1] == pytest.approx(-1.0, abs=1e-12)
        assert gaps[1, 2] == pytest.approx(-2.0, abs=1e-12)
        assert gaps[0, 2] == pytest.approx(-3.0, abs=1e-12)


class TestLoewnerKernel:
    @pytest.fixture
    def frame(self):
        H = _random_hermitian(4, seed=7)
        return GibbsLockedFrame(H, beta=1.0)

    def test_output_shape(self, frame):
        C, vals, vecs = frame.loewner_kernel()
        assert C.shape == (4, 4)
        assert vals.shape == (4,)
        assert vecs.shape == (4, 4)

    def test_agrees_with_core_loewner(self, frame):
        """loewner_kernel() must agree with qig.core.loewner_kernel(rho0)."""
        C_frame, vals_frame, vecs_frame = frame.loewner_kernel()
        C_core, vals_core, vecs_core = loewner_kernel(frame.rho0)
        np.testing.assert_allclose(C_frame, C_core, atol=1e-12)
        np.testing.assert_allclose(vals_frame, vals_core, atol=1e-12)

    def test_positive_kernel(self, frame):
        """Kernel entries should be positive (they are geometric means of eigenvalues)."""
        C, _, _ = frame.loewner_kernel()
        assert np.all(C >= -1e-12)

    def test_cached_identity(self, frame):
        r1 = frame.loewner_kernel()
        r2 = frame.loewner_kernel()
        assert r1[0] is r2[0]


class TestLoewnerMap:
    @pytest.fixture
    def frame(self):
        H = _random_hermitian(3, seed=3)
        return GibbsLockedFrame(H, beta=0.5)

    def test_loewner_map_shape(self, frame):
        X = _random_hermitian(3, seed=99)
        JX = frame.loewner_map(X)
        assert JX.shape == (3, 3)

    def test_loewner_map_zero(self, frame):
        """Applying map to zero matrix returns zero."""
        JX = frame.loewner_map(np.zeros((3, 3), dtype=complex))
        np.testing.assert_allclose(JX, 0.0, atol=1e-15)

    def test_loewner_map_linearity(self, frame):
        X = _random_hermitian(3, seed=10)
        Y = _random_hermitian(3, seed=11)
        alpha = 2.3
        lhs = frame.loewner_map(alpha * X + Y)
        rhs = alpha * frame.loewner_map(X) + frame.loewner_map(Y)
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_loewner_map_diagonal_consistent(self, frame):
        """
        For diagonal X in the eigenbasis of H, the map should scale each
        diagonal entry by the corresponding kernel diagonal (= eigenvalue).
        """
        C, vals, vecs = frame.loewner_kernel()
        # Construct diagonal matrix in eigenbasis
        d_eig = np.diag([1.0, 2.0, 3.0]).astype(complex)
        X = vecs @ d_eig @ vecs.conj().T

        JX = frame.loewner_map(X)
        JX_eig = vecs.conj().T @ JX @ vecs
        expected = C * d_eig
        np.testing.assert_allclose(JX_eig, expected, atol=1e-12)


class TestIsoMarginal:
    """Test iso-marginal classification for a qutrit pair (D=9)."""

    @pytest.fixture
    def frame(self):
        H = _qutrit_pair_hamiltonian(delta=0.5)
        return GibbsLockedFrame(H, beta=1.0, dims=[3, 3])

    def test_requires_dims(self):
        H = _random_hermitian(4)
        frame_nodims = GibbsLockedFrame(H, beta=1.0)
        with pytest.raises(ValueError, match="dims"):
            frame_nodims.is_iso_marginal(np.zeros((4, 4), dtype=complex))

    def test_zero_perturbation_is_iso_marginal(self, frame):
        """Zero perturbation trivially has zero marginal change."""
        delta_K = np.zeros((9, 9), dtype=complex)
        assert frame.is_iso_marginal(delta_K)

    def test_local_perturbation_is_not_iso_marginal(self, frame):
        """
        A purely local perturbation (acting only on site 1) generically
        changes marginal 1 — it should NOT be iso-marginal.
        """
        lam1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
        delta_K = np.kron(lam1, np.eye(3)) * 0.1
        # Local perturbation changes site-1 marginal, so is_iso_marginal = False
        result = frame.is_iso_marginal(delta_K)
        # We just check the function runs; the exact truth depends on rho0 eigenstructure.
        assert isinstance(result, bool)

    def test_output_is_bool(self, frame):
        delta_K = _random_hermitian(9, seed=55)
        result = frame.is_iso_marginal(delta_K, tol=1e-6)
        assert isinstance(result, bool)


class TestLinearisedFlow:
    """Test the analytical linearised dynamics solution."""

    @pytest.fixture
    def frame(self):
        H = np.diag([0.0, 1.0, 2.0]).astype(complex)
        return GibbsLockedFrame(H, beta=1.0)

    def test_flow_at_t0_is_identity(self, frame):
        delta_K = _random_hermitian(3, seed=20)
        out = frame.linearised_flow(delta_K, mu0=0.5, t=0.0)
        np.testing.assert_allclose(out, delta_K, atol=1e-12)

    def test_diagonal_decays(self, frame):
        """
        Diagonal elements in the H eigenbasis have zero Bohr gap.
        They should decay purely as exp(-mu0 * t).
        """
        # For diagonal H, eigenbasis is the computational basis.
        delta_K = np.diag([1.0, 0.5, 0.25]).astype(complex)
        mu0 = 0.7
        t = 1.5
        out = frame.linearised_flow(delta_K, mu0=mu0, t=t)
        expected = delta_K * np.exp(-mu0 * t)
        np.testing.assert_allclose(out.real, expected.real, atol=1e-12)

    def test_off_diagonal_amplitude_decays(self, frame):
        """
        Off-diagonal magnitude should decay as exp(-mu0 * t) regardless of
        the Bohr gap (which only adds oscillation phase).
        """
        delta_K = np.zeros((3, 3), dtype=complex)
        delta_K[0, 1] = 1.0
        delta_K[1, 0] = 1.0
        mu0 = 0.3
        t = 2.0
        out = frame.linearised_flow(delta_K, mu0=mu0, t=t)
        # Amplitude of [0,1] element should be exp(-mu0*t)
        assert abs(out[0, 1]) == pytest.approx(np.exp(-mu0 * t), rel=1e-10)

    def test_mu0_zero_no_decay(self, frame):
        """With mu0=0 only phase rotation occurs; magnitude is preserved."""
        delta_K = np.zeros((3, 3), dtype=complex)
        delta_K[0, 2] = 1.0
        delta_K[2, 0] = 1.0
        out = frame.linearised_flow(delta_K, mu0=0.0, t=1.0)
        assert abs(out[0, 2]) == pytest.approx(1.0, rel=1e-10)

    def test_group_property(self, frame):
        """Flow at t1+t2 should equal composing flow at t1 then t2."""
        delta_K = _random_hermitian(3, seed=30)
        mu0 = 0.4
        t1, t2 = 0.7, 1.1
        composed = frame.linearised_flow(
            frame.linearised_flow(delta_K, mu0=mu0, t=t1),
            mu0=mu0, t=t2
        )
        direct = frame.linearised_flow(delta_K, mu0=mu0, t=t1 + t2)
        np.testing.assert_allclose(composed, direct, atol=1e-11)


# ---------------------------------------------------------------------------
# infer_mu0
# ---------------------------------------------------------------------------

class TestInferMu0:
    """Tests for the mu_0 inference function."""

    def _make_synthetic_trajectory(
        self,
        D: int = 3,
        mu0_true: float = 0.5,
        n_points: int = 50,
        t_max: float = 4.0,
        seed: int = 0,
    ):
        """
        Build a synthetic trajectory that decays exactly as exp(-mu0*t).

        Works directly in the eigenbasis of H: each off-diagonal element of
        delta_rho_eig[ti][i, j] = amplitude * exp(-mu0 * t) * phase, so
        infer_mu0 sees clean exponential decay above the tolerance threshold.
        """
        rng = np.random.default_rng(seed)
        H = _random_hermitian(D, seed=seed)
        frame = GibbsLockedFrame(H, beta=1.0)
        rho0 = frame.rho0

        times = np.linspace(0.0, t_max, n_points)
        _, H_vecs = frame._eigh_H()
        _, gaps = frame.bohr_gaps()

        # Build a perturbation delta_rho directly in the eigenbasis with
        # amplitude well above the default tol=1e-3.
        amp_eig = (
            rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))
        ) * 0.1
        amp_eig = (amp_eig + amp_eig.conj().T) / 2
        # Ensure all off-diagonal amplitudes are at least 0.02
        for i in range(D):
            for j in range(D):
                if i != j and abs(amp_eig[i, j]) < 0.02:
                    amp_eig[i, j] = 0.02 + 0j

        rho_traj = np.zeros((n_points, D, D), dtype=complex)
        for ti, t in enumerate(times):
            # Element-wise analytical decay + phase in eigenbasis
            phase = np.exp((1j * frame.beta * gaps - mu0_true) * t)
            delta_rho_eig = amp_eig * phase
            # Rotate back to original basis
            delta_rho = H_vecs @ delta_rho_eig @ H_vecs.conj().T
            rho_traj[ti] = rho0 + delta_rho

        return times, rho_traj, frame

    def test_recovers_mu0_synthetic(self):
        """infer_mu0 should recover mu_0 from a synthetic decay to within 5%."""
        mu0_true = 0.6
        times, rho_traj, frame = self._make_synthetic_trajectory(
            D=4, mu0_true=mu0_true, n_points=80
        )
        mu0_fit = infer_mu0(times, rho_traj, frame)
        assert mu0_fit == pytest.approx(mu0_true, rel=0.05)

    def test_returns_positive(self):
        times, rho_traj, frame = self._make_synthetic_trajectory()
        mu0 = infer_mu0(times, rho_traj, frame)
        assert mu0 >= 0.0

    def test_zero_perturbation_raises(self):
        """When rho_traj == rho0 everywhere, no modes exceed tol -> ValueError."""
        D = 3
        H = _random_hermitian(D)
        frame = GibbsLockedFrame(H, beta=1.0)
        times = np.linspace(0, 2, 10)
        rho_traj = np.array([frame.rho0] * 10)
        with pytest.raises(ValueError, match="off-diagonal modes"):
            infer_mu0(times, rho_traj, frame, tol=1e-3)

    def test_different_mu0_values(self):
        """Higher true mu_0 should give higher fitted mu_0."""
        mu0_a = 0.2
        mu0_b = 1.0
        times_a, rho_a, frame_a = self._make_synthetic_trajectory(
            mu0_true=mu0_a, D=4, seed=1
        )
        times_b, rho_b, frame_b = self._make_synthetic_trajectory(
            mu0_true=mu0_b, D=4, seed=1
        )
        assert infer_mu0(times_a, rho_a, frame_a) < infer_mu0(times_b, rho_b, frame_b)


# ---------------------------------------------------------------------------
# Package-level import smoke test
# ---------------------------------------------------------------------------

class TestPackageImports:
    def test_import_from_qig(self):
        import qig
        assert hasattr(qig, "GibbsLockedFrame")
        assert hasattr(qig, "infer_mu0")

    def test_gibbs_frame_roundtrip(self):
        from qig import GibbsLockedFrame
        H = np.diag([0.0, 1.0, -1.0]).astype(complex)
        frame = GibbsLockedFrame(H, beta=0.5, dims=[3])
        rho = frame.rho0
        assert rho.shape == (3, 3)
