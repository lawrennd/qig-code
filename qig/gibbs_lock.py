"""
Gibbs-locked frame API for the quantum inaccessible game.

A Gibbs-locked frame is a background state K_0 = beta * H that is a fixed point
of the linearised reversible dynamics: because [K_0, H] = 0 by construction, the
commutator term i[delta_K, K_0] generates pure phase rotation in the eigenbasis of
H without displacing K_0.

This module provides:
- GibbsLockedFrame: encapsulates K_0 = beta * H with its spectral geometry.
- infer_mu0: fits the uniform decay rate mu_0 from an off-diagonal trajectory.

These objects are the primary API for the analysis in::

    the-inaccessible-game-hamiltonian.tex

References
----------
- Lawrence (2026), "Gibbs-Lock and the Emergence of Hamiltonian Structure
  in the Inaccessible Game".
- qig.core.loewner_kernel: the divided-difference kernel applied at rho0.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.linalg import eigh, expm
from scipy.optimize import curve_fit

from qig.core import loewner_kernel, partial_trace


class GibbsLockedFrame:
    """
    Encapsulate a Gibbs-locked background K_0 = beta * H.

    Parameters
    ----------
    H : array, shape (D, D)
        Traceless Hermitian generator.  Its eigenbasis defines the preferred
        spectral decomposition; Bohr gaps are eigenvalue differences of H.
    beta : float
        Inverse temperature / coupling strength (real, nonzero).
    dims : list of int, optional
        Subsystem dimensions for partial-trace operations
        (e.g. [3, 3] for a qutrit pair).  Required for ``is_iso_marginal``.

    Attributes
    ----------
    H : array
    beta : float
    dims : list or None
    D : int
        Hilbert-space dimension.
    """

    def __init__(
        self,
        H: np.ndarray,
        beta: float,
        dims: Optional[list] = None,
    ) -> None:
        H = np.asarray(H, dtype=complex)
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError("H must be a square 2-D array")
        self.H = H
        self.beta = float(beta)
        self.dims = list(dims) if dims is not None else None
        self.D = H.shape[0]

        # Cache expensive decompositions lazily
        self._rho0: Optional[np.ndarray] = None
        self._H_vals: Optional[np.ndarray] = None
        self._H_vecs: Optional[np.ndarray] = None
        self._loewner: Optional[Tuple] = None

    # ------------------------------------------------------------------
    # Spectral decomposition of H
    # ------------------------------------------------------------------

    def _eigh_H(self) -> Tuple[np.ndarray, np.ndarray]:
        """Eigenvalues and eigenvectors of H (sorted ascending)."""
        if self._H_vals is None:
            self._H_vals, self._H_vecs = eigh(self.H)
        return self._H_vals, self._H_vecs

    # ------------------------------------------------------------------
    # Background state
    # ------------------------------------------------------------------

    @property
    def rho0(self) -> np.ndarray:
        """
        Gibbs state rho0 = exp(-beta * H) / tr(exp(-beta * H)).

        Computed via eigendecomposition for numerical stability.
        """
        if self._rho0 is None:
            vals, vecs = self._eigh_H()
            # exp eigenvalues, subtract max for numerical stability
            exp_vals = np.exp(-self.beta * vals)
            exp_vals = exp_vals / exp_vals.sum()
            self._rho0 = (vecs * exp_vals) @ vecs.conj().T
        return self._rho0

    # ------------------------------------------------------------------
    # Bohr gaps
    # ------------------------------------------------------------------

    def bohr_gaps(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return eigenvalues of H and the matrix of Bohr gaps.

        Returns
        -------
        eps : array, shape (D,)
            Eigenvalues of H sorted ascending.
        gaps : array, shape (D, D)
            gaps[i, j] = eps[i] - eps[j].  Diagonal is zero.
        """
        eps, _ = self._eigh_H()
        gaps = eps[:, None] - eps[None, :]
        return eps, gaps

    # ------------------------------------------------------------------
    # Loewner kernel
    # ------------------------------------------------------------------

    def loewner_kernel(self, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loewner divided-difference kernel at rho0.

        Thin wrapper around ``qig.core.loewner_kernel(self.rho0)``.

        Returns
        -------
        C : array, shape (D, D)
            Kernel matrix C[i, j] = c(lambda_i, lambda_j) in the eigenbasis
            of rho0 (= eigenbasis of H for a Gibbs-locked frame).
        vals : array, shape (D,)
            Eigenvalues of rho0 (ascending, clipped to >= 1e-14).
        vecs : array, shape (D, D)
            Eigenvectors of rho0 as columns.
        """
        if self._loewner is None:
            self._loewner = loewner_kernel(self.rho0, tol=tol)
        return self._loewner

    def loewner_map(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the Loewner map J_{rho0}(X) to a matrix X.

        In the eigenbasis of rho0::

            (J_{rho0}(X))_{ij} = C[i, j] * X_{ij}

        Parameters
        ----------
        X : array, shape (D, D)
            Input matrix (same basis as H and rho0).

        Returns
        -------
        JX : array, shape (D, D)
        """
        C, _, vecs = self.loewner_kernel()
        X_eig = vecs.conj().T @ X @ vecs
        JX_eig = C * X_eig
        return vecs @ JX_eig @ vecs.conj().T

    # ------------------------------------------------------------------
    # Iso-marginal test
    # ------------------------------------------------------------------

    def is_iso_marginal(
        self,
        delta_K: np.ndarray,
        tol: float = 1e-10,
    ) -> bool:
        """
        True if delta_K is iso-marginal to first order.

        A perturbation delta_K is iso-marginal if the induced first-order
        change in every marginal density matrix vanishes::

            tr_{!=k}(J_{rho0}(delta_K)) approx 0   for all k.

        Requires ``dims`` to be set.

        Parameters
        ----------
        delta_K : array, shape (D, D)
            Perturbation of the modular generator in the same basis as H.
        tol : float
            Threshold for the Frobenius norm of each marginal perturbation.

        Returns
        -------
        bool
        """
        if self.dims is None:
            raise ValueError(
                "dims must be set on GibbsLockedFrame to test iso-marginal condition"
            )
        delta_rho = self.loewner_map(delta_K)
        for k in range(len(self.dims)):
            drho_k = partial_trace(delta_rho, self.dims, keep=k)
            if np.linalg.norm(drho_k, "fro") > tol:
                return False
        return True

    # ------------------------------------------------------------------
    # Analytical linearised flow
    # ------------------------------------------------------------------

    def linearised_flow(
        self,
        delta_K: np.ndarray,
        mu0: float,
        t: float,
    ) -> np.ndarray:
        """
        Analytical element-wise solution of the linearised GENERIC equation.

        In the eigenbasis of H the equation decouples element-by-element::

            d/dt (delta_K)_{ij} = (i * beta * Delta_eps_{ij} - mu0) * (delta_K)_{ij}

        with solution::

            (delta_K(t))_{ij} = exp((i * beta * Delta_eps_{ij} - mu0) * t)
                                  * (delta_K(0))_{ij}.

        Parameters
        ----------
        delta_K : array, shape (D, D)
            Initial perturbation in the same basis as H.
        mu0 : float
            Uniform decay rate (>= 0).
        t : float
            Evolution time.

        Returns
        -------
        delta_K_t : array, shape (D, D)
            Evolved perturbation at time t (in the same basis as H).
        """
        _, gaps = self.bohr_gaps()
        _, H_vecs = self._eigh_H()

        # Rotate to eigenbasis of H
        dK_eig = H_vecs.conj().T @ delta_K @ H_vecs

        # Element-wise evolution factor
        phase = np.exp((1j * self.beta * gaps - mu0) * t)
        dK_eig_t = phase * dK_eig

        # Rotate back
        return H_vecs @ dK_eig_t @ H_vecs.conj().T

    # ------------------------------------------------------------------
    # Gibbs-lock verification
    # ------------------------------------------------------------------

    def gibbs_lock_residual(self) -> float:
        """
        Return ||[K0, H]||_F as a check of the Gibbs-lock condition.

        Should be zero (< 1e-12) by construction when H is the same
        matrix used to define K0 = beta * H.
        """
        K0 = self.beta * self.H
        comm = K0 @ self.H - self.H @ K0
        return float(np.linalg.norm(comm, "fro"))

    def __repr__(self) -> str:
        return (
            f"GibbsLockedFrame(D={self.D}, beta={self.beta:.4g}, "
            f"dims={self.dims})"
        )


# ---------------------------------------------------------------------------
# mu_0 inference
# ---------------------------------------------------------------------------

def infer_mu0(
    times: np.ndarray,
    rho_trajectory: np.ndarray,
    frame: GibbsLockedFrame,
    tol: float = 1e-3,
) -> float:
    """
    Estimate the uniform decay rate mu_0 from an off-diagonal trajectory.

    Fits exp(-mu0 * t) to the phase-stripped magnitude of off-diagonal
    elements of delta_rho(t) = rho(t) - rho0, averaged over all (i,j) pairs
    whose magnitude at t=0 exceeds `tol`.

    The Gibbs-lock picture predicts that each off-diagonal coherence decays as::

        |delta_rho_{ij}(t)| = |delta_rho_{ij}(0)| * exp(-mu0 * t)

    independently of the Bohr gap Delta_eps_{ij}, so a single exponential fit
    to the mean magnitude extracts mu_0.

    Parameters
    ----------
    times : array, shape (T,)
        Time points corresponding to the trajectory.
    rho_trajectory : array, shape (T, D, D)
        Density matrices at each time step.
    frame : GibbsLockedFrame
        The background Gibbs-locked frame defining rho0 and the eigenbasis.
    tol : float
        Minimum |delta_rho_{ij}(0)| for a mode to be included in the fit.

    Returns
    -------
    mu0 : float
        Fitted uniform decay rate (>= 0).

    Raises
    ------
    ValueError
        If no off-diagonal modes have initial magnitude above `tol`.
    RuntimeError
        If the exponential fit fails to converge.
    """
    times = np.asarray(times, dtype=float)
    rho_traj = np.asarray(rho_trajectory)
    T, D, _ = rho_traj.shape
    rho0 = frame.rho0

    # Eigenbasis of H (= eigenbasis of rho0 for a Gibbs-locked frame)
    _, H_vecs = frame._eigh_H()

    # delta_rho in the H eigenbasis at each time step
    delta_rho_eig = np.zeros((T, D, D), dtype=complex)
    for ti in range(T):
        dr = rho_traj[ti] - rho0
        delta_rho_eig[ti] = H_vecs.conj().T @ dr @ H_vecs

    # Collect off-diagonal mode magnitudes over time
    mode_traces = []
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            mag = np.abs(delta_rho_eig[:, i, j])
            if mag[0] > tol:
                # Normalise so each mode starts at 1
                mode_traces.append(mag / mag[0])

    if not mode_traces:
        raise ValueError(
            f"No off-diagonal modes with initial magnitude > {tol}. "
            "Check that rho_trajectory differs from rho0, or reduce tol."
        )

    # Average over modes: should all decay as exp(-mu0 * t)
    mean_decay = np.mean(np.array(mode_traces), axis=0)

    # Fit exp(-mu0 * t)
    def _exp_decay(t: np.ndarray, mu0: float) -> np.ndarray:
        return np.exp(-mu0 * t)

    try:
        popt, _ = curve_fit(
            _exp_decay,
            times,
            mean_decay,
            p0=[1.0],
            bounds=(0.0, np.inf),
        )
    except RuntimeError as exc:
        raise RuntimeError(f"Exponential fit for mu_0 failed: {exc}") from exc

    return float(popt[0])


__all__ = ["GibbsLockedFrame", "infer_mu0"]
