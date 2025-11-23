"""
Core utilities for the quantum inaccessible game.

This module contains the basic state-manipulation and entropy helpers
used throughout the quantum inaccessible game code:

- partial traces
- von Neumann entropy
- construction of locally maximally entangled (LME) states
- marginal entropies
"""

from typing import Tuple

import numpy as np
from scipy.linalg import eigh


def partial_trace(rho: np.ndarray, dims: list, keep: int) -> np.ndarray:
    """
    Compute partial trace over all subsystems except 'keep'.

    Parameters
    ----------
    rho : array, shape (D, D)
        Density matrix for composite system
    dims : list of int
        Dimensions of each subsystem [d1, d2, ...]
    keep : int
        Index of subsystem to keep (0-indexed)

    Returns
    -------
    rho_reduced : array, shape (d_keep, d_keep)
        Reduced density matrix
    """
    n_sys = len(dims)
    D = np.prod(dims)
    assert rho.shape == (D, D), "rho shape mismatch"

    # Reshape to separate subsystems: (d0, d1, ..., dn) x (d0, d1, ..., dn)
    shape = dims + dims
    rho_tensor = rho.reshape(shape)

    rho_reduced = np.zeros((dims[keep], dims[keep]), dtype=complex)

    for idx_keep in range(dims[keep]):
        for idx_keep_conj in range(dims[keep]):
            # Sum over all configurations of other subsystems
            for multi_idx in np.ndindex(*[dims[i] for i in range(n_sys) if i != keep]):
                # Build full index for the kept + other subsystems
                full_idx = []
                other_idx_pos = 0
                for i in range(n_sys):
                    if i == keep:
                        full_idx.append(idx_keep)
                    else:
                        full_idx.append(multi_idx[other_idx_pos])
                        other_idx_pos += 1

                # Conjugate indices
                full_idx_conj = []
                other_idx_pos = 0
                for i in range(n_sys):
                    if i == keep:
                        full_idx_conj.append(idx_keep_conj)
                    else:
                        full_idx_conj.append(multi_idx[other_idx_pos])
                        other_idx_pos += 1

                rho_reduced[idx_keep, idx_keep_conj] += rho_tensor[
                    tuple(full_idx + full_idx_conj)
                ]

    return rho_reduced


def von_neumann_entropy(rho: np.ndarray, regularisation: float = 1e-14) -> float:
    """
    Compute von Neumann entropy S(rho) = -Tr(rho log rho).

    Parameters
    ----------
    rho : array, shape (d, d)
        Density matrix
    regularisation : float
        Small value added to eigenvalues to avoid log(0)

    Returns
    -------
    entropy : float
        Von Neumann entropy
    """
    # Get eigenvalues (they're real for Hermitian matrices)
    eigvals = np.real(eigh(rho, eigvals_only=True))

    # Filter out negative eigenvalues due to numerical errors
    eigvals = np.maximum(eigvals, 0.0)

    # Regularise to avoid log(0)
    eigvals_reg = eigvals + regularisation

    # Compute entropy: -sum(p * log(p))
    entropy = -np.sum(eigvals * np.log(eigvals_reg))

    return entropy


def create_lme_state(n_sites: int, d: int) -> Tuple[np.ndarray, list]:
    """
    Create a locally maximally entangled (LME) state.

    For even n_sites, creates n_sites/2 maximally entangled pairs.
    For odd n_sites, leaves one site pure.

    Parameters
    -----------
    n_sites : int
        Number of sites/subsystems
    d : int
        Local dimension at each site

    Returns
    --------
    rho : array, shape (d**n_sites, d**n_sites)
        LME state density matrix
    dims : list of int
        Dimensions [d, d, ..., d]
    """
    dims = [d] * n_sites
    D = d**n_sites

    # Create maximally entangled pairs
    n_pairs = n_sites // 2

    # Start with zero state
    psi = np.zeros(D, dtype=complex)

    if n_sites % 2 == 0:
        # All sites paired
        for indices in np.ndindex(*dims):
            # Check if pairs match: (i0==i1, i2==i3, ...)
            paired = all(indices[2 * k] == indices[2 * k + 1] for k in range(n_pairs))
            if paired:
                flat_idx = np.ravel_multi_index(indices, dims)
                psi[flat_idx] = 1.0 / np.sqrt(d**n_pairs)
    else:
        # Odd number: leave last site in |0>
        for indices in np.ndindex(*dims):
            paired = all(indices[2 * k] == indices[2 * k + 1] for k in range(n_pairs))
            last_zero = indices[-1] == 0
            if paired and last_zero:
                flat_idx = np.ravel_multi_index(indices, dims)
                psi[flat_idx] = 1.0 / np.sqrt(d**n_pairs)

    # Normalise
    psi = psi / np.linalg.norm(psi)

    # Convert to density matrix
    rho = np.outer(psi, psi.conj())

    return rho, dims


def marginal_entropies(rho: np.ndarray, dims: list) -> np.ndarray:
    """
    Compute marginal entropies for all subsystems.

    Parameters
    -----------
    rho : array, shape (D, D)
        Joint density matrix
    dims : list of int
        Dimensions of subsystems

    Returns
    --------
    h : array, shape (n_sites,)
        Marginal entropies [h_1, h_2, ...]
    """
    n_sites = len(dims)
    h = np.zeros(n_sites)

    for i in range(n_sites):
        rho_i = partial_trace(rho, dims, keep=i)
        h[i] = von_neumann_entropy(rho_i)

    return h


def generic_decomposition(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose Jacobian into symmetric and antisymmetric parts.

    M = S + A where S = (M + M^T)/2, A = (M - M^T)/2

    Parameters
    ----------
    M : array, shape (n, n)
        Jacobian matrix

    Returns
    -------
    S : array, shape (n, n)
        Symmetric part (dissipative)
    A : array, shape (n, n)
        Antisymmetric part (conservative)
    """
    S = 0.5 * (M + M.T)
    A = 0.5 * (M - M.T)
    return S, A


__all__ = [
    "partial_trace",
    "von_neumann_entropy",
    "create_lme_state",
    "marginal_entropies",
    "generic_decomposition",
]



