"""
Operator bases for entangled pairs in Matrix Exponential Families.

This module provides generators for su(d²) Lie algebras, which act on
the Hilbert space of a pair of d-level systems (e.g., two qubits for d=2).
These operators can generate entangled states, unlike local operators.
"""

import numpy as np
from typing import List, Optional, Tuple


def gell_mann_generators(d: int) -> List[np.ndarray]:
    """
    Generate the d²-1 Gell-Mann matrices for su(d).
    
    These are traceless Hermitian matrices that generalize the Pauli matrices.
    For d=2, this gives the three Pauli matrices.
    For d=3, this gives the eight Gell-Mann matrices.
    
    The construction follows the standard pattern:
    
    - Symmetric matrices: ``|j⟩⟨k| + |k⟩⟨j|`` for j < k
    - Antisymmetric matrices: ``-i(|j⟩⟨k| - |k⟩⟨j|)`` for j < k  
    - Diagonal matrices: ``sum_{l=0}^{j-1} |l⟩⟨l| - j|j⟩⟨j|`` for j = 1,...,d-1
    
    Parameters
    ----------
    d : int
        Dimension of the Hilbert space
        
    Returns
    -------
    List[np.ndarray]
        List of d²-1 traceless Hermitian matrices of size d×d
        
    Notes
    -----
    The matrices are normalized to have Tr(λ_a λ_b) = 2δ_{ab}.
    """
    generators = []
    
    # Symmetric matrices: |j⟩⟨k| + |k⟩⟨j| for j < k
    for j in range(d):
        for k in range(j + 1, d):
            sym = np.zeros((d, d), dtype=complex)
            sym[j, k] = 1
            sym[k, j] = 1
            generators.append(sym)
    
    # Antisymmetric matrices: -i(|j⟩⟨k| - |k⟩⟨j|) for j < k
    for j in range(d):
        for k in range(j + 1, d):
            asym = np.zeros((d, d), dtype=complex)
            asym[j, k] = -1j
            asym[k, j] = 1j
            generators.append(asym)
    
    # Diagonal matrices: sqrt(2/(j(j+1))) * (sum_{l=0}^{j-1} |l⟩⟨l| - j|j⟩⟨j|)
    for j in range(1, d):
        diag = np.zeros((d, d), dtype=complex)
        # Sum over l = 0, ..., j-1
        for l in range(j):
            diag[l, l] = 1
        # Subtract j times |j⟩⟨j|
        diag[j, j] = -j
        # Normalize: Tr(diag²) = j + j² = j(j+1)
        # We want Tr = 2, so multiply by sqrt(2/(j(j+1)))
        diag = diag * np.sqrt(2.0 / (j * (j + 1)))
        generators.append(diag)
    
    # Verify we have the right number
    assert len(generators) == d**2 - 1, f"Expected {d**2-1} generators, got {len(generators)}"
    
    # Verify all are traceless and Hermitian
    for i, g in enumerate(generators):
        assert np.abs(np.trace(g)) < 1e-10, f"Generator {i} not traceless: Tr = {np.trace(g)}"
        assert np.allclose(g, g.conj().T), f"Generator {i} not Hermitian"
    
    return generators


def pair_basis_generators(d: int) -> List[np.ndarray]:
    """
    Generate su(d²) generators for a pair of d-level systems.
    
    These act on the d²-dimensional Hilbert space of a pair (e.g., two qubits
    give d=2, d²=4, with 15 generators in su(4)).
    
    Parameters
    ----------
    d : int
        Dimension of each individual system (e.g., 2 for qubits, 3 for qutrits)
        
    Returns
    -------
    List[np.ndarray]
        List of d⁴-1 traceless Hermitian matrices of size d²×d²
        
    Examples
    --------
    >>> # For a qubit pair
    >>> generators = pair_basis_generators(d=2)
    >>> len(generators)
    15
    >>> generators[0].shape
    (4, 4)
    """
    return gell_mann_generators(d**2)


def bell_state(d: int, k: int = 0) -> np.ndarray:
    """
    Create a maximally entangled state for a pair of d-level systems.
    
    Returns the state vector ``|Φₖ⟩ = (1/√d) ∑_{j=0}^{d-1} |j, (j+k) mod d⟩``.
    
    For k=0, this is the standard Bell state ``(1/√d) ∑_j |jj⟩``.
    
    Parameters
    ----------
    d : int
        Dimension of each subsystem
    k : int, default=0
        Bell state index (0 to d-1). Different k give orthogonal states.
        k=0: ``|00⟩ + |11⟩ + ...`` (standard)
        k=1: ``|01⟩ + |12⟩ + |20⟩ + ...`` (cyclic shift)
        
    Returns
    -------
    np.ndarray
        State vector of length d², normalized
        
    Notes
    -----
    All d Bell states are maximally entangled with the same properties:
    - Global purity: S(ρ) = 0
    - Marginals: ρ_A = ρ_B = I/d
    - Entanglement: maximal for dimension d
    
    The states are mutually orthogonal: ``⟨Φₖ|Φₗ⟩ = δₖₗ``
        
    Examples
    --------
    >>> psi = bell_state(d=2, k=0)  # |00⟩ + |11⟩
    >>> psi
    array([0.70710678+0.j, 0.        +0.j, 0.        +0.j, 0.70710678+0.j])
    
    >>> psi = bell_state(d=2, k=1)  # |01⟩ + |10⟩
    >>> psi
    array([0.        +0.j, 0.70710678+0.j, 0.70710678+0.j, 0.        +0.j])
    """
    if not 0 <= k < d:
        raise ValueError(f"k must be in range [0, d-1]=[0, {d-1}], got {k}")
    
    psi = np.zeros(d**2, dtype=complex)
    for j in range(d):
        # |j, (j+k) mod d⟩ corresponds to index j*d + ((j+k) mod d)
        second_index = (j + k) % d
        psi[j * d + second_index] = 1.0
    psi = psi / np.sqrt(d)
    return psi


def bell_state_density_matrix(d: int) -> np.ndarray:
    """
    Create the density matrix of a maximally entangled pair state.
    
    Returns ``ρ = |Φ⟩⟨Φ|`` where ``|Φ⟩ = (1/√d) ∑_j |jj⟩``.
    
    This is a pure state (Tr(ρ²) = 1) that is globally pure but locally
    maximally mixed: both marginals have entropy log(d).
    
    Parameters
    ----------
    d : int
        Dimension of each subsystem
        
    Returns
    -------
    np.ndarray
        Density matrix of size d²×d², positive semidefinite with trace 1
        
    Notes
    -----
    Properties of the Bell state:
    - Global purity: S(ρ) = 0
    - Local marginals: ρ_A = ρ_B = I/d (maximally mixed)
    - Marginal entropies: S(ρ_A) = S(ρ_B) = log(d)
    - Mutual information: I = 2log(d) (maximal)
    """
    psi = bell_state(d)
    return np.outer(psi, psi.conj())


def multi_pair_basis(n_pairs: int, d: int) -> Tuple[List[np.ndarray], List[int]]:
    """
    Generate direct sum operator basis for n independent pairs.
    
    Creates operators of the form F_α^(k) = I ⊗ ... ⊗ F_α ⊗ ... ⊗ I,
    where F_α is at position k.
    
    Parameters
    ----------
    n_pairs : int
        Number of independent pairs
    d : int
        Dimension of each subsystem (e.g., 2 for qubits)
        
    Returns
    -------
    operators : List[np.ndarray]
        List of n(d⁴-1) operators, each of size (d²)ⁿ × (d²)ⁿ
    pair_indices : List[int]
        Index of which pair each operator acts on
        
    Notes
    -----
    The resulting Fisher metric will be block-diagonal, with one block
    per pair. Cross-pair elements vanish: G_{(k,i),(k',j)} = 0 for k≠k'.
    
    Examples
    --------
    >>> # Two qubit pairs
    >>> ops, indices = multi_pair_basis(n_pairs=2, d=2)
    >>> len(ops)
    30
    >>> ops[0].shape
    (16, 16)
    >>> indices[:15]  # First 15 operators act on pair 0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    >>> indices[15:] # Next 15 operators act on pair 1
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    """
    # Get generators for a single pair
    pair_generators = pair_basis_generators(d)
    n_params_per_pair = len(pair_generators)
    
    # Total Hilbert space dimension
    D = d**(2 * n_pairs)
    
    # Size of each pair's Hilbert space
    D_pair = d**2
    
    operators = []
    pair_indices = []
    
    for k in range(n_pairs):
        for F_alpha in pair_generators:
            # Build I ⊗ ... ⊗ F_α ⊗ ... ⊗ I
            # Start with F_α for pair k
            op = F_alpha
            
            # Tensor with identity for pairs before k
            for _ in range(k):
                op = np.kron(np.eye(D_pair), op)
            
            # Tensor with identity for pairs after k
            for _ in range(n_pairs - k - 1):
                op = np.kron(op, np.eye(D_pair))
            
            operators.append(op)
            pair_indices.append(k)
    
    assert len(operators) == n_pairs * n_params_per_pair
    assert len(pair_indices) == len(operators)
    
    return operators, pair_indices


def product_of_bell_states(
    n_pairs: int, 
    d: int, 
    bell_indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Create a product state of n maximally entangled pairs.
    
    Returns ``|Ψ⟩ = |Φₖ₁⟩⊗|Φₖ₂⟩⊗...⊗|Φₖₙ⟩`` where each ``|Φₖ⟩`` is a Bell state.
    
    Parameters
    ----------
    n_pairs : int
        Number of pairs
    d : int
        Dimension of each subsystem
    bell_indices : list of int, optional
        Which Bell state (k=0,...,d-1) to use for each pair.
        If None, uses k=0 (standard Bell state) for all pairs.
        
    Returns
    -------
    np.ndarray
        State vector of length (d²)ⁿ
        
    Notes
    -----
    This is the "origin" of the inaccessible game. Properties:
    - Globally pure: S(ρ) = 0
    - All 2n marginals maximally mixed: S(ρ_i) = log(d) for all i
    - Pairs are entangled, but no cross-pair entanglement
    - Total marginal entropy: C = 2n·log(d)
    
    Different ``bell_indices`` give orthogonal product states that share
    the same constraint value C and marginal entropies, but represent
    different "origins" in the exponential family.
    
    Examples
    --------
    >>> # Standard: |Φ₀⟩⊗|Φ₀⟩ = (|00⟩+|11⟩)⊗(|00⟩+|11⟩)
    >>> psi = product_of_bell_states(n_pairs=2, d=2)
    
    >>> # Mixed: |Φ₀⟩⊗|Φ₁⟩ = (|00⟩+|11⟩)⊗(|01⟩+|10⟩)
    >>> psi = product_of_bell_states(n_pairs=2, d=2, bell_indices=[0, 1])
    """
    if bell_indices is None:
        bell_indices = [0] * n_pairs
    
    if len(bell_indices) != n_pairs:
        raise ValueError(
            f"bell_indices must have length {n_pairs}, got {len(bell_indices)}"
        )
    
    for i, k in enumerate(bell_indices):
        if not 0 <= k < d:
            raise ValueError(
                f"bell_indices[{i}]={k} out of range [0, {d-1}]"
            )
    
    # Tensor product of Bell states with specified indices
    result = bell_state(d, k=bell_indices[0])
    for i in range(1, n_pairs):
        result = np.kron(result, bell_state(d, k=bell_indices[i]))
    
    return result


def generalised_bell_basis(d: int) -> np.ndarray:
    """
    Construct the full generalised Bell basis for a pair of d-level systems.

    The generalised Bell states are defined as

    .. math::

        |\\Phi_{mn}\\rangle = \\frac{1}{\\sqrt{d}} \\sum_{k=0}^{d-1}
        \\omega^{km}\\, |k\\rangle \\otimes |(k+n)\\bmod d\\rangle,
        \\quad \\omega = e^{2\\pi i/d}.

    Parameters
    ----------
    d : int
        Dimension of each subsystem (so the pair lives in :math:`\\mathbb{C}^{d^2}`).

    Returns
    -------
    U : np.ndarray, shape (d², d²)
        Unitary matrix whose column ``m * d + n`` is ``|Φ_{mn}⟩``.
        Satisfies ``U.conj().T @ U = I`` (verified internally).

    Notes
    -----
    The existing :func:`bell_state` covers only the ``m=0`` sub-family
    (cyclic-shift Bell states).  This function returns all d² orthonormal
    Bell states, spanning a complete basis for the pair Hilbert space.

    Every column ``|Φ_{mn}⟩`` has exactly maximally mixed marginals:
    ``Tr_B(|Φ_{mn}⟩⟨Φ_{mn}|) = I/d``.

    Examples
    --------
    >>> U = generalised_bell_basis(d=2)
    >>> U.shape
    (4, 4)
    >>> import numpy as np
    >>> np.allclose(U.conj().T @ U, np.eye(4))
    True
    """
    omega = np.exp(2j * np.pi / d)
    D = d * d
    U = np.zeros((D, D), dtype=complex)
    for m in range(d):
        for n in range(d):
            col = m * d + n
            for k in range(d):
                row = k * d + (k + n) % d
                U[row, col] = omega ** (k * m) / np.sqrt(d)
    assert np.allclose(U.conj().T @ U, np.eye(D)), \
        "generalised_bell_basis: result is not unitary"
    return U


def near_bell_hamiltonian(d: int, delta: float = 0.1) -> np.ndarray:
    """
    Hermitian Hamiltonian diagonal in the generalised Bell basis, suitable
    for near-Bell Gibbs-state experiments.

    The spectrum is:

    - ``|Φ_{00}⟩``  eigenvalue ``0``     (ground state, reference Bell state)
    - ``|Φ_{01}⟩``  eigenvalue ``delta`` (one neighbouring level)
    - all others    eigenvalue ``1``

    Because every Bell state has exactly maximally mixed marginals
    (``ρ_A = I/d``), the Gibbs state ``ρ(β) = exp(-βH)/Z`` satisfies:

    - **Exact LME** at every temperature (marginals = ``I/d`` for all β).
    - **Approaches pure Bell state** as β → ∞.
    - **Gibbs-locked**: ``K_0 = βH`` commutes with ``H`` trivially.

    Parameters
    ----------
    d : int
        Dimension of each subsystem.
    delta : float, default 0.1
        Energy gap of the neighbouring Bell level above the ground state.

    Returns
    -------
    H : np.ndarray, shape (d², d²)
        Hermitian matrix with the spectrum described above.

    Examples
    --------
    >>> H = near_bell_hamiltonian(d=2, delta=0.1)
    >>> H.shape
    (4, 4)
    >>> import numpy as np
    >>> np.allclose(H, H.conj().T)
    True
    """
    U = generalised_bell_basis(d)
    D = d * d
    # Spectrum: ground state 0, one neighbour at delta, rest at 1
    spectrum = np.ones(D)
    spectrum[0] = 0.0      # |Φ_{00}⟩ = column 0*d+0 = 0
    spectrum[1] = delta    # |Φ_{01}⟩ = column 0*d+1 = 1
    H = U @ np.diag(spectrum) @ U.conj().T
    # Symmetrise to eliminate floating-point anti-Hermitian residuals
    H = 0.5 * (H + H.conj().T)
    return H


def near_bell_gibbs_frame(d: int, delta: float = 0.1, beta: float = 5.0):
    """
    Convenience constructor for the canonical near-Bell testbed.

    Builds a :class:`~qig.gibbs_lock.GibbsLockedFrame` from the near-Bell
    Hamiltonian returned by :func:`near_bell_hamiltonian`.

    The resulting frame has:

    - Gibbs state exactly LME (marginals ``= I/d``) for every ``beta``.
    - Gibbs-lock residual ``||[K_0, H]||_F < 1e-12`` (trivially satisfied
      since ``K_0 = β H``).
    - Pure-Bell limit as ``beta → ∞``.

    Parameters
    ----------
    d : int
        Dimension of each subsystem.
    delta : float, default 0.1
        Energy gap passed to :func:`near_bell_hamiltonian`.
    beta : float, default 5.0
        Inverse temperature for the Gibbs state.

    Returns
    -------
    frame : GibbsLockedFrame
        Fully initialised frame centred near the Bell-state boundary.

    Examples
    --------
    >>> frame = near_bell_gibbs_frame(d=2, delta=0.1, beta=5.0)
    >>> frame.gibbs_lock_residual() < 1e-10
    True
    """
    from .gibbs_lock import GibbsLockedFrame
    H = near_bell_hamiltonian(d, delta)
    return GibbsLockedFrame(H, beta=beta, dims=[d, d])


if __name__ == "__main__":
    # Test su(4) generators for qubit pair
    print("Testing su(4) generators for qubit pair:")
    generators = pair_basis_generators(d=2)
    print(f"Number of generators: {len(generators)} (expected 15)")
    print(f"Shape: {generators[0].shape} (expected (4,4))")
    
    # Test Bell state
    print("\nTesting Bell state:")
    psi = bell_state(d=2)
    print(f"Bell state: {psi}")
    print(f"Norm: {np.linalg.norm(psi):.6f} (expected 1)")
    
    rho = bell_state_density_matrix(d=2)
    print(f"Purity Tr(ρ²): {np.trace(rho @ rho):.6f} (expected 1)")
    
    # Test marginals
    rho_A = np.array([[rho[0,0] + rho[1,1], rho[0,2] + rho[1,3]],
                      [rho[2,0] + rho[3,1], rho[2,2] + rho[3,3]]])
    print(f"Marginal ρ_A:\n{rho_A}")
    print(f"Should be I/2: {np.allclose(rho_A, np.eye(2)/2)}")
    
    # Test multi-pair basis
    print("\nTesting multi-pair basis for 2 qubit pairs:")
    ops, indices = multi_pair_basis(n_pairs=2, d=2)
    print(f"Number of operators: {len(ops)} (expected 30)")
    print(f"Shape: {ops[0].shape} (expected (16,16))")
    print(f"Pair indices (first 5): {indices[:5]}")
    print(f"Pair indices (last 5): {indices[-5:]}")

