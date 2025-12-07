"""
Operator bases for entangled pairs in quantum exponential families.

This module provides generators for su(d²) Lie algebras, which act on
the Hilbert space of a pair of d-level systems (e.g., two qubits for d=2).
These operators can generate entangled states, unlike local operators.
"""

import numpy as np
from typing import List, Tuple


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


def bell_state(d: int) -> np.ndarray:
    """
    Create a maximally entangled state for a pair of d-level systems.
    
    Returns the state vector ``|Φ⟩ = (1/√d) ∑_{j=0}^{d-1} |jj⟩``.
    
    For d=2 (qubits), this is the Bell state ``(|00⟩ + |11⟩)/√2``.
    For d=3 (qutrits), this is ``(|00⟩ + |11⟩ + |22⟩)/√3``.
    
    Parameters
    ----------
    d : int
        Dimension of each subsystem
        
    Returns
    -------
    np.ndarray
        State vector of length d², normalized
        
    Examples
    --------
    >>> psi = bell_state(d=2)
    >>> psi
    array([0.70710678+0.j, 0.        +0.j, 0.        +0.j, 0.70710678+0.j])
    """
    psi = np.zeros(d**2, dtype=complex)
    for j in range(d):
        # |jj⟩ corresponds to index j*d + j in the tensor product basis
        psi[j * d + j] = 1.0
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


def product_of_bell_states(n_pairs: int, d: int) -> np.ndarray:
    """
    Create a product state of n maximally entangled pairs.
    
    Returns ``|Ψ⟩ = |Φ⟩⊗|Φ⟩⊗...⊗|Φ⟩`` where ``|Φ⟩`` is the Bell state for dimension d.
    
    Parameters
    ----------
    n_pairs : int
        Number of pairs
    d : int
        Dimension of each subsystem
        
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
    """
    psi = bell_state(d)
    
    # Tensor product of n copies
    result = psi
    for _ in range(n_pairs - 1):
        result = np.kron(result, psi)
    
    return result


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

