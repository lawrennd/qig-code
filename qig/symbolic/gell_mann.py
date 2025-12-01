"""
Symbolic Gell-Mann matrices and SU(3) structure constants.

This module provides SymPy symbolic representations of the Gell-Mann matrices
(generators of SU(3)) and their structure constants, enabling analytic
computation of quantum information geometric quantities.
"""

import sympy as sp
from sympy import I, sqrt, Rational
from typing import List


def symbolic_gell_mann_matrices() -> List[sp.Matrix]:
    """
    Return symbolic Gell-Mann matrices (generators of SU(3)).
    
    The eight traceless Hermitian Gell-Mann matrices λ_k satisfy:
    - Hermitian: λ_k† = λ_k
    - Traceless: Tr(λ_k) = 0
    - Normalization: Tr(λ_a λ_b) = 2δ_ab
    - Commutation: [λ_a, λ_b] = 2i Σ_c f_abc λ_c
    
    Returns
    -------
    gm : List[sp.Matrix]
        List of 8 symbolic 3×3 matrices
        
    Notes
    -----
    These match the numerical implementation in qig.exponential_family.gell_mann_matrices()
    exactly. The symbolic form enables:
    - Analytic computation of commutators
    - Symbolic verification of Lie algebra relations
    - Derivation of analytic GENERIC decomposition components
    
    Examples
    --------
    >>> from qig.symbolic import symbolic_gell_mann_matrices
    >>> import sympy as sp
    >>> gm = symbolic_gell_mann_matrices()
    >>> len(gm)
    8
    >>> gm[0]  # λ_1
    Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    >>> sp.simplify(sp.trace(gm[0]))
    0
    >>> (gm[0].H == gm[0])  # Hermitian
    True
    
    References
    ----------
    - Gell-Mann (1962): The Eightfold Way
    - quantum-generic-decomposition.tex: Section 2
    """
    gm = []
    
    # λ1 and λ2 (off-diagonal in 01 block)
    gm.append(sp.Matrix([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ]))
    
    gm.append(sp.Matrix([
        [0, -I, 0],
        [I, 0, 0],
        [0, 0, 0]
    ]))
    
    # λ3 (diagonal in 01 block)
    gm.append(sp.Matrix([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 0]
    ]))
    
    # λ4 and λ5 (off-diagonal in 02 block)
    gm.append(sp.Matrix([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ]))
    
    gm.append(sp.Matrix([
        [0, 0, -I],
        [0, 0, 0],
        [I, 0, 0]
    ]))
    
    # λ6 and λ7 (off-diagonal in 12 block)
    gm.append(sp.Matrix([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ]))
    
    gm.append(sp.Matrix([
        [0, 0, 0],
        [0, 0, -I],
        [0, I, 0]
    ]))
    
    # λ8 (diagonal, normalized)
    gm.append(sp.Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -2]
    ]) / sqrt(3))
    
    return gm


def symbolic_su3_structure_constants() -> sp.Array:
    """
    Return symbolic SU(3) structure constants f_abc.
    
    The structure constants satisfy: [λ_a, λ_b] = 2i Σ_c f_abc λ_c
    
    Returns
    -------
    f_abc : sp.Array, shape (8, 8, 8)
        Symbolic structure constants (real valued)
        
    Notes
    -----
    Properties:
    - Antisymmetric: f_abc = -f_bac
    - Real valued: f_abc ∈ ℝ
    - Jacobi identity: Σ_d (f_abd f_dce + f_bcd f_dae + f_cad f_dbe) = 0
    
    Non-zero values (up to permutations and sign):
    - f_123 = 1
    - f_147 = f_246 = f_257 = f_345 = 1/2
    - f_156 = f_367 = -1/2
    - f_458 = f_678 = √3/2
    
    This matches qig.reference_data.get_su3_structure_constants() exactly.
    The symbolic form enables analytic manipulation in GENERIC decomposition.
    
    Examples
    --------
    >>> from qig.symbolic import symbolic_su3_structure_constants
    >>> import sympy as sp
    >>> f = symbolic_su3_structure_constants()
    >>> f[0, 1, 2]  # f_123
    1
    >>> f[0, 3, 6]  # f_147
    1/2
    >>> f[0, 4, 5]  # f_156
    -1/2
    >>> f[3, 4, 7]  # f_458
    sqrt(3)/2
    
    References
    ----------
    - Gell-Mann (1962): SU(3) structure constants
    - quantum-generic-decomposition.tex: Equations (99-107)
    - Particle Data Group: SU(3) algebra reference
    """
    # Initialize 8×8×8 array of zeros
    f = sp.MutableDenseNDimArray.zeros(8, 8, 8)
    
    # Note: Using 0-based indexing (subtract 1 from physical indices)
    
    # f_123 = 1
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        f[i, j, k] = 1
    for i, j, k in [(1, 0, 2), (0, 2, 1), (2, 1, 0)]:
        f[i, j, k] = -1
    
    # f_147 = 1/2
    for i, j, k in [(0, 3, 6), (3, 6, 0), (6, 0, 3)]:
        f[i, j, k] = Rational(1, 2)
    for i, j, k in [(3, 0, 6), (0, 6, 3), (6, 3, 0)]:
        f[i, j, k] = Rational(-1, 2)
    
    # f_156 = -1/2
    for i, j, k in [(0, 4, 5), (4, 5, 0), (5, 0, 4)]:
        f[i, j, k] = Rational(-1, 2)
    for i, j, k in [(4, 0, 5), (0, 5, 4), (5, 4, 0)]:
        f[i, j, k] = Rational(1, 2)
    
    # f_246 = 1/2
    for i, j, k in [(1, 3, 5), (3, 5, 1), (5, 1, 3)]:
        f[i, j, k] = Rational(1, 2)
    for i, j, k in [(3, 1, 5), (1, 5, 3), (5, 3, 1)]:
        f[i, j, k] = Rational(-1, 2)
    
    # f_257 = 1/2
    for i, j, k in [(1, 4, 6), (4, 6, 1), (6, 1, 4)]:
        f[i, j, k] = Rational(1, 2)
    for i, j, k in [(4, 1, 6), (1, 6, 4), (6, 4, 1)]:
        f[i, j, k] = Rational(-1, 2)
    
    # f_345 = 1/2
    for i, j, k in [(2, 3, 4), (3, 4, 2), (4, 2, 3)]:
        f[i, j, k] = Rational(1, 2)
    for i, j, k in [(3, 2, 4), (2, 4, 3), (4, 3, 2)]:
        f[i, j, k] = Rational(-1, 2)
    
    # f_367 = -1/2
    for i, j, k in [(2, 5, 6), (5, 6, 2), (6, 2, 5)]:
        f[i, j, k] = Rational(-1, 2)
    for i, j, k in [(5, 2, 6), (2, 6, 5), (6, 5, 2)]:
        f[i, j, k] = Rational(1, 2)
    
    # f_458 = √3/2
    for i, j, k in [(3, 4, 7), (4, 7, 3), (7, 3, 4)]:
        f[i, j, k] = sqrt(3) / 2
    for i, j, k in [(4, 3, 7), (3, 7, 4), (7, 4, 3)]:
        f[i, j, k] = -sqrt(3) / 2
    
    # f_678 = √3/2
    for i, j, k in [(5, 6, 7), (6, 7, 5), (7, 5, 6)]:
        f[i, j, k] = sqrt(3) / 2
    for i, j, k in [(6, 5, 7), (5, 7, 6), (7, 6, 5)]:
        f[i, j, k] = -sqrt(3) / 2
    
    return f


def verify_gell_mann_properties(gm: List[sp.Matrix]) -> dict:
    """
    Verify that symbolic Gell-Mann matrices satisfy required properties.
    
    Parameters
    ----------
    gm : List[sp.Matrix]
        List of 8 symbolic Gell-Mann matrices
        
    Returns
    -------
    results : dict
        Dictionary with verification results:
        - 'all_hermitian': bool
        - 'all_traceless': bool
        - 'normalization': bool (Tr(λ_a λ_b) = 2δ_ab)
        - 'details': list of any failures
        
    Examples
    --------
    >>> from qig.symbolic import symbolic_gell_mann_matrices, verify_gell_mann_properties
    >>> gm = symbolic_gell_mann_matrices()
    >>> results = verify_gell_mann_properties(gm)
    >>> results['all_hermitian']
    True
    >>> results['all_traceless']
    True
    >>> results['normalization']
    True
    """
    results = {
        'all_hermitian': True,
        'all_traceless': True,
        'normalization': True,
        'details': []
    }
    
    # Check Hermiticity
    for i, λ in enumerate(gm):
        if λ.H != λ:
            results['all_hermitian'] = False
            results['details'].append(f"λ_{i+1} not Hermitian")
    
    # Check tracelessness
    for i, λ in enumerate(gm):
        trace = sp.simplify(sp.trace(λ))
        if trace != 0:
            results['all_traceless'] = False
            results['details'].append(f"λ_{i+1} has trace {trace}")
    
    # Check normalization: Tr(λ_a λ_b) = 2δ_ab
    for i in range(len(gm)):
        for j in range(len(gm)):
            trace = sp.simplify(sp.trace(gm[i] * gm[j]))
            expected = 2 if i == j else 0
            if trace != expected:
                results['normalization'] = False
                results['details'].append(
                    f"Tr(λ_{i+1} λ_{j+1}) = {trace}, expected {expected}"
                )
    
    return results


def commutator(A: sp.Matrix, B: sp.Matrix) -> sp.Matrix:
    """
    Compute commutator [A, B] = AB - BA.
    
    Parameters
    ----------
    A, B : sp.Matrix
        Symbolic matrices
        
    Returns
    -------
    comm : sp.Matrix
        Commutator [A, B]
    """
    return A * B - B * A


def verify_structure_constants(gm: List[sp.Matrix], f_abc: sp.Array) -> dict:
    """
    Verify [λ_a, λ_b] = 2i Σ_c f_abc λ_c for all a, b.
    
    Parameters
    ----------
    gm : List[sp.Matrix]
        Symbolic Gell-Mann matrices
    f_abc : sp.Array
        Symbolic structure constants
        
    Returns
    -------
    results : dict
        Verification results with 'all_correct' bool and list of 'errors'
        
    Examples
    --------
    >>> from qig.symbolic import (
    ...     symbolic_gell_mann_matrices,
    ...     symbolic_su3_structure_constants,
    ...     verify_structure_constants
    ... )
    >>> gm = symbolic_gell_mann_matrices()
    >>> f = symbolic_su3_structure_constants()
    >>> results = verify_structure_constants(gm, f)
    >>> results['all_correct']
    True
    """
    results = {'all_correct': True, 'errors': []}
    
    n = len(gm)
    for a in range(n):
        for b in range(n):
            # Compute commutator [λ_a, λ_b]
            comm = commutator(gm[a], gm[b])
            
            # Reconstruct from structure constants: 2i Σ_c f_abc λ_c
            reconstructed = sp.zeros(3, 3)
            for c in range(n):
                reconstructed += 2 * I * f_abc[a, b, c] * gm[c]
            
            # Check if they match
            diff = sp.simplify(comm - reconstructed)
            if diff != sp.zeros(3, 3):
                results['all_correct'] = False
                max_diff = max(abs(complex(x)) for x in diff)
                results['errors'].append(
                    f"[λ_{a+1}, λ_{b+1}] mismatch, max diff: {max_diff}"
                )
    
    return results

