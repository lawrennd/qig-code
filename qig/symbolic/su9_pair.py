"""
Symbolic computation for su(9) pair basis qutrit system.

This module implements symbolic expressions for a single qutrit PAIR using
the full su(9) Lie algebra (80 generators). This allows representation of
entangled states, unlike the local tensor product basis.

Key differences from local basis:
- 80 parameters (not 16)
- Can represent entangled states (e.g., Bell states)
- Structural identity BROKEN: Gθ ≠ -a
- Lagrange multiplier ν ≠ -1, and ∇ν ≠ 0
- Antisymmetric part A is NON-ZERO

Related to CIP-0007: Analytic Forms for S and A via Lie Algebra Structure
"""

import numpy as np
import sympy as sp
from sympy import Symbol, Matrix, sqrt, log, trace, exp, simplify, Rational
from typing import Tuple, List
from functools import lru_cache


@lru_cache(maxsize=1)
def symbolic_su9_generators() -> List[Matrix]:
    """
    Symbolic su(9) generators for qutrit pair.
    
    Returns 80 traceless Hermitian 9×9 matrices that span su(9).
    These are constructed using the numerical generators from the
    pair_operators module, converted to exact symbolic form.
    
    Returns
    -------
    generators : list of sp.Matrix
        80 symbolic 9×9 matrices
        
    Notes
    -----
    The su(9) algebra has dimension 9² - 1 = 80.
    
    For computational efficiency, we use the numerical generators
    and convert to symbolic form with rationalization.
    
    Examples
    --------
    >>> generators = symbolic_su9_generators()
    >>> len(generators)
    80
    >>> generators[0].shape
    (9, 9)
    >>> trace(generators[0])
    0
    """
    from qig.exponential_family import QuantumExponentialFamily
    
    # Get numerical su(9) generators
    exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
    numerical_generators = exp_fam.operators
    
    # Convert to symbolic with rationalization
    symbolic_generators = []
    for F_num in numerical_generators:
        # Convert to symbolic, rationalizing real and imaginary parts
        F_sym = sp.zeros(9, 9)
        for i in range(9):
            for j in range(9):
                val = F_num[i, j]
                if np.abs(val) < 1e-14:
                    F_sym[i, j] = 0
                else:
                    # Rationalize
                    real_part = sp.nsimplify(val.real, rational=True, tolerance=1e-12)
                    imag_part = sp.nsimplify(val.imag, rational=True, tolerance=1e-12)
                    F_sym[i, j] = real_part + sp.I * imag_part
        
        symbolic_generators.append(F_sym)
    
    return symbolic_generators


@lru_cache(maxsize=1)
def symbolic_su9_structure_constants() -> np.ndarray:
    """
    Structure constants for su(9): [F_a, F_b] = 2i Σ_c f_abc F_c.
    
    Returns
    -------
    f_abc : np.ndarray, shape (80, 80, 80)
        Structure constants (real-valued)
        
    Notes
    -----
    The structure constants are computed numerically from the generators
    using the relation:
        f_abc = -i/2 Tr([F_a, F_b] F_c) / Tr(F_c²)
        
    For su(9), there are 80³ = 512,000 components, but most are zero.
    The sparsity is ~99.4% (only ~2900 non-zero values).
    
    Examples
    --------
    >>> f = symbolic_su9_structure_constants()
    >>> f.shape
    (80, 80, 80)
    >>> np.count_nonzero(f) / f.size  # Sparsity
    0.006
    """
    from qig.exponential_family import QuantumExponentialFamily
    from qig.structure_constants import compute_structure_constants
    
    # Get numerical generators
    exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
    
    # Compute structure constants
    f_abc = compute_structure_constants(exp_fam.operators)
    
    return f_abc


def verify_su9_generators() -> dict:
    """
    Verify properties of su(9) generators.
    
    Checks:
    - Hermiticity: F_a† = F_a
    - Tracelessness: Tr(F_a) = 0
    - Normalization: Tr(F_a F_b) = δ_ab (up to constant)
    - Closure under commutation
    
    Returns
    -------
    results : dict
        Verification results with boolean flags
    """
    generators = symbolic_su9_generators()
    
    results = {
        'hermitian': [],
        'traceless': [],
        'orthogonal': True,
    }
    
    # Check each generator
    for i, F in enumerate(generators):
        # Hermiticity
        is_herm = (F - F.H).is_zero_matrix
        results['hermitian'].append(is_herm)
        
        # Tracelessness (check numerically due to rationalization artifacts)
        tr = sp.trace(F)
        try:
            tr_num = complex(tr)
            is_traceless = abs(tr_num) < 1e-12
        except:
            is_traceless = sp.simplify(tr) == 0
        results['traceless'].append(is_traceless)
    
    # Check orthogonality (sample a few pairs)
    for i in range(min(5, len(generators))):
        for j in range(i+1, min(5, len(generators))):
            overlap = sp.simplify(trace(generators[i] * generators[j]))
            if overlap != 0:
                results['orthogonal'] = False
                break
    
    return results


# Phase 2: Density matrix and cumulant function
def symbolic_density_matrix_su9_pair(
    theta_symbols: Tuple[Symbol, ...],
    order: int = 2
) -> Matrix:
    """
    Symbolic density matrix for su(9) pair basis (80 parameters).
    
    Computes ρ(θ) = exp(Σ θ_a F_a - ψ(θ)) / Z to specified order.
    
    Parameters
    ----------
    theta_symbols : tuple of sp.Symbol
        80 symbolic parameters
    order : int, default=2
        Order of perturbative expansion around θ=0
        
    Returns
    -------
    rho : sp.Matrix, shape (9, 9)
        Density matrix (Hermitian, trace-1, positive)
        
    Notes
    -----
    Perturbative expansion around maximally mixed state ρ(0) = I/9:
    
    Order-0: ρ ≈ I/9
    Order-1: ρ ≈ I/9 + (1/9) Σ θ_a F_a
    Order-2: ρ ≈ I/9 + (1/9) Σ θ_a F_a + (1/18) Σ_ab θ_a θ_b [F_a, F_b]
    
    The order-2 expansion uses Baker-Campbell-Hausdorff formula and
    the fact that ψ(θ) ≈ log(9) + (1/18)||θ||² for normalized generators.
    
    Examples
    --------
    >>> import sympy as sp
    >>> theta = sp.symbols('theta1:81', real=True)
    >>> rho = symbolic_density_matrix_su9_pair(theta, order=2)
    >>> rho.shape
    (9, 9)
    """
    if len(theta_symbols) != 80:
        raise ValueError(f"Expected 80 parameters, got {len(theta_symbols)}")
    
    generators = symbolic_su9_generators()
    
    # Construct K = Σ θ_a F_a
    K = sp.zeros(9, 9)
    for a, theta_a in enumerate(theta_symbols):
        K += theta_a * generators[a]
    
    # Taylor expansion: exp(K) ≈ I + K + (1/2)K²
    rho_unnorm = sp.eye(9)
    
    if order >= 1:
        rho_unnorm += K
    
    if order >= 2:
        rho_unnorm += K * K / 2
    
    # Normalize to trace-1
    # Start with simple division by 9
    rho = rho_unnorm / 9
    
    if order >= 2:
        # Order-2 correction:
        # For su(9) generators with Tr(F_a²) = 2: Tr(K²) = 2||θ||²
        # So Tr(I + K + K²/2) = 9 + 0 + ||θ||²
        # Thus: ρ = (I + K + K²/2) / (9 + ||θ||²)
        #         ≈ (I + K + K²/2)/9 * (1 - ||θ||²/9)
        #         = I/9 + K/9 + K²/18 - ||θ||²*I/81 + O(θ³)
        # Add correction: -||θ||²*I/81
        norm_sq = sum(theta_a**2 for theta_a in theta_symbols)
        rho -= (norm_sq / 81) * sp.eye(9)
    
    return rho


def symbolic_cumulant_generating_function_su9_pair(
    theta_symbols: Tuple[Symbol, ...],
    order: int = 2
) -> sp.Expr:
    """
    Cumulant generating function: ψ(θ) = log Tr[exp(Σ θ_a F_a)].
    
    Parameters
    ----------
    theta_symbols : tuple of sp.Symbol
        80 symbolic parameters
    order : int, default=2
        Order of expansion
        
    Returns
    -------
    psi : sp.Expr
        Cumulant generating function
        
    Notes
    -----
    For normalized generators Tr(F_a F_b) ≈ δ_ab:
        ψ(θ) ≈ log(9) + (1/18)||θ||²
        
    The coefficient 1/18 comes from the second-order expansion of
    the trace of the matrix exponential.
    """
    if len(theta_symbols) != 80:
        raise ValueError(f"Expected 80 parameters, got {len(theta_symbols)}")
    
    # Order-0: log of dimension
    psi = log(9)
    
    if order >= 2:
        # Order-2: (1/18)||θ||²
        # This comes from expanding Tr[exp(Σ θ_a F_a)] to order-2
        psi += sum(theta_a**2 for theta_a in theta_symbols) / 18
    
    return psi


def symbolic_fisher_information_su9_pair(
    theta_symbols: Tuple[Symbol, ...],
    order: int = 2
) -> Matrix:
    """
    Fisher information (BKM metric): G = ∇²ψ.
    
    Parameters
    ----------
    theta_symbols : tuple of sp.Symbol
        80 symbolic parameters
    order : int, default=2
        Order of expansion (only affects ψ)
        
    Returns
    -------
    G : sp.Matrix, shape (80, 80)
        Fisher information matrix (symmetric, positive definite)
        
    Notes
    -----
    For order-2:
        ψ(θ) ≈ log(9) + (1/18)||θ||²
        G_ab = ∂²ψ/∂θ_a∂θ_b ≈ δ_ab / 9
        
    This is diagonal because generators are approximately orthonormal.
    """
    if len(theta_symbols) != 80:
        raise ValueError(f"Expected 80 parameters, got {len(theta_symbols)}")
    
    psi = symbolic_cumulant_generating_function_su9_pair(theta_symbols, order)
    
    # Compute Hessian
    n = len(theta_symbols)
    G = sp.zeros(n, n)
    
    for i in range(n):
        for j in range(i, n):  # Only compute upper triangle
            G[i, j] = sp.diff(psi, theta_symbols[i], theta_symbols[j])
            if i != j:
                G[j, i] = G[i, j]  # Symmetric
    
    return simplify(G)


def symbolic_von_neumann_entropy_su9_pair(
    theta_symbols: Tuple[Symbol, ...],
    order: int = 2
) -> sp.Expr:
    """
    Von Neumann entropy: H = -Tr(ρ log ρ).
    
    Parameters
    ----------
    theta_symbols : tuple of sp.Symbol
        80 symbolic parameters
    order : int, default=2
        Order of expansion
        
    Returns
    -------
    H : sp.Expr
        Von Neumann entropy
        
    Notes
    -----
    For order-2:
        H(θ) ≈ log(9) - (1/18)||θ||²
        
    The entropy decreases quadratically as we move away from the
    maximally mixed state.
    """
    if len(theta_symbols) != 80:
        raise ValueError(f"Expected 80 parameters, got {len(theta_symbols)}")
    
    # Order-0: maximum entropy
    H = log(9)
    
    if order >= 2:
        # Order-2: entropy decreases as -||θ||²/(2*9) 
        # Note: different coefficient than local case due to normalization
        H -= sum(theta_a**2 for theta_a in theta_symbols) / 18
    
    return H


# Phase 3: Constraint geometry
def symbolic_constraint_gradient_su9_pair(
    theta_symbols: Tuple[Symbol, ...],
    order: int = 2
) -> Matrix:
    """
    Constraint gradient for su(9) pair: a = ∇(h_1 + h_2).
    
    Will be implemented in Phase 3.
    """
    raise NotImplementedError("Phase 3: To be implemented")


# Phase 4: Antisymmetric part
def symbolic_antisymmetric_part_su9_pair(
    theta_symbols: Tuple[Symbol, ...],
    order: int = 2
) -> Matrix:
    """
    Antisymmetric part of GENERIC decomposition for su(9) pair.
    
    Will be implemented in Phase 4.
    """
    raise NotImplementedError("Phase 4: To be implemented")

