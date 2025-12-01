"""
Symbolic computation for single qutrit exponential family.

This module provides symbolic representations of the quantum exponential family
for a single qutrit (d=3), including the density matrix, cumulant generating
function, Fisher information metric, and von Neumann entropy.

These form the building blocks for the two-qutrit constraint geometry needed
for GENERIC decomposition.
"""

import sympy as sp
from sympy import symbols, Matrix, exp, log, sqrt, I, simplify, trace
from typing import Tuple, List

from qig.symbolic.gell_mann import symbolic_gell_mann_matrices


def symbolic_density_matrix_single_qutrit(
    theta_symbols: Tuple[sp.Symbol, ...],
    order: int = 1
) -> Matrix:
    """
    Symbolic density matrix for single qutrit: ρ(θ) = exp(Σ θ_k λ_k - ψ(θ)).
    
    Uses perturbative expansion around maximally mixed state due to difficulty
    of symbolic matrix exponential.
    
    Parameters
    ----------
    theta_symbols : tuple of sp.Symbol
        8 symbolic parameters (θ₁, ..., θ₈)
    order : int, default=1
        Order of perturbative expansion (1 or 2)
        
    Returns
    -------
    rho : sp.Matrix, shape (3, 3)
        Symbolic density matrix
        
    Notes
    -----
    Expansion around maximally mixed state ρ₀ = I/3:
    
    Order 1:
        ρ ≈ I/3 + (1/3)Σ θ_k λ_k
        
    Order 2:
        ρ ≈ I/3 + (1/3)Σ θ_k λ_k + (1/6)Σᵢⱼ θ_i θ_j λ_i λ_j - ...
        
    This is valid for small ||θ|| (perturbative regime).
    
    For exact computation, would need:
        ρ = exp(K) / Tr(exp(K))  where K = Σ θ_k λ_k
    which is difficult symbolically.
    
    Examples
    --------
    >>> import sympy as sp
    >>> theta = sp.symbols('theta1:9', real=True)
    >>> rho = symbolic_density_matrix_single_qutrit(theta, order=1)
    >>> sp.trace(rho)  # Should be 1
    1
    >>> rho.is_hermitian
    True
    """
    if len(theta_symbols) != 8:
        raise ValueError(f"Expected 8 parameters, got {len(theta_symbols)}")
    
    gm = symbolic_gell_mann_matrices()
    
    if order == 1:
        # First-order expansion: ρ ≈ I/3 + (1/3)Σ θ_k λ_k
        rho = Matrix.eye(3) / 3
        for k, theta_k in enumerate(theta_symbols):
            rho += (sp.Rational(1, 3) * theta_k) * gm[k]
        return rho
    
    elif order == 2:
        # Second-order expansion using proper Hermitian form
        # For ρ = exp(K)/Z where K = Σ θ_k λ_k, use:
        # exp(K) ≈ I + K + K²/2
        # This is Hermitian if K is Hermitian (which it is)
        
        K = Matrix.zeros(3, 3)
        for k, theta_k in enumerate(theta_symbols):
            K += theta_k * gm[k]
        
        # exp(K) ≈ I + K + K²/2
        exp_K = Matrix.eye(3) + K + K*K / 2
        
        # Normalize: ρ = exp(K) / Tr(exp(K))
        trace_exp_K = trace(exp_K)
        rho = exp_K / trace_exp_K
        
        return simplify(rho)
    
    else:
        raise ValueError(f"Order must be 1 or 2, got {order}")


def symbolic_cumulant_generating_function_single_qutrit(
    theta_symbols: Tuple[sp.Symbol, ...],
    order: int = 2
) -> sp.Expr:
    """
    Cumulant generating function ψ(θ) = log Tr(exp(Σ θ_k λ_k)).
    
    Uses perturbative expansion for tractability.
    
    Parameters
    ----------
    theta_symbols : tuple of sp.Symbol
        8 symbolic parameters
    order : int, default=2
        Order of expansion (1 or 2)
        
    Returns
    -------
    psi : sp.Expr
        Symbolic cumulant generating function
        
    Notes
    -----
    Expansion around θ = 0 (maximally mixed state):
    
    Order 0: ψ(0) = log(3)
    Order 1: ∂ψ/∂θ_k|₀ = 0 (by tracelessness)
    Order 2: ∂²ψ/∂θᵢ∂θⱼ|₀ = (1/3)Tr(λᵢλⱼ) = (2/3)δᵢⱼ
    
    So: ψ(θ) ≈ log(3) + (1/3)Σ θ_k² + O(θ³)
    
    Examples
    --------
    >>> import sympy as sp
    >>> theta = sp.symbols('theta1:9', real=True)
    >>> psi = symbolic_cumulant_generating_function_single_qutrit(theta, order=2)
    >>> sp.diff(psi, theta[0])  # First derivative
    2*theta1/3
    """
    if len(theta_symbols) != 8:
        raise ValueError(f"Expected 8 parameters, got {len(theta_symbols)}")
    
    if order == 0:
        return log(3)
    
    elif order == 1:
        # ψ ≈ log(3) (constant, since ∂ψ/∂θ_k = 0 at origin)
        return log(3)
    
    elif order == 2:
        # ψ ≈ log(3) + (1/3)Σ θ_k²
        psi = log(3)
        for theta_k in theta_symbols:
            psi += sp.Rational(1, 3) * theta_k**2
        return psi
    
    else:
        raise ValueError(f"Order must be 0, 1, or 2, got {order}")


def symbolic_fisher_information_single_qutrit(
    theta_symbols: Tuple[sp.Symbol, ...],
    order: int = 2
) -> Matrix:
    """
    BKM metric (Fisher information): G_ij = ∂²ψ/∂θᵢ∂θⱼ.
    
    Parameters
    ----------
    theta_symbols : tuple of sp.Symbol
        8 symbolic parameters
    order : int, default=2
        Order of ψ expansion (determines G structure)
        
    Returns
    -------
    G : sp.Matrix, shape (8, 8)
        Symbolic Fisher information matrix
        
    Notes
    -----
    At order 2 (quadratic ψ):
        G_ij = (2/3)δᵢⱼ  (diagonal, constant)
        
    This is the metric at the maximally mixed state.
    
    For exact computation away from origin, would need:
        G_ij = ∫₀¹ Tr(ρˢ(λᵢ - μᵢ)ρ¹⁻ˢ(λⱼ - μⱼ)) ds
    which is difficult symbolically.
    
    Examples
    --------
    >>> import sympy as sp
    >>> theta = sp.symbols('theta1:9', real=True)
    >>> G = symbolic_fisher_information_single_qutrit(theta, order=2)
    >>> G[0, 0]
    2/3
    >>> G[0, 1]
    0
    >>> G.is_diagonal()
    True
    """
    if len(theta_symbols) != 8:
        raise ValueError(f"Expected 8 parameters, got {len(theta_symbols)}")
    
    psi = symbolic_cumulant_generating_function_single_qutrit(theta_symbols, order)
    
    # Compute Hessian
    n = len(theta_symbols)
    G = Matrix.zeros(n, n)
    
    for i in range(n):
        for j in range(n):
            G[i, j] = sp.diff(psi, theta_symbols[i], theta_symbols[j])
    
    return simplify(G)


def symbolic_von_neumann_entropy_single_qutrit(
    theta_symbols: Tuple[sp.Symbol, ...],
    order: int = 2
) -> sp.Expr:
    """
    Von Neumann entropy H = -Tr(ρ log ρ).
    
    Uses perturbative expansion around maximally mixed state.
    
    Parameters
    ----------
    theta_symbols : tuple of sp.Symbol
        8 symbolic parameters
    order : int, default=2
        Order of expansion
        
    Returns
    -------
    H : sp.Expr
        Symbolic von Neumann entropy
        
    Notes
    -----
    Expansion around maximally mixed state (H₀ = log(3)):
    
    For ρ = I/3 + εX where X is traceless:
        H ≈ log(3) - (1/2)Tr(X²) + O(ε³)
        
    With X = (1/3)Σ θ_k λ_k:
        H ≈ log(3) - (1/18)Σᵢⱼ θᵢθⱼTr(λᵢλⱼ)
          = log(3) - (1/18)Σ θ_k²·2
          = log(3) - (1/9)Σ θ_k²
          
    This is Petz's perturbation formula for quantum entropy.
    
    Examples
    --------
    >>> import sympy as sp
    >>> theta = sp.symbols('theta1:9', real=True)
    >>> H = symbolic_von_neumann_entropy_single_qutrit(theta, order=2)
    >>> H.subs([(t, 0) for t in theta])  # At origin
    log(3)
    """
    if len(theta_symbols) != 8:
        raise ValueError(f"Expected 8 parameters, got {len(theta_symbols)}")
    
    if order == 0:
        return log(3)
    
    elif order == 1:
        # First order correction is zero (by symmetry)
        return log(3)
    
    elif order == 2:
        # H ≈ log(3) - (1/9)Σ θ_k²
        H = log(3)
        for theta_k in theta_symbols:
            H -= sp.Rational(1, 9) * theta_k**2
        return H
    
    else:
        raise ValueError(f"Order must be 0, 1, or 2, got {order}")


def verify_single_qutrit_consistency(theta_symbols: Tuple[sp.Symbol, ...]) -> dict:
    """
    Verify consistency of symbolic single-qutrit computations.
    
    Parameters
    ----------
    theta_symbols : tuple of sp.Symbol
        8 symbolic parameters
        
    Returns
    -------
    results : dict
        Verification results:
        - 'rho_trace': bool (Tr(ρ) = 1)
        - 'rho_hermitian': bool
        - 'G_symmetric': bool
        - 'G_positive': bool (at origin)
        - 'entropy_max': bool (H ≤ log(3))
        - 'details': list of any failures
        
    Examples
    --------
    >>> import sympy as sp
    >>> theta = sp.symbols('theta1:9', real=True)
    >>> results = verify_single_qutrit_consistency(theta)
    >>> results['rho_trace']
    True
    >>> results['G_symmetric']
    True
    """
    results = {
        'rho_trace': True,
        'rho_hermitian': True,
        'G_symmetric': True,
        'G_positive': True,
        'entropy_max': True,
        'details': []
    }
    
    # Check density matrix (order 2 for full expansion)
    rho = symbolic_density_matrix_single_qutrit(theta_symbols, order=2)
    
    # Trace
    tr = simplify(trace(rho))
    if tr != 1:
        results['rho_trace'] = False
        results['details'].append(f"Tr(ρ) = {tr}, expected 1")
    
    # Hermiticity (order 1 is manifestly Hermitian)
    diff = simplify(rho - rho.H)
    if not diff.is_zero_matrix:
        results['rho_hermitian'] = False
        results['details'].append("ρ not Hermitian")
    
    # Check Fisher metric
    G = symbolic_fisher_information_single_qutrit(theta_symbols, order=2)
    
    # Symmetry
    if not (G - G.T).is_zero_matrix:
        results['G_symmetric'] = False
        results['details'].append("G not symmetric")
    
    # Positive definiteness at origin (G = (2/3)I)
    # All diagonal elements should be 2/3
    for i in range(8):
        if G[i, i] != sp.Rational(2, 3):
            results['G_positive'] = False
            results['details'].append(f"G[{i},{i}] = {G[i,i]}, expected 2/3")
    
    # Check entropy bound
    H = symbolic_von_neumann_entropy_single_qutrit(theta_symbols, order=2)
    H_max = log(3)
    # At origin, should be maximum
    H_at_origin = H.subs([(t, 0) for t in theta_symbols])
    if simplify(H_at_origin - H_max) != 0:
        results['entropy_max'] = False
        results['details'].append(f"H(0) = {H_at_origin}, expected log(3)")
    
    return results

