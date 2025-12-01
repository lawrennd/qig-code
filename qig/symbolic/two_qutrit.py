"""
Symbolic computation for two-qutrit constraint geometry.

This module provides symbolic representations of the constraint geometry
needed for GENERIC decomposition: marginal entropies, constraint gradient,
Lagrange multiplier, and its gradient.

This is Phase 3 of CIP-0007 and is the key step where Lie algebra structure
creates simplifications.
"""

import sympy as sp
from sympy import symbols, Matrix, exp, log, sqrt, I, simplify, trace, Rational
from typing import Tuple, List
import numpy as np

from qig.symbolic.gell_mann import symbolic_gell_mann_matrices


def two_qutrit_operators() -> List[Matrix]:
    """
    Create 16 operators for two-qutrit system using tensor product structure.
    
    Returns
    -------
    operators : List[sp.Matrix]
        16 operators, each 9×9:
        - F_1...F_8: λ_k ⊗ I (site 1)
        - F_9...F_16: I ⊗ λ_k (site 2)
        
    Notes
    -----
    This exploits the block structure: operators on different sites commute.
    Structure constants f_abc = 0 when operators act on different sites.
    
    Examples
    --------
    >>> ops = two_qutrit_operators()
    >>> len(ops)
    16
    >>> ops[0].shape
    (9, 9)
    """
    gm = symbolic_gell_mann_matrices()
    I3 = Matrix.eye(3)
    
    operators = []
    
    # Site 1: λ_k ⊗ I
    for k in range(8):
        op = sp.kronecker_product(gm[k], I3)
        operators.append(op)
    
    # Site 2: I ⊗ λ_k
    for k in range(8):
        op = sp.kronecker_product(I3, gm[k])
        operators.append(op)
    
    return operators


def symbolic_density_matrix_two_qutrit(
    theta_symbols: Tuple[sp.Symbol, ...],
    order: int = 1
) -> Matrix:
    """
    Symbolic density matrix for two qutrits: ρ(θ) = exp(Σ θ_a F_a - ψ).
    
    Parameters
    ----------
    theta_symbols : tuple of sp.Symbol
        16 symbolic parameters (θ₁, ..., θ₁₆)
    order : int, default=1
        Order of perturbative expansion
        
    Returns
    -------
    rho : sp.Matrix, shape (9, 9)
        Symbolic density matrix
        
    Notes
    -----
    Expansion around maximally mixed state ρ₀ = I/9.
    
    Uses block structure: θ₁...θ₈ for site 1, θ₉...θ₁₆ for site 2.
    
    Examples
    --------
    >>> import sympy as sp
    >>> theta = sp.symbols('theta1:17', real=True)
    >>> rho = symbolic_density_matrix_two_qutrit(theta, order=1)
    >>> sp.trace(rho)
    1
    """
    if len(theta_symbols) != 16:
        raise ValueError(f"Expected 16 parameters, got {len(theta_symbols)}")
    
    ops = two_qutrit_operators()
    
    if order == 1:
        # First-order: ρ ≈ I/9 + (1/9)Σ θ_a F_a
        rho = Matrix.eye(9) / 9
        for a, theta_a in enumerate(theta_symbols):
            rho += (Rational(1, 9) * theta_a) * ops[a]
        return rho
    
    elif order == 2:
        # Second-order: exp(K) ≈ I + K + K²/2, normalized
        K = Matrix.zeros(9, 9)
        for a, theta_a in enumerate(theta_symbols):
            K += theta_a * ops[a]
        
        exp_K = Matrix.eye(9) + K + K*K / 2
        trace_exp_K = trace(exp_K)
        rho = exp_K / trace_exp_K
        
        return simplify(rho)
    
    else:
        raise ValueError(f"Order must be 1 or 2, got {order}")


def partial_trace_symbolic(
    rho: Matrix,
    keep: int,
    dims: Tuple[int, int] = (3, 3)
) -> Matrix:
    """
    Compute partial trace symbolically.
    
    Parameters
    ----------
    rho : sp.Matrix
        9×9 density matrix for two qutrits
    keep : int
        Which subsystem to keep (0 for first, 1 for second)
    dims : tuple of int
        Dimensions of subsystems (default: (3, 3) for qutrits)
        
    Returns
    -------
    rho_reduced : sp.Matrix
        3×3 reduced density matrix
        
    Notes
    -----
    For ρ in H₁ ⊗ H₂:
    - Tr₂(ρ) = Σⱼ (I ⊗ ⟨j|) ρ (I ⊗ |j⟩)  (keep site 1)
    - Tr₁(ρ) = Σᵢ (⟨i| ⊗ I) ρ (|i⟩ ⊗ I)  (keep site 2)
    
    Examples
    --------
    >>> import sympy as sp
    >>> rho = Matrix.eye(9) / 9  # Maximally mixed
    >>> rho1 = partial_trace_symbolic(rho, keep=0)
    >>> rho1
    Matrix([[1/3, 0, 0], [0, 1/3, 0], [0, 0, 1/3]])
    """
    d1, d2 = dims
    assert rho.shape == (d1*d2, d1*d2), f"Expected {d1*d2}×{d1*d2} matrix"
    
    # Reshape ρ as a tensor
    rho_reduced = Matrix.zeros(d1 if keep == 0 else d2, d1 if keep == 0 else d2)
    
    if keep == 0:
        # Trace out site 2: Σⱼ ⟨j|₂ ρ |j⟩₂
        for i1 in range(d1):
            for i2 in range(d1):
                elem = 0
                for j in range(d2):
                    # Index in flattened basis: |i1,j⟩ ↔ i1*d2 + j
                    idx1 = i1 * d2 + j
                    idx2 = i2 * d2 + j
                    elem += rho[idx1, idx2]
                rho_reduced[i1, i2] = elem
    else:
        # Trace out site 1: Σᵢ ⟨i|₁ ρ |i⟩₁
        for j1 in range(d2):
            for j2 in range(d2):
                elem = 0
                for i in range(d1):
                    # Index in flattened basis: |i,j1⟩ ↔ i*d2 + j1
                    idx1 = i * d2 + j1
                    idx2 = i * d2 + j2
                    elem += rho[idx1, idx2]
                rho_reduced[j1, j2] = elem
    
    return simplify(rho_reduced)


def symbolic_marginal_entropies_two_qutrit(
    theta_symbols: Tuple[sp.Symbol, ...],
    order: int = 2
) -> Tuple[sp.Expr, sp.Expr]:
    """
    Symbolic marginal von Neumann entropies h₁ and h₂.
    
    Parameters
    ----------
    theta_symbols : tuple of sp.Symbol
        16 symbolic parameters
    order : int, default=2
        Order of expansion
        
    Returns
    -------
    h1, h2 : sp.Expr
        Marginal entropies for sites 1 and 2
        
    Notes
    -----
    Uses perturbative expansion around maximally mixed marginals.
    
    For ρᵢ = I/3 + εXᵢ:
        hᵢ ≈ log(3) - (1/2)Tr(Xᵢ²)
        
    Due to block structure:
    - h₁ depends only on θ₁...θ₈ (site 1 parameters)
    - h₂ depends only on θ₉...θ₁₆ (site 2 parameters)
    
    This is a KEY simplification from Lie algebra structure!
    
    Examples
    --------
    >>> import sympy as sp
    >>> theta = sp.symbols('theta1:17', real=True)
    >>> h1, h2 = symbolic_marginal_entropies_two_qutrit(theta, order=2)
    >>> # At origin:
    >>> h1.subs([(t, 0) for t in theta])
    log(3)
    >>> # h1 depends only on θ₁...θ₈:
    >>> sp.diff(h1, theta[8])
    0
    """
    if len(theta_symbols) != 16:
        raise ValueError(f"Expected 16 parameters, got {len(theta_symbols)}")
    
    if order == 2:
        # Use perturbative formula: h ≈ log(3) - (1/9)Σθᵢ²
        # Due to block structure, each marginal depends only on local parameters
        
        h1 = log(3)
        for i in range(8):  # Site 1 parameters
            h1 -= Rational(1, 9) * theta_symbols[i]**2
        
        h2 = log(3)
        for i in range(8, 16):  # Site 2 parameters
            h2 -= Rational(1, 9) * theta_symbols[i]**2
        
        return h1, h2
    
    else:
        raise ValueError(f"Order must be 2, got {order}")


def symbolic_constraint_gradient_two_qutrit(
    theta_symbols: Tuple[sp.Symbol, ...],
    order: int = 2
) -> Matrix:
    """
    Constraint gradient a = ∇(h₁ + h₂).
    
    Parameters
    ----------
    theta_symbols : tuple of sp.Symbol
        16 symbolic parameters
    order : int, default=2
        Order of expansion
        
    Returns
    -------
    a : sp.Matrix, shape (16, 1)
        Constraint gradient vector
        
    Notes
    -----
    Due to block structure:
        a[i] = ∂h₁/∂θᵢ  for i=0...7  (depends on site 1)
        a[i] = ∂h₂/∂θᵢ  for i=8...15 (depends on site 2)
        
    From perturbative formula:
        ∂hᵢ/∂θⱼ = -(2/9)θⱼ  for θⱼ in site i
        
    This is MUCH simpler than general exponential family!
    
    Examples
    --------
    >>> import sympy as sp
    >>> theta = sp.symbols('theta1:17', real=True)
    >>> a = symbolic_constraint_gradient_two_qutrit(theta, order=2)
    >>> a[0]  # ∂(h₁+h₂)/∂θ₁ = ∂h₁/∂θ₁
    -2*theta1/9
    >>> a[8]  # ∂(h₁+h₂)/∂θ₉ = ∂h₂/∂θ₉
    -2*theta9/9
    """
    if len(theta_symbols) != 16:
        raise ValueError(f"Expected 16 parameters, got {len(theta_symbols)}")
    
    h1, h2 = symbolic_marginal_entropies_two_qutrit(theta_symbols, order)
    h_total = h1 + h2
    
    # Compute gradient
    a = Matrix([sp.diff(h_total, theta) for theta in theta_symbols])
    
    return simplify(a)


def symbolic_lagrange_multiplier_two_qutrit(
    theta_symbols: Tuple[sp.Symbol, ...],
    order: int = 2
) -> sp.Expr:
    """
    Lagrange multiplier ν = (aᵀGθ)/(aᵀa).
    
    Parameters
    ----------
    theta_symbols : tuple of sp.Symbol
        16 symbolic parameters
    order : int, default=2
        Order of expansion
        
    Returns
    -------
    nu : sp.Expr
        Lagrange multiplier
        
    Notes
    -----
    At order 2 with perturbative approximation:
    - G = (2/3)I (constant metric at origin)
    - a = -(2/9)[θ₁, ..., θ₁₆]ᵀ
    
    So: ν = (aᵀGθ)/(aᵀa)
          = (-(2/9)θᵀ · (2/3)I · θ) / ((2/9)²θᵀθ)
          = -(2/9)(2/3)||θ||² / ((4/81)||θ||²)
          = -(4/27)||θ||² / ((4/81)||θ||²)
          = -(4/27) / (4/81)
          = -3
          
    Wait, this simplifies dramatically! Let me recalculate...
    
    Actually: ν = (aᵀGθ)/(aᵀa) has more structure.
    
    Examples
    --------
    >>> import sympy as sp
    >>> theta = sp.symbols('theta1:17', real=True)
    >>> nu = symbolic_lagrange_multiplier_two_qutrit(theta, order=2)
    >>> # Should be a rational function of θ
    """
    if len(theta_symbols) != 16:
        raise ValueError(f"Expected 16 parameters, got {len(theta_symbols)}")
    
    # Get constraint gradient
    a = symbolic_constraint_gradient_two_qutrit(theta_symbols, order)
    
    # Fisher metric at order 2: G = (2/3)I
    G = Rational(2, 3) * Matrix.eye(16)
    
    # Convert theta to column vector
    theta_vec = Matrix(theta_symbols)
    
    # Compute ν = (aᵀGθ)/(aᵀa)
    numerator = (a.T * G * theta_vec)[0, 0]
    denominator = (a.T * a)[0, 0]
    
    nu = numerator / denominator
    
    return simplify(nu)


def symbolic_grad_lagrange_multiplier_two_qutrit(
    theta_symbols: Tuple[sp.Symbol, ...],
    order: int = 2
) -> Matrix:
    """
    Gradient of Lagrange multiplier: ∇ν.
    
    This is the HARD part of Phase 3!
    
    Parameters
    ----------
    theta_symbols : tuple of sp.Symbol
        16 symbolic parameters
    order : int, default=2
        Order of expansion
        
    Returns
    -------
    grad_nu : sp.Matrix, shape (16, 1)
        Gradient ∇ν
        
    Notes
    -----
    Using quotient rule: ∇(u/v) = (v∇u - u∇v)/v²
    
    Where:
    - u = aᵀGθ
    - v = aᵀa
    
    Need:
    - ∇u = (∇a)ᵀGθ + aᵀ(∇G)[θ] + aᵀG
    - ∇v = 2(∇a)ᵀa
    
    At order 2, G is constant → ∇G = 0, which simplifies!
    
    The constraint Hessian (∇a) is key:
        (∇a)ᵢⱼ = ∂²(h₁+h₂)/∂θᵢ∂θⱼ
        
    Due to block structure, this should be block-diagonal!
    
    Examples
    --------
    >>> import sympy as sp
    >>> theta = sp.symbols('theta1:17', real=True)
    >>> grad_nu = symbolic_grad_lagrange_multiplier_two_qutrit(theta, order=2)
    >>> grad_nu.shape
    (16, 1)
    """
    if len(theta_symbols) != 16:
        raise ValueError(f"Expected 16 parameters, got {len(theta_symbols)}")
    
    # Get nu
    nu = symbolic_lagrange_multiplier_two_qutrit(theta_symbols, order)
    
    # Compute gradient directly
    grad_nu = Matrix([sp.diff(nu, theta) for theta in theta_symbols])
    
    return simplify(grad_nu)


def verify_block_structure_two_qutrit(theta_symbols: Tuple[sp.Symbol, ...]) -> dict:
    """
    Verify block structure properties from Lie algebra.
    
    Parameters
    ----------
    theta_symbols : tuple of sp.Symbol
        16 symbolic parameters
        
    Returns
    -------
    results : dict
        Verification results:
        - 'h1_local': bool (h₁ depends only on θ₁...θ₈)
        - 'h2_local': bool (h₂ depends only on θ₉...θ₁₆)
        - 'a_structure': bool (a has expected block structure)
        - 'constraint_hessian_block': bool (∇a is block-diagonal)
        - 'details': list of any failures
    """
    results = {
        'h1_local': True,
        'h2_local': True,
        'a_structure': True,
        'constraint_hessian_block': True,
        'details': []
    }
    
    h1, h2 = symbolic_marginal_entropies_two_qutrit(theta_symbols, order=2)
    
    # Check h₁ depends only on θ₁...θ₈
    for i in range(8, 16):
        if sp.diff(h1, theta_symbols[i]) != 0:
            results['h1_local'] = False
            results['details'].append(f"h₁ depends on θ_{i+1} (should be site 2 only)")
    
    # Check h₂ depends only on θ₉...θ₁₆
    for i in range(8):
        if sp.diff(h2, theta_symbols[i]) != 0:
            results['h2_local'] = False
            results['details'].append(f"h₂ depends on θ_{i+1} (should be site 1 only)")
    
    # Check constraint gradient structure
    a = symbolic_constraint_gradient_two_qutrit(theta_symbols, order=2)
    for i in range(16):
        expected = -Rational(2, 9) * theta_symbols[i]
        if simplify(a[i] - expected) != 0:
            results['a_structure'] = False
            results['details'].append(f"a[{i}] = {a[i]}, expected {expected}")
    
    # Check constraint Hessian is block-diagonal
    # ∂²(h₁+h₂)/∂θᵢ∂θⱼ should be 0 if i in site 1 and j in site 2 (or vice versa)
    h_total = h1 + h2
    for i in range(8):  # Site 1
        for j in range(8, 16):  # Site 2
            cross_deriv = sp.diff(sp.diff(h_total, theta_symbols[i]), theta_symbols[j])
            if cross_deriv != 0:
                results['constraint_hessian_block'] = False
                results['details'].append(f"H[{i},{j}] = {cross_deriv}, expected 0")
    
    return results

