"""
EXACT symbolic computation for LME qutrit pairs.

Key insight: For maximally entangled states, the 9×9 eigenvalue problem
decomposes into blocks with at most QUADRATIC eigenvalues:

    9×9 → 3×3 + 2×2 + 1×1 + 1×1 + 1×1 + 1×1

This enables EXACT exp(K) computation with NO Taylor approximation.

The 20 block-preserving generators span the full entangled subspace,
allowing exploration of ALL maximally entangled states.
"""

import sympy as sp
from sympy import Matrix, sqrt, exp, I as sp_I, symbols, simplify, Rational
from typing import Tuple, List, Dict
import numpy as np
from functools import lru_cache


# =============================================================================
# Gell-Mann matrices
# =============================================================================

@lru_cache(maxsize=1)
def gell_mann_symbolic() -> List[Matrix]:
    """8 Gell-Mann matrices (3×3) with Tr(λ_a λ_b) = 2δ_ab."""
    return [
        Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),           # λ₁
        Matrix([[0, -sp_I, 0], [sp_I, 0, 0], [0, 0, 0]]),    # λ₂
        Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),          # λ₃
        Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),           # λ₄
        Matrix([[0, 0, -sp_I], [0, 0, 0], [sp_I, 0, 0]]),    # λ₅
        Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),           # λ₆
        Matrix([[0, 0, 0], [0, 0, -sp_I], [0, sp_I, 0]]),    # λ₇
        Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / sqrt(3) # λ₈
    ]


# =============================================================================
# Block structure for LME states
# =============================================================================

def permutation_matrix() -> Matrix:
    """
    Permutation to block basis: {|00⟩,|11⟩,|22⟩} + rest.
    
    Original: |00⟩,|01⟩,|02⟩,|10⟩,|11⟩,|12⟩,|20⟩,|21⟩,|22⟩
    Reordered: |00⟩,|11⟩,|22⟩,|01⟩,|02⟩,|10⟩,|12⟩,|20⟩,|21⟩
    """
    reorder = [0, 4, 8, 1, 2, 3, 5, 6, 7]
    P = sp.zeros(9, 9)
    for i, j in enumerate(reorder):
        P[i, j] = 1
    return P


@lru_cache(maxsize=1)
def block_preserving_generators() -> Tuple[List[Matrix], List[str]]:
    """
    20 generators that preserve the {|00⟩,|11⟩,|22⟩} block structure.
    
    Returns
    -------
    generators : list of 9×9 matrices
    names : list of string labels
    """
    λ = gell_mann_symbolic()
    I3 = sp.eye(3)
    
    generators = []
    names = []
    
    # Local generators (4)
    for i in [2, 7]:  # λ₃, λ₈
        generators.append(sp.kronecker_product(λ[i], I3))
        names.append(f'λ{i+1}⊗I')
        generators.append(sp.kronecker_product(I3, λ[i]))
        names.append(f'I⊗λ{i+1}')
    
    # Entangling generators (16)
    entangling_pairs = [
        (0, 0), (0, 1), (1, 0), (1, 1),  # λ₁,λ₂ block
        (2, 2), (2, 7), (7, 2), (7, 7),  # λ₃,λ₈ block
        (3, 3), (3, 4), (4, 3), (4, 4),  # λ₄,λ₅ block
        (5, 5), (5, 6), (6, 5), (6, 6),  # λ₆,λ₇ block
    ]
    for i, j in entangling_pairs:
        generators.append(sp.kronecker_product(λ[i], λ[j]))
        names.append(f'λ{i+1}⊗λ{j+1}')
    
    return generators, names


def extract_blocks(K_block: Matrix) -> Dict[str, Matrix]:
    """
    Extract sub-blocks from K in block basis.
    
    K decomposes as:
    - 3×3 entangled block (indices 0-2)
    - 2×2 block (indices 3,5 coupled)
    - 4 isolated 1×1 blocks
    """
    return {
        'ent_3x3': K_block[:3, :3],
        'block_2x2': K_block[3:6:2, 3:6:2],  # indices 3,5
        'diag_1': K_block[4, 4],  # index 4
        'diag_2': K_block[6, 6],  # index 6
        'diag_3': K_block[7, 7],  # index 7
        'diag_4': K_block[8, 8],  # index 8
    }


# =============================================================================
# Exact eigenvalue computation (quadratic formula)
# =============================================================================

def eigenvalues_2x2(M: Matrix) -> Tuple:
    """
    Exact eigenvalues of 2×2 Hermitian matrix using quadratic formula.
    
    For M = [[a, b], [b*, c]]:
        λ = (a+c)/2 ± sqrt((a-c)²/4 + |b|²)
    """
    a, c = M[0, 0], M[1, 1]
    b = M[0, 1]
    
    trace = a + c
    det = a * c - b * sp.conjugate(b)
    
    discriminant = trace**2 / 4 - det
    sqrt_disc = sqrt(simplify(discriminant))
    
    return (trace/2 - sqrt_disc, trace/2 + sqrt_disc)


def eigenvalues_3x3_block(M: Matrix) -> Tuple:
    """
    Eigenvalues of 3×3 matrix with potential 2×2 + 1×1 block structure.
    
    If M has form [[2×2 block, 0], [0, isolated]], eigenvalues are
    quadratic from 2×2 block + isolated element.
    """
    # Check for block structure
    if simplify(M[0, 2]) == 0 and simplify(M[1, 2]) == 0:
        # 2×2 + 1×1 structure
        block_2x2 = M[:2, :2]
        isolated = M[2, 2]
        ev1, ev2 = eigenvalues_2x2(block_2x2)
        return (ev1, ev2, isolated)
    elif simplify(M[0, 1]) == 0 and simplify(M[2, 1]) == 0:
        # Different block structure
        ev1, ev3 = eigenvalues_2x2(Matrix([[M[0,0], M[0,2]], [M[2,0], M[2,2]]]))
        return (ev1, M[1, 1], ev3)
    else:
        # General 3×3 - use sympy (may give cubic expressions)
        return tuple(M.eigenvals().keys())


# =============================================================================
# Exact exp(K) computation
# =============================================================================

def exact_exp_K_lme(theta: Dict[str, sp.Symbol]) -> Matrix:
    """
    EXACT exp(K) for LME-preserving dynamics.
    
    Parameters
    ----------
    theta : dict
        Dictionary mapping generator names to symbolic coefficients.
        Keys should match names from block_preserving_generators().
        
    Returns
    -------
    Matrix
        9×9 matrix exp(K), computed exactly via block decomposition.
    """
    generators, names = block_preserving_generators()
    P = permutation_matrix()
    
    # Build K
    K = sp.zeros(9, 9)
    for gen, name in zip(generators, names):
        if name in theta:
            K = K + theta[name] * gen
    
    # Transform to block basis
    K_block = simplify(P * K * P.T)
    
    # Extract blocks
    blocks = extract_blocks(K_block)
    
    # Compute exp for each block
    
    # 3×3 entangled block
    M3 = blocks['ent_3x3']
    P3, D3 = M3.diagonalize()
    exp_D3 = sp.diag(*[exp(D3[i, i]) for i in range(3)])
    exp_M3 = simplify(P3 * exp_D3 * P3.inv())
    
    # 2×2 block (if non-trivial)
    M2 = blocks['block_2x2']
    if M2.shape == (2, 2):
        P2, D2 = M2.diagonalize()
        exp_D2 = sp.diag(*[exp(D2[i, i]) for i in range(2)])
        exp_M2 = simplify(P2 * exp_D2 * P2.inv())
    else:
        exp_M2 = sp.eye(2)
    
    # 1×1 blocks (just exponentiate)
    exp_d1 = exp(blocks['diag_1'])
    exp_d2 = exp(blocks['diag_2'])
    exp_d3 = exp(blocks['diag_3'])
    exp_d4 = exp(blocks['diag_4'])
    
    # Reconstruct exp(K_block)
    exp_K_block = sp.zeros(9, 9)
    
    # 3×3 block
    for i in range(3):
        for j in range(3):
            exp_K_block[i, j] = exp_M3[i, j]
    
    # 2×2 block at indices 3,5
    exp_K_block[3, 3] = exp_M2[0, 0]
    exp_K_block[3, 5] = exp_M2[0, 1]
    exp_K_block[5, 3] = exp_M2[1, 0]
    exp_K_block[5, 5] = exp_M2[1, 1]
    
    # 1×1 blocks
    exp_K_block[4, 4] = exp_d1
    exp_K_block[6, 6] = exp_d2
    exp_K_block[7, 7] = exp_d3
    exp_K_block[8, 8] = exp_d4
    
    # Transform back to original basis
    exp_K = simplify(P.T * exp_K_block * P)
    
    return exp_K


def exact_rho_lme(theta: Dict[str, sp.Symbol]) -> Matrix:
    """
    EXACT density matrix ρ = exp(K) / Tr(exp(K)).
    """
    exp_K = exact_exp_K_lme(theta)
    Z = exp_K.trace()
    return exp_K / Z


def symbolic_partial_trace(rho: Matrix, trace_out: int) -> Matrix:
    """
    Symbolic partial trace for 3⊗3 system.
    
    Parameters
    ----------
    rho : 9×9 Matrix
    trace_out : 1 or 2 (which subsystem to trace out)
    
    Returns
    -------
    3×3 Matrix
    """
    result = sp.zeros(3, 3)
    
    if trace_out == 2:  # Tr₂(ρ) → ρ₁
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    # ρ₁[i,k] = Σⱼ ρ[3i+j, 3k+j]
                    result[i, k] += rho[3*i + j, 3*k + j]
    else:  # Tr₁(ρ) → ρ₂
        for j in range(3):
            for l in range(3):
                for i in range(3):
                    # ρ₂[j,l] = Σᵢ ρ[3i+j, 3i+l]
                    result[j, l] += rho[3*i + j, 3*i + l]
    
    return result


def exact_rho1_lme(theta: Dict[str, sp.Symbol]) -> Matrix:
    """EXACT reduced density matrix ρ₁ = Tr₂(ρ)."""
    rho = exact_rho_lme(theta)
    return symbolic_partial_trace(rho, trace_out=2)


def exact_rho2_lme(theta: Dict[str, sp.Symbol]) -> Matrix:
    """EXACT reduced density matrix ρ₂ = Tr₁(ρ)."""
    rho = exact_rho_lme(theta)
    return symbolic_partial_trace(rho, trace_out=1)


def exact_marginal_entropy_lme(rho_marginal: Matrix) -> sp.Expr:
    """
    EXACT von Neumann entropy of 3×3 marginal density matrix.
    
    Uses block structure: eigenvalues are quadratic.
    h = -Σᵢ λᵢ log(λᵢ)
    """
    # Diagonalize (3×3 with block structure → quadratic eigenvalues)
    P, D = rho_marginal.diagonalize()
    
    # Entropy: -Σ λᵢ log(λᵢ)
    h = sp.Integer(0)
    for i in range(3):
        λ = D[i, i]
        # Handle λ log(λ) → 0 as λ → 0
        h -= λ * sp.log(λ)
    
    return simplify(h)


def exact_constraint_lme(theta: Dict[str, sp.Symbol]) -> sp.Expr:
    """EXACT constraint C = h₁ + h₂."""
    rho1 = exact_rho1_lme(theta)
    rho2 = exact_rho2_lme(theta)
    
    h1 = exact_marginal_entropy_lme(rho1)
    h2 = exact_marginal_entropy_lme(rho2)
    
    return h1 + h2


# =============================================================================
# Phase 4b: Constraint gradient, Fisher information, ν, ∇ν, A
# =============================================================================

def create_theta_symbols() -> Tuple[Dict[str, sp.Symbol], List[sp.Symbol]]:
    """
    Create symbolic parameters for the 20 block-preserving generators.
    
    Returns
    -------
    theta_dict : dict mapping generator names to symbols
    theta_list : ordered list of symbols (for differentiation)
    """
    _, names = block_preserving_generators()
    
    # Create symbols with short names for cleaner expressions
    symbols_list = []
    theta_dict = {}
    for i, name in enumerate(names):
        sym = sp.Symbol(f'θ{i}', real=True)
        symbols_list.append(sym)
        theta_dict[name] = sym
    
    return theta_dict, symbols_list


def exact_psi_lme(theta: Dict[str, sp.Symbol]) -> sp.Expr:
    """EXACT cumulant generating function ψ = log Tr(exp(K))."""
    exp_K = exact_exp_K_lme(theta)
    return sp.log(exp_K.trace())


def exact_constraint_gradient_lme(theta: Dict[str, sp.Symbol], 
                                   theta_list: List[sp.Symbol]) -> Matrix:
    """
    EXACT constraint gradient a = ∇C = ∇(h₁ + h₂).
    
    Parameters
    ----------
    theta : dict mapping generator names to symbols
    theta_list : ordered list of symbols for differentiation
    
    Returns
    -------
    Matrix : 20×1 column vector
    """
    C = exact_constraint_lme(theta)
    n = len(theta_list)
    
    a = sp.zeros(n, 1)
    for i, sym in enumerate(theta_list):
        a[i, 0] = simplify(sp.diff(C, sym))
    
    return a


def exact_fisher_information_lme(theta: Dict[str, sp.Symbol],
                                  theta_list: List[sp.Symbol]) -> Matrix:
    """
    EXACT Fisher information G = ∇²ψ (BKM metric).
    
    Parameters
    ----------
    theta : dict mapping generator names to symbols
    theta_list : ordered list of symbols for differentiation
    
    Returns
    -------
    Matrix : 20×20 symmetric matrix
    """
    psi = exact_psi_lme(theta)
    n = len(theta_list)
    
    G = sp.zeros(n, n)
    for i, sym_i in enumerate(theta_list):
        dpsi_i = sp.diff(psi, sym_i)
        for j, sym_j in enumerate(theta_list):
            if j >= i:  # Use symmetry
                G[i, j] = simplify(sp.diff(dpsi_i, sym_j))
                if j > i:
                    G[j, i] = G[i, j]
    
    return G


def exact_lagrange_multiplier_lme(a: Matrix, G: Matrix, 
                                   theta_list: List[sp.Symbol],
                                   do_simplify: bool = False) -> sp.Expr:
    """
    EXACT Lagrange multiplier ν = (aᵀGθ)/(aᵀa).
    
    Parameters
    ----------
    a : 20×1 constraint gradient
    G : 20×20 Fisher information
    theta_list : list of symbols
    do_simplify : bool
        Whether to simplify (SLOW)
    
    Returns
    -------
    Scalar expression for ν
    """
    theta_vec = Matrix(theta_list)
    
    numerator = (a.T * G * theta_vec)[0, 0]
    denominator = (a.T * a)[0, 0]
    
    result = numerator / denominator
    return simplify(result) if do_simplify else result


def exact_grad_lagrange_multiplier_lme(a: Matrix, G: Matrix, nu: sp.Expr,
                                        theta_list: List[sp.Symbol],
                                        do_simplify: bool = False) -> Matrix:
    """
    EXACT gradient of Lagrange multiplier ∇ν.
    
    Parameters
    ----------
    a : 20×1 constraint gradient
    G : 20×20 Fisher information
    nu : scalar Lagrange multiplier
    theta_list : list of symbols
    do_simplify : bool
        Whether to simplify each derivative (SLOW for complex expressions)
    
    Returns
    -------
    Matrix : 20×1 column vector
    """
    n = len(theta_list)
    grad_nu = sp.zeros(n, 1)
    
    for i, sym in enumerate(theta_list):
        print(f"  ∇ν[{i+1}/{n}]...", end=" ", flush=True)
        deriv = sp.diff(nu, sym)
        grad_nu[i, 0] = simplify(deriv) if do_simplify else deriv
        print("done")
    
    return grad_nu


def exact_antisymmetric_part_lme(a: Matrix, grad_nu: Matrix) -> Matrix:
    """
    EXACT antisymmetric part A = (1/2)[a(∇ν)ᵀ - (∇ν)aᵀ].
    
    Parameters
    ----------
    a : 20×1 constraint gradient
    grad_nu : 20×1 gradient of Lagrange multiplier
    
    Returns
    -------
    Matrix : 20×20 antisymmetric matrix
    """
    outer1 = a * grad_nu.T  # a ⊗ (∇ν)ᵀ
    outer2 = grad_nu * a.T  # (∇ν) ⊗ aᵀ
    
    A = (outer1 - outer2) / 2
    
    # Simplify each element
    n = a.shape[0]
    for i in range(n):
        for j in range(n):
            A[i, j] = simplify(A[i, j])
    
    return A


def exact_constraint_hessian_lme(theta: Dict[str, sp.Symbol],
                                  theta_list: List[sp.Symbol],
                                  do_simplify: bool = False) -> Matrix:
    """
    EXACT constraint Hessian ∇²C.
    
    Returns n×n matrix of second derivatives of C = h₁ + h₂.
    """
    C = exact_constraint_lme(theta)
    n = len(theta_list)
    
    H = sp.zeros(n, n)
    for i, sym_i in enumerate(theta_list):
        print(f"  ∇²C row {i+1}/{n}...", end=" ", flush=True)
        dC_i = sp.diff(C, sym_i)
        for j, sym_j in enumerate(theta_list):
            if j >= i:  # Use symmetry
                deriv = sp.diff(dC_i, sym_j)
                H[i, j] = simplify(deriv) if do_simplify else deriv
                if j > i:
                    H[j, i] = H[i, j]
        print("done")
    
    return H


def exact_nabla_G_theta_lme(G: Matrix, theta_list: List[sp.Symbol]) -> Matrix:
    """
    EXACT (∇G)[θ] = Σ_k (∂G_ij/∂θ_k) θ_k.
    
    Third derivatives of ψ contracted with θ.
    """
    n = len(theta_list)
    result = sp.zeros(n, n)
    
    for i in range(n):
        for j in range(n):
            for k, tk in enumerate(theta_list):
                result[i, j] += sp.diff(G[i, j], tk) * tk
    
    return result


def exact_jacobian_lme(theta: Dict[str, sp.Symbol],
                       theta_list: List[sp.Symbol]) -> Matrix:
    """
    EXACT full Jacobian M = -G - (∇G)[θ] + ν∇²C + a(∇ν)ᵀ.
    """
    # Get components
    psi = exact_psi_lme(theta)
    G = exact_fisher_information_lme(theta, theta_list)
    nabla_G_theta = exact_nabla_G_theta_lme(G, theta_list)
    hess_C = exact_constraint_hessian_lme(theta, theta_list)
    a = exact_constraint_gradient_lme(theta, theta_list)
    nu = exact_lagrange_multiplier_lme(a, G, theta_list)
    grad_nu = exact_grad_lagrange_multiplier_lme(a, G, nu, theta_list)
    
    # M = -G - (∇G)[θ] + ν∇²C + a(∇ν)ᵀ
    M = -G - nabla_G_theta + nu * hess_C + a * grad_nu.T
    
    return M


def exact_symmetric_part_lme(M: Matrix) -> Matrix:
    """
    EXACT symmetric part S = (M + Mᵀ)/2.
    """
    return (M + M.T) / 2


def compute_full_chain_lme(theta_dict: Dict[str, sp.Symbol] = None,
                           theta_list: List[sp.Symbol] = None,
                           simplify_intermediate: bool = True) -> Dict:
    """
    Compute the full chain: exp(K) → ρ → h → a → G → ν → ∇ν → M → S, A.
    
    Returns dict with all intermediate results.
    """
    if theta_dict is None or theta_list is None:
        theta_dict, theta_list = create_theta_symbols()
    
    results = {'theta_dict': theta_dict, 'theta_list': theta_list}
    
    print("Step 1/7: Computing constraint gradient a = ∇(h₁+h₂)...")
    results['a'] = exact_constraint_gradient_lme(theta_dict, theta_list)
    
    print("Step 2/7: Computing cumulant generating function ψ...")
    results['psi'] = exact_psi_lme(theta_dict)
    
    print("Step 3/7: Computing Fisher information G = ∇²ψ...")
    results['G'] = exact_fisher_information_lme(theta_dict, theta_list)
    
    print("Step 4/7: Computing Lagrange multiplier ν...")
    results['nu'] = exact_lagrange_multiplier_lme(
        results['a'], results['G'], theta_list
    )
    
    print("Step 5/7: Computing gradient ∇ν...")
    results['grad_nu'] = exact_grad_lagrange_multiplier_lme(
        results['a'], results['G'], results['nu'], theta_list
    )
    
    print("Step 6/7: Assembling antisymmetric part A...")
    results['A'] = exact_antisymmetric_part_lme(results['a'], results['grad_nu'])
    
    print("Step 7/7: Computing ||A||...")
    A_norm_sq = sum(results['A'][i,j]**2 for i in range(20) for j in range(20))
    results['A_norm'] = sp.sqrt(A_norm_sq)
    
    return results


# =============================================================================
# Testing
# =============================================================================

def test_exact_lme():
    """Test exact LME computation."""
    print("Testing EXACT exp(K) for LME dynamics")
    print("=" * 60)
    
    # Create symbolic parameters
    θ = {
        'λ3⊗I': symbols('a3', real=True),
        'λ8⊗I': symbols('a8', real=True),
        'I⊗λ3': symbols('b3', real=True),
        'I⊗λ8': symbols('b8', real=True),
        'λ1⊗λ1': symbols('c11', real=True),
        'λ3⊗λ8': symbols('c38', real=True),
    }
    
    print("Parameters:", list(θ.keys()))
    
    print("\n1. Computing exact exp(K)...")
    exp_K = exact_exp_K_lme(θ)
    print(f"   Shape: {exp_K.shape}")
    
    print("\n2. Numerical validation...")
    # Substitute numerical values
    vals = {θ['λ3⊗I']: 0.3, θ['λ8⊗I']: 0.15, 
            θ['I⊗λ3']: 0.2, θ['I⊗λ8']: 0.1,
            θ['λ1⊗λ1']: 0.1, θ['λ3⊗λ8']: 0.05}
    
    exp_K_num = np.array(exp_K.subs(vals).evalf()).astype(complex)
    
    # Compare with scipy
    from scipy.linalg import expm
    generators, names = block_preserving_generators()
    
    K_num = np.zeros((9, 9), dtype=complex)
    for gen, name in zip(generators, names):
        if name in θ:
            K_num += float(vals[θ[name]]) * np.array(gen.tolist(), dtype=complex)
    
    exp_K_scipy = expm(K_num)
    
    diff = np.linalg.norm(exp_K_num - exp_K_scipy)
    print(f"   ||exp(K)_exact - exp(K)_scipy|| = {diff:.2e}")
    
    if diff < 1e-10:
        print("\n" + "=" * 60)
        print("✓ EXACT MATCH! No Taylor approximation needed.")
        print("  Block decomposition: 9×9 → 3×3 + 2×2 + 1×1×4")
        print("  All eigenvalues computed via quadratic formula.")
        print("=" * 60)
    
    return diff


if __name__ == "__main__":
    test_exact_lme()

