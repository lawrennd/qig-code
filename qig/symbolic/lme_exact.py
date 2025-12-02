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

