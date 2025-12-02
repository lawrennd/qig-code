"""
EXACT matrix exponential for LME qutrit pairs.

Key insights for LME + Gell-Mann:
1. Block structure (no |2⟩ coherences) → 2×2 + 1×1 blocks
2. Eigenvalues are QUADRATIC (not cubic) → exact symbolic solution
3. For separable: exp(A⊗I + I⊗B) = exp(A) ⊗ exp(B)

This gives EXACT ρ and ρ₁ with NO Taylor expansion!
"""

import sympy as sp
from sympy import Matrix, sqrt, exp, I as sp_I, symbols, simplify
from typing import Tuple
import numpy as np


def build_block_matrix(theta1, theta2, theta3, theta8) -> Matrix:
    """
    Build 3×3 matrix with block structure from Gell-Mann params.
    
    Only uses λ₁, λ₂, λ₃, λ₈ (no |2⟩ coherences).
    This gives 2×2 + 1×1 block structure.
    
    Returns A where eigenvalues are:
      λ₀ = -2√3·θ₈/3  (isolated)
      λ₁,₂ = √3·θ₈/3 ∓ √(θ₁² + θ₂² + θ₃²)  (from 2×2 block)
    """
    s3 = sqrt(3)
    return Matrix([
        [theta3 + theta8/s3,     theta1 - sp_I*theta2,      0],
        [theta1 + sp_I*theta2,   -theta3 + theta8/s3,       0],
        [0,                       0,                        -2*theta8/s3]
    ])


def exp_block_matrix_exact(theta1, theta2, theta3, theta8) -> Matrix:
    """
    EXACT exp(A) for block structure matrix.
    
    Uses eigendecomposition: exp(A) = U exp(D) U†
    """
    A = build_block_matrix(theta1, theta2, theta3, theta8)
    P, D = A.diagonalize()
    
    exp_D = sp.diag(*[exp(D[i,i]) for i in range(3)])
    exp_A = simplify(P * exp_D * P.inv())
    
    return exp_A


def exact_rho_separable_block(
    theta_local1: Tuple,  # (θ₁, θ₂, θ₃, θ₈) for site 1
    theta_local2: Tuple   # (θ₁, θ₂, θ₃, θ₈) for site 2
) -> Matrix:
    """
    EXACT density matrix for separable state with block structure.
    
    ρ = exp(A⊗I + I⊗B) / Z = exp(A)⊗exp(B) / Z
    
    NO Taylor expansion!
    """
    exp_A = exp_block_matrix_exact(*theta_local1)
    exp_B = exp_block_matrix_exact(*theta_local2)
    
    exp_K = sp.kronecker_product(exp_A, exp_B)
    Z = exp_K.trace()
    
    return exp_K / Z


def exact_rho1_separable_block(theta_local1: Tuple) -> Matrix:
    """
    EXACT reduced density matrix ρ₁ for separable state.
    
    For separable ρ = ρ_A ⊗ ρ_B: ρ₁ = Tr₂(ρ) = ρ_A
    So ρ₁ = exp(A) / Tr(exp(A))
    
    NO Taylor expansion!
    """
    exp_A = exp_block_matrix_exact(*theta_local1)
    Z_A = exp_A.trace()
    
    return simplify(exp_A / Z_A)


def exact_eigenvalues_block(theta1, theta2, theta3, theta8) -> Tuple:
    """
    Exact eigenvalues for block structure matrix.
    
    Returns (λ₀, λ₁, λ₂) where:
      λ₀ = -2√3·θ₈/3
      λ₁ = √3·θ₈/3 - √(θ₁² + θ₂² + θ₃²)
      λ₂ = √3·θ₈/3 + √(θ₁² + θ₂² + θ₃²)
    """
    s3 = sqrt(3)
    r = sqrt(theta1**2 + theta2**2 + theta3**2)
    
    lambda0 = -2*s3*theta8/3
    lambda1 = s3*theta8/3 - r
    lambda2 = s3*theta8/3 + r
    
    return (lambda0, lambda1, lambda2)


def test_exact_implementation():
    """Test the exact implementation."""
    print("Testing EXACT exp(K) implementation")
    print("=" * 50)
    
    # Symbolic parameters
    a1, a2, a3, a8 = symbols('a1 a2 a3 a8', real=True)
    b1, b2, b3, b8 = symbols('b1 b2 b3 b8', real=True)
    
    print("\n1. Computing exact ρ₁ symbolically...")
    rho1 = exact_rho1_separable_block((a1, a2, a3, a8))
    print(f"   Shape: {rho1.shape}")
    
    print("\n2. Numerical validation...")
    vals = {a1: 0.3, a2: 0.2, a3: 0.1, a8: 0.15,
            b1: 0.25, b2: -0.1, b3: 0.2, b8: 0.1}
    
    # Our exact result
    rho1_exact = np.array(rho1.subs(vals).evalf()).astype(complex)
    
    # scipy reference
    from scipy.linalg import expm
    A = build_block_matrix(a1, a2, a3, a8)
    A_num = np.array(A.subs(vals).evalf()).astype(complex)
    exp_A_scipy = expm(A_num)
    rho1_scipy = exp_A_scipy / np.trace(exp_A_scipy)
    
    diff = np.linalg.norm(rho1_exact - rho1_scipy)
    print(f"   ||ρ₁_exact - ρ₁_scipy|| = {diff:.2e}")
    
    print("\n3. Testing full separable state...")
    rho_full = exact_rho_separable_block(
        (a1, a2, a3, a8),
        (b1, b2, b3, b8)
    )
    rho_full_num = np.array(rho_full.subs(vals).evalf()).astype(complex)
    
    # scipy reference for full state
    B = build_block_matrix(b1, b2, b3, b8)
    A_num = np.array(A.subs(vals).evalf()).astype(complex)
    B_num = np.array(B.subs(vals).evalf()).astype(complex)
    K_num = np.kron(A_num, np.eye(3)) + np.kron(np.eye(3), B_num)
    exp_K_scipy = expm(K_num)
    rho_scipy = exp_K_scipy / np.trace(exp_K_scipy)
    
    diff_full = np.linalg.norm(rho_full_num - rho_scipy)
    print(f"   ||ρ_exact - ρ_scipy|| = {diff_full:.2e}")
    
    if diff < 1e-10 and diff_full < 1e-10:
        print("\n" + "=" * 50)
        print("✓ EXACT RESULTS - NO TAYLOR APPROXIMATION!")
        print("  Block structure + eigendecomposition = machine precision")
        print("=" * 50)
    
    return diff, diff_full


if __name__ == "__main__":
    test_exact_implementation()
