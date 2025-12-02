"""
Tests for LME exact symbolic computation.

Uses PRECOMPUTED expressions from qig.symbolic.precomputed.two_param_chain
for fast testing. No recomputation needed!
"""

import numpy as np
import pytest
import sympy as sp
from sympy import symbols


# =============================================================================
# Fast tests - generators only (no symbolic computation)
# =============================================================================

class TestBlockDecomposition:
    """Test the 9×9 → 3×3 + 6×6 block decomposition."""
    
    def test_block_preserving_generators_count(self):
        """Should have exactly 20 block-preserving generators."""
        from qig.symbolic.lme_exact import block_preserving_generators
        generators, names = block_preserving_generators()
        assert len(generators) == 20
        assert len(names) == 20
    
    def test_generators_hermitian(self):
        """All generators should be Hermitian."""
        from qig.symbolic.lme_exact import block_preserving_generators
        generators, _ = block_preserving_generators()
        
        for i, F in enumerate(generators):
            diff = sp.simplify(F - F.H)
            assert diff == sp.zeros(9, 9), f"Generator {i} not Hermitian"
    
    def test_generators_traceless(self):
        """All generators should be traceless."""
        from qig.symbolic.lme_exact import block_preserving_generators
        generators, _ = block_preserving_generators()
        
        for i, F in enumerate(generators):
            tr = sp.simplify(F.trace())
            assert tr == 0, f"Generator {i} has trace {tr}"


# =============================================================================
# Tests using PRECOMPUTED expressions (instant!)
# =============================================================================

class TestPrecomputedExpressions:
    """Test using precomputed symbolic expressions."""
    
    @pytest.fixture
    def precomputed(self):
        """Load precomputed expressions."""
        from qig.symbolic.precomputed.two_param_chain import (
            a, c, G, a_vec, nu, grad_nu, M, S, A
        )
        return {
            'a_sym': a, 'c_sym': c,
            'G': G, 'a_vec': a_vec, 'nu': nu, 'grad_nu': grad_nu,
            'M': M, 'S': S, 'A': A
        }
    
    @pytest.fixture
    def test_values(self, precomputed):
        """Numerical test values."""
        return {precomputed['a_sym']: 0.1, precomputed['c_sym']: 0.2}
    
    def test_nu_not_minus_one(self, precomputed, test_values):
        """With entangling parameter, ν ≠ -1 (but may be close for small θ)."""
        nu_val = float(precomputed['nu'].subs(test_values).evalf())
        # For small θ, ν is close to -1. Key test is that grad_nu ≠ 0.
        assert abs(nu_val + 1) > 1e-6, f"ν = {nu_val} exactly -1"
    
    def test_grad_nu_nonzero(self, precomputed, test_values):
        """With entangling parameter, ∇ν ≠ 0."""
        grad_nu = precomputed['grad_nu']
        grad_nu_num = np.array([float(grad_nu[i,0].subs(test_values).evalf()) 
                                for i in range(2)])
        assert np.linalg.norm(grad_nu_num) > 1e-6
    
    def test_A_nonzero(self, precomputed, test_values):
        """A ≠ 0 with entangling parameter."""
        A = precomputed['A']
        A_num = np.array([[float(A[i,j].subs(test_values).evalf()) for j in range(2)] 
                         for i in range(2)])
        assert np.linalg.norm(A_num) > 1e-6
    
    def test_A_antisymmetric(self, precomputed, test_values):
        """A should satisfy A + Aᵀ = 0."""
        A = precomputed['A']
        A_num = np.array([[float(A[i,j].subs(test_values).evalf()) for j in range(2)] 
                         for i in range(2)])
        assert np.allclose(A_num, -A_num.T)
    
    def test_S_nonzero(self, precomputed, test_values):
        """S ≠ 0 with entangling parameter."""
        S = precomputed['S']
        S_num = np.array([[float(S[i,j].subs(test_values).evalf()) for j in range(2)] 
                         for i in range(2)])
        assert np.linalg.norm(S_num) > 0.1
    
    def test_S_symmetric(self, precomputed, test_values):
        """S should satisfy S = Sᵀ."""
        S = precomputed['S']
        S_num = np.array([[float(S[i,j].subs(test_values).evalf()) for j in range(2)] 
                         for i in range(2)])
        assert np.allclose(S_num, S_num.T)
    
    def test_M_decomposition(self, precomputed, test_values):
        """M should equal S + A."""
        M = precomputed['M']
        S = precomputed['S']
        A = precomputed['A']
        
        M_num = np.array([[float(M[i,j].subs(test_values).evalf()) for j in range(2)] 
                         for i in range(2)])
        S_num = np.array([[float(S[i,j].subs(test_values).evalf()) for j in range(2)] 
                         for i in range(2)])
        A_num = np.array([[float(A[i,j].subs(test_values).evalf()) for j in range(2)] 
                         for i in range(2)])
        
        assert np.allclose(M_num, S_num + A_num)


# =============================================================================
# Single-parameter structural identity test
# =============================================================================

class TestStructuralIdentity:
    """Test that structural identity holds for local parameters only."""
    
    def test_nu_minus_one_for_local(self):
        """For local parameters only, ν = -1."""
        from qig.symbolic.lme_exact import (
            exact_psi_lme,
            exact_fisher_information_lme, exact_constraint_gradient_lme,
            exact_lagrange_multiplier_lme
        )
        
        a = symbols('a', real=True)
        theta = {'λ3⊗I': a}
        theta_list = [a]
        
        G = exact_fisher_information_lme(theta, theta_list)
        a_vec = exact_constraint_gradient_lme(theta, theta_list)
        nu = exact_lagrange_multiplier_lme(a_vec, G, theta_list)
        
        nu_val = float(nu.subs(a, 0.2).evalf())
        assert abs(nu_val + 1) < 1e-10, f"ν = {nu_val} ≠ -1 for local param"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
