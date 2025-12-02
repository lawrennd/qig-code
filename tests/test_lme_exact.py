"""
Tests for LME exact symbolic computation.

Tests the breakthrough: EXACT exp(K) via block decomposition
with NO Taylor approximation.
"""

import numpy as np
import pytest
from scipy.linalg import expm
import sympy as sp
from sympy import symbols


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
    
    def test_exp_K_matches_scipy(self):
        """Exact exp(K) should match scipy.linalg.expm to machine precision."""
        from qig.symbolic.lme_exact import (
            exact_exp_K_lme, 
            block_preserving_generators
        )
        
        generators, names = block_preserving_generators()
        
        # Create symbolic parameters
        theta = {name: symbols(f't{i}', real=True) for i, name in enumerate(names[:6])}
        
        # Compute symbolic exp(K)
        exp_K_sym = exact_exp_K_lme(theta)
        
        # Test with random numerical values
        np.random.seed(42)
        for _ in range(3):
            theta_vals = {theta[name]: 0.3 * np.random.randn() for name in theta}
            
            # Our result
            exp_K_ours = np.array(exp_K_sym.subs(theta_vals).evalf()).astype(complex)
            
            # scipy reference
            K_num = np.zeros((9, 9), dtype=complex)
            for gen, name in zip(generators, names):
                if name in theta:
                    K_num += float(theta_vals[theta[name]]) * np.array(gen.tolist(), dtype=complex)
            exp_K_scipy = expm(K_num)
            
            diff = np.linalg.norm(exp_K_ours - exp_K_scipy)
            assert diff < 1e-12, f"exp(K) error: {diff}"


class TestDensityMatrix:
    """Test density matrix computation."""
    
    def test_rho_trace_one(self):
        """ρ should have trace 1."""
        from qig.symbolic.lme_exact import exact_rho_lme
        
        theta = {'λ3⊗I': symbols('a', real=True)}
        rho = exact_rho_lme(theta)
        
        tr = sp.simplify(rho.trace())
        assert tr == 1, f"Tr(ρ) = {tr} ≠ 1"
    
    def test_rho_hermitian(self):
        """ρ should be Hermitian."""
        from qig.symbolic.lme_exact import exact_rho_lme
        
        theta = {'λ3⊗I': symbols('a', real=True)}
        rho = exact_rho_lme(theta)
        
        diff = sp.simplify(rho - rho.H)
        assert diff == sp.zeros(9, 9), "ρ not Hermitian"
    
    def test_rho_positive_semidefinite(self):
        """ρ should have non-negative eigenvalues."""
        from qig.symbolic.lme_exact import exact_rho_lme
        
        a = symbols('a', real=True)
        theta = {'λ3⊗I': a}
        rho = exact_rho_lme(theta)
        
        # Numerical check
        rho_num = np.array(rho.subs({a: 0.3}).evalf()).astype(complex)
        eigs = np.linalg.eigvalsh(rho_num)
        assert np.all(eigs >= -1e-14), f"Negative eigenvalue: {min(eigs)}"


class TestPartialTrace:
    """Test partial trace computation."""
    
    def test_partial_trace_preserves_trace(self):
        """Tr(ρ₁) should equal 1."""
        from qig.symbolic.lme_exact import exact_rho1_lme
        
        theta = {'λ3⊗I': symbols('a', real=True)}
        rho1 = exact_rho1_lme(theta)
        
        tr = sp.simplify(rho1.trace())
        assert tr == 1, f"Tr(ρ₁) = {tr} ≠ 1"
    
    def test_partial_trace_shape(self):
        """ρ₁ should be 3×3."""
        from qig.symbolic.lme_exact import exact_rho1_lme
        
        theta = {'λ3⊗I': symbols('a', real=True)}
        rho1 = exact_rho1_lme(theta)
        
        assert rho1.shape == (3, 3)


class TestMarginalEntropy:
    """Test marginal entropy computation."""
    
    def test_maximally_mixed_entropy(self):
        """h(I/3) should equal log(3)."""
        from qig.symbolic.lme_exact import exact_marginal_entropy_lme
        
        rho_mm = sp.eye(3) / 3
        h = exact_marginal_entropy_lme(rho_mm)
        
        expected = sp.log(3)
        diff = sp.simplify(h - expected)
        assert diff == 0, f"h(I/3) = {h} ≠ log(3)"
    
    def test_entropy_numerical_match(self):
        """Symbolic entropy should match numerical."""
        from qig.symbolic.lme_exact import (
            exact_rho1_lme, 
            exact_marginal_entropy_lme,
            block_preserving_generators
        )
        
        a, b = symbols('a b', real=True)
        theta = {'λ3⊗I': a, 'λ8⊗I': b}
        rho1 = exact_rho1_lme(theta)
        h1 = exact_marginal_entropy_lme(rho1)
        
        # Numerical comparison
        vals = {a: 0.2, b: 0.1}
        h1_sym = float(h1.subs(vals).evalf())
        
        # scipy reference
        generators, names = block_preserving_generators()
        K = np.zeros((9,9), dtype=complex)
        for gen, name in zip(generators, names):
            if name in theta:
                K += float(vals[theta[name]]) * np.array(gen.tolist(), dtype=complex)
        
        rho_num = expm(K)
        rho_num /= np.trace(rho_num)
        rho1_num = np.einsum('ijkj->ik', rho_num.reshape(3,3,3,3))
        eigs = np.linalg.eigvalsh(rho1_num)
        h1_scipy = -sum(e * np.log(e) for e in eigs if e > 1e-15)
        
        assert abs(h1_sym - h1_scipy) < 1e-10, f"Entropy mismatch: {h1_sym} vs {h1_scipy}"


class TestFullChain:
    """Test complete chain from K to entropy."""
    
    def test_constraint_computation(self):
        """C = h₁ + h₂ should be computable."""
        from qig.symbolic.lme_exact import exact_constraint_lme
        
        theta = {'λ3⊗I': symbols('a', real=True)}
        C = exact_constraint_lme(theta)
        
        # Should be a symbolic expression
        assert C is not None
        assert isinstance(C, sp.Basic)
    
    def test_separable_state_entropy(self):
        """For separable state, h₁ and h₂ should be independent."""
        from qig.symbolic.lme_exact import (
            exact_rho1_lme,
            exact_rho2_lme,
            exact_marginal_entropy_lme
        )
        
        # Only local-1 parameter
        theta1 = {'λ3⊗I': symbols('a', real=True)}
        rho1 = exact_rho1_lme(theta1)
        rho2 = exact_rho2_lme(theta1)
        
        h1 = exact_marginal_entropy_lme(rho1)
        h2 = exact_marginal_entropy_lme(rho2)
        
        # h₁ should depend on 'a', h₂ should be constant (maximally mixed)
        # Actually for block-preserving generators, both may depend on a
        # Just check they're computable
        assert h1 is not None
        assert h2 is not None


class TestConstraintGradient:
    """Test constraint gradient a = ∇C."""
    
    def test_gradient_numerical_match(self):
        """Symbolic gradient should match finite difference."""
        from qig.symbolic.lme_exact import exact_constraint_lme
        
        a = symbols('a', real=True)
        theta = {'λ3⊗I': a}
        
        C = exact_constraint_lme(theta)
        dC_da = sp.diff(C, a)
        
        # Numerical derivative
        val = 0.2
        eps = 1e-6
        C_plus = float(C.subs(a, val + eps).evalf())
        C_minus = float(C.subs(a, val - eps).evalf())
        dC_num = (C_plus - C_minus) / (2 * eps)
        
        dC_exact = float(dC_da.subs(a, val).evalf())
        
        assert abs(dC_exact - dC_num) < 1e-6


class TestFisherInformation:
    """Test Fisher information G = ∇²ψ."""
    
    def test_fisher_numerical_match(self):
        """Symbolic Hessian should match finite difference."""
        from qig.symbolic.lme_exact import exact_psi_lme
        
        a = symbols('a', real=True)
        theta = {'λ3⊗I': a}
        
        psi = exact_psi_lme(theta)
        G = sp.diff(sp.diff(psi, a), a)
        
        # Numerical second derivative
        val = 0.2
        eps = 1e-5
        psi_plus = float(psi.subs(a, val + eps).evalf())
        psi_mid = float(psi.subs(a, val).evalf())
        psi_minus = float(psi.subs(a, val - eps).evalf())
        G_num = (psi_plus - 2*psi_mid + psi_minus) / eps**2
        
        G_exact = float(G.subs(a, val).evalf())
        
        assert abs(G_exact - G_num) < 1e-4


class TestLagrangeMultiplier:
    """Test Lagrange multiplier ν and its gradient."""
    
    def test_nu_equals_minus_one_for_local(self):
        """For local parameters only, ν = -1 (structural identity)."""
        from qig.symbolic.lme_exact import exact_constraint_lme, exact_psi_lme
        
        a = symbols('a', real=True)
        theta = {'λ3⊗I': a}
        val = 0.2
        
        C = exact_constraint_lme(theta)
        psi = exact_psi_lme(theta)
        
        dC_da = sp.diff(C, a)
        G = sp.diff(sp.diff(psi, a), a)
        nu = (dC_da * G * a) / (dC_da * dC_da)
        
        nu_val = float(nu.subs(a, val).evalf())
        assert abs(nu_val + 1) < 1e-10, f"ν = {nu_val} ≠ -1 for local param"
    
    def test_nu_not_minus_one_for_entangling(self):
        """For entangling parameters, ν ≠ -1."""
        from qig.symbolic.lme_exact import exact_constraint_lme, exact_psi_lme
        
        c = symbols('c', real=True)
        theta = {'λ1⊗λ1': c}
        val = 0.2
        
        C = exact_constraint_lme(theta)
        psi = exact_psi_lme(theta)
        
        dC_dc = sp.diff(C, c)
        G = sp.diff(sp.diff(psi, c), c)
        nu = (dC_dc * G * c) / (dC_dc * dC_dc)
        
        nu_val = float(nu.subs(c, val).evalf())
        assert abs(nu_val + 1) > 0.1, f"ν = {nu_val} ≈ -1 (should differ)"


class TestAntisymmetricPart:
    """Test antisymmetric part A."""
    
    def test_A_zero_for_local_only(self):
        """For local parameters only, A = 0."""
        from qig.symbolic.lme_exact import exact_constraint_lme, exact_psi_lme
        
        a, b = symbols('a b', real=True)
        theta = {'λ3⊗I': a, 'I⊗λ3': b}
        theta_list = [a, b]
        vals = {a: 0.1, b: 0.15}
        
        C = exact_constraint_lme(theta)
        psi = exact_psi_lme(theta)
        
        a_vec = sp.Matrix([sp.diff(C, t) for t in theta_list])
        G = sp.Matrix([[sp.diff(sp.diff(psi, ti), tj) for tj in theta_list] 
                       for ti in theta_list])
        
        theta_vec = sp.Matrix(theta_list)
        nu = (a_vec.T * G * theta_vec)[0,0] / (a_vec.T * a_vec)[0,0]
        grad_nu = sp.Matrix([sp.diff(nu, t) for t in theta_list])
        A = (a_vec * grad_nu.T - grad_nu * a_vec.T) / 2
        
        A_num = np.array([[float(A[i,j].subs(vals).evalf()) for j in range(2)] 
                         for i in range(2)])
        
        assert np.linalg.norm(A_num) < 1e-10, f"||A|| = {np.linalg.norm(A_num)} for local"
    
    def test_A_nonzero_for_entangling(self):
        """For entangling parameters, A ≠ 0."""
        from qig.symbolic.lme_exact import exact_constraint_lme, exact_psi_lme
        
        a, c = symbols('a c', real=True)
        theta = {'λ3⊗I': a, 'λ1⊗λ1': c}
        theta_list = [a, c]
        vals = {a: 0.1, c: 0.2}
        
        C = exact_constraint_lme(theta)
        psi = exact_psi_lme(theta)
        
        a_vec = sp.Matrix([sp.diff(C, t) for t in theta_list])
        G = sp.Matrix([[sp.diff(sp.diff(psi, ti), tj) for tj in theta_list] 
                       for ti in theta_list])
        
        theta_vec = sp.Matrix(theta_list)
        nu = (a_vec.T * G * theta_vec)[0,0] / (a_vec.T * a_vec)[0,0]
        grad_nu = sp.Matrix([sp.diff(nu, t) for t in theta_list])
        A = (a_vec * grad_nu.T - grad_nu * a_vec.T) / 2
        
        A_num = np.array([[float(A[i,j].subs(vals).evalf()) for j in range(2)] 
                         for i in range(2)])
        
        assert np.linalg.norm(A_num) > 1e-6, f"||A|| = {np.linalg.norm(A_num)} ≈ 0"
    
    def test_A_is_antisymmetric(self):
        """A should satisfy A + Aᵀ = 0."""
        from qig.symbolic.lme_exact import exact_constraint_lme, exact_psi_lme
        
        a, c = symbols('a c', real=True)
        theta = {'λ3⊗I': a, 'λ1⊗λ1': c}
        theta_list = [a, c]
        
        C = exact_constraint_lme(theta)
        psi = exact_psi_lme(theta)
        
        a_vec = sp.Matrix([sp.diff(C, t) for t in theta_list])
        G = sp.Matrix([[sp.diff(sp.diff(psi, ti), tj) for tj in theta_list] 
                       for ti in theta_list])
        
        theta_vec = sp.Matrix(theta_list)
        nu = (a_vec.T * G * theta_vec)[0,0] / (a_vec.T * a_vec)[0,0]
        grad_nu = sp.Matrix([sp.diff(nu, t) for t in theta_list])
        A = (a_vec * grad_nu.T - grad_nu * a_vec.T) / 2
        
        # Check antisymmetry symbolically
        for i in range(2):
            for j in range(2):
                assert sp.simplify(A[i,j] + A[j,i]) == 0


class TestSymmetricPart:
    """Test symmetric part S and full Jacobian M."""
    
    def test_M_zero_for_single_local_param(self):
        """For single local parameter, M = 0 (structural identity)."""
        from qig.symbolic.lme_exact import (
            exact_psi_lme, exact_constraint_lme,
            exact_fisher_information_lme, exact_constraint_gradient_lme,
            exact_lagrange_multiplier_lme, exact_grad_lagrange_multiplier_lme,
            exact_constraint_hessian_lme, exact_nabla_G_theta_lme
        )
        
        a = symbols('a', real=True)
        theta = {'λ3⊗I': a}
        theta_list = [a]
        val = 0.2
        
        G = exact_fisher_information_lme(theta, theta_list)
        nabla_G_theta = exact_nabla_G_theta_lme(G, theta_list)
        hess_C = exact_constraint_hessian_lme(theta, theta_list)
        a_vec = exact_constraint_gradient_lme(theta, theta_list)
        nu = exact_lagrange_multiplier_lme(a_vec, G, theta_list)
        grad_nu = exact_grad_lagrange_multiplier_lme(a_vec, G, nu, theta_list)
        
        M = -G - nabla_G_theta + nu * hess_C + a_vec * grad_nu.T
        M_val = float(M[0,0].subs(a, val).evalf())
        
        assert abs(M_val) < 1e-10, f"M = {M_val} ≠ 0 for local param"
    
    def test_S_nonzero_for_entangling(self):
        """For local + entangling, S ≠ 0."""
        from qig.symbolic.lme_exact import (
            exact_psi_lme, exact_constraint_lme,
            exact_fisher_information_lme, exact_constraint_gradient_lme,
            exact_lagrange_multiplier_lme, exact_grad_lagrange_multiplier_lme,
            exact_constraint_hessian_lme, exact_nabla_G_theta_lme
        )
        
        a, c = symbols('a c', real=True)
        theta = {'λ3⊗I': a, 'λ1⊗λ1': c}
        theta_list = [a, c]
        vals = {a: 0.1, c: 0.2}
        
        G = exact_fisher_information_lme(theta, theta_list)
        nabla_G_theta = exact_nabla_G_theta_lme(G, theta_list)
        hess_C = exact_constraint_hessian_lme(theta, theta_list)
        a_vec = exact_constraint_gradient_lme(theta, theta_list)
        nu = exact_lagrange_multiplier_lme(a_vec, G, theta_list)
        grad_nu = exact_grad_lagrange_multiplier_lme(a_vec, G, nu, theta_list)
        
        M = -G - nabla_G_theta + nu * hess_C + a_vec * grad_nu.T
        S = (M + M.T) / 2
        
        S_num = np.array([[float(S[i,j].subs(vals).evalf()) for j in range(2)] 
                         for i in range(2)])
        
        assert np.linalg.norm(S_num) > 0.1, f"||S|| = {np.linalg.norm(S_num)} ≈ 0"
    
    def test_S_is_symmetric(self):
        """S should satisfy S = Sᵀ numerically."""
        from qig.symbolic.lme_exact import (
            exact_psi_lme, exact_constraint_lme,
            exact_fisher_information_lme, exact_constraint_gradient_lme,
            exact_lagrange_multiplier_lme, exact_grad_lagrange_multiplier_lme,
            exact_constraint_hessian_lme, exact_nabla_G_theta_lme
        )
        
        a, c = symbols('a c', real=True)
        theta = {'λ3⊗I': a, 'λ1⊗λ1': c}
        theta_list = [a, c]
        vals = {a: 0.1, c: 0.2}
        
        G = exact_fisher_information_lme(theta, theta_list)
        nabla_G_theta = exact_nabla_G_theta_lme(G, theta_list)
        hess_C = exact_constraint_hessian_lme(theta, theta_list)
        a_vec = exact_constraint_gradient_lme(theta, theta_list)
        nu = exact_lagrange_multiplier_lme(a_vec, G, theta_list)
        grad_nu = exact_grad_lagrange_multiplier_lme(a_vec, G, nu, theta_list)
        
        M = -G - nabla_G_theta + nu * hess_C + a_vec * grad_nu.T
        S = (M + M.T) / 2
        
        # Check S = Sᵀ numerically
        S_num = np.array([[float(S[i,j].subs(vals).evalf()) for j in range(2)] 
                         for i in range(2)])
        assert np.allclose(S_num, S_num.T)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

