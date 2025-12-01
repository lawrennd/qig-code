"""
Validation tests for analytic qutrit GENERIC decomposition.

This module tests that symbolic/analytic forms of S and A match numerical
computation to machine precision. Tests are organized to progressively
validate each component of the symbolic derivation.

Related to CIP-0007: Analytic Forms for S and A via Lie Algebra Structure
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

# These imports will be created as part of CIP-0007
# from qig.analytic import (
#     antisymmetric_part_analytical_qutrit,
#     symmetric_part_analytical_qutrit,
# )
from qig.exponential_family import QuantumExponentialFamily
from qig.structure_constants import compute_structure_constants


@pytest.fixture
def qutrit_pair_family():
    """Two-qutrit system with pair basis (16 parameters)."""
    return QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)


@pytest.fixture
def random_states(qutrit_pair_family, n_states=100):
    """Generate random test states in interior of state space."""
    np.random.seed(42)
    n_params = qutrit_pair_family.n_params
    
    # Generate states spanning parameter space
    # Keep small to stay in valid region (not near pure state boundary)
    theta_list = []
    for _ in range(n_states):
        theta = np.random.randn(n_params) * 0.5  # Scale to stay reasonable
        theta_list.append(theta)
    
    return theta_list


class TestSymbolicInfrastructure:
    """Test basic symbolic computation infrastructure."""
    
    def test_symbolic_gell_mann_matrices(self):
        """Test symbolic Gell-Mann matrices match numerical."""
        from qig.symbolic import symbolic_gell_mann_matrices
        from qig.exponential_family import gell_mann_matrices
        import sympy as sp
        
        symbolic_gm = symbolic_gell_mann_matrices()
        numerical_gm = gell_mann_matrices()
        
        for i, (sym, num) in enumerate(zip(symbolic_gm, numerical_gm)):
            # Convert symbolic to numpy
            sym_eval = np.array(sym.tolist()).astype(complex)
            assert_allclose(sym_eval, num, atol=1e-15,
                           err_msg=f"Gell-Mann matrix {i+1} mismatch")
    
    def test_symbolic_structure_constants(self):
        """Test symbolic structure constants match reference values."""
        from qig.symbolic import symbolic_su3_structure_constants
        from qig.reference_data import get_su3_structure_constants
        import sympy as sp
        
        symbolic_f = symbolic_su3_structure_constants()
        reference_f = get_su3_structure_constants()
        
        # Convert symbolic to numpy (need to evaluate each element)
        f_eval = np.zeros((8, 8, 8))
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    f_eval[i, j, k] = float(symbolic_f[i, j, k])
        
        assert_allclose(f_eval, reference_f, atol=1e-15,
                       err_msg="Structure constants mismatch")
    
    def test_gell_mann_properties(self):
        """Test that Gell-Mann matrices satisfy required properties."""
        from qig.symbolic import symbolic_gell_mann_matrices
        from qig.symbolic.gell_mann import verify_gell_mann_properties
        
        gm = symbolic_gell_mann_matrices()
        results = verify_gell_mann_properties(gm)
        
        assert results['all_hermitian'], f"Not all Hermitian: {results['details']}"
        assert results['all_traceless'], f"Not all traceless: {results['details']}"
        assert results['normalization'], f"Normalization failed: {results['details']}"
    
    def test_commutation_relations(self):
        """Test [λ_a, λ_b] = 2i Σ_c f_abc λ_c."""
        from qig.symbolic import symbolic_gell_mann_matrices, symbolic_su3_structure_constants
        from qig.symbolic.gell_mann import verify_structure_constants
        
        gm = symbolic_gell_mann_matrices()
        f = symbolic_su3_structure_constants()
        
        results = verify_structure_constants(gm, f)
        assert results['all_correct'], f"Commutation relations failed: {results['errors']}"
    
    @pytest.mark.skip(reason="Symbolic density matrix not yet implemented (CIP-0007 Phase 2)")
    def test_symbolic_density_matrix_properties(self):
        """Test symbolic ρ satisfies quantum state properties."""
        # This will be implemented in Phase 2
        pass


class TestSingleQutritAnalytic:
    """Test single-qutrit analytic forms (8 parameters)."""
    
    def test_symbolic_density_matrix(self):
        """Test symbolic density matrix has correct properties."""
        from qig.symbolic import symbolic_density_matrix_single_qutrit
        import sympy as sp
        
        theta = sp.symbols('theta1:9', real=True)
        
        # Test order 1
        rho1 = symbolic_density_matrix_single_qutrit(theta, order=1)
        assert rho1.shape == (3, 3), "Should be 3×3 matrix"
        assert sp.simplify(sp.trace(rho1)) == 1, "Trace should be 1"
        assert (rho1 - rho1.H).is_zero_matrix, "Should be Hermitian"
        
        # Test order 2
        rho2 = symbolic_density_matrix_single_qutrit(theta, order=2)
        assert rho2.shape == (3, 3)
        # Note: order 2 is normalized, so trace = 1
    
    def test_symbolic_cumulant_function(self):
        """Test cumulant generating function."""
        from qig.symbolic import symbolic_cumulant_generating_function_single_qutrit
        import sympy as sp
        
        theta = sp.symbols('theta1:9', real=True)
        
        # At origin: ψ(0) = log(3)
        psi = symbolic_cumulant_generating_function_single_qutrit(theta, order=2)
        psi_at_origin = psi.subs([(t, 0) for t in theta])
        assert sp.simplify(psi_at_origin - sp.log(3)) == 0, "ψ(0) should be log(3)"
        
        # First derivative (expectation): should be 0 at origin
        for t in theta:
            dpsi = sp.diff(psi, t)
            dpsi_at_origin = dpsi.subs([(t2, 0) for t2 in theta])
            assert dpsi_at_origin == 0, f"∂ψ/∂{t} should be 0 at origin"
    
    def test_symbolic_fisher_metric(self):
        """Test Fisher information matrix."""
        from qig.symbolic import symbolic_fisher_information_single_qutrit
        import sympy as sp
        
        theta = sp.symbols('theta1:9', real=True)
        G = symbolic_fisher_information_single_qutrit(theta, order=2)
        
        assert G.shape == (8, 8), "Should be 8×8 matrix"
        assert (G - G.T).is_zero_matrix, "Should be symmetric"
        
        # At order 2 (near origin): G = (2/3)I
        assert G.is_diagonal(), "Should be diagonal at order 2"
        for i in range(8):
            assert G[i, i] == sp.Rational(2, 3), f"G[{i},{i}] should be 2/3"
    
    def test_symbolic_entropy(self):
        """Test von Neumann entropy."""
        from qig.symbolic import symbolic_von_neumann_entropy_single_qutrit
        import sympy as sp
        
        theta = sp.symbols('theta1:9', real=True)
        H = symbolic_von_neumann_entropy_single_qutrit(theta, order=2)
        
        # At origin: H(0) = log(3) (maximum entropy)
        H_at_origin = H.subs([(t, 0) for t in theta])
        assert sp.simplify(H_at_origin - sp.log(3)) == 0, "H(0) should be log(3)"
        
        # Entropy should decrease with ||θ||²
        # H ≈ log(3) - (1/9)Σθ²
        # So for small θ > 0, H < log(3)
    
    def test_consistency_verification(self):
        """Test overall consistency of single-qutrit symbolic computations."""
        from qig.symbolic import verify_single_qutrit_consistency
        import sympy as sp
        
        theta = sp.symbols('theta1:9', real=True)
        results = verify_single_qutrit_consistency(theta)
        
        assert results['rho_trace'], f"ρ trace failed: {results['details']}"
        assert results['rho_hermitian'], f"ρ Hermiticity failed: {results['details']}"
        assert results['G_symmetric'], f"G symmetry failed: {results['details']}"
        assert results['G_positive'], f"G positivity failed: {results['details']}"
        assert results['entropy_max'], f"Entropy bound failed: {results['details']}"


class TestTwoQutritConstraintGeometry:
    """Test constraint geometry components: a, ν, ∇ν."""
    
    def test_two_qutrit_operators(self):
        """Test two-qutrit operator basis construction."""
        from qig.symbolic import two_qutrit_operators
        
        ops = two_qutrit_operators()
        assert len(ops) == 16, "Should have 16 operators"
        assert ops[0].shape == (9, 9), "Should be 9×9 matrices"
        
        # Check they're Hermitian
        for i, op in enumerate(ops):
            assert (op - op.H).is_zero_matrix, f"Operator {i} should be Hermitian"
    
    def test_partial_trace(self):
        """Test partial trace symbolic computation."""
        from qig.symbolic import partial_trace_symbolic
        import sympy as sp
        
        # Test on maximally mixed state
        rho = sp.Matrix.eye(9) / 9
        rho1 = partial_trace_symbolic(rho, keep=0)
        rho2 = partial_trace_symbolic(rho, keep=1)
        
        # Should both be I/3
        expected = sp.Matrix.eye(3) / 3
        assert (rho1 - expected).is_zero_matrix, "Tr₂(I/9) should be I/3"
        assert (rho2 - expected).is_zero_matrix, "Tr₁(I/9) should be I/3"
    
    def test_marginal_entropies_block_structure(self):
        """Test that marginal entropies respect block structure."""
        from qig.symbolic import symbolic_marginal_entropies_two_qutrit
        import sympy as sp
        
        theta = sp.symbols('theta1:17', real=True)
        h1, h2 = symbolic_marginal_entropies_two_qutrit(theta, order=2)
        
        # h₁ should depend only on θ₁...θ₈ (site 1)
        for i in range(8, 16):
            assert sp.diff(h1, theta[i]) == 0, f"h₁ should not depend on θ_{i+1}"
        
        # h₂ should depend only on θ₉...θ₁₆ (site 2)
        for i in range(8):
            assert sp.diff(h2, theta[i]) == 0, f"h₂ should not depend on θ_{i+1}"
        
        # At origin, both should be log(3)
        h1_origin = h1.subs([(t, 0) for t in theta])
        h2_origin = h2.subs([(t, 0) for t in theta])
        assert sp.simplify(h1_origin - sp.log(3)) == 0
        assert sp.simplify(h2_origin - sp.log(3)) == 0
    
    def test_constraint_gradient_structure(self):
        """Test constraint gradient has expected structure."""
        from qig.symbolic import symbolic_constraint_gradient_two_qutrit
        import sympy as sp
        
        theta = sp.symbols('theta1:17', real=True)
        a = symbolic_constraint_gradient_two_qutrit(theta, order=2)
        
        assert a.shape == (16, 1), "Should be 16×1 vector"
        
        # Should be a[i] = -(2/9)θᵢ
        for i in range(16):
            expected = -sp.Rational(2, 9) * theta[i]
            assert sp.simplify(a[i] - expected) == 0, f"a[{i}] should be -(2/9)θ_{i+1}"
    
    def test_lagrange_multiplier(self):
        """Test Lagrange multiplier computation."""
        from qig.symbolic import symbolic_lagrange_multiplier_two_qutrit
        import sympy as sp
        
        theta = sp.symbols('theta1:17', real=True)
        nu = symbolic_lagrange_multiplier_two_qutrit(theta, order=2)
        
        # Should be a symbolic expression
        assert isinstance(nu, sp.Expr), "ν should be a symbolic expression"
        
        # At origin, check if ν is well-defined (not 0/0)
        # Actually at origin θ=0, so a=0, making ν=0/0 undefined
        # Test at a non-zero point
        nu_val = nu.subs([(theta[i], 0.1 if i == 0 else 0) for i in range(16)])
        assert nu_val != sp.nan, "ν should be well-defined away from origin"
    
    def test_grad_lagrange_multiplier(self):
        """Test gradient of Lagrange multiplier."""
        from qig.symbolic import symbolic_grad_lagrange_multiplier_two_qutrit
        import sympy as sp
        
        theta = sp.symbols('theta1:17', real=True)
        grad_nu = symbolic_grad_lagrange_multiplier_two_qutrit(theta, order=2)
        
        assert grad_nu.shape == (16, 1), "Should be 16×1 vector"
        
        # Each component should be a symbolic expression
        for i in range(16):
            assert isinstance(grad_nu[i], sp.Expr), f"∇ν[{i}] should be symbolic"
    
    def test_block_structure_verification(self):
        """Test overall block structure properties."""
        from qig.symbolic import verify_block_structure_two_qutrit
        import sympy as sp
        
        theta = sp.symbols('theta1:17', real=True)
        results = verify_block_structure_two_qutrit(theta)
        
        assert results['h1_local'], f"h₁ locality failed: {results['details']}"
        assert results['h2_local'], f"h₂ locality failed: {results['details']}"
        assert results['a_structure'], f"Constraint gradient structure failed: {results['details']}"
        assert results['constraint_hessian_block'], f"Constraint Hessian block structure failed: {results['details']}"


class TestAntisymmetricPartAnalytic:
    """
    Core validation: analytic A matches numerical to machine precision.
    
    This is the main deliverable of CIP-0007 Phase 4.
    """
    
    @pytest.mark.skip(reason="Analytic A not yet implemented (CIP-0007 Phase 4)")
    def test_antisymmetric_part_matches_numerical(self, qutrit_pair_family, random_states):
        """Test analytic A matches numerical for 100 random states."""
        # from qig.analytic import antisymmetric_part_analytical_qutrit
        # 
        # n_passed = 0
        # max_error = 0.0
        # 
        # for i, theta in enumerate(random_states):
        #     A_analytic = antisymmetric_part_analytical_qutrit(theta)
        #     A_numeric = qutrit_pair_family.antisymmetric_part(theta)
        #     
        #     error = np.linalg.norm(A_analytic - A_numeric, 'fro')
        #     max_error = max(max_error, error)
        #     
        #     # Require machine precision agreement
        #     assert_allclose(A_analytic, A_numeric, atol=1e-12,
        #                    err_msg=f"State {i}: A mismatch, error={error:.2e}")
        #     n_passed += 1
        # 
        # print(f"✓ All {n_passed} states passed. Max error: {max_error:.2e}")
        pass
    
    @pytest.mark.skip(reason="Analytic A not yet implemented (CIP-0007 Phase 4)")
    def test_antisymmetric_property(self, random_states):
        """Test that analytic A is antisymmetric: A + Aᵀ = 0."""
        # from qig.analytic import antisymmetric_part_analytical_qutrit
        # 
        # for theta in random_states[:20]:
        #     A = antisymmetric_part_analytical_qutrit(theta)
        #     
        #     # Check antisymmetry symbolically/numerically
        #     sym_part = A + A.T
        #     
        #     assert_allclose(sym_part, np.zeros_like(A), atol=1e-14,
        #                    err_msg="A should be antisymmetric")
        pass
    
    @pytest.mark.skip(reason="Analytic A not yet implemented (CIP-0007 Phase 4)")
    def test_degeneracy_condition_entropy_gradient(self, qutrit_pair_family, random_states):
        """Test degeneracy: A·(-Gθ) ≈ 0."""
        # from qig.analytic import antisymmetric_part_analytical_qutrit
        # 
        # for theta in random_states[:20]:
        #     A = antisymmetric_part_analytical_qutrit(theta)
        #     G = qutrit_pair_family.fisher_information(theta)
        #     
        #     entropy_grad = -G @ theta
        #     degeneracy_vec = A @ entropy_grad
        #     
        #     norm = np.linalg.norm(degeneracy_vec)
        #     assert norm < 1e-6, f"Degeneracy condition violated: ||A·∇H|| = {norm:.2e}"
        pass
    
    @pytest.mark.skip(reason="Analytic A not yet implemented (CIP-0007 Phase 4)")
    def test_block_structure(self, random_states):
        """Test that A has expected block structure for separable states."""
        # from qig.analytic import antisymmetric_part_analytical_qutrit
        # 
        # # For states near product form, expect block structure
        # theta_product = np.zeros(16)
        # theta_product[:8] = np.random.randn(8) * 0.3  # Site 1 only
        # 
        # A = antisymmetric_part_analytical_qutrit(theta_product)
        # 
        # # Check that A has expected block structure
        # # (depends on exact structure - document in implementation)
        # # For tensor product: expect certain blocks to be zero
        # 
        # # Placeholder test structure
        # pass


class TestSymmetricPartAnalytic:
    """Test analytic symmetric part S."""
    
    @pytest.mark.skip(reason="Analytic S not yet implemented (CIP-0007 Phase 5)")
    def test_symmetric_part_matches_numerical(self, qutrit_pair_family, random_states):
        """Test analytic S matches numerical for 100 random states."""
        # from qig.analytic import symmetric_part_analytical_qutrit
        # 
        # for i, theta in enumerate(random_states):
        #     S_analytic = symmetric_part_analytical_qutrit(theta)
        #     S_numeric = qutrit_pair_family.symmetric_part(theta)
        #     
        #     error = np.linalg.norm(S_analytic - S_numeric, 'fro')
        #     
        #     assert_allclose(S_analytic, S_numeric, atol=1e-12,
        #                    err_msg=f"State {i}: S mismatch, error={error:.2e}")
        pass
    
    @pytest.mark.skip(reason="Analytic S not yet implemented (CIP-0007 Phase 5)")
    def test_symmetric_property(self, random_states):
        """Test that analytic S is symmetric: S - Sᵀ = 0."""
        # from qig.analytic import symmetric_part_analytical_qutrit
        # 
        # for theta in random_states[:20]:
        #     S = symmetric_part_analytical_qutrit(theta)
        #     
        #     antisym_part = S - S.T
        #     
        #     assert_allclose(antisym_part, np.zeros_like(S), atol=1e-14,
        #                    err_msg="S should be symmetric")
        pass
    
    @pytest.mark.skip(reason="Analytic S not yet implemented (CIP-0007 Phase 5)")
    def test_degeneracy_condition_constraint(self, qutrit_pair_family, random_states):
        """Test degeneracy: S·a ≈ 0."""
        # from qig.analytic import symmetric_part_analytical_qutrit
        # 
        # for theta in random_states[:20]:
        #     S = symmetric_part_analytical_qutrit(theta)
        #     _, a = qutrit_pair_family.marginal_entropy_constraint(theta)
        #     
        #     degeneracy_vec = S @ a
        #     norm = np.linalg.norm(degeneracy_vec)
        #     
        #     assert norm < 1e-6, f"Degeneracy condition violated: ||S·a|| = {norm:.2e}"
        pass


class TestGenericStructure:
    """Test full GENERIC structure M = S + A."""
    
    @pytest.mark.skip(reason="Analytic forms not yet implemented (CIP-0007 Phase 5)")
    def test_jacobian_decomposition(self, qutrit_pair_family, random_states):
        """Test that M = S + A exactly."""
        # from qig.analytic import (
        #     antisymmetric_part_analytical_qutrit,
        #     symmetric_part_analytical_qutrit,
        # )
        # 
        # for theta in random_states[:20]:
        #     S = symmetric_part_analytical_qutrit(theta)
        #     A = antisymmetric_part_analytical_qutrit(theta)
        #     M_reconstructed = S + A
        #     
        #     M_numeric = qutrit_pair_family.jacobian(theta)
        #     
        #     assert_allclose(M_reconstructed, M_numeric, atol=1e-12,
        #                    err_msg="M = S + A should match numerical Jacobian")
        pass


class TestIntegration:
    """Integration tests: use analytic forms in full GENERIC decomposition."""
    
    @pytest.mark.skip(reason="Integration not yet implemented (CIP-0007 Phase 6)")
    def test_hamiltonian_extraction_from_analytic_A(self, qutrit_pair_family, random_states):
        """Test that Hamiltonian extraction from analytic A works."""
        # from qig.analytic import antisymmetric_part_analytical_qutrit
        # from qig.generic import effective_hamiltonian_coefficients
        # from qig.structure_constants import compute_structure_constants
        # 
        # f_abc = compute_structure_constants(qutrit_pair_family.operators)
        # 
        # for theta in random_states[:10]:
        #     A_analytic = antisymmetric_part_analytical_qutrit(theta)
        #     A_numeric = qutrit_pair_family.antisymmetric_part(theta)
        #     
        #     eta_analytic, _ = effective_hamiltonian_coefficients(A_analytic, theta, f_abc)
        #     eta_numeric, _ = effective_hamiltonian_coefficients(A_numeric, theta, f_abc)
        #     
        #     assert_allclose(eta_analytic, eta_numeric, atol=1e-10,
        #                    err_msg="Hamiltonian coefficients should match")
        pass


class TestPerformance:
    """Performance benchmarks for analytic vs numerical."""
    
    @pytest.mark.skip(reason="Performance tests for future optimization")
    def test_evaluation_time_comparison(self, qutrit_pair_family):
        """Compare evaluation time: analytic vs numerical."""
        import time
        
        # from qig.analytic import antisymmetric_part_analytical_qutrit
        # 
        # theta = np.random.randn(16) * 0.3
        # 
        # # Time analytic
        # start = time.time()
        # for _ in range(100):
        #     A_analytic = antisymmetric_part_analytical_qutrit(theta)
        # time_analytic = time.time() - start
        # 
        # # Time numerical
        # start = time.time()
        # for _ in range(100):
        #     A_numeric = qutrit_pair_family.antisymmetric_part(theta)
        # time_numeric = time.time() - start
        # 
        # print(f"Analytic: {time_analytic:.3f}s, Numeric: {time_numeric:.3f}s")
        # print(f"Ratio: {time_analytic/time_numeric:.2f}×")
        # 
        # # Goal: within 2× of numerical
        # assert time_analytic < 2 * time_numeric, "Analytic should be reasonably fast"
        pass


if __name__ == "__main__":
    """
    Run tests with: pytest tests/test_analytic_qutrit_validation.py -v
    
    As CIP-0007 progresses, unskip tests progressively:
    - Phase 1: TestSymbolicInfrastructure
    - Phase 2: TestSingleQutritAnalytic
    - Phase 3: TestTwoQutritConstraintGeometry
    - Phase 4: TestAntisymmetricPartAnalytic (main deliverable!)
    - Phase 5: TestSymmetricPartAnalytic, TestGenericStructure
    - Phase 6: TestIntegration
    """
    pytest.main([__file__, "-v", "--tb=short"])

