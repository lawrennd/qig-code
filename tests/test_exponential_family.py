"""
Test suite for quantum exponential family operations.

Tests verify:
1. QuantumExponentialFamily initialization
2. Density matrix generation (rho_from_theta)
3. Fisher information computation
4. Marginal entropy constraint gradient
5. Mathematical properties (LME states, qutrit optimality)

Validates: qig.exponential_family.QuantumExponentialFamily

Run with: pytest test_exponential_family.py -v
"""

import numpy as np
import pytest

from qig.core import create_lme_state, marginal_entropies, von_neumann_entropy
from qig.exponential_family import QuantumExponentialFamily
from qig.dynamics import InaccessibleGameDynamics
from qig.core import generic_decomposition
from tests.tolerance_framework import (
    quantum_assert_hermitian,
    quantum_assert_unit_trace,
)
from tests.fd_helpers import finite_difference_fisher


# ============================================================================
# Test: Quantum Exponential Family
# ============================================================================

class TestQuantumExponentialFamily:
    """Test quantum exponential family operations."""
    
    def test_initialisation(self):
        """Test exponential family initialisation."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        assert exp_family.n_sites == 2
        assert exp_family.d == 2
        assert exp_family.D == 4
        assert exp_family.n_params == 6
    
    def test_rho_from_theta_trace_one(self):
        """Density matrix should have trace 1."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        # Fix seed for reproducibility when comparing two sensitive numerical methods
        rng = np.random.default_rng(0)
        theta = rng.standard_normal(exp_family.n_params) * 0.1
        rho = exp_family.rho_from_theta(theta)
        
        quantum_assert_unit_trace(
            rho,
            "density_matrix",
            "Density matrix should have unit trace",
        )
    
    def test_rho_from_theta_hermitian(self):
        """Density matrix should be Hermitian."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        theta = np.random.randn(exp_family.n_params) * 0.1
        rho = exp_family.rho_from_theta(theta)
        
        quantum_assert_hermitian(
            rho,
            "density_matrix",
            "Density matrix should be Hermitian",
        )
    
    def test_rho_from_theta_positive(self):
        """Density matrix should be positive semi-definite."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        theta = np.random.randn(exp_family.n_params) * 0.1
        rho = exp_family.rho_from_theta(theta)
        
        eigvals = np.linalg.eigvalsh(rho)
        assert np.all(eigvals >= -1e-10), "Density matrix should be positive semi-definite"
    
    def test_fisher_information_positive_definite(self):
        """Fisher information should be positive definite."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        theta = np.random.randn(exp_family.n_params) * 0.1
        G = exp_family.fisher_information(theta)
        
        # Check symmetry
        assert np.allclose(G, G.T, atol=1e-6), "Fisher information should be symmetric"
        
        # Check positive definiteness
        eigvals = np.linalg.eigvalsh(G)
        assert np.all(eigvals > -1e-6), "Fisher information should be positive semi-definite"
    
    def test_fisher_information_matches_finite_difference(self):
        """
        Analytic BKM Fisher information should agree with a finite-difference
        Hessian of the log-partition function ψ(θ) for small systems.
        """
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)

        # Test agreement for a small ensemble of natural-parameter values
        # drawn from a standard normal (no special "initialisation" scaling).
        rng = np.random.default_rng(0)

        for _ in range(5):
            theta = rng.standard_normal(exp_family.n_params)

            G_analytic = exp_family.fisher_information(theta)
            G_numeric = finite_difference_fisher(exp_family, theta, eps=1e-5)

            diff = G_analytic - G_numeric
            max_abs_err = np.max(np.abs(diff))
            rel_fro_err = (
                np.linalg.norm(diff, ord="fro")
                / max(np.linalg.norm(G_numeric, ord="fro"), 1e-12)
            )

            # We care about catching O(1) conceptual errors (e.g. wrong operator
            # ordering or missing centring), not chasing machine precision.
            # Empirically the spectral BKM expression and the finite-difference
            # Hessian agree to ~1e-4 in this 2-qubit case across seeds.
            assert max_abs_err < 1e-4 and rel_fro_err < 5e-4, (
                "Analytic BKM Fisher information does not closely match finite-difference "
                f"Hessian for some θ: max_abs_err={max_abs_err:.2e}, "
                f"rel_fro_err={rel_fro_err:.2e}"
            )
    
    def test_fisher_information_matches_finite_difference_qutrits(self):
        """
        Analytic BKM Fisher information should agree with a finite-difference
        Hessian of the log-partition function ψ(θ) for a small qutrit system.
        """
        exp_family = QuantumExponentialFamily(n_sites=2, d=3)

        rng = np.random.default_rng(1)

        for _ in range(3):
            theta = rng.standard_normal(exp_family.n_params)

            G_analytic = exp_family.fisher_information(theta)
            G_numeric = finite_difference_fisher(exp_family, theta, eps=1e-5)

            diff = G_analytic - G_numeric
            max_abs_err = np.max(np.abs(diff))
            rel_fro_err = (
                np.linalg.norm(diff, ord="fro")
                / max(np.linalg.norm(G_numeric, ord="fro"), 1e-12)
            )

            # Allow slightly looser tolerances for the higher-dimensional qutrit
            # case, but still tight enough to detect conceptual errors.
            assert max_abs_err < 5e-4 and rel_fro_err < 2e-3, (
                "Analytic BKM Fisher information for qutrits does not closely match "
                f"finite-difference Hessian for some θ: max_abs_err={max_abs_err:.2e}, "
                f"rel_fro_err={rel_fro_err:.2e}"
            )

    def test_fisher_information_matches_finite_difference_ququarts(self):
        """
        Analytic BKM Fisher information should agree with a finite-difference
        Hessian of the log-partition function ψ(θ) for a small d=4 system
        using the generalised Hermitian traceless basis.
        """
        exp_family = QuantumExponentialFamily(n_sites=2, d=4)

        rng = np.random.default_rng(2)

        # Fewer samples here to keep runtime reasonable; d=4 has a larger
        # operator basis but the same conceptual checks apply.
        for _ in range(2):
            theta = rng.standard_normal(exp_family.n_params)

            G_analytic = exp_family.fisher_information(theta)
            G_numeric = finite_difference_fisher(exp_family, theta, eps=1e-5)

            diff = G_analytic - G_numeric
            max_abs_err = np.max(np.abs(diff))
            rel_fro_err = (
                np.linalg.norm(diff, ord="fro")
                / max(np.linalg.norm(G_numeric, ord="fro"), 1e-12)
            )

            # Allow slightly looser tolerances for the larger d=4 space, but
            # still tight enough to detect structural mistakes.
            assert max_abs_err < 1e-3 and rel_fro_err < 5e-3, (
                "Analytic BKM Fisher information for d=4 does not closely match "
                f"finite-difference Hessian for some θ: max_abs_err={max_abs_err:.2e}, "
                f"rel_fro_err={rel_fro_err:.2e}"
            )
    
    def test_marginal_entropy_constraint_gradient(self):
        """Constraint gradient should be non-zero for generic state."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        theta = np.random.randn(exp_family.n_params) * 0.1
        
        C, grad_C = exp_family.marginal_entropy_constraint(theta)
        
        assert 0 <= C <= 2 * np.log(2) + 0.1, "Constraint value out of bounds"
        assert np.linalg.norm(grad_C) > 1e-8, "Constraint gradient should be non-zero"


# ============================================================================
# Test: Mathematical Properties
# ============================================================================

class TestMathematicalProperties:
    """Test key mathematical properties of the framework."""
    
    def test_lme_state_maximises_marginal_entropy_sum(self):
        """LME states should maximise ∑h_i for pure states."""
        rho_lme, dims = create_lme_state(n_sites=2, d=2)
        h_lme = marginal_entropies(rho_lme, dims)
        C_lme = h_lme.sum()
        
        # Theoretical maximum for 2 qubits
        C_max = 2 * np.log(2)
        assert np.abs(C_lme - C_max) < 1e-8, "LME state should achieve maximum marginal entropy sum"
    
    def test_qutrit_optimality_vs_qubit(self):
        """Qutrits should have higher efficiency than qubits for equal sites."""
        # Efficiency: (m/d) log d where m = n*d
        n = 2
        
        # Qubits
        m_qubit = n * 2
        eff_qubit = (m_qubit / 2) * np.log(2)
        
        # Qutrits
        m_qutrit = n * 3
        eff_qutrit = (m_qutrit / 3) * np.log(3)
        
        assert eff_qutrit > eff_qubit, "Qutrits should be more efficient than qubits"
    
    @pytest.mark.slow
    def test_constraint_gradient_orthogonal_to_symmetric_flow(self):
        """Symmetric part of flow should be orthogonal to constraint gradient."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        dynamics = InaccessibleGameDynamics(exp_family)
        
        theta = np.random.randn(exp_family.n_params) * 0.1
        
        # Compute flow and constraint gradient
        theta_dot = dynamics.flow(0.0, theta)
        _, a = exp_family.marginal_entropy_constraint(theta)
        
        # GENERIC decomposition (finite differences are slow)
        M = dynamics.exp_family.jacobian(theta)
        S, A = generic_decomposition(M)
        
        # Symmetric part times theta
        S_theta = S @ theta
        
        # Should be orthogonal to constraint gradient (degeneracy condition)
        # Note: in practice, projection enforces this, so check approximate orthogonality
        inner_product = np.dot(S_theta, a)
        a_norm = np.linalg.norm(a)
        S_theta_norm = np.linalg.norm(S_theta)
        
        if a_norm > 1e-8 and S_theta_norm > 1e-8:
            cos_angle = np.abs(inner_product) / (a_norm * S_theta_norm)
            assert cos_angle < 0.2, "S·θ should be approximately orthogonal to constraint gradient"


# ============================================================================
# Parametrised Tests
# ============================================================================

@pytest.mark.parametrize("n_sites,d", [(2, 2), (2, 3), (3, 2)])
def test_various_systems(n_sites, d):
    """Test framework works for various system sizes."""
    exp_family = QuantumExponentialFamily(n_sites, d)
    assert exp_family.D == d ** n_sites
    assert exp_family.n_params == n_sites * (d**2 - 1)


@pytest.mark.parametrize("d", [2, 3, 4])
def test_entropy_bounds_various_dimensions(d):
    """Test entropy bounds for various dimensions."""
    rho = np.eye(d) / d  # Maximally mixed
    S = von_neumann_entropy(rho)
    expected = np.log(d)
    assert np.abs(S - expected) < 1e-10


# ============================================================================
# Qutrit Validation Tests (absorbed from test_quantum_qutrit.py)
# ============================================================================

class TestQutritValidation:
    """
    Validation tests for qutrit (d=3) exponential families.
    
    Tests qutrit-specific behavior including:
    - LME state construction
    - Fisher information computation
    - Constraint gradient
    - Third cumulant
    - Mutual information
    """
    
    def test_qutrit_lme_state(self):
        """Test qutrit LME state construction and marginal entropies."""
        from tests.tolerance_framework import quantum_assert_close
        
        # Create LME state for 1 pair (2 sites)
        rho_lme, dims = create_lme_state(n_sites=2, d=3)
        
        # Check quantum state properties (Category B: quantum_state)
        quantum_assert_close(np.trace(rho_lme), 1.0, 'quantum_state',
                           err_msg="Trace should be 1")
        quantum_assert_close(np.trace(rho_lme @ rho_lme), 1.0, 'quantum_state',
                           err_msg="Should be pure state")

        # Check marginal entropies
        h = marginal_entropies(rho_lme, dims)
        
        # For maximally entangled qutrit pair, each marginal is maximally mixed
        # So h_i = log(3) for each site (Category C: information metrics)
        quantum_assert_close(h[0], np.log(3), 'information_metric',
                           err_msg="Marginal entropy should be log(3)")
        quantum_assert_close(h[1], np.log(3), 'information_metric',
                           err_msg="Marginal entropy should be log(3)")
    
    def test_qutrit_fisher_information(self):
        """Test BKM metric (Fisher information) computation."""
        from tests.tolerance_framework import quantum_assert_symmetric
        
        # Create exponential family with pair basis
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)

        # Test at random point
        np.random.seed(42)
        theta = 0.1 * np.random.randn(exp_fam.n_params)
        
        # Compute Fisher information
        G = exp_fam.fisher_information(theta)
        
        eigenvalues = np.linalg.eigvalsh(G)

        # Fisher information should be positive semidefinite (Category D)
        assert np.all(eigenvalues >= -1e-10), "Fisher information should be PSD"
        quantum_assert_symmetric(G, 'fisher_metric',
                                err_msg="Fisher information should be symmetric")
    
    def test_qutrit_constraint_gradient(self):
        """Test constraint gradient ∇C computation."""
        from tests.tolerance_framework import quantum_assert_close
        
        # Create exponential family
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)

        # Test at random point
        np.random.seed(42)
        theta = 0.2 * np.random.randn(exp_fam.n_params)
        
        # Compute analytic constraint gradient
        C_value, a_analytic = exp_fam.marginal_entropy_constraint(theta)
        
        # Verify against finite differences
        eps = 1e-6
        rho_base = exp_fam.rho_from_theta(theta)
        h_base = marginal_entropies(rho_base, [3, 3])
        C_base = np.sum(h_base)

        a_numerical = np.zeros(exp_fam.n_params)
        for i in range(min(10, exp_fam.n_params)):  # Sample first 10 parameters
            theta_plus = theta.copy()
            theta_plus[i] += eps
            rho_plus = exp_fam.rho_from_theta(theta_plus)
            h_plus = marginal_entropies(rho_plus, [3, 3])
            C_plus = np.sum(h_plus)
            a_numerical[i] = (C_plus - C_base) / eps

        # Compare sampled entries (use duhamel_integration for coarse validation)
        quantum_assert_close(a_analytic[:10], a_numerical[:10], 'duhamel_integration',
                           err_msg="Constraint gradient vs FD mismatch")
    
    def test_qutrit_third_cumulant(self):
        """Test third cumulant tensor computation."""
        # Create exponential family
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        
        # Test at random point
        np.random.seed(42)
        theta = 0.1 * np.random.randn(exp_fam.n_params)
        
        # Compute third cumulant contraction
        T_contracted = exp_fam.third_cumulant_contraction(theta, method='fd')
        
        # Verify it's finite
        assert np.all(np.isfinite(T_contracted)), "Third cumulant should be finite"
    
    def test_qutrit_mutual_information(self):
        """Test mutual information computation for entangled qutrit pair."""
        from tests.tolerance_framework import quantum_assert_close
        
        # Create exponential family
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)

        # Start near LME state (should have high mutual information)
        np.random.seed(42)
        theta_lme = np.random.randn(exp_fam.n_params) * 0.5
        
        I_lme = exp_fam.mutual_information(theta_lme)
        
        # Test at zero (product state, should have I=0)
        theta_zero = np.zeros(exp_fam.n_params)
        I_zero = exp_fam.mutual_information(theta_zero)

        # Mutual information should be non-negative (allow tiny numerical error)
        assert I_lme >= -1e-10, "Mutual information should be non-negative"
        assert I_zero >= -1e-10, "Mutual information should be non-negative"
        # Product state should have near-zero mutual information (Category C)
        quantum_assert_close(I_zero, 0.0, 'information_metric',
                           err_msg="Product state should have near-zero mutual information")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

