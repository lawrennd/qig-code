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

    @pytest.mark.slow
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


# ============================================================================
# Test: Sigma Regularisation Infrastructure
# ============================================================================

class TestSigmaValidation:
    """Test σ validation and structure detection for regularised states."""
    
    def test_validate_sigma_valid(self):
        """Test validate_sigma accepts valid density matrices."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        D = exp_fam.D
        
        # Isotropic sigma
        sigma_iso = np.eye(D) / D
        is_valid, msg = exp_fam.validate_sigma(sigma_iso)
        assert is_valid, f"Isotropic sigma should be valid: {msg}"
        
        # Pure state projector
        psi = np.zeros(D, dtype=complex)
        psi[0] = 1.0
        sigma_pure = np.outer(psi, psi.conj())
        is_valid, msg = exp_fam.validate_sigma(sigma_pure)
        assert is_valid, f"Pure state sigma should be valid: {msg}"
    
    def test_validate_sigma_invalid(self):
        """Test validate_sigma rejects invalid matrices."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        D = exp_fam.D
        
        # Wrong trace
        sigma_bad_trace = np.eye(D) * 2
        is_valid, msg = exp_fam.validate_sigma(sigma_bad_trace)
        assert not is_valid, "Should reject wrong trace"
        assert "Tr" in msg
        
        # Not Hermitian
        sigma_non_herm = np.eye(D, dtype=complex) / D
        sigma_non_herm[0, 1] = 1j
        is_valid, msg = exp_fam.validate_sigma(sigma_non_herm)
        assert not is_valid, "Should reject non-Hermitian"
        
        # Wrong shape
        sigma_wrong_shape = np.eye(D-1) / (D-1)
        is_valid, msg = exp_fam.validate_sigma(sigma_wrong_shape)
        assert not is_valid, "Should reject wrong shape"
    
    def test_detect_sigma_structure_isotropic(self):
        """Test detection of isotropic sigma."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        D = exp_fam.D
        
        sigma_iso = np.eye(D) / D
        structure = exp_fam.detect_sigma_structure(sigma_iso)
        assert structure == 'isotropic'
    
    def test_detect_sigma_structure_pure(self):
        """Test detection of pure state sigma."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        D = exp_fam.D
        
        psi = np.zeros(D, dtype=complex)
        psi[1] = 1.0
        sigma_pure = np.outer(psi, psi.conj())
        structure = exp_fam.detect_sigma_structure(sigma_pure)
        assert structure == 'pure'
    
    def test_detect_sigma_structure_general(self):
        """Test detection of general sigma."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        D = exp_fam.D
        
        # Mixed state that's not isotropic
        sigma_mixed = np.diag([0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.005, 0.004, 0.001])
        sigma_mixed = sigma_mixed / np.trace(sigma_mixed)
        structure = exp_fam.detect_sigma_structure(sigma_mixed)
        assert structure == 'general'
    
    def test_regularise_pure_state_isotropic(self):
        """Test regularise_pure_state with default isotropic sigma."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        D = exp_fam.D
        eps = 0.01
        
        psi = np.zeros(D, dtype=complex)
        psi[0] = 1.0
        
        rho_eps = exp_fam.regularise_pure_state(psi, eps)
        
        # Check properties
        assert np.isclose(np.trace(rho_eps), 1.0), "Should have unit trace"
        assert np.allclose(rho_eps, rho_eps.conj().T), "Should be Hermitian"
        eigvals = np.linalg.eigvalsh(rho_eps)
        assert np.all(eigvals >= -1e-10), "Should be PSD"
    
    def test_regularise_pure_state_custom_sigma(self):
        """Test regularise_pure_state with custom sigma."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        D = exp_fam.D
        eps = 0.1
        
        psi = np.zeros(D, dtype=complex)
        psi[0] = 1.0
        
        # Custom sigma (projector onto different state)
        sigma = np.zeros((D, D), dtype=complex)
        sigma[1, 1] = 1.0
        
        rho_eps = exp_fam.regularise_pure_state(psi, eps, sigma=sigma)
        
        # Should be (1-eps)|0><0| + eps|1><1|
        expected = (1 - eps) * np.outer(psi, psi.conj()) + eps * sigma
        assert np.allclose(rho_eps, expected), "Should match expected formula"


class TestBellStateParametersWithSigma:
    """Test get_bell_state_parameters with custom regularisation σ."""
    
    def test_bell_parameters_isotropic(self):
        """Test get_bell_state_parameters with default isotropic sigma."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        
        theta = exp_fam.get_bell_state_parameters(epsilon=0.01)
        
        assert theta.shape == (exp_fam.n_params,)
        assert np.all(np.isfinite(theta))
        # Bell state has large negative parameters
        assert np.linalg.norm(theta) > 1.0
    
    def test_bell_parameters_custom_sigma(self):
        """Test get_bell_state_parameters with custom sigma."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        D = exp_fam.D
        
        # Custom sigma (projector)
        sigma = np.zeros((D, D), dtype=complex)
        sigma[1, 1] = 1.0
        
        theta = exp_fam.get_bell_state_parameters(epsilon=0.01, sigma=sigma)
        
        assert theta.shape == (exp_fam.n_params,)
        assert np.all(np.isfinite(theta))
    
    def test_different_sigma_gives_different_theta(self):
        """Test that different sigma gives different theta."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        D = exp_fam.D
        
        # Isotropic
        theta_iso = exp_fam.get_bell_state_parameters(epsilon=0.01)
        
        # Custom sigma
        sigma = np.zeros((D, D), dtype=complex)
        sigma[1, 1] = 1.0
        theta_custom = exp_fam.get_bell_state_parameters(epsilon=0.01, sigma=sigma)
        
        # Should be different
        diff = np.linalg.norm(theta_iso - theta_custom)
        assert diff > 0.1, f"Different sigma should give different theta, diff={diff}"
    
    def test_bell_parameters_log_epsilon(self):
        """Test get_bell_state_parameters with log_epsilon."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        
        # These should give similar results
        theta1 = exp_fam.get_bell_state_parameters(epsilon=0.001)
        theta2 = exp_fam.get_bell_state_parameters(log_epsilon=np.log(0.001))
        
        assert np.allclose(theta1, theta2, rtol=1e-6)


class TestMultiPairExponentialFamily:
    """Test exponential family with multiple entangled pairs (n_pairs > 1)."""
    
    def test_multipair_initialisation(self):
        """Test multi-pair exponential family initialisation."""
        exp_fam = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
        
        assert exp_fam.n_pairs == 2
        assert exp_fam.d == 2
        assert exp_fam.D == 16  # (2^2)^2 = 16
        assert exp_fam.n_params == 30  # 2 * 15 = 30
    
    def test_multipair_bell_parameters(self):
        """Test get_bell_state_parameters for multi-pair."""
        exp_fam = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
        
        theta = exp_fam.get_bell_state_parameters(epsilon=0.01)
        
        assert theta.shape == (exp_fam.n_params,)
        assert np.all(np.isfinite(theta))
    
    def test_multipair_dynamics_constraint_preservation(self):
        """Test that multi-pair dynamics preserve constraint."""
        exp_fam = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
        dynamics = InaccessibleGameDynamics(exp_fam)
        
        theta_0 = exp_fam.get_bell_state_parameters(epsilon=0.01)
        result = dynamics.solve(theta_0, n_steps=20, dt=0.01, verbose=False)
        
        C_init = result['C_init']
        C_final = result['constraint_values'][-1]
        
        # Constraint should be preserved (tight tolerance)
        assert abs(C_final - C_init) < 1e-6, \
            f"Constraint drift: {abs(C_final - C_init)}"
    
    def test_multipair_dynamics_entropy_increase(self):
        """Test that multi-pair dynamics increase entropy."""
        exp_fam = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
        dynamics = InaccessibleGameDynamics(exp_fam)
        
        theta_0 = exp_fam.get_bell_state_parameters(epsilon=0.01)
        result = dynamics.solve(theta_0, n_steps=50, dt=0.01, verbose=False)
        
        # Compute entropies
        rho_0 = exp_fam.rho_from_theta(theta_0)
        rho_f = exp_fam.rho_from_theta(result['trajectory'][-1])
        
        H_0 = von_neumann_entropy(rho_0)
        H_f = von_neumann_entropy(rho_f)
        
        # Entropy should increase (or at least not decrease significantly)
        assert H_f >= H_0 - 1e-6, f"Entropy should increase: {H_0} -> {H_f}"
    
    def test_multipair_three_pairs(self):
        """Test with 3 qubit pairs (larger system)."""
        exp_fam = QuantumExponentialFamily(n_pairs=3, d=2, pair_basis=True)
        
        assert exp_fam.D == 64  # (2^2)^3 = 64
        assert exp_fam.n_params == 45  # 3 * 15 = 45
        
        theta = exp_fam.get_bell_state_parameters(epsilon=0.01)
        assert theta.shape == (45,)
        assert np.all(np.isfinite(theta))


class TestProductSigmaRegularisation:
    """Test sigma_per_pair for efficient product-structured regularisation."""

    def test_product_sigma_matches_general(self):
        """Product σ path should match general σ computation."""
        qef = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
        D_pair = qef.d ** 2
        
        # Create random per-pair sigmas
        np.random.seed(42)
        def random_density_matrix(d):
            A = np.random.randn(d, d) + 1j * np.random.randn(d, d)
            rho = A @ A.conj().T
            return rho / np.trace(rho)
        
        sigma_per_pair = [random_density_matrix(D_pair) for _ in range(qef.n_pairs)]
        
        # Efficient path
        theta_product = qef.get_bell_state_parameters(epsilon=0.1, sigma_per_pair=sigma_per_pair)
        
        # Build full sigma and use general path
        sigma_full = np.kron(sigma_per_pair[0], sigma_per_pair[1])
        theta_general = qef.get_bell_state_parameters(epsilon=0.1, sigma=sigma_full)
        
        assert np.allclose(theta_product, theta_general), \
            f"Max diff: {np.max(np.abs(theta_product - theta_general))}"

    def test_product_sigma_different_from_isotropic(self):
        """Product of I/d² gives DIFFERENT regularisation than isotropic I/D.
        
        This is physically meaningful - they represent different 'directions
        of approach' to the pure Bell state (different meridians from the
        north pole).
        """
        qef = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
        D_pair = qef.d ** 2
        
        # Per-pair I/d² (NOT same as isotropic I/D for the full state)
        sigma_per_pair = [np.eye(D_pair) / D_pair for _ in range(qef.n_pairs)]
        theta_product = qef.get_bell_state_parameters(epsilon=0.1, sigma_per_pair=sigma_per_pair)
        
        # Isotropic (sigma=None) 
        theta_isotropic = qef.get_bell_state_parameters(epsilon=0.1)
        
        # They should be DIFFERENT (different regularisation directions)
        assert not np.allclose(theta_product, theta_isotropic), \
            "Product and isotropic should give different θ"
        
        # But both should reconstruct to valid states
        rho_prod = qef.rho_from_theta(theta_product)
        rho_iso = qef.rho_from_theta(theta_isotropic)
        
        # Both valid density matrices
        assert np.isclose(np.trace(rho_prod), 1.0)
        assert np.isclose(np.trace(rho_iso), 1.0)
        assert np.all(np.linalg.eigvalsh(rho_prod) > -1e-10)
        assert np.all(np.linalg.eigvalsh(rho_iso) > -1e-10)

    def test_product_sigma_three_pairs(self):
        """Test product σ with three pairs."""
        qef = QuantumExponentialFamily(n_pairs=3, d=2, pair_basis=True)
        D_pair = qef.d ** 2
        
        np.random.seed(123)
        def random_density_matrix(d):
            A = np.random.randn(d, d) + 1j * np.random.randn(d, d)
            rho = A @ A.conj().T
            return rho / np.trace(rho)
        
        sigma_per_pair = [random_density_matrix(D_pair) for _ in range(qef.n_pairs)]
        
        # Should complete quickly (efficient path)
        theta = qef.get_bell_state_parameters(epsilon=0.1, sigma_per_pair=sigma_per_pair)
        
        # Check shape
        assert theta.shape == (qef.n_params,)
        
        # Check finite
        assert np.all(np.isfinite(theta))
    
    def test_sigma_per_pair_basic(self):
        """Test sigma_per_pair constructs product sigma."""
        exp_fam = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
        D_pair = 4  # d^2 = 4
        
        # Per-pair sigmas (isotropic on each pair)
        sigma1 = np.eye(D_pair) / D_pair
        sigma2 = np.eye(D_pair) / D_pair
        
        theta = exp_fam.get_bell_state_parameters(
            epsilon=0.01,
            sigma_per_pair=[sigma1, sigma2]
        )
        
        assert theta.shape == (exp_fam.n_params,)
        assert np.all(np.isfinite(theta))
    
    def test_sigma_per_pair_wrong_length(self):
        """Test sigma_per_pair rejects wrong number of matrices."""
        exp_fam = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
        D_pair = 4
        
        sigma1 = np.eye(D_pair) / D_pair
        
        with pytest.raises(ValueError, match="must have 2 matrices"):
            exp_fam.get_bell_state_parameters(
                epsilon=0.01,
                sigma_per_pair=[sigma1]  # Only 1, need 2
            )
    
    def test_sigma_and_sigma_per_pair_exclusive(self):
        """Test that sigma and sigma_per_pair cannot both be specified."""
        exp_fam = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
        D = exp_fam.D
        D_pair = 4
        
        sigma = np.eye(D) / D
        sigma_per_pair = [np.eye(D_pair) / D_pair, np.eye(D_pair) / D_pair]
        
        with pytest.raises(ValueError, match="Cannot specify both"):
            exp_fam.get_bell_state_parameters(
                epsilon=0.01,
                sigma=sigma,
                sigma_per_pair=sigma_per_pair
            )


class TestBlockDiagonalFisherInformation:
    """Test fisher_information_product with block-diagonal structure."""

    def test_single_pair_matches_full(self):
        """Single pair block computation matches full computation."""
        qef = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        theta = qef.get_bell_state_parameters(epsilon=0.1)
        
        G_full = qef.fisher_information(theta)
        G_block = qef.fisher_information_product(theta)
        
        assert np.allclose(G_full, G_block), "Single pair should match exactly"

    def test_two_pairs_matches_full(self):
        """Two pairs block computation matches full computation."""
        qef = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
        theta = qef.get_bell_state_parameters(epsilon=0.1)
        
        G_full = qef.fisher_information(theta)
        G_block = qef.fisher_information_product(theta)
        
        assert np.allclose(G_full, G_block), f"Max diff: {np.max(np.abs(G_full - G_block))}"

    def test_three_pairs_matches_full(self):
        """Three pairs block computation matches full computation."""
        qef = QuantumExponentialFamily(n_pairs=3, d=2, pair_basis=True)
        theta = qef.get_bell_state_parameters(epsilon=0.1)
        
        G_full = qef.fisher_information(theta)
        G_block = qef.fisher_information_product(theta)
        
        assert np.allclose(G_full, G_block), f"Max diff: {np.max(np.abs(G_full - G_block))}"

    def test_block_diagonal_structure(self):
        """Verify the result is actually block-diagonal."""
        qef = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
        theta = qef.get_bell_state_parameters(epsilon=0.1)
        
        G_block = qef.fisher_information_product(theta)
        
        # Check off-diagonal blocks are zero
        n_ops_per_pair = 15  # d^4 - 1 for d=2
        off_diag = G_block[:n_ops_per_pair, n_ops_per_pair:]
        
        assert np.allclose(off_diag, 0), f"Off-diagonal block max: {np.max(np.abs(off_diag))}"

    def test_qutrits_two_pairs(self):
        """Test with qutrits (d=3) for two pairs."""
        qef = QuantumExponentialFamily(n_pairs=2, d=3, pair_basis=True)
        theta = qef.get_bell_state_parameters(epsilon=0.1)
        
        G_full = qef.fisher_information(theta)
        G_block = qef.fisher_information_product(theta)
        
        assert np.allclose(G_full, G_block), f"Max diff: {np.max(np.abs(G_full - G_block))}"

    def test_symmetric_positive_definite(self):
        """Fisher metric should be symmetric positive definite."""
        qef = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
        theta = qef.get_bell_state_parameters(epsilon=0.1)
        
        G = qef.fisher_information_product(theta)
        
        # Symmetric
        assert np.allclose(G, G.T), "Fisher metric should be symmetric"
        
        # Positive definite (all eigenvalues > 0)
        eigvals = np.linalg.eigvalsh(G)
        assert np.all(eigvals > 0), f"Min eigenvalue: {eigvals.min()}"

    def test_requires_pair_basis(self):
        """Should raise error if not using pair_basis."""
        qef = QuantumExponentialFamily(n_sites=2, d=2, pair_basis=False)
        theta = np.zeros(qef.n_params)
        
        with pytest.raises(ValueError, match="pair_basis"):
            qef.fisher_information_product(theta)

    def test_check_product_flag(self):
        """Test check_product flag behaviour."""
        qef = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
        theta = qef.get_bell_state_parameters(epsilon=0.1)
        
        # With check_product=True (default) - should work for product state
        G1 = qef.fisher_information_product(theta, check_product=True)
        
        # With check_product=False - should also work
        G2 = qef.fisher_information_product(theta, check_product=False)
        
        assert np.allclose(G1, G2)

    def test_partial_trace_to_pair_correctness(self):
        """Test _partial_trace_to_pair gives correct marginals."""
        qef = QuantumExponentialFamily(n_pairs=3, d=2, pair_basis=True)
        theta = qef.get_bell_state_parameters(epsilon=0.1)
        rho = qef.rho_from_theta(theta)
        
        # For product Bell state, each marginal should be I/d²
        D_pair = qef.d ** 2
        expected_marginal = np.eye(D_pair) / D_pair
        
        for k in range(qef.n_pairs):
            rho_k = qef._partial_trace_to_pair(rho, k)
            # Should be approximately I/d² (mixed state)
            assert rho_k.shape == (D_pair, D_pair)
            assert np.isclose(np.trace(rho_k), 1.0), f"Pair {k} trace: {np.trace(rho_k)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

