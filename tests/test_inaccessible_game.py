#!/usr/bin/env python3
"""
Test suite for quantum inaccessible game validation framework.

Tests verify:
1. Quantum state utilities (entropy, partial trace, LME states)
2. Operator basis construction (Pauli, Gell-Mann)
3. Exponential family operations
4. Constrained dynamics (constraint preservation, entropy increase)
5. GENERIC decomposition
6. Numerical stability and edge cases

Run with: pytest test_inaccessible_game.py -v

For quick tests (skip slow ones):
    pytest test_inaccessible_game.py -v -m "not slow"

To run only slow tests:
    pytest test_inaccessible_game.py -v -m "slow"
"""

import numpy as np
import pytest

from qig.core import (
    partial_trace,
    von_neumann_entropy,
    create_lme_state,
    marginal_entropies,
)
from qig.exponential_family import (
    QuantumExponentialFamily,
    pauli_basis,
    gell_mann_matrices,
    qutrit_basis,
    create_operator_basis,
)
from qig.dynamics import InaccessibleGameDynamics
from qig.core import generic_decomposition
from tests.tolerance_framework import (
    quantum_assert_close,
    quantum_assert_scalar_close,
    quantum_assert_hermitian,
    quantum_assert_unit_trace,
)

# Configure pytest markers
pytest.mark.slow = pytest.mark.slow


# ============================================================================
# Test: Quantum State Utilities
# ============================================================================

class TestQuantumStateUtilities:
    """Test basic quantum state operations."""
    
    def test_von_neumann_entropy_pure_state(self):
        """Pure state should have zero entropy."""
        psi = np.array([1, 0], dtype=complex)
        rho = np.outer(psi, psi.conj())
        S = von_neumann_entropy(rho)
        quantum_assert_scalar_close(
            S,
            0.0,
            "entropy",
            "Pure state entropy should be ~0",
            atol=1e-10,
            rtol=0.0,
        )
    
    def test_von_neumann_entropy_maximally_mixed(self):
        """Maximally mixed state should have entropy log(d)."""
        d = 3
        rho = np.eye(d) / d
        S = von_neumann_entropy(rho)
        expected = np.log(d)
        quantum_assert_scalar_close(
            S,
            expected,
            "entropy",
            f"Maximally mixed entropy should be log({d})",
            atol=1e-10,
            rtol=0.0,
        )
    
    def test_von_neumann_entropy_bounds(self):
        """Entropy should satisfy 0 ≤ S ≤ log(d)."""
        d = 4
        # Random density matrix
        A = np.random.randn(d, d) + 1j * np.random.randn(d, d)
        rho = A @ A.conj().T
        rho = rho / np.trace(rho)
        
        S = von_neumann_entropy(rho)
        assert 0 <= S <= np.log(d) + 1e-8, "Entropy out of bounds"
    
    def test_partial_trace_two_qubits(self):
        """Test partial trace for Bell state."""
        # Bell state: |Φ+⟩ = (|00⟩ + |11⟩)/√2
        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho = np.outer(psi, psi.conj())
        
        # Trace out second qubit
        rho_1 = partial_trace(rho, dims=[2, 2], keep=0)
        
        # Should be maximally mixed: I/2
        expected = np.eye(2) / 2
        quantum_assert_close(
            rho_1,
            expected,
            "density_matrix",
            "Bell state marginal should be I/2",
            atol=1e-10,
            rtol=0.0,
        )
    
    def test_partial_trace_preserves_trace(self):
        """Partial trace should preserve trace = 1."""
        d1, d2 = 2, 3
        D = d1 * d2
        A = np.random.randn(D, D) + 1j * np.random.randn(D, D)
        rho = A @ A.conj().T
        rho = rho / np.trace(rho)
        
        rho_1 = partial_trace(rho, dims=[d1, d2], keep=0)
        rho_2 = partial_trace(rho, dims=[d1, d2], keep=1)
        
        quantum_assert_unit_trace(
            rho_1,
            "density_matrix",
            "Partial trace should preserve unit trace (subsystem 1)",
        )
        quantum_assert_unit_trace(
            rho_2,
            "density_matrix",
            "Partial trace should preserve unit trace (subsystem 2)",
        )
    
    def test_create_lme_state_two_qubits(self):
        """LME state for 2 qubits should be Bell state."""
        rho, dims = create_lme_state(n_sites=2, d=2)
        
        # Should be pure
        quantum_assert_scalar_close(
            np.trace(rho @ rho),
            1.0,
            "purity",
            "LME state should be pure",
            atol=1e-10,
            rtol=0.0,
        )
        
        # Marginals should be maximally mixed
        h = marginal_entropies(rho, dims)
        expected = np.log(2)
        quantum_assert_close(
            h,
            np.array([expected, expected]),
            "marginal_entropy",
            "Marginals should be maximally mixed",
            atol=1e-10,
            rtol=0.0,
        )
    
    def test_create_lme_state_three_qutrits(self):
        """LME state for 3 qutrits should have correct marginal entropies."""
        rho, dims = create_lme_state(n_sites=3, d=3)
        
        # Should be pure
        purity = np.trace(rho @ rho).real
        quantum_assert_scalar_close(
            purity,
            1.0,
            "purity",
            "LME state should be pure",
            atol=1e-10,
            rtol=0.0,
        )
        
        # Check marginal entropy sum (one site will be pure for odd n)
        h = marginal_entropies(rho, dims)
        # Two sites paired: 2*log(3), one site pure: 0
        expected_sum = 2 * np.log(3)
        quantum_assert_scalar_close(
            h.sum(),
            expected_sum,
            "marginal_entropy",
            "Marginal entropy sum incorrect",
            atol=1e-8,
            rtol=0.0,
        )


# ============================================================================
# Test: Operator Bases
# ============================================================================

class TestOperatorBases:
    """Test Pauli and Gell-Mann operator construction."""
    
    def test_pauli_basis_hermitian(self):
        """Pauli operators should be Hermitian."""
        ops = pauli_basis(site=0, n_sites=2)
        for op in ops:
            quantum_assert_hermitian(
                op,
                "density_matrix",
                "Pauli operators should be Hermitian",
            )
    
    def test_pauli_basis_traceless(self):
        """Pauli operators should be traceless."""
        ops = pauli_basis(site=0, n_sites=2)
        for op in ops:
            quantum_assert_scalar_close(
                np.trace(op),
                0.0,
                "trace",
                "Pauli operators should be traceless",
                atol=1e-10,
                rtol=0.0,
            )
    
    def test_pauli_commutation_relations(self):
        """Check [σ_x, σ_y] = 2iσ_z at single site."""
        ops = pauli_basis(site=0, n_sites=1)
        X, Y, Z = ops
        
        # [X, Y] = 2iZ
        commutator = X @ Y - Y @ X
        expected = 2j * Z
        quantum_assert_close(
            commutator,
            expected,
            "commutator",
            "Pauli commutation relation failed",
            atol=1e-10,
            rtol=0.0,
        )
    
    def test_gell_mann_hermitian(self):
        """Gell-Mann matrices should be Hermitian."""
        gm = gell_mann_matrices()
        for G in gm:
            quantum_assert_hermitian(
                G,
                "density_matrix",
                "Gell-Mann matrices should be Hermitian",
            )
    
    def test_gell_mann_traceless(self):
        """Gell-Mann matrices should be traceless."""
        gm = gell_mann_matrices()
        for G in gm:
            quantum_assert_scalar_close(
                np.trace(G),
                0.0,
                "trace",
                "Gell-Mann matrices should be traceless",
                atol=1e-10,
                rtol=0.0,
            )
    
    def test_operator_basis_count(self):
        """Operator basis should have correct number of elements."""
        # Qubits: 3 operators per site
        ops_qubits, _ = create_operator_basis(n_sites=2, d=2)
        assert len(ops_qubits) == 6, "2 qubits should have 6 operators"
        
        # Qutrits: 8 operators per site
        ops_qutrits, _ = create_operator_basis(n_sites=2, d=3)
        assert len(ops_qutrits) == 16, "2 qutrits should have 16 operators"


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

        def numerical_fisher_information(theta_vec, eps: float = 1e-5):
            """Finite-difference Hessian of ψ(θ) = log Z(θ)."""
            n = exp_family.n_params
            G_num = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    theta_pp = theta_vec.copy()
                    theta_pp[i] += eps
                    theta_pp[j] += eps

                    theta_pm = theta_vec.copy()
                    theta_pm[i] += eps
                    theta_pm[j] -= eps

                    theta_mp = theta_vec.copy()
                    theta_mp[i] -= eps
                    theta_mp[j] += eps

                    theta_mm = theta_vec.copy()
                    theta_mm[i] -= eps
                    theta_mm[j] -= eps

                    psi_pp = exp_family.log_partition(theta_pp)
                    psi_pm = exp_family.log_partition(theta_pm)
                    psi_mp = exp_family.log_partition(theta_mp)
                    psi_mm = exp_family.log_partition(theta_mm)

                    G_ij = (psi_pp - psi_pm - psi_mp + psi_mm) / (4 * eps**2)
                    G_num[i, j] = G_ij
                    G_num[j, i] = G_ij
            return G_num

        # Test agreement for a small ensemble of natural-parameter values
        # drawn from a standard normal (no special "initialisation" scaling).
        rng = np.random.default_rng(0)

        for _ in range(5):
            theta = rng.standard_normal(exp_family.n_params)

            G_analytic = exp_family.fisher_information(theta)
            G_numeric = numerical_fisher_information(theta, eps=1e-5)

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

        def numerical_fisher_information(theta_vec, eps: float = 1e-5):
            """Finite-difference Hessian of ψ(θ) = log Z(θ) for qutrits."""
            n = exp_family.n_params
            G_num = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    theta_pp = theta_vec.copy()
                    theta_pp[i] += eps
                    theta_pp[j] += eps

                    theta_pm = theta_vec.copy()
                    theta_pm[i] += eps
                    theta_pm[j] -= eps

                    theta_mp = theta_vec.copy()
                    theta_mp[i] -= eps
                    theta_mp[j] += eps

                    theta_mm = theta_vec.copy()
                    theta_mm[i] -= eps
                    theta_mm[j] -= eps

                    psi_pp = exp_family.log_partition(theta_pp)
                    psi_pm = exp_family.log_partition(theta_pm)
                    psi_mp = exp_family.log_partition(theta_mp)
                    psi_mm = exp_family.log_partition(theta_mm)

                    G_ij = (psi_pp - psi_pm - psi_mp + psi_mm) / (4 * eps**2)
                    G_num[i, j] = G_ij
                    G_num[j, i] = G_ij
            return G_num

        rng = np.random.default_rng(1)

        for _ in range(3):
            theta = rng.standard_normal(exp_family.n_params)

            G_analytic = exp_family.fisher_information(theta)
            G_numeric = numerical_fisher_information(theta, eps=1e-5)

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

        def numerical_fisher_information(theta_vec, eps: float = 1e-5):
            """Finite-difference Hessian of ψ(θ) = log Z(θ) for d=4."""
            n = exp_family.n_params
            G_num = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    theta_pp = theta_vec.copy()
                    theta_pp[i] += eps
                    theta_pp[j] += eps

                    theta_pm = theta_vec.copy()
                    theta_pm[i] += eps
                    theta_pm[j] -= eps

                    theta_mp = theta_vec.copy()
                    theta_mp[i] -= eps
                    theta_mp[j] += eps

                    theta_mm = theta_vec.copy()
                    theta_mm[i] -= eps
                    theta_mm[j] -= eps

                    psi_pp = exp_family.log_partition(theta_pp)
                    psi_pm = exp_family.log_partition(theta_pm)
                    psi_mp = exp_family.log_partition(theta_mp)
                    psi_mm = exp_family.log_partition(theta_mm)

                    G_ij = (psi_pp - psi_pm - psi_mp + psi_mm) / (4 * eps**2)
                    G_num[i, j] = G_ij
                    G_num[j, i] = G_ij
            return G_num

        rng = np.random.default_rng(2)

        # Fewer samples here to keep runtime reasonable; d=4 has a larger
        # operator basis but the same conceptual checks apply.
        for _ in range(2):
            theta = rng.standard_normal(exp_family.n_params)

            G_analytic = exp_family.fisher_information(theta)
            G_numeric = numerical_fisher_information(theta, eps=1e-5)

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
# Test: Constrained Dynamics
# ============================================================================

class TestConstrainedDynamics:
    """Test constrained maximum entropy production dynamics."""
    
    def test_initialisation(self):
        """Test dynamics initialisation."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        dynamics = InaccessibleGameDynamics(exp_family)
        assert dynamics.time_mode == 'affine'
    
    def test_time_mode_setting(self):
        """Test time mode switching."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        dynamics = InaccessibleGameDynamics(exp_family)
        
        dynamics.set_time_mode('entropy')
        assert dynamics.time_mode == 'entropy'
        
        dynamics.set_time_mode('real')
        assert dynamics.time_mode == 'real'
    
    def test_flow_tangent_to_constraint(self):
        """Flow should be tangent to constraint manifold: a^T · θ̇ = 0."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        dynamics = InaccessibleGameDynamics(exp_family)
        
        theta = np.random.randn(exp_family.n_params) * 0.01  # Smaller for speed
        theta_dot = dynamics.flow(0.0, theta)
        
        _, a = exp_family.marginal_entropy_constraint(theta)
        
        tangency = np.dot(a, theta_dot)
        assert np.abs(tangency) < 1e-5, "Flow should be tangent to constraint manifold"
    
    def test_integration_preserves_constraint(self):
        """Integration should preserve marginal entropy constraint."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        dynamics = InaccessibleGameDynamics(exp_family)
        
        theta_0 = np.random.randn(exp_family.n_params) * 0.01
        # Ultra-short integration for speed
        solution = dynamics.integrate(theta_0, (0, 0.05), n_points=3)
        
        assert solution['success'], "Integration should succeed"
        
        # Check constraint preservation
        constraint_violations = np.abs(solution['constraint'] - solution['constraint'][0])
        max_violation = np.max(constraint_violations)
        
        assert max_violation < 5e-2, f"Constraint violation too large: {max_violation}"
    
    def test_entropy_monotonic_increase(self):
        """Joint entropy should increase monotonically."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        dynamics = InaccessibleGameDynamics(exp_family)
        
        theta_0 = np.random.randn(exp_family.n_params) * 0.01
        # Ultra-short integration
        solution = dynamics.integrate(theta_0, (0, 0.05), n_points=3)
        
        H = solution['H']
        dH = np.diff(H)
        
        # Allow small numerical violations
        negative_count = np.sum(dH < -1e-4)
        assert negative_count == 0, f"Entropy decreased at {negative_count} points"
    
    @pytest.mark.slow
    def test_constrained_maxent_dynamics(self):
        """In entropy time, dH/dt should be exactly 1."""
        # Use entangled pair where entropy time actually matters
        exp_family = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        dynamics = InaccessibleGameDynamics(exp_family)
        dynamics.set_time_mode('entropy')

        np.random.seed(24)
        theta_0 = np.random.randn(exp_family.n_params) * 0.05
        # Very reduced time and points for speed (entropy time is slow)
        solution = dynamics.integrate(theta_0, (0, 0.1), n_points=5)

        # Compute dH/dt
        dH = np.diff(solution['H'])
        dt = np.diff(solution['time'])
        dH_dt = dH / dt
        
        # Should be close to 1 (with wide tolerance for short integration)
        mean_rate = np.mean(dH_dt)
        print(f"Entropy production rate: {mean_rate}")
        assert 1-1e-6 < mean_rate < 1+1e-6, f"Entropy production rate unreasonable: {mean_rate}"

    @pytest.mark.slow
    def test_constrained_maxent_dynamics(self):
        """Test constrained maximum entropy dynamics using stable gradient descent."""
        # Use entangled pairs where the constrained optimization actually has work to do
        exp_family = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)  # 1 entangled pair
        dynamics = InaccessibleGameDynamics(exp_family)

        np.random.seed(24)
        theta_0 = np.random.randn(exp_family.n_params)

        # Run with sufficient iterations to check convergence behavior
        solution = dynamics.solve_constrained_maxent(theta_0, n_steps=100000, dt=1e-5,
                                                    convergence_tol=1e-6, project=True, project_every=200,
                                                    use_entropy_time=True)

        # Debug: Check final flow norms
        print(f"Final flow norms (last 10): {solution['flow_norms'][-10:]}")
        print(f"Min flow norm: {np.min(solution['flow_norms'])}")
        print(f"Max flow norm: {np.max(solution['flow_norms'])}")

        # Debug: Check constraint values
        print(f"Constraint values (last 10): {solution['constraint_values'][-10:]}")
        print(f"Initial C: {solution['C_init']:.6f}")
        print(f"Final constraint violation: {np.abs(solution['constraint_values'][-1] - solution['C_init']):.2e}")

        # Check that joint entropy increased (second law of thermodynamics)
        # Use the fast exponential family method
        H_init = exp_family.von_neumann_entropy(theta_0)
        H_final = exp_family.von_neumann_entropy(solution['trajectory'][-1])

        entropy_increase = H_final - H_init
        print(f"Joint entropy increase: {entropy_increase:.6f}")

        # Should increase (second law)
        assert entropy_increase > 0.01, f"Entropy should increase: ΔH = {entropy_increase}"

        # Check constraint preservation (more lenient for very tight convergence)
        constraint_violation = np.max(np.abs(solution['constraint_values'] - solution['C_init']))
        print(f"Max constraint violation: {constraint_violation:.2e}")
        # Allow slightly more drift for ultra-tight convergence
        assert constraint_violation < 2e-4, f"Constraint not preserved: violation = {constraint_violation}"

        # Check entropy monotonicity - should increase throughout
        # Sample entropy at various points during trajectory
        n_check_points = min(10, len(solution['trajectory']))
        check_indices = np.linspace(0, len(solution['trajectory'])-1, n_check_points, dtype=int)

        entropies = []
        for idx in check_indices:
            theta_check = solution['trajectory'][idx]
            H_check = exp_family.von_neumann_entropy(theta_check)
            entropies.append(H_check)

        # Check monotonic increase
        for i in range(1, len(entropies)):
            assert entropies[i] >= entropies[i-1] - 1e-10, \
                f"Entropy decreased: H[{check_indices[i-1]}]={entropies[i-1]:.6f}, H[{check_indices[i]}]={entropies[i]:.6f}"

        # Should eventually converge to tight tolerance
        assert solution['converged'], f"Dynamics should converge to 1e-6 tolerance. Min flow norm: {np.min(solution['flow_norms']):.6e}"


# ============================================================================
# Test: GENERIC Decomposition
# ============================================================================

class TestGENERICDecomposition:
    """Test GENERIC decomposition analysis."""
    
    def test_generic_decomposition_symmetric_antisymmetric(self):
        """Test that S is symmetric and A is antisymmetric."""
        M = np.random.randn(5, 5)
        S, A = generic_decomposition(M)
        
        assert np.allclose(S, S.T, atol=1e-10), "S should be symmetric"
        assert np.allclose(A, -A.T, atol=1e-10), "A should be antisymmetric"
    
    def test_generic_decomposition_reconstruction(self):
        """Test that M = S + A."""
        M = np.random.randn(5, 5)
        S, A = generic_decomposition(M)
        
        M_reconstructed = S + A
        assert np.allclose(M, M_reconstructed, atol=1e-10), "M should equal S + A"
    
    def test_jacobian_shape(self):
        """Jacobian should have correct shape."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        dynamics = InaccessibleGameDynamics(exp_family)
        theta = np.random.randn(exp_family.n_params) * 0.1
        
        M = dynamics.exp_family.jacobian(theta)
        
        expected_shape = (exp_family.n_params, exp_family.n_params)
        assert M.shape == expected_shape, f"Jacobian shape should be {expected_shape}"


# ============================================================================
# Test: Numerical Stability
# ============================================================================

class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_zero_parameters(self):
        """System should handle zero natural parameters (maximally mixed)."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        theta_0 = np.zeros(exp_family.n_params)
        rho = exp_family.rho_from_theta(theta_0)
        
        # Should be maximally mixed: I/D
        expected = np.eye(exp_family.D) / exp_family.D
        assert np.allclose(rho, expected, atol=1e-8), "Zero parameters should give maximally mixed"
    
    def test_small_parameters(self):
        """System should handle very small parameters."""
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        theta_0 = np.random.randn(exp_family.n_params) * 1e-8
        
        dynamics = InaccessibleGameDynamics(exp_family)
        solution = dynamics.integrate(theta_0, (0, 0.1), n_points=10)
        
        assert solution['success'], "Integration with small parameters should succeed"
    
    def test_different_seeds_reproducible(self):
        """Same seed should give reproducible results."""
        np.random.seed(42)
        exp_family1 = QuantumExponentialFamily(n_sites=2, d=2)
        theta1 = np.random.randn(exp_family1.n_params) * 0.1
        
        np.random.seed(42)
        exp_family2 = QuantumExponentialFamily(n_sites=2, d=2)
        theta2 = np.random.randn(exp_family2.n_params) * 0.1
        
        assert np.allclose(theta1, theta2), "Same seed should give same parameters"


# ============================================================================
# Test: Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests using qig library."""
    
    @pytest.mark.slow
    def test_full_validation_two_qubits(self):
        """Full validation pipeline for 2 qubits (1 entangled pair)."""
        # Create system with pair basis (1 pair = 2 sites)
        exp_family = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        dynamics = InaccessibleGameDynamics(exp_family)
        
        # Random initial state
        theta_0 = np.random.randn(exp_family.n_params) * 0.1
        
        # Very short integration for testing speed
        solution = dynamics.integrate(theta_0, (0, 0.1), n_points=3)
        
        # Verify integration succeeded
        assert solution['success'], "Integration should succeed"
        
        # Verify constraint preservation
        constraint_violations = np.abs(solution['constraint'] - solution['constraint'][0])
        max_violation = np.max(constraint_violations)
        assert max_violation < 5e-2, f"Constraint violation too large: {max_violation}"
        
        # Verify entropy increase
        delta_H = solution['H'][-1] - solution['H'][0]
        assert delta_H >= -1e-6, f"Entropy should not decrease: ΔH = {delta_H}"
    
    @pytest.mark.slow
    def test_full_validation_two_qutrits(self):
        """Full validation pipeline for 2 qutrits (1 entangled pair)."""
        # Create system with pair basis (1 pair = 2 sites)
        exp_family = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        dynamics = InaccessibleGameDynamics(exp_family)
        
        # Random initial state
        theta_0 = np.random.randn(exp_family.n_params) * 0.1
        
        # Very short integration for testing speed
        solution = dynamics.integrate(theta_0, (0, 0.1), n_points=3)
        
        # Verify integration succeeded
        assert solution['success'], "Integration should succeed"
        
        # Verify constraint preservation
        constraint_violations = np.abs(solution['constraint'] - solution['constraint'][0])
        max_violation = np.max(constraint_violations)
        assert max_violation < 5e-2, f"Constraint violation too large: {max_violation}"


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
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

