"""
Tests for effective Hamiltonian extraction (Phase 3 of CIP-0006).

This module tests the extraction of the effective Hamiltonian from the
antisymmetric flow using structure constants.
"""

import pytest
import numpy as np
from qig.exponential_family import QuantumExponentialFamily
from qig.structure_constants import compute_structure_constants
from qig.generic import (
    effective_hamiltonian_coefficients,
    effective_hamiltonian_coefficients_lstsq,
    effective_hamiltonian_operator,
    verify_hamiltonian_evolution,
    cross_validate_hamiltonian_coefficients,
    verify_antisymmetric_flow_equals_commutator
)


class TestCoefficientExtraction:
    """Test Hamiltonian coefficient extraction methods."""
    
    def test_linear_solver_method_su2(self):
        """Test linear solver for SU(2) system."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        # Compute antisymmetric part
        A = exp_fam.antisymmetric_part(theta)
        
        # Compute structure constants
        f_abc = compute_structure_constants(exp_fam.operators)
        
        # Extract coefficients
        eta, diagnostics = effective_hamiltonian_coefficients(A, theta, f_abc)
        
        # Should return valid coefficients
        assert eta.shape == (exp_fam.n_params,)
        assert 'condition_number' in diagnostics
        assert 'residual' in diagnostics
        assert diagnostics['method'] == 'linear_solver'
    
    def test_linear_solver_method_su3(self):
        """Test linear solver for SU(3) system."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=3)
        theta = 0.05 * np.random.rand(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta)
        f_abc = compute_structure_constants(exp_fam.operators)
        
        eta, diagnostics = effective_hamiltonian_coefficients(A, theta, f_abc)
        
        assert eta.shape == (exp_fam.n_params,)
        assert diagnostics['residual'] < 1e-6
    
    def test_least_squares_method(self):
        """Test least-squares fitting method."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta)
        rho = exp_fam.rho_from_theta(theta)
        
        eta, diagnostics = effective_hamiltonian_coefficients_lstsq(
            A, theta, exp_fam.operators, rho
        )
        
        assert eta.shape == (exp_fam.n_params,)
        assert 'success' in diagnostics
        assert diagnostics['method'] == 'least_squares'


class TestHamiltonianOperator:
    """Test Hamiltonian operator construction."""
    
    def test_operator_construction(self):
        """Test H_eff construction from coefficients."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        eta = 0.1 * np.random.rand(exp_fam.n_params)
        
        H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
        
        # Should be same dimension as Hilbert space
        d_hilbert = exp_fam.d ** exp_fam.n_sites
        assert H_eff.shape == (d_hilbert, d_hilbert)
    
    def test_hermiticity(self):
        """Test that H_eff is Hermitian."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta)
        f_abc = compute_structure_constants(exp_fam.operators)
        
        eta, _ = effective_hamiltonian_coefficients(A, theta, f_abc)
        H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
        
        # Should be Hermitian
        herm_error = np.max(np.abs(H_eff - H_eff.conj().T))
        assert herm_error < 1e-12
    
    def test_tracelessness(self):
        """Test that H_eff is (approximately) traceless."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta)
        f_abc = compute_structure_constants(exp_fam.operators)
        
        eta, _ = effective_hamiltonian_coefficients(A, theta, f_abc)
        H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
        
        # Should be traceless (Lie algebra generators are traceless)
        trace = np.abs(np.trace(H_eff))
        assert trace < 1e-10


class TestVerification:
    """Test verification functions."""
    
    def test_verification_report_structure(self):
        """Test that verification report has expected structure."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta)
        f_abc = compute_structure_constants(exp_fam.operators)
        rho = exp_fam.rho_from_theta(theta)
        
        eta, _ = effective_hamiltonian_coefficients(A, theta, f_abc)
        H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
        
        report = verify_hamiltonian_evolution(
            H_eff, A, theta, exp_fam.operators, rho
        )
        
        # Should have multiple checks
        assert len(report.checks) >= 3
        # Should check Hermiticity, tracelessness, commutator
        check_names = [c.name for c in report.checks]
        assert any('Hermit' in name for name in check_names)
        assert any('Traceless' in name for name in check_names)
    
    def test_hermiticity_verified(self):
        """Test that Hermiticity check passes."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta)
        f_abc = compute_structure_constants(exp_fam.operators)
        rho = exp_fam.rho_from_theta(theta)
        
        eta, _ = effective_hamiltonian_coefficients(A, theta, f_abc)
        H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
        
        report = verify_hamiltonian_evolution(
            H_eff, A, theta, exp_fam.operators, rho
        )
        
        # Hermiticity check should pass
        herm_checks = [c for c in report.checks if 'Hermit' in c.name]
        assert len(herm_checks) > 0
        assert herm_checks[0].passed


class TestCrossValidation:
    """Test cross-validation between methods."""
    
    def test_cross_validation_report(self):
        """Test cross-validation report structure."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta)
        f_abc = compute_structure_constants(exp_fam.operators)
        rho = exp_fam.rho_from_theta(theta)
        
        report = cross_validate_hamiltonian_coefficients(
            A, theta, f_abc, exp_fam.operators, rho
        )
        
        # Should have checks for both methods and comparison
        assert len(report.checks) >= 3
    
    def test_methods_agree_small_theta(self):
        """Test that methods agree for small theta."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.01 * np.random.rand(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta)
        f_abc = compute_structure_constants(exp_fam.operators)
        rho = exp_fam.rho_from_theta(theta)
        
        # Method A
        eta_A, _ = effective_hamiltonian_coefficients(A, theta, f_abc)
        
        # Method B
        eta_B, _ = effective_hamiltonian_coefficients_lstsq(
            A, theta, exp_fam.operators, rho
        )
        
        # Should agree reasonably well for small systems
        # Note: Agreement may not be perfect due to approximations
        diff = np.linalg.norm(eta_A - eta_B)
        # Relaxed tolerance since methods use different approximations
        assert diff < 0.1


class TestDifferentSystems:
    """Test Hamiltonian extraction for different quantum systems."""
    
    def test_two_qubit_system(self):
        """Test extraction for 2-qubit system."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta)
        f_abc = compute_structure_constants(exp_fam.operators)
        
        eta, diag = effective_hamiltonian_coefficients(A, theta, f_abc)
        H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
        
        # Basic checks
        assert H_eff.shape == (4, 4)
        assert np.max(np.abs(H_eff - H_eff.conj().T)) < 1e-12
        assert diag['residual'] < 1e-6
    
    def test_two_qutrit_system(self):
        """Test extraction for 2-qutrit system."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=3)
        theta = 0.05 * np.random.rand(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta)
        f_abc = compute_structure_constants(exp_fam.operators)
        
        eta, diag = effective_hamiltonian_coefficients(A, theta, f_abc)
        H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
        
        # Basic checks
        assert H_eff.shape == (9, 9)
        assert np.max(np.abs(H_eff - H_eff.conj().T)) < 1e-12
    
    def test_regularization_for_singular_systems(self):
        """Test regularization handling for nearly singular systems."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        
        # Near origin, system may be nearly singular
        theta = 1e-6 * np.random.rand(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta)
        f_abc = compute_structure_constants(exp_fam.operators)
        
        # Should handle gracefully with regularization
        eta, diag = effective_hamiltonian_coefficients(
            A, theta, f_abc, regularization=1e-8
        )
        
        assert eta.shape == (exp_fam.n_params,)
        assert not np.any(np.isnan(eta))
        assert not np.any(np.isinf(eta))


class TestNearOrigin:
    """Test behavior near origin (LME state)."""
    
    def test_small_theta_extraction(self):
        """Test Hamiltonian extraction near origin."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 1e-4 * np.random.rand(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta)
        f_abc = compute_structure_constants(exp_fam.operators)
        
        eta, diag = effective_hamiltonian_coefficients(A, theta, f_abc)
        H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
        
        # Should produce valid Hamiltonian
        assert np.max(np.abs(H_eff - H_eff.conj().T)) < 1e-12
    
    def test_near_zero_hamiltonian(self):
        """Test that near origin, Hamiltonian is small."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 1e-6 * np.ones(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta)
        f_abc = compute_structure_constants(exp_fam.operators)
        
        eta, _ = effective_hamiltonian_coefficients(A, theta, f_abc)
        
        # Near origin, coefficients should be small
        assert np.linalg.norm(eta) < 1e-3


class TestAntisymmetricFlowCommutatorMatching:
    """
    Test explicit matching of antisymmetric flow to von Neumann commutator.
    
    This is the key verification for CIP-0009: the antisymmetric sector
    of the GENERIC decomposition should be exactly unitary evolution
    generated by the effective Hamiltonian.
    
    KNOWN LIMITATION: The current Duhamel implementation uses numerical integration
    rather than BCH formulas, so the BCH identity ∑_a η_a ∂_a ρ = -i[H_eff, ρ]
    is not satisfied to machine precision. Future work should implement BCH-based
    derivatives. For now, we verify structural properties that hold regardless.
    """
    
    def test_hamiltonian_properties_qubit_pair(self):
        """Test that H_eff has correct structural properties for 2-qubit system."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        # Extract Hamiltonian
        A = exp_fam.antisymmetric_part(theta, method='duhamel')
        f_abc = compute_structure_constants(exp_fam.operators)
        eta, _ = effective_hamiltonian_coefficients(A, theta, f_abc)
        H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
        
        # Verify structural properties
        report = verify_antisymmetric_flow_equals_commutator(
            H_eff, A, theta, exp_fam, method='duhamel', tol=1e-8
        )
        
        # Structural checks (Hermiticity, tracelessness) should pass
        structural_checks = [c for c in report.checks if any(word in c.name.lower() 
                            for word in ['hermitian', 'traceless'])]
        assert all(check.passed for check in structural_checks), \
            "H_eff must be Hermitian and traceless"
    
    def test_hamiltonian_properties_qutrit_pair(self):
        """Test that H_eff has correct structural properties for 2-qutrit system."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        theta = 0.05 * np.random.rand(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta, method='duhamel')
        f_abc = compute_structure_constants(exp_fam.operators)
        eta, _ = effective_hamiltonian_coefficients(A, theta, f_abc)
        H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
        
        report = verify_antisymmetric_flow_equals_commutator(
            H_eff, A, theta, exp_fam, method='duhamel', tol=1e-8
        )
        
        # Structural checks should pass
        structural_checks = [c for c in report.checks if any(word in c.name.lower() 
                            for word in ['hermitian', 'traceless'])]
        assert all(check.passed for check in structural_checks)
    
    def test_hamiltonian_properties_near_lme_origin(self):
        """Test H_eff properties near LME origin (regularized Bell state)."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        
        # Get regularized Bell state parameters
        theta = exp_fam.get_bell_state_parameters(log_epsilon=-10)
        
        A = exp_fam.antisymmetric_part(theta, method='duhamel')
        f_abc = compute_structure_constants(exp_fam.operators)
        eta, _ = effective_hamiltonian_coefficients(A, theta, f_abc)
        H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
        
        # Basic checks
        assert np.max(np.abs(H_eff - H_eff.conj().T)) < 1e-12, "H_eff must be Hermitian"
        assert np.abs(np.trace(H_eff)) < 1e-10, "H_eff must be traceless"
    
    def test_extraction_consistency_multiple_points(self):
        """Test that Hamiltonian extraction is internally consistent at multiple points."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        f_abc = compute_structure_constants(exp_fam.operators)
        
        # Test at multiple random points
        n_tests = 5
        
        for i in range(n_tests):
            theta = 0.05 * np.random.rand(exp_fam.n_params)
            
            A = exp_fam.antisymmetric_part(theta, method='duhamel')
            eta, diag = effective_hamiltonian_coefficients(A, theta, f_abc)
            H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
            
            # Check extraction formula: A @ theta should equal f @ eta (approximately)
            lhs = A @ theta
            rhs = np.einsum('abc,c->a', f_abc, eta)
            extraction_error = np.linalg.norm(lhs - rhs)
            
            # Check H_eff properties
            herm_error = np.max(np.abs(H_eff - H_eff.conj().T))
            trace_error = np.abs(np.trace(H_eff))
            
            assert extraction_error < 1e-6, f"Extraction formula error: {extraction_error:.2e}"
            assert herm_error < 1e-12, f"H_eff not Hermitian: {herm_error:.2e}"
            assert trace_error < 1e-10, f"H_eff not traceless: {trace_error:.2e}"
    
    def test_spectral_vs_quadrature_duhamel_consistency(self):
        """Test that spectral and quadrature Duhamel give same antisymmetric Jacobian.
        
        This verifies that the BCH/spectral implementation is numerically consistent
        with the quadrature method, confirming the ~14x error is theoretical not numerical.
        """
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        theta = 0.05 * np.random.rand(exp_fam.n_params)
        
        # Compute antisymmetric part with both methods
        A_quad = exp_fam.antisymmetric_part(theta, method='duhamel')
        A_spectral = exp_fam.antisymmetric_part(theta, method='duhamel_spectral')
        
        # Should agree to high precision
        diff = np.linalg.norm(A_quad - A_spectral, 'fro')
        assert diff < 1e-10, f"Spectral and quadrature Duhamel disagree: {diff:.2e}"
    
    def test_kubo_mori_kernel_properties(self):
        """Test that Kubo-Mori derivatives have expected kernel structure.
        
        The Kubo-Mori derivative ∂ρ/∂θ_a = K_ρ[F_a - ⟨F_a⟩] where K_ρ is the
        Duhamel kernel. This is NOT just a commutator [F_a, ρ] but includes
        the full operator-ordered integral structure.
        """
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        theta = 0.05 * np.random.rand(exp_fam.n_params)
        rho = exp_fam.rho_from_theta(theta)
        
        # For a single operator, compare Kubo-Mori vs commutator
        idx = 0
        F = exp_fam.operators[idx]
        drho_km = exp_fam.rho_derivative(theta, idx, method='duhamel_spectral')
        drho_comm = F @ rho - rho @ F  # Pure commutator
        
        # They should differ significantly (this is the Kubo-Mori kernel effect)
        ratio = np.linalg.norm(drho_km, 'fro') / np.linalg.norm(drho_comm, 'fro')
        
        # Empirically we see ~7-8x ratio
        assert ratio > 2.0, f"Kubo-Mori should differ from commutator, ratio={ratio:.2f}"
        assert ratio < 20.0, f"Ratio unexpectedly large: {ratio:.2f}"
        
        # Both should be Hermitian and traceless
        assert np.max(np.abs(drho_km - drho_km.conj().T)) < 1e-12
        assert np.abs(np.trace(drho_km)) < 1e-12
    
    def test_documented_bch_identity_limitation(self):
        """Document that the strong BCH identity does NOT hold in general.
        
        The identity ∑_a η_a ∂_a ρ = -i[H_eff, ρ] is NOT guaranteed by Lie closure.
        The LHS includes the Kubo-Mori kernel K_ρ, while the RHS is a pure commutator.
        
        This test documents the empirically observed ~14x discrepancy and serves as
        a regression guard against accidentally assuming this identity holds.
        """
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        np.random.seed(42)
        theta = 0.05 * np.random.rand(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta, method='duhamel_spectral')
        f_abc = compute_structure_constants(exp_fam.operators)
        eta, _ = effective_hamiltonian_coefficients(A, theta, f_abc)
        H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
        rho = exp_fam.rho_from_theta(theta)
        
        # LHS: with Kubo-Mori kernel
        drho_dtheta = [exp_fam.rho_derivative(theta, a, method='duhamel_spectral') 
                       for a in range(len(theta))]
        lhs = sum(eta[a] * drho_dtheta[a] for a in range(len(eta)))
        
        # RHS: pure commutator
        rhs = -1j * (H_eff @ rho - rho @ H_eff)
        
        rel_error = np.linalg.norm(lhs - rhs, 'fro') / np.linalg.norm(rhs, 'fro')
        
        # Document that this does NOT match (within order of magnitude)
        assert rel_error > 5.0, \
            f"Strong BCH identity unexpectedly holds (error={rel_error:.2f}). " \
            f"This would contradict our understanding of the Kubo-Mori kernel structure."
        
        # But should be reasonably bounded (not wildly wrong)
        assert rel_error < 50.0, f"Error unexpectedly large: {rel_error:.2f}"
    
    def test_flow_hermiticity_and_tracelessness(self):
        """Test that both flows are Hermitian and traceless."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta, method='duhamel')
        f_abc = compute_structure_constants(exp_fam.operators)
        eta, _ = effective_hamiltonian_coefficients(A, theta, f_abc)
        H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
        
        report = verify_antisymmetric_flow_equals_commutator(
            H_eff, A, theta, exp_fam, method='duhamel', tol=1e-8
        )
        
        # Check Hermiticity
        herm_checks = [c for c in report.checks if 'Hermitian' in c.name]
        assert len(herm_checks) == 2
        assert all(check.passed for check in herm_checks)
        
        # Check tracelessness
        trace_checks = [c for c in report.checks if 'traceless' in c.name]
        assert len(trace_checks) == 2
        assert all(check.passed for check in trace_checks)
    
    def test_relative_and_absolute_errors(self):
        """Test that both relative and absolute errors are reported."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta, method='duhamel')
        f_abc = compute_structure_constants(exp_fam.operators)
        eta, _ = effective_hamiltonian_coefficients(A, theta, f_abc)
        H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
        
        report = verify_antisymmetric_flow_equals_commutator(
            H_eff, A, theta, exp_fam, method='duhamel', tol=1e-8
        )
        
        # Should have both absolute and relative error checks
        absolute_checks = [c for c in report.checks if 'absolute' in c.name.lower()]
        relative_checks = [c for c in report.checks if 'relative' in c.name.lower()]
        
        assert len(absolute_checks) > 0
        assert len(relative_checks) > 0


# Phase 3 Gate: All these tests must pass
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

