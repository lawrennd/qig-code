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
    cross_validate_hamiltonian_coefficients
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


# Phase 3 Gate: All these tests must pass
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

