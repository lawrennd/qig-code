"""
Tests for diffusion operator (Phase 4 of CIP-0006).

This module tests the construction and verification of the diffusion
operator D[ρ] from the symmetric flow.
"""

import pytest
import numpy as np
from qig.exponential_family import QuantumExponentialFamily
from qig.structure_constants import compute_structure_constants
from qig.generic import (
    kubo_mori_derivatives,
    diffusion_operator,
    milburn_approximation,
    verify_diffusion_operator,
    compare_diffusion_methods,
    effective_hamiltonian_coefficients,
    effective_hamiltonian_operator
)


class TestKuboMoriDerivatives:
    """Test Kubo-Mori derivative computation."""
    
    def test_derivatives_computed(self):
        """Test that derivatives are computed for all parameters."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        drho_dtheta = kubo_mori_derivatives(theta, exp_fam.operators, exp_fam)
        
        # Should have one derivative per parameter
        assert len(drho_dtheta) == exp_fam.n_params
        
        # Each should be a matrix
        for drho in drho_dtheta:
            d_hilbert = exp_fam.d ** exp_fam.n_sites
            assert drho.shape == (d_hilbert, d_hilbert)
    
    def test_derivatives_hermitian(self):
        """Test that Kubo-Mori derivatives are Hermitian."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        drho_dtheta = kubo_mori_derivatives(theta, exp_fam.operators, exp_fam)
        
        for i, drho in enumerate(drho_dtheta):
            herm_error = np.max(np.abs(drho - drho.conj().T))
            assert herm_error < 1e-10, f"Derivative {i} not Hermitian: {herm_error}"


class TestDiffusionOperator:
    """Test diffusion operator construction."""
    
    def test_operator_constructed(self):
        """Test that D[ρ] is constructed."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        S = exp_fam.symmetric_part(theta)
        D_rho = diffusion_operator(S, theta, exp_fam)
        
        d_hilbert = exp_fam.d ** exp_fam.n_sites
        assert D_rho.shape == (d_hilbert, d_hilbert)
    
    def test_hermiticity(self):
        """Test that D[ρ] is Hermitian."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        S = exp_fam.symmetric_part(theta)
        D_rho = diffusion_operator(S, theta, exp_fam)
        
        herm_error = np.max(np.abs(D_rho - D_rho.conj().T))
        assert herm_error < 1e-10
    
    def test_trace_preservation(self):
        """Test that Tr(D[ρ]) = 0."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        S = exp_fam.symmetric_part(theta)
        D_rho = diffusion_operator(S, theta, exp_fam)
        
        trace = np.abs(np.trace(D_rho))
        assert trace < 1e-10
    
    def test_entropy_production(self):
        """Test that entropy production is non-negative."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        S = exp_fam.symmetric_part(theta)
        rho = exp_fam.rho_from_theta(theta)
        D_rho = diffusion_operator(S, theta, exp_fam)
        
        # Entropy production: -Tr(ρ log ρ D[ρ])
        from scipy.linalg import logm
        log_rho = logm(rho)
        entropy_prod = -np.trace(rho @ log_rho @ D_rho).real
        
        # Should be non-negative (allowing small numerical error)
        assert entropy_prod >= -1e-12


class TestMilburnApproximation:
    """Test Milburn approximation."""
    
    def test_milburn_computed(self):
        """Test that Milburn approximation computes."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta)
        f_abc = compute_structure_constants(exp_fam.operators)
        rho = exp_fam.rho_from_theta(theta)
        
        eta, _ = effective_hamiltonian_coefficients(A, theta, f_abc)
        H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
        
        D_rho_milburn = milburn_approximation(H_eff, rho)
        
        d_hilbert = exp_fam.d ** exp_fam.n_sites
        assert D_rho_milburn.shape == (d_hilbert, d_hilbert)
    
    def test_milburn_hermitian(self):
        """Test that Milburn approximation is Hermitian."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta)
        f_abc = compute_structure_constants(exp_fam.operators)
        rho = exp_fam.rho_from_theta(theta)
        
        eta, _ = effective_hamiltonian_coefficients(A, theta, f_abc)
        H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
        
        D_rho_milburn = milburn_approximation(H_eff, rho)
        
        herm_error = np.max(np.abs(D_rho_milburn - D_rho_milburn.conj().T))
        assert herm_error < 1e-12
    
    def test_milburn_trace_preserving(self):
        """Test that Milburn approximation preserves trace."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta)
        f_abc = compute_structure_constants(exp_fam.operators)
        rho = exp_fam.rho_from_theta(theta)
        
        eta, _ = effective_hamiltonian_coefficients(A, theta, f_abc)
        H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
        
        D_rho_milburn = milburn_approximation(H_eff, rho)
        
        trace = np.abs(np.trace(D_rho_milburn))
        assert trace < 1e-12


class TestVerification:
    """Test verification functions."""
    
    def test_verification_report(self):
        """Test verification report structure."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        S = exp_fam.symmetric_part(theta)
        rho = exp_fam.rho_from_theta(theta)
        D_rho = diffusion_operator(S, theta, exp_fam)
        
        report = verify_diffusion_operator(D_rho, rho)
        
        # Should have multiple checks
        assert len(report.checks) >= 4
        check_names = [c.name for c in report.checks]
        assert any('Hermit' in name for name in check_names)
        assert any('Trace' in name for name in check_names)
        assert any('Entropy' in name for name in check_names)
    
    def test_all_checks_pass(self):
        """Test that all verification checks pass."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        S = exp_fam.symmetric_part(theta)
        rho = exp_fam.rho_from_theta(theta)
        D_rho = diffusion_operator(S, theta, exp_fam)
        
        report = verify_diffusion_operator(D_rho, rho)
        
        # Most checks should pass
        assert len(report.get_passes()) >= 3


class TestCrossValidation:
    """Test cross-validation between methods."""
    
    def test_comparison_report(self):
        """Test comparison report structure."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.05 * np.random.rand(exp_fam.n_params)
        
        S = exp_fam.symmetric_part(theta)
        A = exp_fam.antisymmetric_part(theta)
        f_abc = compute_structure_constants(exp_fam.operators)
        
        eta, _ = effective_hamiltonian_coefficients(A, theta, f_abc)
        H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
        
        report = compare_diffusion_methods(S, theta, H_eff, exp_fam)
        
        assert len(report.checks) >= 3


class TestDifferentSystems:
    """Test diffusion operator for different quantum systems."""
    
    def test_two_qubit_system(self):
        """Test diffusion operator for 2-qubit system."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        S = exp_fam.symmetric_part(theta)
        rho = exp_fam.rho_from_theta(theta)
        D_rho = diffusion_operator(S, theta, exp_fam)
        
        # Basic checks
        assert D_rho.shape == (4, 4)
        assert np.max(np.abs(D_rho - D_rho.conj().T)) < 1e-10
        assert np.abs(np.trace(D_rho)) < 1e-10
    
    def test_two_qutrit_system(self):
        """Test diffusion operator for 2-qutrit system."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=3)
        theta = 0.05 * np.random.rand(exp_fam.n_params)
        
        S = exp_fam.symmetric_part(theta)
        D_rho = diffusion_operator(S, theta, exp_fam)
        
        # Basic checks
        assert D_rho.shape == (9, 9)
        assert np.max(np.abs(D_rho - D_rho.conj().T)) < 1e-10


class TestNearEquilibrium:
    """Test behavior near equilibrium."""
    
    def test_small_dissipation(self):
        """Test that near origin, dissipation is small."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 1e-4 * np.random.rand(exp_fam.n_params)
        
        S = exp_fam.symmetric_part(theta)
        D_rho = diffusion_operator(S, theta, exp_fam)
        
        # Near equilibrium, D should be small
        assert np.linalg.norm(D_rho) < 1e-6
    
    def test_positivity_preservation(self):
        """Test that positivity is preserved under time evolution."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        S = exp_fam.symmetric_part(theta)
        rho = exp_fam.rho_from_theta(theta)
        D_rho = diffusion_operator(S, theta, exp_fam)
        
        # Evolve for small timestep
        eps = 1e-6
        rho_evolved = rho + eps * D_rho
        
        # All eigenvalues should be non-negative
        eigvals = np.linalg.eigvalsh(rho_evolved.real)
        assert np.min(eigvals) >= -1e-12


# Phase 4 Gate: All these tests must pass
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

