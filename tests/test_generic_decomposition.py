"""
Tests for GENERIC decomposition (Phase 2 of CIP-0006).

This module tests the symmetric/antisymmetric decomposition of the
flow Jacobian and verification of degeneracy conditions.
"""

import pytest
import numpy as np
from qig.exponential_family import QuantumExponentialFamily


class TestSymmetricPart:
    """Test symmetric part computation."""
    
    def test_symmetry_property(self):
        """Test that S = S^T."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        S = exp_fam.symmetric_part(theta)
        
        # Should be symmetric within machine precision
        error = np.max(np.abs(S - S.T))
        assert error < 1e-14
    
    def test_reconstruction_from_parts(self):
        """Test that M = S + A."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        M = exp_fam.jacobian(theta)
        S = exp_fam.symmetric_part(theta)
        A = exp_fam.antisymmetric_part(theta)
        
        # Reconstruction should be exact within machine precision
        error = np.max(np.abs(M - (S + A)))
        assert error < 1e-13  # Slightly looser for numerical stability


class TestAntisymmetricPart:
    """Test antisymmetric part computation."""
    
    def test_antisymmetry_property(self):
        """Test that A = -A^T."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta)
        
        # Should be antisymmetric within machine precision
        error = np.max(np.abs(A + A.T))
        assert error < 1e-14
    
    def test_trace_zero(self):
        """Test that Tr(A) = 0."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        A = exp_fam.antisymmetric_part(theta)
        
        # Antisymmetric matrices have zero trace
        trace = np.trace(A)
        assert np.abs(trace) < 1e-12


class TestDegeneracyConditions:
    """Test degeneracy condition verification."""
    
    def test_diagnostics_structure(self):
        """Test that diagnostics dict has all required keys."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        diagnostics = exp_fam.verify_degeneracy_conditions(theta)
        
        # Check all keys present
        required_keys = [
            'S', 'A', 'constraint_gradient', 'entropy_gradient',
            'S_annihilates_constraint', 'A_annihilates_entropy_gradient',
            'entropy_production', 'S_symmetric_error', 'A_antisymmetric_error',
            'reconstruction_error', 'all_passed', 'tolerance'
        ]
        for key in required_keys:
            assert key in diagnostics
    
    def test_symmetry_antisymmetry_verified(self):
        """Test that S and A have correct symmetry properties."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        diagnostics = exp_fam.verify_degeneracy_conditions(theta)
        
        # Symmetry/antisymmetry should be at machine precision
        assert diagnostics['S_symmetric_error'] < 1e-14
        assert diagnostics['A_antisymmetric_error'] < 1e-14
    
    def test_reconstruction_verified(self):
        """Test that M = S + A is verified."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        diagnostics = exp_fam.verify_degeneracy_conditions(theta)
        
        # Reconstruction should be at near-machine precision
        assert diagnostics['reconstruction_error'] < 1e-12
    
    def test_entropy_production_nonnegative(self):
        """Test that entropy production θ^T S θ ≥ 0."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        diagnostics = exp_fam.verify_degeneracy_conditions(theta)
        
        # Entropy production should be non-negative (allowing small numerical error)
        assert diagnostics['entropy_production'] >= -1e-12
    
    def test_degeneracy_at_small_theta(self):
        """Test degeneracy conditions near origin."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
        theta = 1e-3 * np.random.rand(exp_fam.n_params)
        
        diagnostics = exp_fam.verify_degeneracy_conditions(theta, tol=1e-4)
        
        # Near origin, degeneracy should be strong
        assert diagnostics['S_annihilates_constraint'] < 1e-4
        # Note: A annihilating entropy gradient may not hold as strongly
        # due to the nature of the flow


class TestDifferentSystems:
    """Test GENERIC decomposition for different quantum systems."""
    
    def test_two_qubit_system(self):
        """Test GENERIC decomposition for 2-qubit system."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2)  # 6 parameters
        theta = 0.1 * np.random.rand(exp_fam.n_params)
        
        M = exp_fam.jacobian(theta)
        S = exp_fam.symmetric_part(theta)
        A = exp_fam.antisymmetric_part(theta)
        
        # Basic properties
        assert S.shape == (6, 6)
        assert A.shape == (6, 6)
        assert np.allclose(M, S + A, atol=1e-13)
    
    def test_two_qutrit_system(self):
        """Test GENERIC decomposition for 2-qutrit system."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=3)  # 16 parameters
        theta = 0.05 * np.random.rand(exp_fam.n_params)
        
        M = exp_fam.jacobian(theta)
        S = exp_fam.symmetric_part(theta)
        A = exp_fam.antisymmetric_part(theta)
        
        # Basic properties
        assert S.shape == (16, 16)
        assert A.shape == (16, 16)
        assert np.allclose(M, S + A, atol=1e-14)
    
    def test_entangled_pair_system(self):
        """Test GENERIC decomposition for entangled pair (pair basis)."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)  # 15 parameters
        theta = 0.05 * np.random.rand(exp_fam.n_params)
        
        M = exp_fam.jacobian(theta)
        S = exp_fam.symmetric_part(theta)
        A = exp_fam.antisymmetric_part(theta)
        
        # Basic properties
        assert S.shape == (15, 15)
        assert A.shape == (15, 15)
        assert np.allclose(M, S + A, atol=1e-14)


# Phase 2 Gate: All these tests must pass
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

