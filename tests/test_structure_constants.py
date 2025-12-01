"""
Tests for structure constants computation (Phase 1 of CIP-0006).

This module tests structure constant computation for Lie algebras
and verifies against reference data.
"""

import pytest
import numpy as np
from qig.structure_constants import (
    compute_structure_constants,
    verify_lie_algebra,
    verify_jacobi_identity,
    verify_antisymmetry,
    verify_all_properties,
    compute_and_cache_structure_constants,
    get_cached_structure_constants,
    cache_structure_constants
)
from qig.reference_data import get_reference_structure_constants
from qig.exponential_family import pauli_basis, qutrit_basis
from qig.pair_operators import gell_mann_generators


class TestStructureConstantsComputation:
    """Test basic structure constant computation."""
    
    def test_pauli_matrices_su2(self):
        """Test structure constants for Pauli matrices (SU(2))."""
        # Get Pauli operators for single qubit
        operators = pauli_basis(0, 1)  # Site 0, 1 site total
        
        # Compute structure constants
        f_abc = compute_structure_constants(operators)
        
        # Should be 3x3x3
        assert f_abc.shape == (3, 3, 3)
        
        # Should be real
        assert np.allclose(f_abc.imag, 0.0)
        f_abc = f_abc.real
        
        # Check specific values: f_123 = 1 (and cyclic)
        assert np.abs(f_abc[0, 1, 2] - 1.0) < 1e-10
        assert np.abs(f_abc[1, 2, 0] - 1.0) < 1e-10
        assert np.abs(f_abc[2, 0, 1] - 1.0) < 1e-10
        
        # Check anti-cyclic: f_213 = -1
        assert np.abs(f_abc[1, 0, 2] + 1.0) < 1e-10
    
    def test_gell_mann_matrices_su3(self):
        """Test structure constants for Gell-Mann matrices (SU(3))."""
        # Get Gell-Mann generators
        operators = gell_mann_generators(3)
        
        # Compute structure constants
        f_abc = compute_structure_constants(operators)
        
        # Should be 8x8x8
        assert f_abc.shape == (8, 8, 8)
        
        # Should be real
        assert np.allclose(f_abc.imag, 0.0, atol=1e-10)
    
    def test_empty_operators(self):
        """Test handling of empty operator list."""
        operators = []
        f_abc = compute_structure_constants(operators)
        assert f_abc.size == 0
    
    def test_mismatched_dimensions(self):
        """Test error on mismatched operator dimensions."""
        op1 = np.eye(2)
        op2 = np.eye(3)
        
        with pytest.raises(ValueError, match="expected"):
            compute_structure_constants([op1, op2])
    
    def test_zero_norm_operator(self):
        """Test error on operator with zero norm."""
        op1 = np.zeros((2, 2))
        op2 = np.eye(2)
        
        with pytest.raises(ValueError, match="near-zero norm"):
            compute_structure_constants([op1, op2])


class TestVerification:
    """Test verification functions."""
    
    def test_verify_antisymmetry_pauli(self):
        """Test antisymmetry verification for Pauli matrices."""
        operators = pauli_basis(0, 1)
        f_abc = compute_structure_constants(operators)
        
        report = verify_antisymmetry(f_abc)
        assert report.all_passed()
        assert len(report.get_failures()) == 0
    
    def test_verify_antisymmetry_gell_mann(self):
        """Test antisymmetry verification for Gell-Mann matrices."""
        operators = gell_mann_generators(3)
        f_abc = compute_structure_constants(operators)
        
        report = verify_antisymmetry(f_abc)
        assert report.all_passed()
    
    def test_verify_jacobi_identity_pauli(self):
        """Test Jacobi identity for Pauli matrices."""
        operators = pauli_basis(0, 1)
        f_abc = compute_structure_constants(operators)
        
        report = verify_jacobi_identity(f_abc)
        assert report.all_passed()
        
        # Should have very small violations
        max_violation = report.checks[0].value
        assert max_violation < 1e-10
    
    def test_verify_jacobi_identity_gell_mann(self):
        """Test Jacobi identity for Gell-Mann matrices."""
        operators = gell_mann_generators(3)
        f_abc = compute_structure_constants(operators)
        
        report = verify_jacobi_identity(f_abc, tol=1e-8)
        assert report.all_passed()
    
    def test_verify_lie_algebra_pauli(self):
        """Test commutator verification for Pauli matrices."""
        operators = pauli_basis(0, 1)
        f_abc = compute_structure_constants(operators)
        
        report = verify_lie_algebra(operators, f_abc)
        assert report.all_passed()
    
    def test_verify_lie_algebra_gell_mann(self):
        """Test commutator verification for Gell-Mann matrices."""
        operators = gell_mann_generators(3)
        f_abc = compute_structure_constants(operators)
        
        report = verify_lie_algebra(operators, f_abc)
        assert report.all_passed()
    
    def test_verify_all_properties_pauli(self):
        """Test all verification checks for Pauli matrices."""
        operators = pauli_basis(0, 1)
        f_abc = compute_structure_constants(operators)
        
        report = verify_all_properties(f_abc, operators, "SU(2) Pauli")
        assert report.all_passed()
        assert len(report.checks) == 3  # Antisymmetry, Jacobi, Commutators
    
    def test_verify_all_properties_gell_mann(self):
        """Test all verification checks for Gell-Mann matrices."""
        operators = gell_mann_generators(3)
        f_abc = compute_structure_constants(operators)
        
        report = verify_all_properties(f_abc, operators, "SU(3) Gell-Mann")
        assert report.all_passed()


class TestCrossValidation:
    """Test cross-validation against reference data."""
    
    def test_pauli_vs_reference_su2(self):
        """Cross-validate computed SU(2) against reference."""
        operators = pauli_basis(0, 1)
        f_computed = compute_structure_constants(operators)
        f_reference = get_reference_structure_constants("su2")
        
        # Should match within tight tolerance
        max_error = np.max(np.abs(f_computed - f_reference))
        assert max_error < 1e-10
    
    def test_gell_mann_vs_reference_su3(self):
        """Cross-validate computed SU(3) against reference."""
        # Use the Gell-Mann matrices from exponential_family
        # which match the normalization used in reference_data
        from qig.exponential_family import gell_mann_matrices
        operators = gell_mann_matrices()
        f_computed = compute_structure_constants(operators)
        f_reference = get_reference_structure_constants("su3")
        
        # Should match within tight tolerance
        max_error = np.max(np.abs(f_computed - f_reference))
        assert max_error < 1e-10


class TestTensorProductStructure:
    """Test structure constants for tensor product bases."""
    
    def test_two_qubit_operators(self):
        """Test structure constants for 2-qubit system."""
        # Get operators for 2 qubits (6 operators total)
        ops_site0 = pauli_basis(0, 2)  # 3 operators
        ops_site1 = pauli_basis(1, 2)  # 3 operators
        operators = ops_site0 + ops_site1
        
        f_abc = compute_structure_constants(operators)
        
        # Should be 6x6x6
        assert f_abc.shape == (6, 6, 6)
        
        # Operators on different sites should commute
        # f_abc should be zero when a,b are on different sites
        for a in range(3):  # Site 0
            for b in range(3, 6):  # Site 1
                for c in range(6):
                    # [site0, site1] = 0, so f_abc should be ~0
                    assert np.abs(f_abc[a, b, c]) < 1e-10
                    assert np.abs(f_abc[b, a, c]) < 1e-10
    
    def test_two_qutrit_operators(self):
        """Test structure constants for 2-qutrit system."""
        # Get operators for 2 qutrits (16 operators total)
        ops_site0 = qutrit_basis(0, 2)  # 8 operators
        ops_site1 = qutrit_basis(1, 2)  # 8 operators
        operators = ops_site0 + ops_site1
        
        f_abc = compute_structure_constants(operators)
        
        # Should be 16x16x16
        assert f_abc.shape == (16, 16, 16)
        
        # Operators on different sites should commute
        for a in range(8):  # Site 0
            for b in range(8, 16):  # Site 1
                for c in range(16):
                    assert np.abs(f_abc[a, b, c]) < 1e-10


class TestCaching:
    """Test caching mechanism."""
    
    def test_cache_and_retrieve(self):
        """Test caching structure constants."""
        operators = pauli_basis(0, 1)
        f_abc = compute_structure_constants(operators)
        
        # Cache
        cache_structure_constants("test_su2", f_abc)
        
        # Retrieve
        f_cached = get_cached_structure_constants("test_su2")
        assert f_cached is not None
        assert np.allclose(f_cached, f_abc)
    
    def test_cache_returns_none_for_missing(self):
        """Test that cache returns None for missing entries."""
        f_cached = get_cached_structure_constants("nonexistent_algebra")
        assert f_cached is None
    
    def test_compute_and_cache(self):
        """Test compute_and_cache_structure_constants."""
        operators = pauli_basis(0, 1)
        
        # First call should compute
        f_abc = compute_and_cache_structure_constants(operators, "test_pauli")
        
        # Second call should use cache
        f_abc_2 = compute_and_cache_structure_constants(operators, "test_pauli")
        
        assert np.allclose(f_abc, f_abc_2)
    
    def test_force_recompute(self):
        """Test force_recompute flag."""
        operators = pauli_basis(0, 1)
        
        # Compute and cache
        f_abc = compute_and_cache_structure_constants(operators, "test_force")
        
        # Modify cache (to test that force_recompute ignores it)
        cache_structure_constants("test_force", np.zeros_like(f_abc))
        
        # Force recompute
        f_abc_new = compute_and_cache_structure_constants(
            operators, "test_force", force_recompute=True
        )
        
        # Should match original, not zeros
        assert not np.allclose(f_abc_new, 0.0)
        assert np.allclose(f_abc_new, f_abc)


# Phase 1 Gate: All these tests must pass
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

