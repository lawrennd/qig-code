"""
Core Physics Tests for MEPP
===========================

Test the essential physics components of the MEPP implementation.
"""

import numpy as np
import pytest
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mepp import MEPPSimulator


class TestBellPairCreation:
    """Test Bell pair state creation."""
    
    def test_qubit_bell_pairs(self):
        """Test Bell pair creation for qubits."""
        sim = MEPPSimulator(n_qubits=4, d=2)
        state = sim._create_bell_pairs_state()
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert np.abs(norm - 1.0) < 1e-10, f"Qubit state not normalized: {norm}"
        
        # Check purity (should be pure state)
        rho = np.outer(state, state.conj())
        purity = np.trace(rho @ rho)
        assert np.abs(purity - 1.0) < 1e-10, f"Qubit state not pure: {purity}"
        
        # Check initial entropy (should be 0 for pure state)
        full_entropy = sim.von_neumann_entropy(rho)
        assert np.abs(full_entropy) < 1e-10, f"Pure state should have zero entropy: {full_entropy}"
    
    def test_qutrit_bell_pairs(self):
        """Test Bell pair creation for qutrits."""
        sim = MEPPSimulator(n_qubits=4, d=3)
        state = sim._create_bell_pairs_state()
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert np.abs(norm - 1.0) < 1e-10, f"Qutrit state not normalized: {norm}"
        
        # Check purity
        rho = np.outer(state, state.conj())
        purity = np.trace(rho @ rho)
        assert np.abs(purity - 1.0) < 1e-10, f"Qutrit state not pure: {purity}"
        
        # Check initial entropy (should be 0 for pure state)
        full_entropy = sim.von_neumann_entropy(rho)
        assert np.abs(full_entropy) < 1e-10, f"Pure state should have zero entropy: {full_entropy}"


class TestEntropyCalculations:
    """Test entropy calculation methods."""
    
    def test_von_neumann_entropy(self):
        """Test von Neumann entropy calculation."""
        sim = MEPPSimulator(n_qubits=4, d=2)
        
        # Test pure state entropy (should be 0)
        state = sim._create_bell_pairs_state()
        rho = np.outer(state, state.conj())
        
        full_entropy = sim.von_neumann_entropy(rho)
        assert np.abs(full_entropy) < 1e-10, f"Pure state should have zero entropy: {full_entropy}"
        
        # Test maximally mixed state entropy
        dim = rho.shape[0]
        rho_mixed = np.eye(dim) / dim
        mixed_entropy = sim.von_neumann_entropy(rho_mixed)
        expected_entropy = np.log(dim)
        assert np.abs(mixed_entropy - expected_entropy) < 1e-10, f"Mixed state entropy wrong: {mixed_entropy} != {expected_entropy}"
    
    def test_coarse_grained_entropy(self):
        """Test coarse-grained entropy calculation."""
        sim = MEPPSimulator(n_qubits=4, d=2)
        
        # Create initial state
        state = sim._create_bell_pairs_state()
        rho = np.outer(state, state.conj())
        
        # Test coarse-grained entropy
        cg_entropy = sim._compute_coarse_grained_entropy(rho, 2)
        assert cg_entropy >= 0, f"Coarse-grained entropy should be non-negative: {cg_entropy}"
        
        # Test that coarse-grained entropy is less than or equal to full entropy
        full_entropy = sim.von_neumann_entropy(rho)
        assert cg_entropy <= full_entropy + 1e-10, f"Coarse-grained entropy should not exceed full entropy"


class TestGellMannMatrices:
    """Test Gell-Mann matrix properties."""
    
    def test_gell_mann_properties(self):
        """Test Gell-Mann matrix mathematical properties."""
        sim = MEPPSimulator(n_qubits=2, d=3)
        
        # Test individual Gell-Mann matrices
        lambda1 = sim._qutrit_pauli('x')  # λ1
        lambda2 = sim._qutrit_pauli('y')  # λ2
        lambda3 = sim._qutrit_z()         # λ3
        
        # Check traceless property
        assert np.abs(np.trace(lambda1)) < 1e-10, "λ1 should be traceless"
        assert np.abs(np.trace(lambda2)) < 1e-10, "λ2 should be traceless"
        assert np.abs(np.trace(lambda3)) < 1e-10, "λ3 should be traceless"
        
        # Check Hermitian property
        assert np.allclose(lambda1, lambda1.conj().T), "λ1 should be Hermitian"
        assert np.allclose(lambda2, lambda2.conj().T), "λ2 should be Hermitian"
        assert np.allclose(lambda3, lambda3.conj().T), "λ3 should be Hermitian"
        
        # Check normalization (should have norm √2)
        expected_norm = np.sqrt(2)
        assert np.abs(np.linalg.norm(lambda1) - expected_norm) < 1e-10, f"λ1 norm wrong: {np.linalg.norm(lambda1)}"
        assert np.abs(np.linalg.norm(lambda2) - expected_norm) < 1e-10, f"λ2 norm wrong: {np.linalg.norm(lambda2)}"
        assert np.abs(np.linalg.norm(lambda3) - expected_norm) < 1e-10, f"λ3 norm wrong: {np.linalg.norm(lambda3)}"


class TestUnitaryGates:
    """Test that generated gates are unitary."""
    
    @pytest.mark.parametrize("d", [2, 3])
    @pytest.mark.parametrize("stage", ["dephasing", "isolation"])
    def test_gate_unitarity(self, d, stage):
        """Test that generated gates are unitary."""
        sim = MEPPSimulator(n_qubits=4, d=d)
        
        # Generate multiple gates to test
        for _ in range(5):
            gate, qudits, alpha = sim._generate_random_gate(0.1, stage)
            
            # Check unitarity: U†U = I
            gate_adjoint = gate.conj().T
            unitary_check = gate @ gate_adjoint
            max_deviation = np.max(np.abs(unitary_check - np.eye(gate.shape[0])))
            
            assert max_deviation < 1e-10, f"Gate not unitary, max deviation: {max_deviation}"
            
            # Check determinant is 1 (up to numerical precision)
            det = np.linalg.det(gate)
            assert np.abs(np.abs(det) - 1.0) < 1e-10, f"Gate determinant not 1: {det}" 