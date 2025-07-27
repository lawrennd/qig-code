"""
Thermalization Tests for MEPP
=============================

Test the thermalization process and stages.
"""

import numpy as np
import pytest
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mepp import MEPPSimulator


class TestDephasingStage:
    """Test dephasing stage entropy production."""
    
    @pytest.mark.parametrize("d", [2, 3])
    def test_dephasing_entropy_increase(self, d):
        """Test that dephasing increases entropy."""
        sim = MEPPSimulator(n_qubits=4, d=d)
        
        # Create initial state
        state = sim._create_bell_pairs_state()
        rho = np.outer(state, state.conj())
        
        initial_entropy = sim._compute_coarse_grained_entropy(rho, 2)
        
        # Apply dephasing
        for step in range(5):
            rho = sim._apply_dephasing_channel_v2(rho, 8, 0.1, step)
            entropy = sim._compute_coarse_grained_entropy(rho, 2)
            
            # Entropy should increase or stay the same
            assert entropy >= initial_entropy - 1e-10, f"Entropy decreased during dephasing: {entropy} < {initial_entropy}"
        
        final_entropy = sim._compute_coarse_grained_entropy(rho, 2)
        assert final_entropy > initial_entropy, f"Dephasing should increase entropy: {final_entropy} <= {initial_entropy}"
    
    @pytest.mark.parametrize("d", [2, 3])
    def test_dephasing_preserves_trace(self, d):
        """Test that dephasing preserves trace."""
        sim = MEPPSimulator(n_qubits=4, d=d)
        
        # Create initial state
        state = sim._create_bell_pairs_state()
        rho = np.outer(state, state.conj())
        
        initial_trace = np.trace(rho)
        
        # Apply dephasing
        for step in range(5):
            rho = sim._apply_dephasing_channel_v2(rho, 8, 0.1, step)
            trace = np.trace(rho)
            
            # Trace should be preserved
            assert np.abs(trace - initial_trace) < 1e-10, f"Trace not preserved: {trace} != {initial_trace}"


class TestIsolationStage:
    """Test isolation stage entropy production."""
    
    @pytest.mark.parametrize("d", [2, 3])
    def test_isolation_entropy_increase(self, d):
        """Test that isolation increases entropy."""
        sim = MEPPSimulator(n_qubits=4, d=d)
        
        # Create state after dephasing
        state = sim._create_bell_pairs_state()
        rho = np.outer(state, state.conj())
        
        # Apply some dephasing first
        for step in range(3):
            rho = sim._apply_dephasing_channel_v2(rho, 8, 0.1, step)
        
        initial_entropy = sim._compute_coarse_grained_entropy(rho, 2)
        
        # Apply isolation gates
        for step in range(5):
            rho = sim.apply_gate_block(rho, 8, 0.1, 'isolation', step, 2)
            entropy = sim._compute_coarse_grained_entropy(rho, 2)
            
            # Entropy should increase or stay the same
            assert entropy >= initial_entropy - 1e-10, f"Entropy decreased during isolation: {entropy} < {initial_entropy}"
        
        final_entropy = sim._compute_coarse_grained_entropy(rho, 2)
        assert final_entropy > initial_entropy, f"Isolation should increase entropy: {final_entropy} <= {initial_entropy}"
    
    @pytest.mark.parametrize("d", [2, 3])
    def test_isolation_preserves_trace(self, d):
        """Test that isolation preserves trace."""
        sim = MEPPSimulator(n_qubits=4, d=d)
        
        # Create state after dephasing
        state = sim._create_bell_pairs_state()
        rho = np.outer(state, state.conj())
        
        # Apply some dephasing first
        for step in range(3):
            rho = sim._apply_dephasing_channel_v2(rho, 8, 0.1, step)
        
        initial_trace = np.trace(rho)
        
        # Apply isolation gates
        for step in range(5):
            rho = sim.apply_gate_block(rho, 8, 0.1, 'isolation', step, 2)
            trace = np.trace(rho)
            
            # Trace should be preserved
            assert np.abs(trace - initial_trace) < 1e-10, f"Trace not preserved: {trace} != {initial_trace}"


class TestFullThermalization:
    """Test complete thermalization process."""
    
    @pytest.mark.parametrize("d", [2, 3])
    def test_thermalization_efficiency(self, d):
        """Test that thermalization achieves reasonable efficiency."""
        sim = MEPPSimulator(n_qubits=4, d=d)
        
        # Create initial state
        state = sim._create_bell_pairs_state()
        rho = np.outer(state, state.conj())
        
        initial_entropy = sim._compute_coarse_grained_entropy(rho, 2)
        
        # Apply dephasing
        for step in range(5):
            rho = sim._apply_dephasing_channel_v2(rho, 8, 0.1, step)
        
        dephasing_entropy = sim._compute_coarse_grained_entropy(rho, 2)
        assert dephasing_entropy > initial_entropy, "Dephasing should increase entropy"
        
        # Apply isolation
        for step in range(5):
            rho = sim.apply_gate_block(rho, 8, 0.1, 'isolation', step, 2)
        
        final_entropy = sim._compute_coarse_grained_entropy(rho, 2)
        assert final_entropy > dephasing_entropy, "Isolation should increase entropy further"
        
        # Check reasonable efficiency (should be >10%)
        max_entropy = 4 * np.log(d)  # 4 qudits
        efficiency = final_entropy / max_entropy
        assert efficiency > 0.1, f"Thermalization efficiency too low: {efficiency:.1%}"
    
    @pytest.mark.parametrize("d", [2, 3])
    def test_thermalization_monotonicity(self, d):
        """Test that entropy increases monotonically during thermalization."""
        sim = MEPPSimulator(n_qubits=4, d=d)
        
        # Create initial state
        state = sim._create_bell_pairs_state()
        rho = np.outer(state, state.conj())
        
        previous_entropy = sim._compute_coarse_grained_entropy(rho, 2)
        
        # Apply dephasing
        for step in range(5):
            rho = sim._apply_dephasing_channel_v2(rho, 8, 0.1, step)
            entropy = sim._compute_coarse_grained_entropy(rho, 2)
            
            # Entropy should not decrease
            assert entropy >= previous_entropy - 1e-10, f"Entropy decreased during dephasing: {entropy} < {previous_entropy}"
            previous_entropy = entropy
        
        # Apply isolation
        for step in range(5):
            rho = sim.apply_gate_block(rho, 8, 0.1, 'isolation', step, 2)
            entropy = sim._compute_coarse_grained_entropy(rho, 2)
            
            # Entropy should not decrease
            assert entropy >= previous_entropy - 1e-10, f"Entropy decreased during isolation: {entropy} < {previous_entropy}"
            previous_entropy = entropy


class TestThermalizationProperties:
    """Test properties of the thermalization process."""
    
    @pytest.mark.parametrize("d", [2, 3])
    def test_density_matrix_properties(self, d):
        """Test that density matrix properties are preserved."""
        sim = MEPPSimulator(n_qubits=4, d=d)
        
        # Create initial state
        state = sim._create_bell_pairs_state()
        rho = np.outer(state, state.conj())
        
        # Apply thermalization
        for step in range(3):
            rho = sim._apply_dephasing_channel_v2(rho, 8, 0.1, step)
        
        for step in range(3):
            rho = sim.apply_gate_block(rho, 8, 0.1, 'isolation', step, 2)
        
        # Check density matrix properties
        trace = np.trace(rho)
        assert np.abs(trace - 1.0) < 1e-10, f"Density matrix trace not 1: {trace}"
        
        # Check Hermitian property
        assert np.allclose(rho, rho.conj().T), "Density matrix not Hermitian"
        
        # Check positive semidefinite (eigenvalues >= 0)
        eigenvals = np.linalg.eigvals(rho)
        assert np.all(eigenvals >= -1e-10), f"Density matrix not positive semidefinite: {eigenvals}"
    
    @pytest.mark.parametrize("d", [2, 3])
    def test_entropy_bounds(self, d):
        """Test that entropy stays within reasonable bounds."""
        sim = MEPPSimulator(n_qubits=4, d=d)
        
        # Create initial state
        state = sim._create_bell_pairs_state()
        rho = np.outer(state, state.conj())
        
        # Apply thermalization
        for step in range(3):
            rho = sim._apply_dephasing_channel_v2(rho, 8, 0.1, step)
        
        for step in range(3):
            rho = sim.apply_gate_block(rho, 8, 0.1, 'isolation', step, 2)
        
        # Check entropy bounds
        entropy = sim._compute_coarse_grained_entropy(rho, 2)
        max_entropy = 4 * np.log(d)  # 4 qudits
        
        assert entropy >= 0, f"Entropy negative: {entropy}"
        assert entropy <= max_entropy + 1e-10, f"Entropy exceeds maximum: {entropy} > {max_entropy}" 