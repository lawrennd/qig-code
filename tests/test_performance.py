"""
Performance Tests for MEPP
==========================

Test performance characteristics and benchmarks.
"""

import time
import numpy as np
import pytest
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mepp import MEPPSimulator


class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.performance
    @pytest.mark.parametrize("n_qudits", [2, 4, 6])
    @pytest.mark.parametrize("d", [2, 3])
    def test_thermalization_speed(self, n_qudits, d):
        """Test thermalization speed for different system sizes."""
        sim = MEPPSimulator(n_qubits=n_qudits, d=d)
        
        # Create initial state
        state = sim._create_bell_pairs_state()
        rho = np.outer(state, state.conj())
        
        # Time dephasing
        start_time = time.time()
        for step in range(5):
            rho = sim._apply_dephasing_channel_v2(rho, 8, 0.1, step)
        dephasing_time = time.time() - start_time
        
        # Time isolation
        start_time = time.time()
        for step in range(5):
            rho = sim.apply_gate_block(rho, 8, 0.1, 'isolation', step, 2)
        isolation_time = time.time() - start_time
        
        total_time = dephasing_time + isolation_time
        
        # Check reasonable performance (should complete within 30 seconds)
        assert total_time < 30, f"Thermalization too slow: {total_time:.2f}s for {n_qudits} qudits (d={d})"
        
        # Check that isolation is the bottleneck (as expected)
        assert isolation_time > dephasing_time, f"Isolation should be slower than dephasing: {isolation_time:.2f}s vs {dephasing_time:.2f}s"
    
    @pytest.mark.performance
    @pytest.mark.parametrize("d", [2, 3])
    def test_memory_usage(self, d):
        """Test memory usage for different system sizes."""
        sim = MEPPSimulator(n_qubits=4, d=d)
        
        # Create initial state
        state = sim._create_bell_pairs_state()
        rho = np.outer(state, state.conj())
        
        # Check memory usage of density matrix
        matrix_size = rho.nbytes / (1024 * 1024)  # MB
        expected_size = (d**4) * 16 / (1024 * 1024)  # 16 bytes per complex number
        
        # Memory usage should be reasonable
        assert matrix_size < 100, f"Density matrix too large: {matrix_size:.1f}MB for d={d}"
    
    @pytest.mark.performance
    @pytest.mark.parametrize("d", [2, 3])
    def test_gate_generation_speed(self, d):
        """Test gate generation speed."""
        sim = MEPPSimulator(n_qubits=4, d=d)
        
        # Time gate generation
        start_time = time.time()
        for _ in range(100):
            gate, qudits, alpha = sim._generate_random_gate(0.1, 'isolation')
        generation_time = time.time() - start_time
        
        # Should generate 100 gates in reasonable time
        assert generation_time < 10, f"Gate generation too slow: {generation_time:.2f}s for 100 gates"
    
    @pytest.mark.performance
    @pytest.mark.parametrize("d", [2, 3])
    def test_entropy_calculation_speed(self, d):
        """Test entropy calculation speed."""
        sim = MEPPSimulator(n_qubits=4, d=d)
        
        # Create a mixed state
        state = sim._create_bell_pairs_state()
        rho = np.outer(state, state.conj())
        
        # Apply some dephasing to create mixed state
        for step in range(3):
            rho = sim._apply_dephasing_channel_v2(rho, 8, 0.1, step)
        
        # Time entropy calculations
        start_time = time.time()
        for _ in range(100):
            entropy = sim._compute_coarse_grained_entropy(rho, 2)
        calculation_time = time.time() - start_time
        
        # Should calculate 100 entropies in reasonable time
        assert calculation_time < 10, f"Entropy calculation too slow: {calculation_time:.2f}s for 100 calculations"


class TestScalability:
    """Test scalability characteristics."""
    
    @pytest.mark.slow
    @pytest.mark.parametrize("d", [2, 3])
    def test_large_system_scalability(self, d):
        """Test performance with larger systems."""
        # Test with larger system (but not too large to avoid timeout)
        n_qudits = 6 if d == 2 else 4  # Smaller for qutrits due to higher dimensionality
        
        sim = MEPPSimulator(n_qubits=n_qudits, d=d)
        
        # Create initial state
        state = sim._create_bell_pairs_state()
        rho = np.outer(state, state.conj())
        
        # Time full thermalization
        start_time = time.time()
        
        # Apply dephasing
        for step in range(3):
            rho = sim._apply_dephasing_channel_v2(rho, 8, 0.1, step)
        
        # Apply isolation
        for step in range(3):
            rho = sim.apply_gate_block(rho, 8, 0.1, 'isolation', step, 2)
        
        total_time = time.time() - start_time
        
        # Check that larger systems don't explode in time
        max_time = 60 if d == 2 else 120  # More time for qutrits
        assert total_time < max_time, f"Large system too slow: {total_time:.2f}s for {n_qudits} qudits (d={d})"
        
        # Check that we get reasonable entropy
        final_entropy = sim._compute_coarse_grained_entropy(rho, 2)
        max_entropy = n_qudits * np.log(d)
        efficiency = final_entropy / max_entropy
        
        assert efficiency > 0.05, f"Large system efficiency too low: {efficiency:.1%}"


class TestTensorNetworkPerformance:
    """Test tensor network performance characteristics."""
    
    @pytest.mark.parametrize("d", [2, 3])
    def test_tensor_network_usage(self, d):
        """Test that tensor networks are used appropriately."""
        sim = MEPPSimulator(n_qubits=4, d=d)
        
        # Check if tensor networks are enabled for appropriate systems
        if d == 3:  # Qutrits should use tensor networks
            assert sim.use_tensor_networks, "Qutrits should use tensor networks"
        else:  # Qubits might not need tensor networks for small systems
            # This depends on the threshold, but we can check the property exists
            assert hasattr(sim, 'use_tensor_networks'), "Tensor network property should exist"
    
    @pytest.mark.performance
    @pytest.mark.parametrize("d", [2, 3])
    def test_partial_trace_performance(self, d):
        """Test partial trace performance."""
        sim = MEPPSimulator(n_qubits=4, d=d)
        
        # Create a mixed state
        state = sim._create_bell_pairs_state()
        rho = np.outer(state, state.conj())
        
        # Apply some dephasing
        for step in range(3):
            rho = sim._apply_dephasing_channel_v2(rho, 8, 0.1, step)
        
        # Time partial trace
        start_time = time.time()
        for _ in range(50):
            rho_reduced = sim._partial_trace(rho, 2)
        trace_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert trace_time < 10, f"Partial trace too slow: {trace_time:.2f}s for 50 operations"


class TestBenchmarks:
    """Benchmark tests for performance comparison."""
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("d", [2, 3])
    def test_thermalization_benchmark(self, d, benchmark):
        """Benchmark full thermalization process."""
        sim = MEPPSimulator(n_qubits=4, d=d)
        
        def thermalization():
            # Create initial state
            state = sim._create_bell_pairs_state()
            rho = np.outer(state, state.conj())
            
            # Apply dephasing
            for step in range(3):
                rho = sim._apply_dephasing_channel_v2(rho, 8, 0.1, step)
            
            # Apply isolation
            for step in range(3):
                rho = sim.apply_gate_block(rho, 8, 0.1, 'isolation', step, 2)
            
            return sim._compute_coarse_grained_entropy(rho, 2)
        
        result = benchmark(thermalization)
        
        # Check that we get reasonable entropy
        max_entropy = 4 * np.log(d)
        efficiency = result / max_entropy
        assert efficiency > 0.1, f"Benchmark efficiency too low: {efficiency:.1%}"
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("d", [2, 3])
    def test_gate_generation_benchmark(self, d, benchmark):
        """Benchmark gate generation."""
        sim = MEPPSimulator(n_qubits=4, d=d)
        
        def generate_gates():
            gates = []
            for _ in range(10):
                gate, qudits, alpha = sim._generate_random_gate(0.1, 'isolation')
                gates.append((gate, qudits, alpha))
            return gates
        
        result = benchmark(generate_gates)
        
        # Check that we get the expected number of gates
        assert len(result) == 10, f"Expected 10 gates, got {len(result)}" 