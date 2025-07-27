"""
Pytest configuration and shared fixtures for MEPP tests.
"""

import pytest
import numpy as np
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mepp import MEPPSimulator


@pytest.fixture
def qubit_simulator():
    """Create a qubit simulator for testing."""
    return MEPPSimulator(n_qubits=4, d=2)


@pytest.fixture
def qutrit_simulator():
    """Create a qutrit simulator for testing."""
    return MEPPSimulator(n_qubits=4, d=3)


@pytest.fixture
def mixed_qubit_state(qubit_simulator):
    """Create a mixed qubit state for testing."""
    sim = qubit_simulator
    
    # Create initial state
    state = sim._create_bell_pairs_state()
    rho = np.outer(state, state.conj())
    
    # Apply some dephasing to create mixed state
    for step in range(3):
        rho = sim._apply_dephasing_channel_v2(rho, 8, 0.1, step)
    
    return rho


@pytest.fixture
def mixed_qutrit_state(qutrit_simulator):
    """Create a mixed qutrit state for testing."""
    sim = qutrit_simulator
    
    # Create initial state
    state = sim._create_bell_pairs_state()
    rho = np.outer(state, state.conj())
    
    # Apply some dephasing to create mixed state
    for step in range(3):
        rho = sim._apply_dephasing_channel_v2(rho, 8, 0.1, step)
    
    return rho


@pytest.fixture
def pure_qubit_state(qubit_simulator):
    """Create a pure qubit state for testing."""
    sim = qubit_simulator
    state = sim._create_bell_pairs_state()
    return np.outer(state, state.conj())


@pytest.fixture
def pure_qutrit_state(qutrit_simulator):
    """Create a pure qutrit state for testing."""
    sim = qutrit_simulator
    state = sim._create_bell_pairs_state()
    return np.outer(state, state.conj())


# Custom markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "physics: marks tests as physics validation tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmark tests"
    ) 