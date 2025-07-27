"""
Simple test to verify the pytest setup works.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


def test_import_mepp():
    """Test that we can import the mepp module."""
    try:
        from mepp import MEPPSimulator
        assert MEPPSimulator is not None
        print("✅ Successfully imported MEPPSimulator")
    except ImportError as e:
        pytest.fail(f"Failed to import MEPPSimulator: {e}")


def test_basic_math():
    """Test basic math operations."""
    assert 2 + 2 == 4
    assert np.array([1, 2, 3]).sum() == 6


def test_numpy_import():
    """Test numpy import."""
    arr = np.array([1, 2, 3])
    assert len(arr) == 3
    assert arr[0] == 1 