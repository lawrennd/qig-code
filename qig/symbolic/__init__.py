"""
Symbolic computation module for quantum information geometry.

This module provides SymPy-based symbolic computation tools for
quantum exponential families, particularly for exploiting Lie algebra
structure to derive analytic forms of GENERIC decomposition components.

Related to CIP-0007: Analytic Forms for S and A via Lie Algebra Structure
"""

from qig.symbolic.su9_pair import (
    symbolic_su9_generators,
    symbolic_su9_structure_constants,
    verify_su9_generators,
)

__all__ = [
    "symbolic_su9_generators",
    "symbolic_su9_structure_constants",
    "verify_su9_generators",
]

