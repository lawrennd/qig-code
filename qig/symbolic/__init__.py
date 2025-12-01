"""
Symbolic computation module for quantum information geometry.

This module provides SymPy-based symbolic computation tools for
quantum exponential families, particularly for exploiting Lie algebra
structure to derive analytic forms of GENERIC decomposition components.

Related to CIP-0007: Analytic Forms for S and A via Lie Algebra Structure
"""

from qig.symbolic.gell_mann import (
    symbolic_gell_mann_matrices,
    symbolic_su3_structure_constants,
)

__all__ = [
    "symbolic_gell_mann_matrices",
    "symbolic_su3_structure_constants",
]

