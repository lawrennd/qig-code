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

from qig.symbolic.single_qutrit import (
    symbolic_density_matrix_single_qutrit,
    symbolic_cumulant_generating_function_single_qutrit,
    symbolic_fisher_information_single_qutrit,
    symbolic_von_neumann_entropy_single_qutrit,
    verify_single_qutrit_consistency,
)

__all__ = [
    "symbolic_gell_mann_matrices",
    "symbolic_su3_structure_constants",
    "symbolic_density_matrix_single_qutrit",
    "symbolic_cumulant_generating_function_single_qutrit",
    "symbolic_fisher_information_single_qutrit",
    "symbolic_von_neumann_entropy_single_qutrit",
    "verify_single_qutrit_consistency",
]

