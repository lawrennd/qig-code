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

from qig.symbolic.two_qutrit import (
    two_qutrit_operators,
    symbolic_density_matrix_two_qutrit,
    partial_trace_symbolic,
    symbolic_marginal_entropies_two_qutrit,
    symbolic_constraint_gradient_two_qutrit,
    symbolic_lagrange_multiplier_two_qutrit,
    symbolic_grad_lagrange_multiplier_two_qutrit,
    verify_block_structure_two_qutrit,
)

__all__ = [
    "symbolic_gell_mann_matrices",
    "symbolic_su3_structure_constants",
    "symbolic_density_matrix_single_qutrit",
    "symbolic_cumulant_generating_function_single_qutrit",
    "symbolic_fisher_information_single_qutrit",
    "symbolic_von_neumann_entropy_single_qutrit",
    "verify_single_qutrit_consistency",
    "two_qutrit_operators",
    "symbolic_density_matrix_two_qutrit",
    "partial_trace_symbolic",
    "symbolic_marginal_entropies_two_qutrit",
    "symbolic_constraint_gradient_two_qutrit",
    "symbolic_lagrange_multiplier_two_qutrit",
    "symbolic_grad_lagrange_multiplier_two_qutrit",
    "verify_block_structure_two_qutrit",
]

