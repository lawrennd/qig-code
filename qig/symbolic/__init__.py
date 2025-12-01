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
    symbolic_density_matrix_su9_pair,
    symbolic_cumulant_generating_function_su9_pair,
    symbolic_fisher_information_su9_pair,
    symbolic_von_neumann_entropy_su9_pair,
    symbolic_partial_trace_su9_pair,
    symbolic_marginal_entropies_su9_pair,
    symbolic_constraint_gradient_su9_pair,
    symbolic_lagrange_multiplier_su9_pair,
    symbolic_grad_lagrange_multiplier_su9_pair,
)

__all__ = [
    "symbolic_su9_generators",
    "symbolic_su9_structure_constants",
    "verify_su9_generators",
    "symbolic_density_matrix_su9_pair",
    "symbolic_cumulant_generating_function_su9_pair",
    "symbolic_fisher_information_su9_pair",
    "symbolic_von_neumann_entropy_su9_pair",
    "symbolic_partial_trace_su9_pair",
    "symbolic_marginal_entropies_su9_pair",
    "symbolic_constraint_gradient_su9_pair",
    "symbolic_lagrange_multiplier_su9_pair",
    "symbolic_grad_lagrange_multiplier_su9_pair",
]

