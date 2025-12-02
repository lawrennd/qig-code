"""
Symbolic computation module for quantum information geometry.

This module provides exact symbolic computation tools for quantum exponential
families, exploiting Lie algebra structure to derive analytic forms of GENERIC
decomposition components.

RECOMMENDED: Use `lme_exact` for EXACT computation (no Taylor approximation).
LEGACY: Use `su9_taylor_approximation` only if you need all 80 parameters.

Related to CIP-0007: Analytic Forms for S and A via Lie Algebra Structure
"""

# =============================================================================
# RECOMMENDED: LME Exact Computation (NO Taylor approximation)
# =============================================================================
from qig.symbolic.lme_exact import (
    # Generators
    gell_mann_symbolic,
    block_preserving_generators,
    permutation_matrix,
    
    # Core computation
    exact_exp_K_lme,
    exact_rho_lme,
    exact_rho1_lme,
    exact_rho2_lme,
    
    # Entropy
    exact_marginal_entropy_lme,
    exact_constraint_lme,
)

# =============================================================================
# LEGACY: Taylor approximation (only if you need all 80 parameters)
# =============================================================================
from qig.symbolic.su9_taylor_approximation import (
    symbolic_su9_generators,
    symbolic_su9_structure_constants,
    symbolic_density_matrix_su9_pair,
    symbolic_fisher_information_su9_pair,
    symbolic_constraint_gradient_su9_pair,
    symbolic_antisymmetric_part_su9_pair,
)

__all__ = [
    # Recommended (exact)
    "gell_mann_symbolic",
    "block_preserving_generators",
    "permutation_matrix",
    "exact_exp_K_lme",
    "exact_rho_lme",
    "exact_rho1_lme",
    "exact_rho2_lme",
    "exact_marginal_entropy_lme",
    "exact_constraint_lme",
    
    # Legacy (Taylor approximation)
    "symbolic_su9_generators",
    "symbolic_su9_structure_constants",
    "symbolic_density_matrix_su9_pair",
    "symbolic_fisher_information_su9_pair",
    "symbolic_constraint_gradient_su9_pair",
    "symbolic_antisymmetric_part_su9_pair",
]
