"""
qig: Quantum Inaccessible Game core package.

This package provides a structured namespace for the quantum inaccessible
game implementation, refactoring and consolidating functionality that was
originally developed in `inaccessible_game_quantum.py` and related scripts.

Initially, only low-level utilities live here; higher-level classes will
be migrated incrementally as CIP-0001 is implemented.
"""

from .core import (
    partial_trace,
    von_neumann_entropy,
    create_lme_state,
    marginal_entropies,
    loewner_kernel,
)
from .gibbs_lock import GibbsLockedFrame, infer_mu0
from .pair_operators import (
    gell_mann_generators,
    pair_basis_generators,
    bell_state,
    bell_state_density_matrix,
    multi_pair_basis,
    product_of_bell_states,
    generalised_bell_basis,
    near_bell_hamiltonian,
    near_bell_gibbs_frame,
)

__all__ = [
    "partial_trace",
    "von_neumann_entropy",
    "create_lme_state",
    "marginal_entropies",
    "loewner_kernel",
    "GibbsLockedFrame",
    "infer_mu0",
    "gell_mann_generators",
    "pair_basis_generators",
    "bell_state",
    "bell_state_density_matrix",
    "multi_pair_basis",
    "product_of_bell_states",
    "generalised_bell_basis",
    "near_bell_hamiltonian",
    "near_bell_gibbs_frame",
]


