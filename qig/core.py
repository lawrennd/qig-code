"""
Core utilities for the quantum inaccessible game.

For now this module simply re-exports the core functions from
`inaccessible_game_quantum.py` to provide a stable namespace. As
refactoring proceeds (CIP-0001), implementations can be migrated here.
"""

from ..inaccessible_game_quantum import (
    partial_trace,
    von_neumann_entropy,
    create_lme_state,
    marginal_entropies,
)

__all__ = [
    "partial_trace",
    "von_neumann_entropy",
    "create_lme_state",
    "marginal_entropies",
]


