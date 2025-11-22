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
)

__all__ = [
    "partial_trace",
    "von_neumann_entropy",
    "create_lme_state",
    "marginal_entropies",
]


