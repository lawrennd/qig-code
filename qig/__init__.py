"""
qig: Quantum Inaccessible Game core package.

This package provides a structured namespace for the quantum inaccessible
game implementation, refactoring and consolidating functionality that was
originally developed in `inaccessible_game_quantum.py` and related scripts.

Initially, the modules in `qig` re-export the existing functionality to
preserve behaviour. As CIP-0001 is implemented, code will be migrated
incrementally into `qig.core`, `qig.exponential_family`, `qig.dynamics`,
and `qig.analysis`.
"""

from .core import (
    partial_trace,
    von_neumann_entropy,
    create_lme_state,
    marginal_entropies,
)

from .exponential_family import QuantumExponentialFamily
from .dynamics import InaccessibleGameDynamics

__all__ = [
    "partial_trace",
    "von_neumann_entropy",
    "create_lme_state",
    "marginal_entropies",
    "QuantumExponentialFamily",
    "InaccessibleGameDynamics",
]


