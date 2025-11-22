"""
Constrained dynamics and GENERIC-like structure for the quantum inaccessible game.

This module currently re-exports `InaccessibleGameDynamics` from
`inaccessible_game_quantum.py` to provide a stable namespace. As CIP-0001
is implemented, the class definition and any supporting helpers will be
incrementally migrated here.
"""

from ..inaccessible_game_quantum import InaccessibleGameDynamics

__all__ = ["InaccessibleGameDynamics"]


