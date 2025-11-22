"""
Quantum exponential family and BKM metric interface.

This module currently re-exports `QuantumExponentialFamily` from
`inaccessible_game_quantum.py` to provide a stable namespace for the
refactored code. As CIP-0001 is implemented, the class definition and
supporting utilities can be migrated here.
"""

from ..inaccessible_game_quantum import QuantumExponentialFamily

__all__ = ["QuantumExponentialFamily"]


