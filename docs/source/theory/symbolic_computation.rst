Symbolic Computation for Qutrits
=================================

This document describes the symbolic computation approach for deriving
analytic expressions in the quantum inaccessible game for qutrit systems.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

The ``qig.symbolic`` module provides exact symbolic expressions for the
GENERIC decomposition of qutrit pair systems. Unlike numerical computation,
symbolic expressions:

- Reveal the geometric structure of the dynamics
- Enable verification of theoretical properties
- Provide insight into how Lie algebra structure creates simplifications

The main result is that the antisymmetric part A ≠ 0 for the su(9) pair
basis, proving the existence of Hamiltonian (reversible) dynamics.

The su(9) Pair Basis
--------------------

For a pair of qutrits (d=3), we use the full su(9) Lie algebra:

- *80 generators** (compared to 16 for local su(3)⊗su(3) basis)
- *Can represent entangled states* including Bell states
- *Structural identity does not hold*: :math:`G\theta \neq -a` (unlike separable states)

This breaking of the structural identity is what allows :math:`A \neq 0`.

Key Optimisation: SU(3) Block Structure
---------------------------------------

A crucial optimisation exploits the Lie algebra structure of SU(3).

The Block Structure
^^^^^^^^^^^^^^^^^^^

Reduced density matrices of qutrit pairs have a special form:

.. math::

   \rho_1 = \begin{pmatrix} a & b & 0 \\ b & c & 0 \\ 0 & 0 & d \end{pmatrix}

This is a 2×2 block plus 1×1 block structure.

Lie-Algebraic Origin
^^^^^^^^^^^^^^^^^^^^

This structure comes from SU(3)'s canonical subgroup
decomposition.

.. math::

   \mathfrak{su}(3) = \mathfrak{su}(2) \oplus \mathfrak{u}(1) \oplus \text{(ladder operators)}

The 8 Gell-Mann matrices :math:`\lambda_1, \ldots, \lambda_8` split as:

+----------------+---------------------------+-------------------------------------+
| Type           | Generators                | Action                              |
+================+===========================+=====================================+
| SU(2)          | :math:`\lambda_1,2,3`     | Mix :math:`|0\rangle, |1\rangle`    |
+----------------+---------------------------+-------------------------------------+
| U(1)           | :math:`\lambda_8`         | Diagonal; separates (0,1) from 2    |
+----------------+---------------------------+-------------------------------------+
| Ladder         | :math:`\lambda_4,5,6,7`   | Mix :math:`|2\rangle` with others   |
+----------------+---------------------------+-------------------------------------+

**Key insight**: Any state lacking coherences with :math:`|2\rangle` has
**zero coefficients** for the ladder operators, automatically giving the
2×2 + 1 block form.

**For the quantum inaccessible game with Gell-Mann basis**, this block structure
applies to the reduced density matrices obtained by partial trace. Specifically:

- The LME (maximally mixed) starting state: :math:`\rho = I/3`
- Partial traces of maximally entangled states (Bell-like states)
- States along the constrained dynamics trajectory
- Any qutrit state diagonal in the computational basis

This covers the states encountered in the inaccessible game analysis.

Computational Benefit
^^^^^^^^^^^^^^^^^^^^^

With block structure, eigenvalues come from a **quadratic** (not cubic) formula:

.. math::

   \lambda_{1,2} = \frac{(a+c) \pm \sqrt{(a-c)^2 + 4b^2}}{2}, \quad \lambda_3 = d

This makes symbolic differentiation **~100× faster** than the general case.

References
^^^^^^^^^^

- Byrd & Khaneja, *Phys. Rev. A* **68** (2003)
- Kimura, *Phys. Lett. A* **314** (2003)
- Gamel, *Phys. Rev. A* **93**, 062320 (2016)

Available Methods
-----------------

Entropy Computation
^^^^^^^^^^^^^^^^^^^

Two methods are available for computing marginal entropies:

1. **Exact with block structure** (``method='exact'``, default):
   
   - Uses quadratic eigenvalue formula
   - Fast differentiation (~0.08s)
   - Ratio to numerical: 1.0001 (essentially exact)

2. **Taylor approximation** (``method='taylor'``):
   
   - Uses :math:`H(\rho) \approx \log(d) - \frac{d}{2}\text{Tr}[(\rho - I/d)^2]`
   - Polynomial expressions (fastest differentiation)
   - ~1% error for small :math:`\theta`

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   from qig.symbolic import (
       symbolic_constraint_gradient_su9_pair,
       symbolic_lagrange_multiplier_su9_pair,
       symbolic_antisymmetric_part_su9_pair,
   )
   import sympy as sp
   
   # Create symbolic parameters (4 active, rest zero)
   theta = sp.symbols('theta1:5', real=True)
   theta_full = tuple(list(theta) + [0] * 76)
   
   # Compute constraint gradient (exact method by default)
   a = symbolic_constraint_gradient_su9_pair(theta_full, method='exact')
   
   # Compute Lagrange multiplier and its gradient
   nu = symbolic_lagrange_multiplier_su9_pair(theta_full)
   
   # Compute antisymmetric part A
   A = symbolic_antisymmetric_part_su9_pair(theta_full)

Caching
^^^^^^^

Expensive symbolic computations are cached to disk in ``qig/symbolic/_cache/``.
The first run may take seconds to minutes; subsequent runs load instantly.

Validation
----------

Symbolic expressions are validated against numerical computation:

.. code-block:: bash

   python examples/symbolic_vs_numerical_demo.py

Key validations:

- Constraint gradient ``a``: ratio to numerical = 1.0001
- Antisymmetric part ``A ≠ 0``: confirms Hamiltonian dynamics
- Signs and structure match numerical computation

Planned Extensions
------------------

Qubits (d=2)
^^^^^^^^^^^^

The next priority is implementing symbolic computation for qubit pairs:

- su(4) basis: 15 parameters (simpler than 80 for qutrits)
- Reduced :math:`\rho` is 2×2: eigenvalues always trivial (quadratic)
- No block structure optimization needed—inherently simple

See Also
--------

- :doc:`generic_structure` - GENERIC decomposition theory
- :doc:`quantum_exponential_families` - Quantum exponential family background
- CIP-0007 in the repository for implementation details

