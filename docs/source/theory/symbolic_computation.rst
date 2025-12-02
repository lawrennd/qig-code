Symbolic Computation for Qutrits
=================================

This document describes the symbolic computation approach for deriving
analytic expressions in the quantum inaccessible game for qutrit systems.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

The ``qig.symbolic`` module provides **exact** symbolic expressions for the
GENERIC decomposition of qutrit pair systems. A key breakthrough is that
for maximally entangled states, we can compute exp(K) exactly with **no
Taylor approximation**.

Key results:

- The antisymmetric part A ≠ 0 for entangled states
- **Exact exp(K)** via block decomposition: 9×9 → 3×3 + 2×2 + 1×1×4
- **20 block-preserving generators** span the full entangled subspace
- All eigenvalues are at most **quadratic** (no cubic equations)

The su(9) Pair Basis
--------------------

For a pair of qutrits (d=3), we use the full su(9) Lie algebra:

- *80 generators** (compared to 16 for local su(3)⊗su(3) basis)
- *Can represent entangled states* including Bell states
- *Structural identity does not hold*: :math:`G\theta \neq -a` (unlike separable states)

This breaking of the structural identity is what allows :math:`A \neq 0`.

LME Block Decomposition (Key Breakthrough)
------------------------------------------

For locally maximally entangled (LME) states, the full 9×9 eigenvalue problem
decomposes into smaller blocks, all with at most **quadratic** eigenvalues.

The Entangled Subspace
^^^^^^^^^^^^^^^^^^^^^^

LME states like :math:`|\psi\rangle = \frac{1}{\sqrt{3}}(|00\rangle + |11\rangle + |22\rangle)`
live in the 3-dimensional subspace :math:`\{|00\rangle, |11\rangle, |22\rangle\}`.

In a reordered basis, the 9×9 K matrix becomes block diagonal:

.. math::

   K_{\text{block}} = \begin{pmatrix} K_{3\times 3} & 0 \\ 0 & K_{6\times 6} \end{pmatrix}

The 6×6 block further decomposes into 2×2 + 1×1×4.

Block-Preserving Generators
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**20 generators** preserve this block structure:

- **4 local**: :math:`\lambda_3 \otimes I`, :math:`I \otimes \lambda_3`, 
  :math:`\lambda_8 \otimes I`, :math:`I \otimes \lambda_8`
- **16 entangling**: :math:`\lambda_1 \otimes \lambda_1`, :math:`\lambda_1 \otimes \lambda_2`, etc.

These span the **full** entangled subspace (rank 9), enabling exploration of
all maximally entangled states while maintaining exact computation.

Reduced Density Matrix Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reduced density matrices also have a special block form:

.. math::

   \rho_1 = \begin{pmatrix} a & b & 0 \\ b^* & c & 0 \\ 0 & 0 & d \end{pmatrix}

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

With block structure, eigenvalues come from a quadratic (not cubic) formula:

.. math::

   \lambda_{1,2} = \frac{(a+c) \pm \sqrt{(a-c)^2 + 4b^2}}{2}, \quad \lambda_3 = d

This makes symbolic differentiation ~100× faster than the general case.

References
^^^^^^^^^^

- Byrd & Khaneja, *Phys. Rev. A* **68** (2003)
- Kimura, *Phys. Lett. A* **314** (2003)
- Gamel, *Phys. Rev. A* **93**, 062320 (2016)

Available Methods
-----------------

Two approaches are available:

1. **LME Exact** (``qig.symbolic.lme_exact``, recommended for LME dynamics):
   
   - Uses block decomposition: 9×9 → 3×3 + 2×2 + 1×1×4
   - **No Taylor approximation** - machine precision (~10⁻¹⁵)
   - Works for all 20 block-preserving generators

2. **General su(9)** (``qig.symbolic.su9_pair``):
   
   - Uses Taylor expansion for exp(K)
   - ~1% error at order 2, ~0.0008% at order 6
   - Works for all 80 generators

Usage Example: LME Exact
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from qig.symbolic.lme_exact import (
       exact_exp_K_lme,
       exact_rho_lme,
       block_preserving_generators,
   )
   import sympy as sp
   
   # Get available generators
   generators, names = block_preserving_generators()
   print(f"20 block-preserving generators: {names[:4]}...")
   
   # Create symbolic parameters
   theta = {
       'λ3⊗I': sp.Symbol('a3', real=True),
       'λ8⊗I': sp.Symbol('a8', real=True),
       'λ1⊗λ1': sp.Symbol('c11', real=True),
   }
   
   # EXACT exp(K) - no Taylor approximation!
   exp_K = exact_exp_K_lme(theta)
   
   # EXACT density matrix
   rho = exact_rho_lme(theta)

Caching
^^^^^^^

Expensive symbolic computations are cached to disk in ``qig/symbolic/_cache/``.
The first run may take seconds to minutes; subsequent runs load instantly.

Validation
----------

The LME exact method is validated against scipy:

.. code-block:: python

   # All tests pass with machine precision
   # ||exp(K)_exact - exp(K)_scipy|| ~ 10⁻¹⁵

Key validations:

- Block decomposition: 9×9 → 3×3 + 2×2 + 1×1×4 ✓
- Eigenvalues at most quadratic ✓
- 20 generators span full entangled subspace (rank 9) ✓
- Machine precision agreement with scipy.linalg.expm ✓

Current Status
--------------

**Complete:**

- EXACT exp(K) for LME dynamics via block decomposition
- Antisymmetric part A ≠ 0 (proves Hamiltonian dynamics exist)
- 20 block-preserving generators identified

**In progress (CIP-0007):**

- Symmetric part S and full Jacobian M
- Qubit (d=2) implementation

See Also
--------

- :doc:`generic_structure` - GENERIC decomposition theory
- :doc:`quantum_exponential_families` - Quantum exponential family background
- CIP-0007 in the repository for implementation details

