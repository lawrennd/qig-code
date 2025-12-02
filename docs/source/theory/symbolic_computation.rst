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

Natural Parameter Interpretation
--------------------------------

The natural parameters θ in the quantum exponential family have specific meanings:

- **θ = 0**: Maximally mixed state (ρ = I/D)
- **θ → -∞** (large negative): Locally maximally entangled (LME/Bell) states

For LME states, regularization (ε ~ 10⁻³) keeps parameters finite but large:
||θ|| ~ 3 with some components around -3.

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

- The maximally mixed state: :math:`\rho = I/3` (corresponds to θ = 0)
- LME (Bell) states: :math:`|\Phi^+\rangle` (corresponds to θ → -∞)
- Partial traces of maximally entangled states
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

2. **General su(9)** (``qig.symbolic.su9_taylor_approximation``):
   
   - Uses Taylor expansion for exp(K)
   - ~1% error at order 2, ~0.0008% at order 6
   - Works for all 80 generators

Usage Example: LME Exact
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from qig.symbolic.lme_exact import (
       exact_exp_K_lme,
       exact_rho_lme,
       exact_constraint_lme,
       exact_psi_lme,
       block_preserving_generators,
   )
   import sympy as sp
   
   # Get available generators
   generators, names = block_preserving_generators()
   print(f"20 block-preserving generators: {names[:4]}...")
   
   # Create symbolic parameters
   a = sp.Symbol('a', real=True)  # local
   c = sp.Symbol('c', real=True)  # entangling
   theta = {'λ3⊗I': a, 'λ1⊗λ1': c}
   theta_list = [a, c]
   
   # EXACT exp(K) - no Taylor approximation!
   exp_K = exact_exp_K_lme(theta)
   
   # EXACT constraint C = h₁ + h₂
   C = exact_constraint_lme(theta)
   
   # Constraint gradient a = ∇C
   a_vec = sp.Matrix([sp.diff(C, t) for t in theta_list])
   
   # Fisher information G = ∇²ψ
   psi = exact_psi_lme(theta)
   G = sp.Matrix([[sp.diff(sp.diff(psi, ti), tj) 
                   for tj in theta_list] for ti in theta_list])
   
   # Lagrange multiplier ν = (aᵀGθ)/(aᵀa)
   theta_vec = sp.Matrix(theta_list)
   nu = (a_vec.T * G * theta_vec)[0,0] / (a_vec.T * a_vec)[0,0]
   
   # Antisymmetric part A = (1/2)[a(∇ν)ᵀ - (∇ν)aᵀ]
   grad_nu = sp.Matrix([sp.diff(nu, t) for t in theta_list])
   A = (a_vec * grad_nu.T - grad_nu * a_vec.T) / 2

Precomputed Expressions
^^^^^^^^^^^^^^^^^^^^^^^

For fast evaluation, precomputed symbolic expressions are available in
``qig/symbolic/precomputed/``. These were generated once and saved to Python files:

.. code-block:: python

   from qig.symbolic.precomputed.two_param_chain import (
       a, c,  # symbolic parameters
       G, nu, grad_nu,  # intermediate quantities
       M, S, A,  # Jacobian and its decomposition
   )
   
   # Evaluate at specific values (e.g., LME scale)
   vals = {a: 2.0, c: 2.0}
   nu_val = float(nu.subs(vals))
   A_num = A.subs(vals)

Caching
^^^^^^^

Expensive symbolic computations are cached to disk in ``qig/symbolic/_cache/``.
The first run may take seconds to minutes; subsequent runs load instantly.

Validation
----------

Run tests with:

.. code-block:: bash

   pytest tests/test_lme_exact.py -v

Key validations:

- **exp(K)**: matches scipy.linalg.expm to ~10⁻¹² ✓
- **Constraint gradient a**: matches finite difference ✓
- **Fisher info G**: matches numerical Hessian ✓
- **ν for local params**: equals -1 (structural identity) ✓
- **ν for entangling params**: ≠ -1 ✓
- **A for local params**: equals 0 ✓
- **A for entangling params**: ≠ 0 (Hamiltonian dynamics!) ✓
- **A antisymmetry**: A + Aᵀ = 0 ✓
- **S symmetry**: S = Sᵀ ✓
- **M decomposition**: M = S + A ✓

Current Status
--------------

**Complete (CIP-0007):**

- EXACT exp(K) for LME dynamics via block decomposition
- EXACT density matrix ρ and reduced density matrices ρ₁, ρ₂
- EXACT marginal entropies h₁, h₂
- EXACT constraint gradient a = ∇(h₁ + h₂)
- EXACT Fisher information G = ∇²ψ
- EXACT Lagrange multiplier ν = (aᵀGθ)/(aᵀa)
- EXACT gradient ∇ν
- **EXACT antisymmetric part A = (1/2)[a(∇ν)ᵀ - (∇ν)aᵀ]**
- **EXACT constraint Hessian ∇²C**
- **EXACT (∇G)[θ] term**
- **EXACT full Jacobian M = -G - (∇G)[θ] + ν∇²C + a(∇ν)ᵀ**
- **EXACT symmetric part S = (M + Mᵀ)/2**
- 20 block-preserving generators identified
- Precomputed expressions for 2-parameter subset

**Key results:**

- Local parameters only: ν = -1, ∇ν = 0, A = 0 (structural identity holds)
- With entangling parameters: ν ≠ -1, ∇ν ≠ 0, **A ≠ 0** (Hamiltonian dynamics!)
- Results verified at LME scale (||θ|| ~ 3): A ≠ 0 for mixed local+entangling params

**Planned:**

- Qubit (d=2) implementation
- Extraction of effective Hamiltonian H_eff from A
- Extraction of diffusion operator D[ρ] from S

See Also
--------

- :doc:`generic_structure` - GENERIC decomposition theory
- :doc:`quantum_exponential_families` - Quantum exponential family background
- CIP-0007 in the repository for implementation details
