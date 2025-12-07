Module: qig.symbolic
====================

The symbolic computation subpackage for exact GENERIC decomposition.

Overview
--------

The ``qig.symbolic`` package provides exact symbolic computation for qutrit
pair systems at the LME (Locally Maximally Entangled) origin. Key features:

- **Exact exp(K)** via block decomposition (no Taylor approximation)
- **Block structure**: 9×9 → 3×3 + 2×2 + 1×1×4
- **Numeric-symbolic bridge** for connecting exponential family θ to block params

Submodules
----------

qig.symbolic.lme_exact
^^^^^^^^^^^^^^^^^^^^^^

Core module for exact LME computations.

.. automodule:: qig.symbolic.lme_exact
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
-------------

Block Decomposition
^^^^^^^^^^^^^^^^^^^

.. autofunction:: qig.symbolic.lme_exact.exact_exp_K_lme

.. autofunction:: qig.symbolic.lme_exact.exact_rho_lme

.. autofunction:: qig.symbolic.lme_exact.exact_constraint_lme

.. autofunction:: qig.symbolic.lme_exact.exact_psi_lme

Generators
^^^^^^^^^^

.. autofunction:: qig.symbolic.lme_exact.block_preserving_generators

.. autofunction:: qig.symbolic.lme_exact.permutation_matrix

.. autofunction:: qig.symbolic.lme_exact.extract_blocks

Numeric Bridge
^^^^^^^^^^^^^^

.. autofunction:: qig.symbolic.lme_exact.numeric_lme_blocks_from_theta

Example Usage
-------------

Basic symbolic computation:

.. code-block:: python

   from qig.symbolic.lme_exact import (
       exact_exp_K_lme,
       exact_constraint_lme,
       block_preserving_generators,
   )
   import sympy as sp
   
   # Create symbolic parameters
   a = sp.Symbol('a', real=True)
   c = sp.Symbol('c', real=True)
   theta = {'λ3⊗I': a, 'λ1⊗λ1': c}
   
   # Exact computations
   exp_K = exact_exp_K_lme(theta)
   C = exact_constraint_lme(theta)

Bridging numeric and symbolic:

.. code-block:: python

   from qig.symbolic.lme_exact import numeric_lme_blocks_from_theta
   from qig.exponential_family import QuantumExponentialFamily
   
   # Numeric θ from exponential family
   qef = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
   theta = qef.get_bell_state_parameters(log_epsilon=-20)
   
   # Extract symbolic-style blocks
   blocks = numeric_lme_blocks_from_theta(theta, qef.operators)
   print(blocks['ent_3x3'])  # 3×3 entangled block

See Also
--------

- :doc:`/theory/symbolic_computation` - Theory and mathematical background
- ``examples/lme_numeric_symbolic_bridge.ipynb`` - Tutorial notebook

