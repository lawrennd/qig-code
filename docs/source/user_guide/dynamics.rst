Constrained Dynamics
====================

*This section is under development.*

Introduction
------------

The **qig** package implements constrained quantum dynamics that preserve marginal entropies.

Creating Dynamics
-----------------

.. code-block:: python

   from qig.dynamics import InaccessibleGameDynamics
   from qig.exponential_family import QuantumExponentialFamily
   
   exp_fam = QuantumExponentialFamily(d=3)
   dynamics = InaccessibleGameDynamics(exp_fam)

Solving Constrained Dynamics
-----------------------------

.. code-block:: python

   import numpy as np
   
   # Initial state
   theta_0 = np.random.randn(exp_fam.n_params) * 0.1
   
   # Solve constrained maximum entropy dynamics
   result = dynamics.solve_constrained_maxent(
       theta_init=theta_0,
       n_steps=1000,
       dt=0.001
   )

See Also
--------

* :mod:`qig.dynamics` - API reference
* :doc:`../theory/inaccessible_game` - Theoretical framework

