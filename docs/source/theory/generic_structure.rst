GENERIC Structure
=================

*This section is under development.*

General Equation for Non-Equilibrium Reversible-Irreversible Coupling
----------------------------------------------------------------------

GENERIC provides a geometric decomposition of dynamics into reversible and irreversible parts.

Decomposition
-------------

The dynamics can be written as:

.. math::

   \dot{\theta} = F(\theta) = F_{\text{rev}} + F_{\text{irr}}

where the Jacobian decomposes as:

.. math::

   M = \frac{\partial F}{\partial \theta} = S + A

with:

* :math:`S` = symmetric part (dissipation)
* :math:`A` = antisymmetric part (circulation)

For Constrained Quantum Dynamics
---------------------------------

The constrained dynamics have the form:

.. math::

   \dot{\theta} = -G(\theta)\theta + \nu(\theta) a(\theta)

where:

* :math:`G` is the Fisher information metric
* :math:`a = \nabla C` is the constraint gradient
* :math:`\nu` is the Lagrange multiplier

See Also
--------

* :mod:`qig.dynamics` - Dynamics implementation
* :mod:`qig.core` - ``generic_decomposition()`` function

