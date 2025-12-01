The Inaccessible Game
======================

*This section is under development.*

Information-Geometric Framework
--------------------------------

The inaccessible game framework studies dynamics constrained to preserve marginal entropies.

Key Concepts:

* **Constraint manifold**: States with fixed marginal entropies
* **Inaccessible region**: States violating the constraint
* **Information gain**: Increase in total entropy
* **Accessibility**: Whether information gain is compatible with constraint

The Central Question
--------------------

Given a constraint :math:`C(\theta) = \text{const}` (marginal entropies fixed):

* Which directions increase total entropy :math:`S(\theta)`?
* Which of those directions preserve the constraint?
* What is the "inaccessible" entropy gain?

Geometric Interpretation
-------------------------

The dynamics follow a constrained gradient flow on the constraint manifold,
using the Fisher information metric as the Riemannian structure.

See Also
--------

* :mod:`qig.dynamics` - Constrained dynamics implementation
* :doc:`generic_structure` - GENERIC decomposition

