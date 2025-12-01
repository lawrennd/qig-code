Constraint Manifolds
====================

*This section is under development.*

Marginal Entropy Constraints
-----------------------------

The inaccessible game framework constrains dynamics to preserve marginal entropies.

Constraint Function
~~~~~~~~~~~~~~~~~~~

.. math::

   C(\theta) = \sum_i h_i(\theta)

where :math:`h_i` are marginal entropies.

Computing Constraints
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   exp_fam = QuantumExponentialFamily(d=3, include_pairs=True)
   theta = np.random.randn(exp_fam.n_params) * 0.1
   
   C, a = exp_fam.marginal_entropy_constraint(theta)
   # C: constraint value
   # a: constraint gradient ∇C

See Also
--------

* :mod:`qig.exponential_family` - Constraint methods
* :doc:`../theory/inaccessible_game` - Constraint theory

