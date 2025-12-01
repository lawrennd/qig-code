Fisher Information and BKM Metric
==================================

*This section is under development.*

The Bogoliubov-Kubo-Mori Metric
--------------------------------

The quantum Fisher information is defined using the Bogoliubov-Kubo-Mori (BKM) inner product:

.. math::

   G_{ab}(\theta) = \int_0^1 \text{tr}\left[\rho^s F_a \rho^{1-s} F_b\right] ds

Properties:

* Symmetric: :math:`G_{ab} = G_{ba}`
* Positive semidefinite
* Reduces to classical Fisher information for commuting operators

Quantum Covariance
------------------

The BKM metric can be expressed as a quantum covariance:

.. math::

   G_{ab} = \text{cov}_{BKM}(F_a, F_b)

This generalizes the classical covariance to non-commuting operators.

See Also
--------

* :mod:`qig.exponential_family` - ``fisher_information()`` method

