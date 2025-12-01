Quantum Exponential Families
=============================

*This section is under development.*

Mathematical Framework
----------------------

A quantum exponential family is defined by:

.. math::

   \rho(\theta) = \exp\left(\sum_a \theta_a F_a - \psi(\theta)\right)

where:

* :math:`\rho(\theta)` is a density matrix (positive semidefinite, trace 1)
* :math:`F_a` are Hermitian operators (generators)
* :math:`\theta = (\theta_1, \ldots, \theta_n)` are natural parameters
* :math:`\psi(\theta) = \log \text{Tr}[\exp(\sum_a \theta_a F_a)]` is the log-partition function

Properties
----------

* **Convexity**: :math:`\psi(\theta)` is strictly convex
* **Duality**: Expectation parameters :math:`\eta_a = \langle F_a \rangle`
* **Fisher metric**: :math:`G_{ab} = \text{Cov}(F_a, F_b)` where covariance uses BKM inner product

See Also
--------

* :mod:`qig.exponential_family` - Implementation

