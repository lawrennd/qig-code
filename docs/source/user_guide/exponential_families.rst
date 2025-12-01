Quantum Exponential Families
=============================

*This section is under development.*

Introduction
------------

Quantum exponential families represent quantum states in the form:

.. math::

   \rho(\theta) = \exp\left(\sum_a \theta_a F_a - \psi(\theta)\right)

where:

* :math:`F_a` are Hermitian operators (generators)
* :math:`\theta_a` are natural parameters
* :math:`\psi(\theta)` is the log-partition function

Creating an Exponential Family
-------------------------------

.. code-block:: python

   from qig.exponential_family import QuantumExponentialFamily
   
   # Qutrit with Gell-Mann operators
   exp_fam = QuantumExponentialFamily(d=3, basis_type='gell-mann')
   
   # With entangling pair operators
   exp_fam_pairs = QuantumExponentialFamily(
       d=3,
       basis_type='gell-mann',
       include_pairs=True
   )

See Also
--------

* :mod:`qig.exponential_family` - API reference
* :doc:`../theory/quantum_exponential_families` - Mathematical theory

