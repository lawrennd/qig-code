Quick Start
===========

This quick start guide demonstrates the basic usage of **qig** for quantum information geometry.

Creating a Quantum Exponential Family
--------------------------------------

The core object is :class:`~qig.exponential_family.QuantumExponentialFamily`, which represents
a quantum state as :math:`\rho(\theta) = \exp(\sum_a \theta_a F_a - \psi(\theta))`.

.. code-block:: python

   from qig.exponential_family import QuantumExponentialFamily
   import numpy as np
   
   # Create a qutrit (d=3) exponential family with Gell-Mann operators
   exp_fam = QuantumExponentialFamily(d=3, basis_type='gell-mann')
   
   # Natural parameters (8 for a qutrit with traceless operators)
   theta = np.zeros(exp_fam.n_params)
   
   # Get the density matrix
   rho = exp_fam.rho_from_theta(theta)
   print(f"Density matrix shape: {rho.shape}")

Computing the Fisher Information Metric
----------------------------------------

The Fisher information metric (BKM metric) is computed using the Kubo-Mori formula:

.. code-block:: python

   # Compute the Fisher information matrix G(θ)
   G = exp_fam.fisher_information(theta)
   
   print(f"Fisher metric shape: {G.shape}")
   print(f"Metric is symmetric: {np.allclose(G, G.T)}")

Constrained Dynamics
--------------------

Study dynamics constrained to preserve marginal entropies:

.. code-block:: python

   from qig.dynamics import InaccessibleGameDynamics
   
   # Create dynamics object
   dynamics = InaccessibleGameDynamics(exp_fam)
   
   # Initial state (small perturbation from origin)
   theta_0 = np.random.randn(exp_fam.n_params) * 0.1
   
   # Integrate dynamics
   result = dynamics.integrate(
       theta_0,
       t_span=(0.0, 1.0),
       n_points=100
   )
   
   print(f"Integrated {len(result['theta'])} time steps")
   print(f"Final constraint value: {result['C'][-1]:.6f}")

Working with Entanglement
--------------------------

For studying entanglement, use the pair operator basis:

.. code-block:: python

   # Create exponential family with pair operators
   exp_fam_pairs = QuantumExponentialFamily(
       d=3,
       basis_type='gell-mann',
       include_pairs=True
   )
   
   print(f"With pairs: {exp_fam_pairs.n_params} parameters")
   
   # Compute mutual information
   theta = np.random.randn(exp_fam_pairs.n_params) * 0.1
   rho = exp_fam_pairs.rho_from_theta(theta)
   
   # Get marginal entropies
   from qig.core import marginal_entropies
   h = marginal_entropies(rho, dims=[3, 3])
   mutual_info = h[0] + h[1] - (-np.trace(rho @ np.log(rho)))
   
   print(f"Mutual information: {mutual_info:.6f} nats")

Next Steps
----------

* Explore the :doc:`user_guide/index` for detailed tutorials
* Read the :doc:`api/index` for complete API documentation
* See the :doc:`theory/index` for mathematical background

