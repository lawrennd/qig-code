qig.core
========

Core quantum information utilities for density matrices and entropy calculations.

This module provides the fundamental building blocks used throughout the library:

* **Partial trace** — :func:`~qig.core.partial_trace`: reduce a bipartite density
  matrix to one subsystem.
* **Von Neumann entropy** — :func:`~qig.core.von_neumann_entropy`: compute
  :math:`S(\rho) = -\operatorname{tr}(\rho \log \rho)`.
* **Marginal entropies** — :func:`~qig.core.marginal_entropies`: joint and
  subsystem entropies for bipartite systems.
* **LME state** — :func:`~qig.core.create_lme_state`: construct a
  Lindblad-Master-Equation (maximally mixed) state with specified marginals.
* **GENERIC decomposition** — :func:`~qig.core.generic_decomposition`: split a
  flow matrix :math:`M` into symmetric (dissipative) and antisymmetric
  (reversible) parts.
* **Loewner kernel** — :func:`~qig.core.loewner_kernel`: compute the
  Kubo-Mori / BKM divided-difference kernel matrix in the eigenbasis of a
  density matrix. This is the central object linking exponential-family geometry
  to the BKM Fisher information metric.

  .. math::

     C_{ij} = \begin{cases}
       \dfrac{\lambda_i - \lambda_j}{\log \lambda_i - \log \lambda_j} &
       \lambda_i \neq \lambda_j \\[6pt]
       \lambda_i & \lambda_i = \lambda_j
     \end{cases}

  In the LME limit all :math:`\lambda_i \to 1/D` and every entry
  :math:`C_{ij} \to 1/D`, recovering flat Fisher geometry.

.. automodule:: qig.core
   :members:
   :undoc-members:
   :show-inheritance:

