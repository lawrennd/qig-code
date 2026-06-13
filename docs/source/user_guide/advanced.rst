Advanced Topics
===============

*This section is under development.*

Entanglement with Pair Operators
---------------------------------

For studying entanglement, use pair operator bases that can create correlations.

Kubo-Mori (Duhamel) Derivatives
---------------------------------

For a quantum exponential family :math:`\rho(\theta) = e^H` with
:math:`H = \sum_a \theta_a F_a - \psi(\theta) I`, the exact derivative is

.. math::

   \frac{\partial \rho}{\partial \theta_a}
   = \int_0^1 e^{(1-s)H} \bigl(F_a - \langle F_a \rangle I\bigr) e^{sH}\,\mathrm{d}s.

This is the Kubo-Mori (Duhamel) derivative, and it is what
:meth:`~qig.exponential_family.QuantumExponentialFamily.rho_derivative` computes.

**Choosing a method**

Three numerical strategies are available via the ``method`` argument:

.. list-table::
   :header-rows: 1
   :widths: 15 25 20 40

   * - ``method``
     - Cost
     - Accuracy
     - When to use
   * - ``'duhamel_block'`` *(default)*
     - 1 ``expm`` on a :math:`2n \times 2n` matrix
     - Machine precision
     - Robust default; handles ill-conditioned :math:`H`
   * - ``'duhamel_spectral'``
     - 1 ``eigh`` + kernel application
     - Machine precision
     - When eigendecomposition is cheap and :math:`H` is well-conditioned
   * - ``'duhamel'``
     - 50 ``expm`` calls (trapezoid)
     - ~10⁻⁵
     - Validation / cross-checking only

The **block method** (default since January 2026) implements Higham's identity:

.. math::

   \exp\begin{pmatrix} H & F \\ 0 & H \end{pmatrix}
   =
   \begin{pmatrix} e^H & D\exp_H[F] \\ 0 & e^H \end{pmatrix}

so the Duhamel integral is "compiled away" into a single matrix exponential.
This avoids explicit quadrature and eigendecomposition, and delegates all
numerical work to the highly-optimised ``scipy.linalg.expm`` (Padé + scaling
and squaring).

**Higher-order derivatives and cumulants**

The block trick extends to 2nd and 3rd Fréchet derivatives via 3×3 / 4×4 block
matrices. These are available as utilities in :mod:`qig.duhamel` for computing:

* the **Hessian** of :math:`\psi(\theta)` (2nd cumulant / BKM Fisher metric),
* the **3rd cumulant contraction** :math:`(\nabla G)[\theta]` for small systems.

Third-Order Cumulants
---------------------

The third cumulant :math:`\nabla G = \nabla \nabla \nabla \psi` describes how
the Fisher metric varies across the manifold. It is available via the block
Fréchet method in :mod:`qig.duhamel` and is used internally for validation.

Optimisation on Manifolds
--------------------------

Constrained optimisation techniques for quantum state spaces are implemented in
:mod:`qig.dynamics` and :mod:`qig.generic_decomposition`.

See Also
--------

* :mod:`qig.duhamel` — Duhamel derivative implementations and API reference
* :doc:`../api/duhamel` — Method comparison table
* :mod:`qig.pair_operators` — Entangling operator bases

