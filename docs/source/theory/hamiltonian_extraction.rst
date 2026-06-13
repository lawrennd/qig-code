Hamiltonian Extraction from Antisymmetric GENERIC Flow
=======================================================

The antisymmetric sector of the GENERIC decomposition must be unitary — a
consequence of the categorical axioms on reversible, entropy-conserving
morphisms. This page explains how that unitary sector is explicitly mapped to
an effective Hamiltonian :math:`H_\text{eff}(\theta)` such that the reversible
density-matrix dynamics take the von Neumann form

.. math::

   \dot{\rho}_\text{rev} = -i[H_\text{eff}(\theta), \rho(\theta)].

Mathematical Background
-----------------------

GENERIC Decomposition
~~~~~~~~~~~~~~~~~~~~~

The Jacobian of the constrained flow :math:`\dot{\theta} = F(\theta)` splits as

.. math::

   M(\theta) = S(\theta) + A(\theta),

where :math:`S = \tfrac{1}{2}(M + M^T)` is the symmetric (irreversible, dissipative)
part and :math:`A = \tfrac{1}{2}(M - M^T)` is the antisymmetric (reversible,
entropy-conserving) part.

The antisymmetric part generates circulation on the constraint manifold and,
in finite-dimensional examples, must correspond to a Hamiltonian flow.

Lie-Algebraic Structure
~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`\{F_a\}` be a Hermitian operator basis that is **Lie-closed**:

.. math::

   [F_a, F_b] = 2i \sum_c f_{abc} F_c,

where :math:`f_{abc}` are the real structure constants of the corresponding Lie
algebra (e.g. :math:`\mathfrak{su}(d)` for qutrits, :math:`\mathfrak{su}(4)` for qubit pairs).

An antisymmetric flow :math:`A` acting on natural parameters :math:`\theta` induces
a density-matrix vector field. For this to equal the commutator flow of some
Hamiltonian :math:`H_\text{eff} = \sum_c \eta_c F_c`, the coefficients :math:`\eta_c`
must satisfy the linear system

.. math::

   (A \theta)_b = \sum_{a,c} f_{abc}\, \eta_c(\theta) \, \theta_a,

which can be written in matrix form as

.. math::

   A\,\theta = \mathbf{f} \cdot \eta(\theta),

where :math:`\mathbf{f}_{bc} = \sum_a f_{abc}\, \theta_a` is determined by the
structure constants and the current parameter point.

Extraction Algorithm
---------------------

Given a point :math:`\theta^*` on the constraint manifold:

1. **Compute** the Jacobian :math:`M(\theta^*)` and extract
   :math:`A = \tfrac{1}{2}(M - M^T)`.

2. **Solve** the linear system :math:`\mathbf{f} \cdot \eta = A\,\theta` for the
   coefficient vector :math:`\eta \in \mathbb{R}^{\dim\mathfrak{g}}`.
   For well-conditioned bases this is a standard least-squares problem.

3. **Build** the effective Hamiltonian:

   .. math::

      H_\text{eff}(\theta) = \sum_c \eta_c(\theta)\, F_c.

4. **Verify** that :math:`H_\text{eff}` is Hermitian and traceless (as it should
   be for an element of :math:`\mathfrak{su}(d^2)`).

Implementation
--------------

The extraction is available in :mod:`qig.generic_decomposition`:

.. code-block:: python

   from qig.generic_decomposition import (
       effective_hamiltonian_coefficients,
       effective_hamiltonian_operator,
       verify_antisymmetric_flow_equals_commutator,
   )
   from qig.structure_constants import structure_constants
   from qig.core import generic_decomposition

   # At a parameter point theta_star:
   M = exp_fam.jacobian(theta_star)
   S, A = generic_decomposition(M)

   # Extract structure constants for the chosen basis
   f_abc = structure_constants(operators)           # shape (n_ops, n_ops, n_ops)

   # Solve A @ theta = f @ eta for eta
   eta = effective_hamiltonian_coefficients(A, theta_star, f_abc)

   # Build H_eff = sum_c eta_c F_c
   H_eff = effective_hamiltonian_operator(eta, operators)

   print(f"H_eff Hermitian: {np.allclose(H_eff, H_eff.conj().T)}")
   print(f"H_eff traceless: {abs(np.trace(H_eff)) < 1e-10}")

A complete worked example is in
``examples/effective_hamiltonian_derivation.ipynb`` and the qutrit-pair case
is demonstrated in ``examples/origin_two_qutrit_worked_example.ipynb``.

Validation and Limitations
---------------------------

Structural properties that hold exactly:

- :math:`H_\text{eff}` is **Hermitian** (verified to machine precision).
- :math:`H_\text{eff}` is **traceless** (by construction from :math:`\mathfrak{su}` basis).
- The extraction identity :math:`A\,\theta = \mathbf{f} \cdot \eta` holds to
  :math:`\sim 10^{-6}` relative error for qutrit-pair examples.

A subtlety about the Kubo-Mori kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A natural (but incorrect) expectation is that the induced density-matrix flow
should satisfy the **strong BCH identity**

.. math::

   \sum_a \eta_a \frac{\partial\rho}{\partial\theta_a}
   \stackrel{?}{=} -i[H_\text{eff}, \rho].

Numerical investigation shows this is **false** — with :math:`\sim 14\times`
relative error for both numerical and analytical (spectral) Duhamel methods.
The left-hand side includes the Kubo-Mori kernel
:math:`K_\rho = f(\operatorname{ad}_H)` where
:math:`f(z) = (e^z - 1)/z`, while the right-hand side is a pure commutator.
The two differ by a factor of :math:`\sim 7`–:math:`8\times` due to the
operator-ordered integral structure of :math:`\partial\rho/\partial\theta_a`.

**The correct statement** is weaker: the antisymmetric parameter-space flow
corresponds to a Hamiltonian in the sense that :math:`H_\text{eff}` generates
the correct symplectic structure on the manifold, but the naive identification
of Kubo-Mori derivatives with commutator brackets fails because the
Kubo-Mori inner product and the standard Hilbert-Schmidt inner product differ.

This is a valuable theoretical clarification, not a limitation of the
implementation. Tests in ``tests/test_generic_hamiltonian.py`` guard
against reintroduction of the over-strong assumption.

See Also
--------

* :mod:`qig.generic_decomposition` — GENERIC decomposition and Hamiltonian extraction API
* :mod:`qig.structure_constants` — Lie structure constants :math:`f_{abc}`
* :doc:`generic_structure` — GENERIC decomposition overview
* :doc:`../api/duhamel` — Kubo-Mori (Duhamel) derivatives
* :func:`qig.core.loewner_kernel` — Kubo-Mori divided-difference kernel

References
----------

- **Paper**: *The Inaccessible Game* — sections on GENERIC decomposition,
  categorical forcing of unitarity, and Hamiltonian reconstruction from
  :math:`A` and :math:`f_{abc}`.
- **CIP-0009**: Explicit Hamiltonian Extraction from Antisymmetric GENERIC Flow.
- **Notebook**: ``examples/effective_hamiltonian_derivation.ipynb`` — symbolic
  derivation and numerical validation.
