Hamiltonian Extraction from Antisymmetric GENERIC Flow
=======================================================

The antisymmetric sector of the GENERIC decomposition must be unitary — a
consequence of the categorical axioms on reversible, entropy-conserving
morphisms. This page explains how that unitary sector is explicitly mapped to
an effective Hamiltonian :math:`H_\text{eff}(\theta)` such that the reversible
density-matrix dynamics take the von Neumann form

.. math::

   \dot{\rho}_\text{rev} = -i[H_\text{eff}(\theta), \rho(\theta)].

Gibbs-Locked Frames
-------------------

The central object of the Hamiltonian-emergence paper is a **Gibbs-locked frame**
:math:`K_0 = \beta H`.  A Gibbs-locked frame is a fixed point of the linearised
reversible dynamics: because :math:`[K_0, H] = [H, H] = 0` by construction, the
commutator term :math:`i[\delta K, K_0]` in the linearised GENERIC equation
generates only phase rotation in the eigenbasis of :math:`H`, without displacing
:math:`K_0` itself.

The background Gibbs state is

.. math::

   \rho_0 = \frac{e^{-\beta H}}{\operatorname{tr}(e^{-\beta H})},

and the **Bohr gaps** are the eigenvalue differences of :math:`H`:

.. math::

   \Delta\epsilon_{ij} = \epsilon_i - \epsilon_j,
   \qquad H \,|\!\epsilon_i\rangle = \epsilon_i\,|\!\epsilon_i\rangle.

Linearised dynamics and the uniform decay rate :math:`\mu_0`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Near a Gibbs-locked background the linearised GENERIC equation decouples
element-by-element in the eigenbasis of :math:`H`:

.. math::

   \frac{d}{dt}(\delta\rho)_{ij}
   = \bigl(i\,\beta\,\Delta\epsilon_{ij} - \mu_0\bigr)\,(\delta\rho)_{ij},

with analytical solution

.. math::

   (\delta\rho)_{ij}(t)
   = (\delta\rho)_{ij}(0)\,
     \exp\!\bigl[(i\,\beta\,\Delta\epsilon_{ij} - \mu_0)\,t\bigr].

Here :math:`\mu_0 \geq 0` is the **uniform decay rate** that controls how
quickly off-diagonal coherences decay.  It can be inferred from a trajectory
via :func:`qig.gibbs_lock.infer_mu0`, which strips the oscillatory phase factor
and fits the mean off-diagonal magnitude to :math:`e^{-\mu_0 t}`.

Loewner-kernel pushforward
~~~~~~~~~~~~~~~~~~~~~~~~~~

In modular-generator coordinates, perturbations are described by :math:`\delta K`.
The corresponding first-order change in the density matrix is given by the
**Loewner (Kubo-Mori) divided-difference map**:

.. math::

   (\delta\rho)_{ij}
   = c(\lambda_i, \lambda_j)\,(\delta K)_{ij},
   \qquad
   c(\lambda_i, \lambda_j)
   = \frac{\lambda_i - \lambda_j}{\log\lambda_i - \log\lambda_j},

where :math:`\lambda_i` are the eigenvalues of :math:`\rho_0` and the
off-diagonal formula is extended by L'Hôpital's rule to
:math:`c(\lambda, \lambda) = \lambda` on the diagonal.

Iso-marginal perturbations are those for which the first-order change in every
subsystem marginal is zero:

.. math::

   \operatorname{tr}_{\neq k}\!\bigl(J_{\rho_0}(\delta K)\bigr) = 0
   \quad \forall\, k.

The :class:`~qig.gibbs_lock.GibbsLockedFrame` class exposes both the kernel
and the iso-marginal test:

.. code-block:: python

   from qig import GibbsLockedFrame

   frame = GibbsLockedFrame(H, beta=2.0, dims=[3, 3])

   # Loewner kernel at rho_0
   C, vals, vecs = frame.loewner_kernel()

   # Map a modular perturbation to density-matrix coordinates
   delta_rho = frame.loewner_map(delta_K)

   # Check iso-marginal condition
   print(frame.is_iso_marginal(delta_K))   # True / False

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

   (A \theta)_r = \sum_{b,c} f_{rbc}\, \theta_b\, \eta_c(\theta),

which can be written in matrix form as

.. math::

   F\,\eta(\theta) = A\,\theta,

where :math:`F_{rc} = \sum_b f_{rbc}\, \theta_b` is the adjoint-action matrix
built from the structure constants and the current parameter point.

.. note::

   **BCH approximation — known limitation.**  The formula above is derived
   under the assumption that the Kubo-Mori derivative
   :math:`\partial\rho/\partial\theta_a` reduces analytically to the commutator
   :math:`[F_a, \rho]` via Baker-Campbell-Hausdorff identities
   (the *strong BCH identity*, see below).  This identity does **not** hold in
   general; the full Kubo-Mori kernel introduces corrections of
   :math:`O(\|\theta\|^2)`.  Consequently, the vector :math:`A\theta` has a
   component outside the column space of :math:`F` of magnitude
   :math:`\sim 0.1\,\|\theta\|^2`, so the system :math:`F\eta = A\theta` is
   **inconsistent**: no choice of :math:`\eta` achieves a residual smaller than
   this BCH floor.  For :math:`\|\theta\|\sim 0.05` the achievable residual is
   :math:`\sim 10^{-3}`, not :math:`10^{-6}`.  The matrix :math:`F` is also
   typically rank-deficient (rank 12 of 15 for the qubit-pair system), reflecting
   the null directions of the adjoint action at finite :math:`\theta`.

Extraction Algorithm
---------------------

Given a point :math:`\theta^*` on the constraint manifold:

1. **Compute** the Jacobian :math:`M(\theta^*)` and extract
   :math:`A = \tfrac{1}{2}(M - M^T)`.

2. **Solve** the linear system :math:`F\,\eta = A\,\theta` for the
   coefficient vector :math:`\eta \in \mathbb{R}^{\dim\mathfrak{g}}`,
   where :math:`F_{rc} = \sum_b f_{rbc}\,\theta_b`.
   Because :math:`F` is rank-deficient and the system is inconsistent
   (BCH residual :math:`\sim 0.1\|\theta\|^2`), the solver uses Tikhonov
   regularisation.

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

   # Solve F @ eta = A @ theta for eta  (F[r,c] = sum_b f[r,b,c] * theta[b])
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
- The extraction formula :math:`F\,\eta \approx A\,\theta` is
  :math:`O(\|\theta\|^2)` accurate; the residual floor is
  :math:`\sim 0.1\,\|\theta\|^2` (e.g. :math:`\sim 10^{-3}` for
  :math:`\|\theta\| \sim 0.05`), not :math:`10^{-6}`.  This arises
  because :math:`A\,\theta` lies partly outside :math:`\operatorname{col}(F)`;
  the exact magnitude of the BCH correction is verified in
  ``tests/test_generic_hamiltonian.py::test_extraction_consistency_multiple_points``.

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
* :mod:`qig.gibbs_lock` — :class:`~qig.gibbs_lock.GibbsLockedFrame` and
  :func:`~qig.gibbs_lock.infer_mu0`
* **Notebook** ``examples/gibbs_lock_hamiltonian_extraction.ipynb`` —
  end-to-end companion for the Hamiltonian-emergence paper.

References
----------

- **Paper**: *The Inaccessible Game* — sections on GENERIC decomposition,
  categorical forcing of unitarity, and Hamiltonian reconstruction from
  :math:`A` and :math:`f_{abc}`.
- **Paper**: *Gibbs-Lock and the Emergence of Hamiltonian Structure in the
  Inaccessible Game* — Gibbs-locked frames, Bohr gaps, iso-marginal perturbations,
  and the uniform decay rate :math:`\mu_0`.
- **CIP-0009**: Explicit Hamiltonian Extraction from Antisymmetric GENERIC Flow.
- **CIP-000C**: Gibbs-Lock API and :math:`\mu_0` Inference.
- **CIP-000D**: Hamiltonian Paper End-to-End Companion notebook.
- **Notebook**: ``examples/effective_hamiltonian_derivation.ipynb`` — symbolic
  derivation and numerical validation.
