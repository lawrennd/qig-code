GENERIC Structure
=================

General Equation for Non-Equilibrium Reversible-Irreversible Coupling
----------------------------------------------------------------------

GENERIC provides a geometric decomposition of dynamics into reversible and irreversible parts.

Decomposition
-------------

The dynamics can be written as:

.. math::

   \dot{\theta} = F(\theta) = F_{\text{rev}} + F_{\text{irr}}

where the Jacobian decomposes as:

.. math::

   M = \frac{\partial F}{\partial \theta} = S + A

with:

* :math:`S` = symmetric part (dissipation)
* :math:`A` = antisymmetric part (circulation/Hamiltonian)

For Constrained Quantum Dynamics
---------------------------------

The constrained dynamics have the form:

.. math::

   \dot{\theta} = -G(\theta)\theta + \nu(\theta) a(\theta)

where:

* :math:`G` is the Fisher information metric
* :math:`a = \nabla C` is the constraint gradient
* :math:`\nu` is the Lagrange multiplier

When Duhamel Integrals Are Needed
----------------------------------

A crucial computational insight concerns when the Duhamel integral formula is required.

The Lie Closure Cancellation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When working with a **Lie-closed operator basis** :math:`\{F_a\}` (where :math:`[F_a, F_b] = 2i\sum_c f_{abc} F_c`),
an algebraic cancellation occurs for **scalar derivatives** in natural parameter space.

This simplification means the Duhamel integral for differentiating matrix exponentials
cancels out when computing gradients of scalar functions like :math:`\psi(\theta) = \log \text{Tr}[e^{\sum_a \theta_a F_a}]`.

✅ No Duhamel Required
~~~~~~~~~~~~~~~~~~~~~~

**For computations in natural parameter space** :math:`\theta`:

1. **Computing the Jacobian** :math:`M`:

   .. math::

      M = -G - (\nabla G)[\theta] + \nu \nabla^2 C + a(\nabla\nu)^T

   - Uses third cumulant :math:`T_{abc} = \partial^3\psi/\partial\theta_a\partial\theta_b\partial\theta_c`
   - This is a **scalar derivative** of :math:`\psi(\theta)`
   - ✅ Lie closure ensures cancellation

2. **Entropy gradient** :math:`\partial S/\partial\theta`:
   - Scalar derivative of von Neumann entropy
   - ✅ Cancellation applies

3. **All flow computations in** :math:`\theta`-**space**:
   - :math:`\dot{\theta} = F(\theta) = -\Pi_\parallel G\theta`
   - GENERIC decomposition :math:`M = S + A`
   - Effective Hamiltonian extraction from :math:`A`
   - ✅ Pure algebraic operations

❌ Duhamel Required
~~~~~~~~~~~~~~~~~~~

**For mapping to density matrix space** :math:`\rho`:

1. **Kubo-Mori derivatives** :math:`\partial\rho/\partial\theta`:

   .. math::

      \frac{\partial\rho}{\partial\theta_a} = \int_0^1 \rho^s (F_a - \langle F_a \rangle I) \rho^{1-s} \, ds

   - This is a **matrix-valued derivative**, not a scalar
   - ❌ No cancellation - full Duhamel integral required
   - Implemented in :mod:`qig.duhamel`

2. **Diffusion operator** :math:`\mathcal{D}[\rho]`:

   .. math::

      \mathcal{D}[\rho] = \sum_a (S \cdot q)_a \frac{\partial\rho}{\partial\theta_a}

   - Maps parameter space flow to density matrix flow
   - ❌ Requires Kubo-Mori derivatives

3. **Full dynamics in density matrix form**:

   .. math::

      \dot{\rho} = -i[H_{\text{eff}}, \rho] + \mathcal{D}[\rho]

   - Master equation representation
   - ❌ Requires :math:`\mathcal{D}[\rho]` which needs Duhamel

Practical Implications
~~~~~~~~~~~~~~~~~~~~~~

**The quantum inaccessible game is played entirely in natural parameter space** :math:`\theta`.

Therefore:

* **Core game dynamics** (NO Duhamel needed):
  
  - Flow computation: :math:`\dot{\theta} = -\Pi_\parallel G\theta`
  - GENERIC decomposition: :math:`M = S + A`
  - Effective Hamiltonian extraction: :math:`\eta` from :math:`A`
  - Constraint enforcement
  - Endpoint detection

* **Duhamel only required for**:
  
  - Visualizing :math:`\rho(t)` (density matrix evolution)
  - Computing :math:`\mathcal{D}[\rho]` for comparison with master equations
  - Connecting to standard quantum optics formulations
  - Physical interpretation in density matrix language

**Implementation consequence**: Structure constant computation, GENERIC decomposition,
and Hamiltonian extraction work purely in parameter space and benefit from the
Lie closure simplification. Only diffusion operator construction requires the
more expensive Duhamel integral computation from :mod:`qig.duhamel`.

See Also
--------

* :mod:`qig.dynamics` - Dynamics implementation
* :mod:`qig.core` - ``generic_decomposition()`` function

