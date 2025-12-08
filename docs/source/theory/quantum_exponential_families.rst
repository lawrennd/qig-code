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

Why Duhamel Integrals Appear
----------------------------

If you are used to classical exponential families, the appearance of
operator-valued Duhamel integrals in the quantum setting can seem mysterious.
In the classical case, sufficient statistics :math:`T_i(x)` commute with each
other, and

.. math::

   p_\theta(x)
   = \exp\Bigl(\sum_i \theta_i T_i(x) - \psi(\theta)\Bigr)

leads directly to

.. math::

   \frac{\partial}{\partial \theta_i} p_\theta(x)
   = \bigl(T_i(x) - \mathbb{E}_\theta[T_i]\bigr)\,p_\theta(x),

so derivatives of the log-partition function and the Fisher metric can be
expressed using ordinary covariances.

In the quantum case, the sufficient statistics are Hermitian operators
:math:`F_i` and in general do not commute with the Hamiltonian
:math:`K(\theta) = \sum_i \theta_i F_i`. Differentiating the matrix
exponential :math:`\exp(K(\theta))` therefore produces the
Wilcox/Duhamel formula

.. math::

   \frac{\partial}{\partial \theta_i} e^{K(\theta)}
   = \int_0^1 e^{(1-s)K(\theta)} F_i e^{sK(\theta)} \,\mathrm{d}s,

and, after centering :math:`F_i` and normalising, the derivative of the density
matrix

.. math::

   \partial_i \rho(\theta)
   = \int_0^1 \rho(\theta)^{1-s}\,\bigl(F_i - \mu_i(\theta)\bigr)\,
     \rho(\theta)^{s}\,\mathrm{d}s

is an operator-ordered integral rather than a simple product. This is the
origin of the Kubo–Mori / BKM metric and higher cumulants: the inner products
and covariances in quantum information geometry are defined with respect to
this non-commutative kernel, not the classical pointwise product.

What Is Special About Our Geometry?
-----------------------------------

Two structural choices make the Duhamel machinery both tractable and
geometrically natural in this project:

* **Lie-closed operator bases**:
  we choose :math:`\{F_a\}` to form a Lie algebra
  :math:`[F_a, F_b] = i \sum_c f_{abc} F_c`. Then the Heisenberg-evolved
  operators :math:`e^{-sK} F_i e^{sK}` stay in the linear span of the
  :math:`F_a`, so the Duhamel kernel becomes a finite-dimensional linear
  operator :math:`K_\rho = f(\mathrm{ad}_H)` on this Lie algebra (with
  :math:`f(z) = (e^z - 1)/z`). In other words, the Duhamel integral does
  not disappear, but it is encoded as a matrix function of the adjoint
  representation rather than an intractable operator integral.

* **Categorical forcing of unitarity**:
  using the categorical framework of Parzygnat and the GENERIC-like
  decomposition, we know a priori that the entropy-conserving (antisymmetric)
  sector of the flow must be implemented by unitary conjugation, hence
  has von Neumann form :math:`\dot{\rho}_{\mathrm{rev}} = -i[H_{\mathrm{eff}},
  \rho]`. The Lie-closed exponential-family coordinates then provide a
  concrete way to express the effective Hamiltonian :math:`H_{\mathrm{eff}}`
  in terms of the antisymmetric tensor :math:`A_{ab}` and the structure
  constants :math:`f_{abc}`, with the Duhamel/BKM kernel absorbed into the
  finite-dimensional map that relates :math:`A` to the Hamiltonian
  coefficients :math:`\eta_a(\theta)`.

Compared to standard presentations of quantum information geometry—which often
start from arbitrary density matrices and modular theory—our framework keeps
the exponential-family viewpoint in the foreground. This makes the role of
natural parameters, the BKM metric, and the Duhamel kernel transparent, and it
explains why Lie-closed coordinates are particularly well adapted to the
categorical/unitary structure of the quantum inaccessible game.

See Also
--------

* :mod:`qig.exponential_family` - Implementation
* :mod:`qig.duhamel` - Duhamel implementations (quadrature and spectral/BCH)

