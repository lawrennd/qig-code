qig.exponential_family
======================

Quantum exponential families and Fisher information geometry.

The core class :class:`qig.exponential_family.QuantumExponentialFamily`
provides several ways to compute the density-matrix derivatives
:math:`\partial\rho / \partial\theta_a`:

- ``method='sld'``:
  symmetric logarithmic derivative approximation
  (fast, ~few-percent error in genuinely quantum, non-commuting cases).
- ``method='duhamel'``:
  high-precision Duhamel / Wilcox formula using numerical quadrature
  over :math:`s \in [0,1]` (slower, but serves as a reference).
- ``method='duhamel_spectral'`` (alias ``'duhamel_bch'``):
  uses the spectral/BCH representation of :math:`H` to evaluate the
  Duhamel integral as a matrix function :math:`f(\mathrm{ad}_H)` with
  :math:`f(z) = (e^z - 1)/z`, avoiding explicit quadrature and matching
  the Lie-closure discussion in the theory sections.

For small finite-dimensional systems (e.g. the qutrit-pair examples used in
the origin paper), the spectral/BCH variant is typically the best choice:
it is as accurate as the quadrature-based Duhamel evaluation and more
efficient, while remaining faithful to the Kubo-Mori / BKM structure.
The ``'duhamel'`` method is retained for validation and for comparison
with legacy code paths.

.. automodule:: qig.exponential_family
   :members:
   :undoc-members:
   :show-inheritance:

