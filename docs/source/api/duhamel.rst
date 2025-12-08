qig.duhamel
===========

Tools for evaluating the Duhamel (Wilcox) formula for quantum exponential-family
derivatives,

.. math::

   \frac{\partial \rho}{\partial \theta_a}
   = \int_0^1 e^{(1-s)H} \bigl(F_a - \langle F_a \rangle I\bigr)
     e^{sH}\,\mathrm{d}s,

where :math:`\rho = e^{H}` with
:math:`H = \sum_a \theta_a F_a - \psi(\theta) I`.

Conceptually there are two evaluation strategies:

- **Quadrature-based Duhamel** (``duhamel_derivative``,
  ``duhamel_derivative_simpson``):

  - Treat the integral over :math:`s \in [0,1]` literally and approximate it
    with trapezoid or Simpson rules.
  - Works for any Hermitian :math:`H`, with accuracy controlled by the
    number of quadrature points.
  - Used as a high-precision reference in tests and for legacy
    ``method='duhamel'`` paths.

- **Spectral/BCH Duhamel** (``duhamel_derivative_spectral``):

  - Use the eigen-decomposition :math:`H = U \operatorname{diag}(\lambda)
    U^\dagger` and evaluate the Fréchet derivative of the exponential
    in that basis.
  - In the eigenbasis of :math:`H` the derivative has closed form

    .. math::

       \bigl(D\exp_H[X]\bigr)_{ij}
       =
       \begin{cases}
         e^{\lambda_i} X_{ii}, &
         i = j,\\[4pt]
         \displaystyle
         X_{ij}\,
         \frac{e^{\lambda_i}-e^{\lambda_j}}{\lambda_i-\lambda_j}, &
         i \neq j,
       \end{cases}

    which can be written as a matrix function :math:`f(\mathrm{ad}_H)` applied
    to :math:`X = F_a - \langle F_a \rangle I`, with
    :math:`f(z) = (e^z - 1)/z`.
  - This realises the Lie-closure/BCH observation from the paper in a concrete
    numerical form: when the operator basis closes under commutation, the
    Duhamel integral becomes a finite-dimensional linear map on the Lie
    algebra, and can be evaluated analytically (up to diagonalisation error)
    without explicit :math:`s`-quadrature.

The quadrature and spectral implementations are numerically cross-validated in
the test suite (see ``TestRhoDerivativeNumerical`` in
``tests/test_pair_exponential_family.py``). For small finite-dimensional
examples the spectral method is typically preferred: it is both faster and
more accurate, and it aligns directly with the adjoint/BCH structure used in
the theory sections of the origin paper.

.. automodule:: qig.duhamel
   :members:
   :undoc-members:
   :show-inheritance:

