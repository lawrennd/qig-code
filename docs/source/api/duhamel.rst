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

Conceptually there are three evaluation strategies:

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

- **Block-matrix Duhamel** (``duhamel_derivative_block``):

  - Uses Higham's block-matrix identity to compute the Fréchet derivative
    via a single :math:`2n \times 2n` matrix exponential:

    .. math::

       \exp\begin{pmatrix} H & F_a - \langle F_a \rangle I \\ 0 & H \end{pmatrix}
       = \begin{pmatrix} e^H & D\exp_H[F_a - \langle F_a \rangle I] \\ 0 & e^H \end{pmatrix}

    The (1,2) block of the result is the Duhamel integral.
  - Avoids both explicit quadrature and eigendecomposition, instead "compiling
    away" the integral into the exponential itself.
  - More robust than spectral method for ill-conditioned :math:`H` where
    eigendecomposition may be unstable.
  - Cost: one call to a highly-optimized ``expm`` routine (Padé approximation
    with scaling and squaring) on a :math:`2n \times 2n` matrix.
  - Best for small to medium systems (:math:`n \leq 100`) when numerical
    robustness is important.

Method Selection
----------------

=========== ===================== ================= ====================
Method      Cost                  Accuracy          Best For
=========== ===================== ================= ====================
Quadrature  50 ``expm`` calls     ~10⁻⁵             Validation
Spectral    1 ``eigh`` + kernel   Machine precision Well-conditioned H
**Block**   1 ``expm`` (2n×2n)    Machine precision **Ill-conditioned H**
SLD         2 evaluations         ~10⁻³             Fast approximation
=========== ===================== ================= ====================

The quadrature, spectral, and block implementations are numerically cross-validated in
the test suite (see ``TestRhoDerivativeNumerical`` in
``tests/test_pair_exponential_family.py`` and ``TestBlockFrechet`` in
``tests/test_block_frechet.py``).

For small finite-dimensional examples, both the spectral and block methods achieve
machine precision. Choose based on your needs:

- **Spectral**: Faster when eigendecomposition is cheap and well-conditioned;
  aligns with the adjoint/BCH structure in theory
- **Block**: More robust for ill-conditioned problems; leverages highly-optimized
  ``expm`` without eigendecomposition

See **CIP-000A** for detailed comparison and implementation notes.

References
----------

- **Higham, N. J. (2008).** *Functions of Matrices: Theory and Computation.* SIAM.
  Chapter 10: The Fréchet Derivative.
- **Al-Mohy, A. H., & Higham, N. J. (2009).** Computing the Fréchet derivative of
  the matrix exponential, with an application to condition number estimation.
  *SIAM J. Matrix Anal. Appl.*, 30(4), 1639–1657.

.. automodule:: qig.duhamel
   :members:
   :undoc-members:
   :show-inheritance:

