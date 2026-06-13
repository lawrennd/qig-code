Testing Tolerance Framework
============================

*This section is under development. Content will be migrated from CIP-0004 documentation.*

The **qig** test suite uses scientifically-derived tolerance categories for numerical validation.

Tolerance Categories
--------------------

**Category A: Machine Precision Operations** (≤ 1e-14)
   Pure algebraic operations with minimal error accumulation

**Category B: Quantum State Properties** (≤ 1e-12)
   Fundamental quantum constraints (unit trace, hermiticity)

**Category C: Entanglement & Information Metrics** (≤ 1e-10)
   Information-theoretic quantities sensitive to eigenvalue ratios

**Category D: Analytical Derivatives** (≤ 1e-8)
   Error propagation in quantum Fisher information metric

**Category E: Numerical Integration** (≤ 1e-6)
   ODE solver convergence and long-time stability

**Category F: Physical Validation** (≤ 1e-4)
   Statistical significance for physical claims

**Category G: BCH Approximation Residuals** (~0.1 · ‖θ‖²)
   The Hamiltonian extraction formula
   :math:`F\,\eta \approx A\,\theta`
   (where :math:`F_{rc} = \sum_b f_{rbc}\,\theta_b`) is only leading-order
   correct.  The Kubo-Mori kernel embedded in :math:`A` contributes
   :math:`O(\|\theta\|^2)` corrections outside :math:`\operatorname{col}(F)`.
   For :math:`\|\theta\|\sim 0.05` the irreducible residual is
   :math:`\sim 10^{-3}`.  Tests that verify this residual use the bound
   :math:`0.5\,\|\theta\|^2` rather than a fixed numerical tolerance.
   See ``tests/test_generic_hamiltonian.py::test_extraction_consistency_multiple_points``
   and CIP-0009 (*Correction note: extraction formula accuracy*).

For complete documentation, see:

* ``tests/tolerance_framework.py`` - Implementation
* ``docs/cip0004_precision_analysis.md`` - Mathematical derivations

See Also
--------

* :doc:`testing` - General testing guidelines

