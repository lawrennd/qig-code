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

For complete documentation, see:

* ``tests/tolerance_framework.py`` - Implementation
* ``docs/cip0004_precision_analysis.md`` - Mathematical derivations

See Also
--------

* :doc:`testing` - General testing guidelines

