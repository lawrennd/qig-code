Testing Documentation
=====================

This document describes the testing infrastructure for the QIG codebase.

Overview
--------

The QIG project has multiple testing layers:

1. **Unit Tests**: ``tests/test_*.py`` - Core functionality tests
2. **Notebook Tests**: ``tests/test_notebook.py`` - Jupyter notebook validation
3. **Integration Tests**: Various validation scripts
4. **GitHub Actions CI/CD**: Automated testing workflows

Test Organization
-----------------

Test Markers
~~~~~~~~~~~~

The test suite uses pytest markers to organize different types of tests:

* **integration**: Integration tests (notebooks, end-to-end scenarios)

  * **Excluded by default** to keep regular test runs fast
  * Run with: ``pytest -m integration``
  * Example: Notebook execution tests

* **slow**: Performance and validation tests that take longer

  * **Included by default** (these are important for correctness)
  * Skip with: ``pytest -m "not slow"``
  * Example: Full validation tests, performance benchmarks

Running Different Test Subsets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Default: All tests except integration (includes slow tests)
   pytest                              # Most tests
   
   # Only integration tests (notebooks, etc.)
   pytest -m integration
   
   # Only slow tests (performance, validation)
   pytest -m slow
   
   # Exclude both slow and integration
   pytest -m "not slow and not integration"
   
   # Only fast unit tests
   pytest -m "not slow and not integration"

Tolerance Framework (CIP-0004)
-------------------------------

The test suite uses scientifically-derived tolerance categories for numerical validation. 
See :doc:`testing_tolerances` for detailed information.

Running Tests
-------------

.. code-block:: bash

   # All tests
   pytest tests/ -v
   
   # Specific test file
   pytest tests/test_pair_exponential_family.py -v
   
   # Specific test
   pytest tests/test_exponential_family.py::TestQuantumExponentialFamily::test_rho_is_hermitian -v

Notebook Testing
----------------

Available Notebooks
~~~~~~~~~~~~~~~~~~~

* ``examples/generate-origin-paper-figures.ipynb`` - Paper figure generation and validation

Running Notebook Tests
~~~~~~~~~~~~~~~~~~~~~~

**Note:** Notebook tests are marked as "integration" and **excluded from default test runs**.

There are two types of notebook tests:

1. **Smoke tests** (``@pytest.mark.integration``):

   * Execute first 8 cells only (imports, setup)
   * Fast: ~10-20 seconds
   * Catches 90% of issues

2. **Full execution** (``@pytest.mark.integration @pytest.mark.slow``):

   * Execute complete notebook
   * Slow: several minutes
   * Full validation

Via pytest (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Run integration tests (smoke tests only, fast)
   pytest -m integration
   
   # Run full notebook execution (slow)
   pytest -m "integration and slow"
   
   # Run specific smoke test
   pytest tests/test_notebook.py::test_notebook_smoke -v

Test Structure
--------------

The test suite is organized to mirror the qig module structure:

Core Library Tests
~~~~~~~~~~~~~~~~~~

* ``test_core_utilities.py`` - State utilities, operator bases, GENERIC
* ``test_exponential_family.py`` - Basic exponential family operations
* ``test_pair_exponential_family.py`` - Pair basis and entanglement

Derivative Tests
~~~~~~~~~~~~~~~~

* ``test_fisher_metric.py`` - Fisher information (BKM) for all operator types
* ``test_constraint_derivatives.py`` - Constraint gradient, Hessian, Lagrange multiplier
* ``test_higher_derivatives.py`` - Jacobian and third cumulant

Dynamics & Integration
~~~~~~~~~~~~~~~~~~~~~~

* ``test_dynamics.py`` - Constrained dynamics, GENERIC decomposition, integration

Performance & Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``test_theta_only_constraint.py`` - θ-only optimization method

Diagnostic & Utility
~~~~~~~~~~~~~~~~~~~~

* ``test_notebook.py`` - Notebook validation tests

For more details on test organization and the rationale behind the structure, see CIP-0004.

See Also
--------

* :doc:`testing_tolerances` - Numerical tolerance framework
* :doc:`contributing` - Contributing guidelines
* :doc:`notebooks` - Notebook development guide

