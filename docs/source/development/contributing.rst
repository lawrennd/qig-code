Contributing Guidelines
=======================

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

Getting Started
---------------

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

   .. code-block:: bash

      git clone https://github.com/lawrennd/qig-code.git
      cd qig-code

3. **Create a virtual environment** (recommended):

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

4. **Install in development mode**:

   .. code-block:: bash

      pip install -r requirements.txt

Development Workflow
--------------------

1. Create a Feature Branch
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git checkout -b feature/your-feature-name

Use descriptive branch names:

* ``feature/new-analysis`` for new features
* ``bugfix/issue-123`` for bug fixes
* ``docs/improve-readme`` for documentation
* ``test/add-coverage`` for test improvements
* ``cip/0003-description`` for new CIPs

2. Make Your Changes
~~~~~~~~~~~~~~~~~~~~

* Write clean, readable code following PEP 8 style guidelines
* Add docstrings for all functions, classes, and modules
* Include type hints where appropriate
* Keep functions focused and modular

3. Add Tests
~~~~~~~~~~~~

All new functionality must include tests:

.. code-block:: python

   # tests/test_your_module.py
   import pytest
   import numpy as np
   from tests.tolerance_framework import quantum_assert_close
   
   class TestYourFeature:
       def test_basic_functionality(self):
           result = your_function(input_data)
           assert result == expected_output
       
       def test_numerical_precision(self):
           # Test with known values
           quantum_assert_close(computed, expected, 'quantum_state')

Run tests locally:

.. code-block:: bash

   pytest tests/ -v

Check coverage:

.. code-block:: bash

   pytest tests/ --cov=qig --cov-report=term

4. Working with Notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~

If you're contributing example notebooks or modifying existing ones:

**First-time setup**: Install and configure ``nbstripout`` to automatically clean notebook outputs:

.. code-block:: bash

   pip install nbstripout  # Already in requirements.txt
   nbstripout --install    # Install git filter

This creates a git filter that automatically strips outputs, execution counts, and metadata from notebooks when you commit.

**Manual stripping**: To manually strip outputs:

.. code-block:: bash

   nbstripout *.ipynb

**Testing notebooks**: Before submitting, ensure your notebook runs cleanly:

.. code-block:: bash

   python test_notebook.py

5. Update Documentation
~~~~~~~~~~~~~~~~~~~~~~~

* Update README.md if adding new features
* Add or update CIPs for architectural changes
* Update documentation for new test procedures
* Add docstrings following NumPy style:

.. code-block:: python

   def compute_fisher_information(theta, operators):
       """
       Compute quantum Fisher information (BKM metric).
       
       Parameters
       ----------
       theta : np.ndarray, shape (n_params,)
           Natural parameters of exponential family
       operators : list of np.ndarray
           Hermitian operators forming the basis
       
       Returns
       -------
       G : np.ndarray, shape (n_params, n_params)
           Fisher information matrix (symmetric, positive semidefinite)
       
       Examples
       --------
       >>> from qig.exponential_family import QuantumExponentialFamily
       >>> exp_fam = QuantumExponentialFamily(d=3)
       >>> theta = np.random.randn(exp_fam.n_params) * 0.1
       >>> G = exp_fam.fisher_information(theta)
       >>> assert G.shape == (exp_fam.n_params, exp_fam.n_params)
       """

6. Commit Your Changes
~~~~~~~~~~~~~~~~~~~~~~

Write clear, descriptive commit messages:

.. code-block:: bash

   git add specific-files.py  # Use surgical adds
   git commit -m "Add feature: brief description
   
   Detailed explanation of:
   - What changed
   - Why it changed  
   - Any breaking changes or migration notes"

**Important**: Follow VibeSafe guidelines:

* ✅ Use surgical ``git add`` for specific files
* ❌ Never use ``git add .`` or ``git add -A``
* ✅ Commit regularly (after planning, before refactoring, after implementation)
* ✅ Reference CIPs/backlog items when relevant

7. Push and Create Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git push origin feature/your-feature-name

Then create a Pull Request on GitHub with:

* Clear title describing the change
* Description of what the PR does
* Reference to any related CIPs or backlog items
* Test results

Project Management (VibeSafe)
------------------------------

This project uses `VibeSafe <https://github.com/lawrennd/vibesafe>`_ for structured project management.

Code Improvement Plans (CIPs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For major architectural changes:

1. Create ``cip/cipXXXX.md`` using the template
2. Document motivation, implementation plan, and status
3. Reference in commits and PRs
4. See CIP-0002 for a complete example

Backlog Tasks
~~~~~~~~~~~~~

For features and tasks:

1. Create ``backlog/category/YYYY-MM-DD_description.md``
2. Use template from ``backlog/task_template.md``
3. Track status (Proposed → Ready → In Progress → Completed)
4. Update index with ``python backlog/update_index.py``

Checking Project Status
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ./whats-next  # Shows CIPs, backlog, git status, next steps

Code Style Guidelines
---------------------

Python Code
~~~~~~~~~~~

* Follow PEP 8 style guide
* Maximum line length: 100 characters
* Use meaningful variable names
* Prefer explicit over implicit

Example:

.. code-block:: python

   # Good
   def compute_mutual_information(rho: np.ndarray, dims: list) -> float:
       """Compute mutual information I = Σh_i - H."""
       # Implementation
       
   # Avoid
   def calc_I(r, d):  # Unclear abbreviations
       # Implementation

Quantum-Specific Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Verify Hermiticity of density matrices and operators
* Check positivity and normalization (Tr(ρ) = 1)
* Use high-precision methods for quantum derivatives (Duhamel formula)
* Test commutation relations for operator bases
* Validate physical bounds (0 ≤ entropy ≤ log(d))

Testing Guidelines
------------------

Test Categories
~~~~~~~~~~~~~~~

1. **Unit tests**: Test individual functions
2. **Integration tests**: Test module interactions
3. **Numerical validation**: Compare analytic vs finite differences
4. **Physical properties**: Test quantum constraints

Test Naming
~~~~~~~~~~~

.. code-block:: python

   def test_fisher_information_is_hermitian():
       """BKM metric should be Hermitian (symmetric for real)."""
       
   def test_density_matrix_is_normalized():
       """Density matrix trace should equal 1."""

Pull Request Checklist
----------------------

Before submitting your PR, ensure:

* Code follows style guidelines
* All pytest tests pass (``pytest tests/ -v``)
* New functionality includes tests
* Documentation is updated
* Commit messages are clear and reference CIPs/backlog
* No merge conflicts with main branch
* Notebooks have outputs stripped (nbstripout)
* Quantum properties validated

Types of Contributions
----------------------

Bug Reports
~~~~~~~~~~~

When reporting bugs, include:

* Python version
* NumPy/SciPy versions
* Minimal reproducible example
* Expected vs actual behavior
* Error traceback

Feature Requests
~~~~~~~~~~~~~~~~

For new features, describe:

* The problem it solves
* Proposed implementation
* Example usage
* Why it fits the project scope

Code Contributions
~~~~~~~~~~~~~~~~~~

Areas where contributions are especially welcome:

* Additional test coverage
* Numerical precision improvements
* Performance optimizations
* Documentation improvements
* Example notebooks

Questions?
----------

* Open an issue for general questions
* Reference :doc:`testing` for testing details
* See the README for project overview

License
-------

This project is licensed under the MIT License.

Copyright (c) 2025 Neil D. Lawrence

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

