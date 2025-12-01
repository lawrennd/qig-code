Installation
============

Requirements
------------

**qig** requires:

* Python ≥ 3.9
* NumPy ≥ 1.21.0
* SciPy ≥ 1.7.0
* Matplotlib ≥ 3.4.0

Optional dependencies:

* nbformat ≥ 5.0.0 (for notebook tests)
* nbconvert ≥ 7.0.0 (for full notebook testing)

Install from source
-------------------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/lawrennd/qig-code.git
   cd qig-code
   pip install -e .

This will install the package in editable mode, allowing you to modify the source code.

Install dependencies
--------------------

Install the required dependencies:

.. code-block:: bash

   pip install -r requirements.txt

For development (including testing):

.. code-block:: bash

   pip install pytest

Verify installation
-------------------

Test that the package is correctly installed:

.. code-block:: python

   import qig
   from qig.exponential_family import QuantumExponentialFamily
   
   # Create a simple qutrit exponential family
   exp_fam = QuantumExponentialFamily(d=3, basis_type='gell-mann')
   print(f"Created exponential family with {exp_fam.n_params} parameters")

Run the test suite:

.. code-block:: bash

   pytest tests/

