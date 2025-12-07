Notebook Development
====================

Working with Jupyter notebooks in the **qig** project.

Available Notebooks
-------------------

The ``examples/`` directory contains tutorial and demonstration notebooks:

**generate-origin-paper-figures.ipynb**
   Interactive demonstration notebook with validation experiments for the
   "Inaccessible Game" paper. Generates figures showing constrained dynamics,
   entropy evolution, and mutual information trajectories.
   
   `Open in Colab <https://colab.research.google.com/github/lawrennd/qig-code/blob/main/examples/generate-origin-paper-figures.ipynb>`_

**symbolic_verification_experiments.ipynb**
   Verification of key theoretical predictions including qutrit optimality,
   constraint linearization, and the structural identity ν = -1.
   
   `Open in Colab <https://colab.research.google.com/github/lawrennd/qig-code/blob/main/examples/symbolic_verification_experiments.ipynb>`_

**lme_numeric_symbolic_bridge.ipynb**
   Tutorial bridging the numeric exponential family (``QuantumExponentialFamily``)
   with the symbolic LME decomposition (``qig.symbolic.lme_exact``). Covers:
   
   - Regularized Bell state construction with ``log_epsilon``
   - Block decomposition of K(θ) in the LME basis
   - Eigenvalue structure at the LME origin
   - Scaling behaviour of natural parameters (``|θ| ~ |log ε|``)
   
   `Open in Colab <https://colab.research.google.com/github/lawrennd/qig-code/blob/main/examples/lme_numeric_symbolic_bridge.ipynb>`_

**entropy_time_analysis.ipynb**
   Analysis of entropy time reparameterisation and its relationship to Fisher
   information geometry. Covers:
   
   - Entropy evolution under different time parameterisations
   - Fisher metric tensor analysis
   - Time dilation effects near entropy extrema
   - Comparison of affine vs entropy time evolution
   
   `Open in Colab <https://colab.research.google.com/github/lawrennd/qig-code/blob/main/examples/entropy_time_analysis.ipynb>`_

**entropy_time_paths.ipynb**
   Exploration of different paths from the LME origin using entropy time. This
   notebook demonstrates that the "boring" game is an artifact of isotropic
   regularisation, not an intrinsic property of the origin. Covers:
   
   - The "north pole" analogy for the LME origin
   - Isotropic vs anisotropic regularisation (different σ choices)
   - The "almost-null" direction of the BKM metric
   - L'Hôpital-style limits in entropy time
   - How different σ choices give different limiting directions
   - Tracing trajectories backward to the origin
   - Physical interpretation: many histories sharing the same asymptotic boundary
   
   `Open in Colab <https://colab.research.google.com/github/lawrennd/qig-code/blob/main/examples/entropy_time_paths.ipynb>`_

**boring_game_dynamics.ipynb**
   Analysis of why the inaccessible game becomes "boring" from the LME origin.
   When starting from a product of Bell states with isotropic regularisation,
   the marginal entropy constraint is automatically satisfied along the entire
   gradient flow, making constrained and unconstrained dynamics identical. Covers:
   
   - Bell state construction and properties
   - Why the LME origin satisfies constraints automatically
   - When the game becomes non-trivial
   - Conditions for constraint activation
   
   `Open in Colab <https://colab.research.google.com/github/lawrennd/qig-code/blob/main/examples/boring_game_dynamics.ipynb>`_

Notebook Output Filtering
--------------------------

The project uses ``nbstripout`` to automatically clean notebook outputs before committing.

Setup:

.. code-block:: bash

   pip install nbstripout
   nbstripout --install

This prevents:

* Large binary data from bloating the repository
* Merge conflicts from execution counts
* Accidentally committing sensitive data

Notebook Testing
----------------

Notebooks can be tested in two modes:

1. **Smoke tests** (fast, ~10-20 seconds): Execute first N cells to verify imports
2. **Full execution** (complete validation): Run entire notebook

Run smoke tests:

.. code-block:: bash

   pytest -m integration tests/test_notebook.py -v

Run full execution tests:

.. code-block:: bash

   pytest -m "integration and slow" tests/test_notebook.py -v

See :doc:`testing` for details on running notebook tests.

See Also
--------

* :doc:`testing` - Testing guidelines (includes notebook testing)
* :doc:`notebook_output_filtering` - Detailed filtering documentation
* :doc:`/theory/symbolic_computation` - Symbolic computation theory

