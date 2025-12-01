Notebook Development
====================

*This section is under development.*

Working with Jupyter notebooks in the **qig** project.

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

1. **Smoke tests** (fast, ~10-20 seconds)
2. **Full execution** (complete validation)

See :doc:`testing` for details on running notebook tests.

See Also
--------

* :doc:`testing` - Testing guidelines (includes notebook testing)
* :doc:`notebook_output_filtering` - Detailed filtering documentation

