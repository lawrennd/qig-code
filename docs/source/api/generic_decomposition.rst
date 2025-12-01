qig.generic_decomposition
=========================

High-Level GENERIC Decomposition Interface
-------------------------------------------

This module provides a user-friendly interface for executing the complete 12-step
GENERIC decomposition procedure systematically.

The :class:`GenericDecomposition` class orchestrates all steps from initial state
to final diagnostics, providing comprehensive results with validation checks.

Classes
-------

.. autoclass:: qig.generic_decomposition.GenericDecomposition
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: qig.generic_decomposition.run_generic_decomposition

The 12-Step Procedure
---------------------

The complete GENERIC decomposition executes these steps systematically:

1. **Initial State**: Density matrix :math:`\rho(\theta)` from parameters
2. **Cumulant Function**: :math:`\psi(\theta) = \log \text{Tr}[e^{\sum_a \theta_a F_a}]`
3. **Mean Parameters**: :math:`\mu = \nabla\psi(\theta)`
4. **Fisher Information**: :math:`G(\theta)` (BKM metric)
5. **Marginal Entropies**: :math:`h_i` for each subsystem
6. **Constraint Gradient**: :math:`a = \nabla C` where :math:`C = \sum_i h_i`
7. **Lagrange Multiplier**: :math:`\nu(\theta)` for constraint enforcement
8. **Lagrange Multiplier Gradient**: :math:`\nabla\nu(\theta)`
9. **Flow Jacobian**: :math:`M = \partial F/\partial\theta`
10. **GENERIC Decomposition**: :math:`M = S + A` (symmetric + antisymmetric)
11. **Effective Hamiltonian**: :math:`H_{\text{eff}} = \sum_a \eta_a F_a` from :math:`A`
12. **Diffusion Operator**: :math:`\mathcal{D}[\rho]` from :math:`S` (optional)
13. **Diagnostics**: Comprehensive validation of all properties

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from qig.exponential_family import QuantumExponentialFamily
    from qig.generic_decomposition import run_generic_decomposition
    
    # Initialize 2-qubit system
    exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
    
    # Choose state near LME origin
    theta = 0.1 * np.random.randn(exp_fam.n_params)
    
    # Run complete decomposition
    results = run_generic_decomposition(
        theta, exp_fam,
        compute_diffusion=False,  # Skip expensive D[ρ] computation
        verbose=True,              # Print progress
        print_summary=True         # Show results summary
    )
    
    # Access results
    print(f"Effective Hamiltonian: {results['H_eff']}")
    print(f"All checks passed: {results['diagnostics']['all_checks_pass']}")

Using the Class Directly
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from qig.generic_decomposition import GenericDecomposition
    
    # Create decomposition object
    decomp = GenericDecomposition(
        exp_fam,
        method='duhamel',          # High-precision derivatives
        compute_diffusion=False
    )
    
    # Execute all steps
    results = decomp.compute_all(theta, verbose=False)
    
    # Print summary
    decomp.print_summary(detailed=True)
    
    # Access specific results
    H_eff = results['H_eff']
    S = results['S']
    A = results['A']
    diagnostics = results['diagnostics']

Diagnostics and Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The diagnostics include automatic validation of all mathematical properties:

.. code-block:: python

    # Check which properties pass
    checks = results['diagnostics']['checks']
    
    for property_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {property_name}")
    
    # Get detailed error metrics
    diag = results['diagnostics']
    print(f"S symmetry error: {diag['S_symmetry_error']:.2e}")
    print(f"H_eff Hermiticity error: {diag['H_eff_hermiticity_error']:.2e}")
    print(f"Degeneracy condition (S): {diag['degeneracy_S_condition']:.2e}")

See Also
--------

* :mod:`qig.generic` - Individual GENERIC components
* :mod:`qig.dynamics.GenericDynamics` - Dynamics with GENERIC monitoring
* :doc:`../user_guide/generic_decomposition` - User guide
* :doc:`../theory/generic_structure` - Theoretical background

