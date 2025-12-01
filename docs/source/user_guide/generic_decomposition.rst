GENERIC Decomposition
=====================

The GENERIC (General Equation for Non-Equilibrium Reversible-Irreversible Coupling)
decomposition separates quantum dynamics into reversible (Hamiltonian) and
irreversible (dissipative) components.

For the quantum inaccessible game, this decomposition reveals the structure of
constrained entropy-maximizing dynamics on the marginal-entropy manifold.

Overview
--------

The flow Jacobian :math:`M = \partial F/\partial\theta` naturally decomposes as:

.. math::

   M = S + A

where:

* :math:`S = \frac{1}{2}(M + M^T)` is the **symmetric part** (dissipative/irreversible)
* :math:`A = \frac{1}{2}(M - M^T)` is the **antisymmetric part** (reversible/Hamiltonian)

From this decomposition, we can extract:

1. **Effective Hamiltonian** :math:`H_{\text{eff}}` from :math:`A`
2. **Diffusion operator** :math:`\mathcal{D}[\rho]` from :math:`S`

The full dynamics then take the form:

.. math::

   \dot{\rho} = -i[H_{\text{eff}}, \rho] + \mathcal{D}[\rho]

Quick Start
-----------

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from qig.exponential_family import QuantumExponentialFamily
    from qig.generic_decomposition import run_generic_decomposition
    
    # Initialize 2-qubit entangled pair
    exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
    
    # Choose a state (near LME origin)
    theta = 0.1 * np.random.randn(exp_fam.n_params)
    
    # Run complete GENERIC decomposition
    results = run_generic_decomposition(
        theta, exp_fam,
        verbose=True,
        print_summary=True
    )

This executes all 12 steps and provides comprehensive diagnostics.

Understanding the Results
~~~~~~~~~~~~~~~~~~~~~~~~~

The results dictionary contains:

.. code-block:: python

    # Information geometry
    results['psi']      # Cumulant generating function ψ(θ)
    results['mu']       # Mean parameters ∇ψ
    results['G']        # Fisher information (BKM metric)
    results['a']        # Constraint gradient
    results['nu']       # Lagrange multiplier
    
    # GENERIC decomposition
    results['M']        # Full Jacobian
    results['S']        # Symmetric part (dissipative)
    results['A']        # Antisymmetric part (reversible)
    
    # Physical operators
    results['H_eff']    # Effective Hamiltonian
    results['eta']      # Hamiltonian coefficients
    results['D_rho']    # Diffusion operator (if computed)
    
    # Diagnostics
    results['diagnostics']  # Validation checks

Step-by-Step Guide
------------------

Step 1: Initialize Exponential Family
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose your system:

.. code-block:: python

    # For qubits (d=2)
    exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
    
    # For qutrits (d=3)
    exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
    
    # For multiple pairs
    exp_fam = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)

Step 2: Choose Initial State
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Select natural parameters :math:`\theta`:

.. code-block:: python

    # At LME origin (maximally entangled)
    theta = np.zeros(exp_fam.n_params)
    
    # Near origin (slightly entangled)
    theta = 0.1 * np.random.randn(exp_fam.n_params)
    
    # Random state
    theta = np.random.randn(exp_fam.n_params)

Step 3: Run Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the convenience function:

.. code-block:: python

    results = run_generic_decomposition(
        theta, exp_fam,
        method='duhamel',          # or 'sld' for faster approximation
        compute_diffusion=False,   # Set True to compute D[ρ] (expensive!)
        verbose=True,              # Print progress
        print_summary=True         # Show summary at end
    )

Or use the class for more control:

.. code-block:: python

    from qig.generic_decomposition import GenericDecomposition
    
    decomp = GenericDecomposition(exp_fam, compute_diffusion=False)
    results = decomp.compute_all(theta, verbose=True)
    decomp.print_summary(detailed=True)

Step 4: Extract Key Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Access the decomposition:

.. code-block:: python

    # Jacobian decomposition
    M = results['M']  # Full Jacobian
    S = results['S']  # Symmetric (dissipative)
    A = results['A']  # Antisymmetric (reversible)
    
    # Verify: M = S + A
    assert np.allclose(M, S + A)
    
    # Effective Hamiltonian
    H_eff = results['H_eff']
    eta = results['eta']  # Coefficients
    
    # Check Hermiticity
    assert np.allclose(H_eff, H_eff.conj().T)

Step 5: Validate Results
~~~~~~~~~~~~~~~~~~~~~~~~~

Check diagnostics:

.. code-block:: python

    diag = results['diagnostics']
    
    # Overall pass/fail
    if diag['all_checks_pass']:
        print("✓ All validation checks passed!")
    else:
        print("✗ Some checks failed:")
        for name, passed in diag['checks'].items():
            if not passed:
                print(f"  - {name}")
    
    # Detailed error metrics
    print(f"S symmetry error: {diag['S_symmetry_error']:.2e}")
    print(f"H_eff Hermiticity: {diag['H_eff_hermiticity_error']:.2e}")

Advanced Usage
--------------

Computing the Diffusion Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The diffusion operator :math:`\mathcal{D}[\rho]` maps the symmetric flow to
density matrix space. **Warning**: This is computationally expensive as it
requires Kubo-Mori derivatives.

.. code-block:: python

    results = run_generic_decomposition(
        theta, exp_fam,
        compute_diffusion=True,  # Enable D[ρ] computation
        verbose=False
    )
    
    D_rho = results['D_rho']
    
    # Properties of D[ρ]
    assert np.allclose(D_rho, D_rho.conj().T)  # Hermitian
    assert abs(np.trace(D_rho)) < 1e-10         # Traceless

Integrating GENERIC Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`qig.dynamics.GenericDynamics` to track GENERIC structure along trajectories:

.. code-block:: python

    from qig.dynamics import GenericDynamics
    
    dyn = GenericDynamics(exp_fam)
    
    # Integrate with monitoring
    result = dyn.integrate_with_monitoring(
        theta_0, (0.0, 1.0), n_points=50,
        compute_diffusion=False
    )
    
    # Access GENERIC structure along trajectory
    H_eff_traj = result['H_eff']           # Hamiltonians
    entropy_prod = result['entropy_production']  # dS/dt
    S_norms = result['S_norm']             # ||S|| over time
    A_norms = result['A_norm']             # ||A|| over time

Separating Reversible and Irreversible Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Integrate only one component:

.. code-block:: python

    # Reversible (Hamiltonian) only
    rev_result = dyn.integrate_reversible(theta_0, (0.0, 1.0))
    
    # Irreversible (dissipative) only
    irr_result = dyn.integrate_irreversible(theta_0, (0.0, 1.0))
    
    # Compare with full dynamics
    full_result = dyn.integrate(theta_0, (0.0, 1.0))

Performance Considerations
--------------------------

Computational Cost
~~~~~~~~~~~~~~~~~~

The main computational bottlenecks are:

1. **Fisher information** :math:`G(\theta)` - :math:`O(n^2 D^2)`
2. **Jacobian** :math:`M` - :math:`O(n^3)` for third cumulants
3. **Diffusion operator** :math:`\mathcal{D}[\rho]` - :math:`O(n D^4)` (very expensive!)

For large systems:

* Skip diffusion operator computation (``compute_diffusion=False``)
* Use ``method='sld'`` for faster (but less accurate) derivatives
* Cache structure constants (computed once per basis)

Accuracy vs Speed
~~~~~~~~~~~~~~~~~

Two methods for derivatives:

.. code-block:: python

    # High accuracy (~10⁻⁶ error), slow
    results = run_generic_decomposition(theta, exp_fam, method='duhamel')
    
    # Fast (~5% error), much faster
    results = run_generic_decomposition(theta, exp_fam, method='sld')

Choose based on your needs:

* Use ``'duhamel'`` for publication-quality results
* Use ``'sld'`` for exploration and prototyping

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Degeneracy conditions fail**

The conditions :math:`S \cdot a \approx 0` and :math:`A \cdot (-G\theta) \approx 0`
may not hold exactly far from equilibrium. This is expected and doesn't indicate
an error in the computation.

**Large constraint gradient norm**

If :math:`\|a\|` is very large, you may be far from the constraint manifold.
Project back onto the constraint or reduce parameter magnitudes.

**Numerical instabilities**

Near pure states or boundaries, numerical precision may degrade. Use:

* Smaller parameter values
* Higher precision (``method='duhamel'``)
* Regularization in structure constant computation

Interpreting Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~

**Always passing** (algebraic properties):

* ``S_symmetric``: :math:`S = S^T`
* ``A_antisymmetric``: :math:`A = -A^T`
* ``M_reconstructs``: :math:`M = S + A`
* ``H_eff_hermitian``: :math:`H_{\text{eff}} = H_{\text{eff}}^\dagger`
* ``H_eff_traceless``: :math:`\text{Tr}(H_{\text{eff}}) = 0`

**May fail** (physical conditions):

* ``degeneracy_S``: :math:`\|S \cdot a\| < \epsilon` (depends on state)
* ``degeneracy_A``: :math:`\|A \cdot (-G\theta)\| < \epsilon` (depends on state)
* ``tangency``: Flow tangent to constraint (numerical precision)

Examples
--------

See the ``examples/`` directory for complete examples:

* ``generic_decomposition_demo.py`` - Basic decomposition
* ``generic_decomposition_complete.py`` - Full workflow with visualizations

See Also
--------

* :doc:`../theory/generic_structure` - Theoretical background
* :doc:`../api/generic` - Low-level API
* :doc:`../api/generic_decomposition` - High-level API
* :doc:`dynamics` - Dynamics integration

