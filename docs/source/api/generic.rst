qig.generic
===========

GENERIC Decomposition Components
---------------------------------

This module provides functions for extracting effective Hamiltonian and diffusion
operators from the GENERIC decomposition of quantum inaccessible game dynamics.

The GENERIC (General Equation for Non-Equilibrium Reversible-Irreversible Coupling)
framework decomposes dynamics into:

- **Reversible (Hamiltonian) part**: :math:`-i[H_{\text{eff}}, \rho]`
- **Irreversible (dissipative) part**: :math:`\mathcal{D}[\rho]`

Effective Hamiltonian Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: qig.generic.effective_hamiltonian_coefficients

.. autofunction:: qig.generic.effective_hamiltonian_coefficients_lstsq

.. autofunction:: qig.generic.effective_hamiltonian_operator

.. autofunction:: qig.generic.cross_validate_hamiltonian_coefficients

Diffusion Operator Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: qig.generic.kubo_mori_derivatives

.. autofunction:: qig.generic.diffusion_operator

.. autofunction:: qig.generic.milburn_approximation

Examples
--------

Extracting Effective Hamiltonian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from qig.exponential_family import QuantumExponentialFamily
    from qig.structure_constants import compute_structure_constants
    from qig.generic import effective_hamiltonian_coefficients, effective_hamiltonian_operator
    
    # Initialize system
    exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
    theta = 0.1 * np.random.randn(exp_fam.n_params)
    
    # Compute structure constants
    f_abc = compute_structure_constants(exp_fam.operators)
    
    # Get antisymmetric part of Jacobian
    A = exp_fam.antisymmetric_part(theta)
    
    # Extract Hamiltonian coefficients
    eta, info = effective_hamiltonian_coefficients(A, theta, f_abc)
    
    # Construct Hamiltonian operator
    H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
    
    print(f"Effective Hamiltonian: {H_eff.shape}")
    print(f"Hermitian: {np.allclose(H_eff, H_eff.conj().T)}")

Computing Diffusion Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from qig.generic import diffusion_operator
    
    # Get symmetric part of Jacobian
    S = exp_fam.symmetric_part(theta)
    
    # Compute diffusion operator (expensive!)
    D_rho = diffusion_operator(S, theta, exp_fam, method='duhamel')
    
    print(f"Diffusion operator: {D_rho.shape}")
    print(f"Hermitian: {np.allclose(D_rho, D_rho.conj().T)}")
    print(f"Traceless: {abs(np.trace(D_rho)) < 1e-10}")

See Also
--------

* :mod:`qig.structure_constants` - Lie algebra structure constants
* :mod:`qig.generic_decomposition` - High-level interface
* :mod:`qig.dynamics` - Dynamics integration with GENERIC monitoring

