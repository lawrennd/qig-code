Structure Constants Module
===========================

.. automodule:: qig.structure_constants
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

This module provides functions to compute and verify structure constants for Lie algebras,
which are fundamental for the GENERIC decomposition procedure.

For a Lie algebra with generators {F_a}, the structure constants f_abc satisfy:

.. math::

   [F_a, F_b] = 2i \sum_c f_{abc} F_c

The structure constants encode all commutation relations of the algebra and are
used to extract the effective Hamiltonian in the GENERIC decomposition.

Examples
--------

Computing Structure Constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Pauli matrices (SU(2)):

.. code-block:: python

    from qig.exponential_family import pauli_basis
    from qig.structure_constants import compute_structure_constants
    
    # Get Pauli operators for single qubit
    operators = pauli_basis(0, 1)
    
    # Compute structure constants
    f_abc = compute_structure_constants(operators)
    
    # Should be 3x3x3 for SU(2)
    print(f_abc.shape)  # (3, 3, 3)
    
    # Check specific value: f_123 = 1
    print(f_abc[0, 1, 2])  # 1.0

For Gell-Mann matrices (SU(3)):

.. code-block:: python

    from qig.exponential_family import gell_mann_matrices
    from qig.structure_constants import compute_structure_constants
    
    # Get Gell-Mann generators
    operators = gell_mann_matrices()
    
    # Compute structure constants
    f_abc = compute_structure_constants(operators)
    
    # Should be 8x8x8 for SU(3)
    print(f_abc.shape)  # (8, 8, 8)

Verification
~~~~~~~~~~~~

Verify all properties of structure constants:

.. code-block:: python

    from qig.structure_constants import (
        compute_structure_constants,
        verify_all_properties
    )
    from qig.exponential_family import pauli_basis
    
    operators = pauli_basis(0, 1)
    f_abc = compute_structure_constants(operators)
    
    # Verify antisymmetry, Jacobi identity, and commutator relations
    report = verify_all_properties(f_abc, operators, "SU(2)")
    report.print_summary()
    
    if report.all_passed():
        print("All verifications passed!")

Cross-Validation Against Reference Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare computed structure constants with reference values:

.. code-block:: python

    from qig.structure_constants import compute_structure_constants
    from qig.reference_data import get_reference_structure_constants
    from qig.exponential_family import pauli_basis
    import numpy as np
    
    # Compute structure constants
    operators = pauli_basis(0, 1)
    f_computed = compute_structure_constants(operators)
    
    # Get reference
    f_reference = get_reference_structure_constants("su2")
    
    # Compare
    max_error = np.max(np.abs(f_computed - f_reference))
    print(f"Max error: {max_error:.2e}")  # Should be < 1e-10

Caching for Performance
~~~~~~~~~~~~~~~~~~~~~~~~

Cache structure constants for reuse:

.. code-block:: python

    from qig.structure_constants import compute_and_cache_structure_constants
    from qig.exponential_family import pauli_basis
    
    operators = pauli_basis(0, 1)
    
    # First call computes and caches
    f_abc = compute_and_cache_structure_constants(operators, "my_pauli")
    
    # Second call uses cache (fast!)
    f_abc_cached = compute_and_cache_structure_constants(operators, "my_pauli")

Tensor Product Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~

For systems with multiple sites, operators on different sites commute:

.. code-block:: python

    from qig.exponential_family import pauli_basis
    from qig.structure_constants import compute_structure_constants
    import numpy as np
    
    # 2-qubit system: 3 operators per site = 6 total
    ops_site0 = pauli_basis(0, 2)
    ops_site1 = pauli_basis(1, 2)
    operators = ops_site0 + ops_site1
    
    f_abc = compute_structure_constants(operators)
    
    # Verify operators on different sites commute
    for a in range(3):  # Site 0
        for b in range(3, 6):  # Site 1
            for c in range(6):
                # f_abc should be ~0 when a,b on different sites
                assert np.abs(f_abc[a, b, c]) < 1e-10

Mathematical Properties
-----------------------

Antisymmetry
~~~~~~~~~~~~

The structure constants are antisymmetric in the first two indices:

.. math::

   f_{abc} = -f_{bac}

This follows from the antisymmetry of the commutator: [F_a, F_b] = -[F_b, F_a].

Jacobi Identity
~~~~~~~~~~~~~~~

The structure constants satisfy the Jacobi identity:

.. math::

   \sum_d (f_{abd} f_{dce} + f_{bcd} f_{dae} + f_{cad} f_{dbe}) = 0

This is equivalent to the operator Jacobi identity:

.. math::

   [F_a, [F_b, F_c]] + [F_b, [F_c, F_a]] + [F_c, [F_a, F_b]] = 0

Real Values
~~~~~~~~~~~

For Hermitian generators, the structure constants are real:

.. math::

   f_{abc} \in \mathbb{R}

See Also
--------

- :doc:`../development/validation_framework`: Validation utilities used for verification
- :doc:`../api/reference_data`: Reference structure constants for SU(2) and SU(3)
- :doc:`../api/generic`: GENERIC decomposition using structure constants

