Origin Regularisation
=====================

This document explains the regularisation of pure states in the quantum
exponential family, focusing on the physical meaning of the regularisation
matrix σ and its computational implications.

The North Pole Analogy
----------------------

The Local Maximum Entropy (LME) origin, a product of Bell states, behaves like
a coordinate singularity at the north pole of a sphere.:

- **Many meridians, one pole**: Just as infinitely many lines of longitude
  converge at the north pole, infinitely many distinct trajectories through
  state space converge at the LME origin.

- **Different histories**: Each trajectory represents a different "direction
  of approach" to the pure state boundary. The regularisation matrix σ
  encodes which direction we came from.

- **Coordinate singularity**: At the pole, longitude becomes undefined.
  Similarly, at the LME origin, the natural parameters θ → -∞ and the
  direction of departure becomes ambiguous without regularisation.

This is why we write:

.. math::

    \rho_\varepsilon = (1-\varepsilon)|\Psi\rangle\langle\Psi| + \varepsilon \sigma

The matrix σ specifies which "meridian" we're on—different σ give different
limiting directions as ε → 0.


Valid Regularisation Matrices
-----------------------------

For σ to define a valid regularisation direction, it must be a density matrix.

1. **Hermitian**: σ = σ†
2. **Positive semidefinite**: All eigenvalues ≥ 0
3. **Unit trace**: Tr(σ) = 1

The code provides validation:

.. code-block:: python

    from qig.exponential_family import QuantumExponentialFamily
    
    qef = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
    
    # Check if σ is valid
    is_valid, message = qef.validate_sigma(sigma)
    if not is_valid:
        raise ValueError(f"Invalid σ: {message}")

Structure Detection
^^^^^^^^^^^^^^^^^^^

The code automatically detects the structure of σ:

.. code-block:: python

    structure = qef.detect_sigma_structure(sigma)
    # Returns: 'isotropic', 'product', 'pure', or 'general'


Efficiency Implications
-----------------------

The choice of σ has significant computational implications:

+----------------+---------------------------+--------------------------+-------------+
| σ type         | Eigenstructure            | Fisher metric            | Complexity  |
+================+===========================+==========================+=============+
| I/D (isotropic)| Trivial (2 eigenvalues)   | Block-diagonal           | **O(n)**    |
+----------------+---------------------------+--------------------------+-------------+
| ⊗ᵢ σᵢ (product)| Per-pair analytic         | Block-diagonal           | **O(n·d⁶)** |
+----------------+---------------------------+--------------------------+-------------+
| General        | Full eigendecomposition   | Full computation         | O(D³)       |
+----------------+---------------------------+--------------------------+-------------+

Where:
- n = number of pairs
- d = local dimension
- D = d^(2n) = total Hilbert space dimension

For n=3 qutrit pairs: D=729, so O(D³) ≈ 387 million operations, while
O(n·d⁶) ≈ 2000 operations—a **200,000× speedup**.


Isotropic Regularisation (σ = I/D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest choice, giving maximally symmetric departure:

.. code-block:: python

    # Default: isotropic regularisation
    theta = qef.get_bell_state_parameters(epsilon=1e-6)

**Properties**:

- Fastest computation (analytic formulas)
- Symmetric departure from origin
- Often "boring" dynamics (see ``boring_game_dynamics.ipynb``)
- Block-diagonal Fisher information


Product Regularisation (σ = σ₁⊗...⊗σₙ)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For independent per-pair perturbations:

.. code-block:: python

    # Per-pair regularisation (efficient)
    sigma_per_pair = [sigma_1, sigma_2, sigma_3]  # Each d²×d²
    theta = qef.get_bell_state_parameters(
        epsilon=1e-6,
        sigma_per_pair=sigma_per_pair
    )

**Properties**:

- O(n·d⁶) complexity (efficient)
- Pairs depart independently
- Correlations emerge through constraint dynamics
- Block-diagonal Fisher information


General Regularisation
^^^^^^^^^^^^^^^^^^^^^^

For arbitrary σ (including entangled):

.. code-block:: python

    # General σ (may be expensive)
    theta = qef.get_bell_state_parameters(
        epsilon=1e-6,
        sigma=sigma_full  # D×D matrix
    )

**Properties**:

- O(D³) complexity (expensive for large n)
- Can encode pre-existing inter-pair correlations
- Full Fisher information computation required

**Warning**: For n≥3 pairs, this becomes impractical. Use ``sigma_per_pair``
for efficient computation when possible.


Physics vs Efficiency Trade-off
-------------------------------

The efficiency requirements impose a **physics assumption**:

+-------------------+-------------------------------+-------------------------+
| Assumption        | Physical meaning              | Computation             |
+===================+===============================+=========================+
| Product σ         | Pairs depart independently    | Efficient O(n)          |
+-------------------+-------------------------------+-------------------------+
| Entangled σ       | Departure couples pairs       | Expensive O(D³)         |
+-------------------+-------------------------------+-------------------------+

**When to use product σ** (efficient):

- Studying emergence of correlations from constraint dynamics
- Pairs have independent local noise/decoherence
- Computational tractability needed

**When to use general σ** (expensive):

- Pre-existing inter-pair coupling in perturbation
- Correlated noise scenarios
- Small systems where O(D³) is acceptable


Block-Diagonal Fisher Information
---------------------------------

For product states, the BKM Fisher metric is block-diagonal:

.. math::

    G = \text{diag}(G_1, G_2, \ldots, G_n)

where each Gₖ is the (d⁴-1)×(d⁴-1) metric for pair k.

Use the efficient computation:

.. code-block:: python

    # Efficient block-diagonal computation
    G = qef.fisher_information_product(theta)
    
    # Compare with full computation (should match for product states)
    G_full = qef.fisher_information(theta)
    assert np.allclose(G, G_full)

Performance comparison (d=2 qubits):

+--------+------+---------+----------+---------+
| n_pairs| D    | Full    | Block    | Speedup |
+========+======+=========+==========+=========+
| 2      | 16   | 5ms     | 10ms     | 0.5×    |
+--------+------+---------+----------+---------+
| 3      | 64   | 575ms   | 168ms    | **3.4×**|
+--------+------+---------+----------+---------+
| 4      | 256  | minutes | 575ms    | **~100×**|
+--------+------+---------+----------+---------+


Different Origins: bell_indices
-------------------------------

The standard LME origin uses |Φ₀⟩⊗|Φ₀⟩⊗... where |Φ₀⟩ = Σⱼ|jj⟩/√d.
But there are d different Bell states per pair:

.. math::

    |\Phi_k\rangle = \frac{1}{\sqrt{d}} \sum_{j=0}^{d-1} |j, (j+k) \mod d\rangle

All share the same properties:

- Maximally entangled
- Marginals = I/d
- Constraint C = 2n·log(d)

Use ``bell_indices`` to select different origins:

.. code-block:: python

    from qig.pair_operators import product_of_bell_states
    
    # Standard origin: |Φ₀⟩⊗|Φ₀⟩
    psi = product_of_bell_states(n_pairs=2, d=2)
    
    # Alternative origin: |Φ₀⟩⊗|Φ₁⟩
    psi = product_of_bell_states(n_pairs=2, d=2, bell_indices=[0, 1])

These represent different "starting points" for the inaccessible game,
all at the same constraint value but with different local structures.


Further Reading
---------------

- **entropy_time_paths.ipynb**: Detailed exploration of different σ and
  the L'Hôpital-style limits that resolve the coordinate singularity
  
- **boring_game_dynamics.ipynb**: Analysis of why isotropic σ gives
  "boring" dynamics where constrained and unconstrained flows coincide

- **CIP-0008**: Implementation details for efficient multi-pair machinery
