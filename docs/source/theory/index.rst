Theory
======

Mathematical background for quantum information geometry and the inaccessible game.

.. toctree::
   :maxdepth: 2
   
   matrix_exponential_families
   fisher_information
   inaccessible_game
   origin_regularisation
   generic_structure
   hamiltonian_extraction
   symbolic_computation

Matrix Exponential Families
-----------------------------

*Coming soon: Mathematical theory of Matrix Exponential Families*

Topics:
- Definition and properties
- Log-partition function :math:`\psi(\theta)`
- Expectation parameters and duality
- Quantum Fisher information

Fisher Information and the BKM Metric
--------------------------------------

*Coming soon: The Bogoliubov-Kubo-Mori metric*

Topics:
- Definition via perturbation theory
- Kubo-Mori inner product
- Relationship to quantum Fisher information
- Geometric properties

The Inaccessible Game Framework
--------------------------------

*Coming soon: Information-geometric perspective on constrained dynamics*

Topics:
- Marginal entropy constraints
- The inaccessible region
- Information gain vs accessibility
- Relationship to thermodynamics

Origin Regularisation
---------------------

The LME origin (product of Bell states) lies at the pure-state boundary
where natural parameters θ → -∞. The regularisation matrix σ encodes the
"direction of approach" to this boundary. Key topics:

- The north pole analogy (coordinate singularity at pure states)
- Valid σ requirements (Hermitian, PSD, unit trace)
- Efficiency implications (isotropic vs product vs general)
- Block-diagonal Fisher information for product states

See :doc:`origin_regularisation` for full details.

GENERIC Structure
-----------------

Decomposition of dynamics into reversible and irreversible parts:
:math:`M = S + A`. Covers Lie closure cancellation, when Duhamel integrals
are required, and the practical consequences for qig computations.

See :doc:`generic_structure` for full details.

Hamiltonian Extraction
-----------------------

How the antisymmetric sector :math:`A` of the GENERIC Jacobian is mapped to
an explicit effective Hamiltonian :math:`H_\text{eff}(\theta) = \sum_c \eta_c F_c`
via the Lie structure constants :math:`f_{abc}`. Explains the extraction
algorithm, the Kubo-Mori kernel subtlety, and validated properties.

See :doc:`hamiltonian_extraction` for full details.

