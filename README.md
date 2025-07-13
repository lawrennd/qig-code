<a target="_blank" href="https://colab.research.google.com/github/lawrennd/the-inaccessible-game/blob/main/origin-evolution.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Information Conservation: The Inaccessible Game and Emergent Physics

## Overview

This project investigates how complex physical laws can emerge from simple information-theoretic principles. The central focus is the "inaccessible game," as developed in the draft paper (`the-inaccessible-game.tex`). The inaccessible game is a zero-player, observer-independent system whose internal state is information-isolated from any observer. Its state variables are exchangeable, and its dynamics are derived from a set of axioms that generalize and extend classical information theory.

The project explores how, starting from these axioms, one can derive:
- Entropy conservation as a fundamental constraint
- The emergence of time and dynamics via entropy production
- The quantum-classical transition as a consequence of information isolation
- The appearance of 'information atoms' (E3 units) and an exclusion principle
- The emergence of geometry and gauge symmetries reminiscent of known physics

## Background

The work started out from attempts to formalise the notion of *information topography* used extensively in *The Atomic Human*. The initial path was inspired by a long interest in Maxwell's Demon which led to recent work on information thermodynamics and information engines. A first exploration was a game known as *Jaynes' World* which instataneously maximised entropy. That honed some mathematical intuitions and the final piece of information conservation emerged as an attempt to deal with the 'cold start' problem that the maximum entropy formalism triggered, i.e. what are the parameters at the begining of the game?

The work builds on foundational ideas in information theory, particularly the axiomatic characterization of entropy by Baez, Fritz, and Leinster. The inaccessible game introduces a fourth axiom—entropic exchangeability—ensuring that the sum of marginal entropies over any finite subset of variables is constant. This leads to a non-parametric, observer-independent system with rich emergent behavior.

Earlier versions of this project explored information geometry and the Fisher information metric as central tools. The current focus, however, is on the axiomatic and dynamical structure of the inaccessible game, as developed in the draft paper. Some previous directions (e.g., exercise-based learning) have been retired in favor of this approach that emerged through working on the exercises to build mathematical understanding.

## The Inaccessible Game: Key Ideas

- *Axiomatic Foundation:*
  - Three axioms from Baez et al. (functoriality, convex linearity, continuity)
  - A new axiom: entropic exchangeability (sum of marginal entropies is constant)
- *Information Isolation:*
  - The system is information-isolated from any observer; mutual information between observer and system is zero
- *Entropy Conservation:*
  - The sum of joint entropy and multiinformation is constant
- *Emergent Dynamics:*
  - Dynamics arise from instantaneous maximization of entropy production, subject to the conservation constraint
  - The system exhibits a transition from a pure quantum state (origin) to a classical, independent state (end)
- *Quantum-Classical Transition:*
  - The transition is characterized by the emergence of maximally entangled units (E3 atoms) and an exclusion principle
  - The structure of emergent gauge symmetries (SU(3), SU(2) x U(1)) mirrors aspects of the Standard Model
- *Emergent Geometry:*
  - The system's evolution can be described in terms of symplectic geometry and Hamiltonian flows on an information manifold

For a detailed, step-by-step development of these ideas, see the main paper:

- `information-conservation/the-inaccessible-game.tex`

## Project Structure

- `the-inaccessible-game.tex`: Main draft of research paper 
- `the-inaccessible-game.bib`: Bibliography for the paper

## Research Status and Participation

This is an active research project. The current focus is on developing, clarifying, and extending the inaccessible game framework as presented in the draft paper. 
## Getting Started

- Read the draft paper (`the-inaccessible-game.tex`) for the latest and most complete exposition of the framework.

## Software Library

This project includes a comprehensive Python software library for quantum information calculations related to the inaccessible game research.

### Core Module: `inxg.py`

The `inxg.py` module provides tensor-based quantum information processing utilities specifically designed for the inaccessible game framework:

#### Key Features
- **Tensor-based quantum operations** using explicit-legs tensor layouts
- **Hybrid quantum-classical algorithms** for entropy evolution
- **Constraint-satisfying gradient flows** for information conservation
- **Robust numerical implementations** with multiple fallback strategies
- **Comprehensive type hints and documentation** for all functions

#### Main Function Categories
1. **State Creation**: Bell pair tensors and multi-qudit initial states
2. **Quantum Operations**: Partial trace, entanglement entropy calculations
3. **Entropy & Gradients**: Von Neumann entropy and gradient computations
4. **Projection & Constraints**: Gradient projection onto constraint manifolds
5. **Classical Operations**: IPF (Sinkhorn) projection and classical entropy ascent
6. **Hybrid Algorithms**: Quantum-to-classical transition workflows
7. **Utilities**: Local dephasing, density matrix normalization

#### Usage Example
```python
import inxg

# Create initial Bell pair state
rho = inxg.create_initial_state(M=2, d=3)

# Calculate entanglement entropy
S_ent = inxg.entanglement_entropy(rho, B_indices=[2, 3], total_qudits=4)

# Run hybrid quantum-classical evolution
final_rho, ent_hist, viol_hist = inxg.gradient_ascent_simulation_tensor(M=2)
```

### Development Structure

The project follows a professional development structure with:

- **CIPs (Code Improvement Plans)**: Documented development roadmap
  - [CIP-0001](./cip/cip0001.md): Core module implementation
  - [CIP-0002](./cip/cip0002.md): Comprehensive test suite
  - [CIP-0003](./cip/cip0003.md): quimb library integration
- **Backlog**: Task tracking for ongoing improvements
- **VibeSafe**: Project management and documentation standards

### Dependencies

Core dependencies include:
- `numpy`: Numerical computations
- `scipy`: Linear algebra operations
- `matplotlib`: Visualization (optional)

Future enhancements planned:
- `quimb`: Advanced tensor network operations and GPU acceleration
- `pytest`: Comprehensive testing framework

## Contributing



## Key References

- Baez, J. C., Fritz, T., & Leinster, T. (2011). A characterization of entropy in terms of information loss. *Entropy*, 13(11), 1945-1957.
- Parzygnat, A. J. (2022). A functorial characterization of von Neumann entropy. *Cahiers de Topologie et Géométrie Différentielle Catégoriques*, 63(1), 89-128.
- Lawrence, N. D. (2025). The Inaccessible Game. *draft paper* (see `the-inaccessible-game.tex`).

---

*"It is IT ..."*
