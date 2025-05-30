# Information Conservation: A Simple Game with Complex Consequences

## Overview

This framework presents an approach to understanding physical laws through the lens of *information theory* and *information geometry*. 

By starting with a simple rule around conservation of information we aim to derive complex physics. 

In the emerging framework the Fisher information matrix takes a key role. We think of it as representing the *information topography*.

The framework is speculative and not full mathematically formalised.

## Background

The work started out from attempts to formalise the notion of *information topography* used extensively in *The Atomic Human*. The initial path was inspired by a long interest in Maxwell's Demon which led to recent work on information thermodynamics and information engines. A first exploration was a game known as *Jaynes' World* which instataneously maximised entropy. That honed some mathematical intuitions and the final piece of information conservation emerged as an attempt to deal with the 'cold start' problem that the maximum entropy formalism triggered, i.e. what are the parameters at the begining of the game?

The information conservation principle is the only rule, but it leads to a set of `effective physics rules' where the *information topography*, as represented by the Fisher information geometry, is one of the primary components in the emergent physics of the game. 

By founding the game on the principles of information, the hope is that the realtionship between the effective physics and more complex emergent phenomena in the game can be interrelated through the common language of information that *The Atomic Human* also builds on.

## Details of The Information Game

The framework centers on an "Information Game" with N variables Z, partitioned into:
- *Latent variables (M)*: Each has marginal entropy S(m_i) = log 2
- *Active variables (X)*: Have Shannon entropy S(x_i) = -∑ p_j log p_j

The fundamental constraint is *information conservation*:
```
∑ S(z_i) = N log 2 = constant
```

This leads to the core conservation law:
```
S(Z) + I(Z) = constant
```
where S(Z) is joint entropy and I(Z) is multi-information.

## Key Insights from the Game

### 1. Information Conservation Constraints Lead Directly to a Quantum Formalism

We demonstrate that quantum mechanics  arises mathematically when classical probability theory encounters fundamental information limitations.

- *Complex amplitudes* emerge from Maximum Entropy principles under information constraints
- *Density matrices* arise as optimal representations of partial information
- *The Born rule* emerges as inevitable under maximum entropy when variables transit from undistinguishable to distinguishable 

### 2. Fisher Information as Universal Substrate

The Fisher information metric provides a unified mathematical language that underpins the framework when leading to the natural emergence of guage symmetries when the continuous information geometry is applied to modelling an intrinsically discrete set of variables.

### 3. Emergence of Space, Time, and Physical Laws

The framework shows how fundamental physics emerges from information optimization:

- *Space and time* emerge from distinguishability requirements and Fisher information geometry
- *Energy conservation* emerges from information conservation through geometric time coordinate transforms
- *Gauge symmetries* arise naturally from the discrete-continuous interface
- *Classical independence* emerges from quantum entanglement via de Finetti's theorem

## Framework Structure

### Educational Materials
- *information-conservation.tex*: Main educational framework with exercises
- *information-conservation-solutions.tex*: Complete solutions and derivations
- *quantum-states-demystified.md*: Short primer on quantum state emergence
- *quantum-mechanics-demystified.md*: Comprehensive primer with advanced topics

### Exercise Progression
1. *Exercise 1*: Classical statistical mechanics and Fisher information foundations
2. *Exercise 2*: Quantum formalism emergence from information constraints
3. *Exercise 3*: Born rule derivation as optimal information extraction
4. *Exercise 4*: Entanglement and the de Finetti theorem
5. *Exercise 5*: Distinguishability, Fisher information geometry, and spacetime emergence
6. *Exercise 6*: Hierarchical partitions and exponential family structures
7. *Exercise 7*: Gradient flow dynamics and resolution limits
8. *Exercise 8*: Action principles, gauge theory, and physical law emergence

### Project Management
- *backlog/*: Task tracking for ongoing improvements
- *cip/*: Code Improvement Plans for major framework changes
- *docs/*: Additional documentation and guides

## Mathematical Foundation

The framework builds from information conservation through successive mathematical structures:

### Core Conservation Law
```
S(Z) + I(Z) = N log 2 = constant
```

### Fisher Information Metric
```
G_ij(θ) = E[∂ log p/∂θ_i ∂ log p/∂θ_j]
```

### Fundamental Flow Equation
```
dθ/dτ = -G(θ)θ
```

### Geometric Time Conversion
```
dt = √|G(θ)| dτ
```

### Action Principle (Entropy Time)
```
S = ∫[S(X,M_d|θ) - I(X,M_d|θ)]dτ
```

### Emergent Physical Action (Geometric Time)  
```
S = ∫[½G_ij(θ)dθ_i/dt dθ_j/dt - V(θ)]dt
```

## Key Results and Discoveries

### Quantum Mechanics Emergence
- *Quantum states* arise as unique solutions to information conservation constraints
- *Entanglement* is the natural default state for maximum correlation
- *Born rule* emerges as optimal information extraction procedure
- *Classical independence* emerges from quantum entanglement through scale separation

### Physical Law Emergence
- *Space and time* emerge from distinguishability requirements
- *Energy conservation* emerges from information conservation through coordinate geometry
- *Gauge symmetries* arise from discrete-continuous interface
- *Standard Model structure* emerges from information optimization under constraints

### Emergence Chain
```
Information Conservation → Gauge Theory → EPI → Physical Laws
```

### Emergence of Quantum Paradoxes
- *Wave-particle duality*: Different information extraction procedures
- *Measurement problem*: Optimal information extraction at scale transitions  
- *Quantum-classical boundary*: Natural scale separation effects via de Finetti
- *Observer effect*: Information extraction necessarily changes information structure

## Applications and Implications

### Physics
- Derives quantum mechanics from information conservation
- Explains emergence of spacetime and physical laws
- Unifies quantum and classical mechanics through scale separation
- Connects gauge theory to information optimization

### Beyond Physics
The framework's emphasis on *dynamic information topography* (dθ/dt ≠ 0) provides tools for understanding:
- Adaptive systems in economics and social science
- Technological disruption as topographical engineering
- Complex system emergence through information constraints

## Getting Started

### For Physicists
Start with Exercise 1 and work through the progression. The framework builds from familiar statistical mechanics concepts.

### For Information Theorists  
Focus on how Fisher information geometry becomes the substrate for physical law emergence.

### For Applied Researchers
Examine how dynamic information topography (dθ/dt ≠ 0) creates complexity in adaptive systems.

## Contributing

This is an active research framework. See the backlog system for current tasks and the CIP process for proposing major improvements.

## Key References

- Information geometry and statistical manifolds
- Fisher information and parameter estimation theory
- Maximum entropy principles in physics
- Quantum mechanics foundations and interpretations
- Gauge theory and fundamental physics
- De Finetti theorem and exchangeability

---

*"It is IT ..."*
