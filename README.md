<a target="_blank" href="https://colab.research.google.com/github/lawrennd/the-inaccessible-game/blob/main/origin-evolution.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# The Inaccessible Game: Information Conservation and Emergent Physics

With nothing more than "conserve the sum of marginal entropies and always follow the steepest‐entropy‐ascent direction," a purely information‑theoretic game reproduces—organically and sequentially—decoherence, colour confinement, an electroweak‑like gauge epoch, Lorentz kinematics, particle‑mass scales, and even quasi‑Pauli exclusion, then winds down to a feature‑free heat‑death.*

## Overview

This project implements the *Maximum Entropy Production Principle (MEPP)* framework using *Steepest-Entropy-Ascent (SEA)* dynamics in *Jaynes natural parameter* coordinates. Starting from pure information-theoretic axioms, the system demonstrates how complex physical laws—including gauge theories, particle physics, and cosmological evolution—can emerge organically from entropy maximization alone.

## Key Take‑aways

| Level                       | Emergent structure                                                                                  | Comment                                                                                                       |
| --------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| *First principles*        | Four information axioms + SEA rule (no Hamiltonian, no spacetime, no particles)                     | Demonstrates you can start from *entropy alone* rather than energy or action.                                 |
| *Stage hierarchy*         | Dephasing → Isolation/Confinement → Long plateau(s)                                                 | Matches cosmological eras: reheating, QCD confinement, long‑lived electroweak phase.                          |
| *Gauge theory*            | Local SU (3) Gauss law, automatic confinement; sloppy Fisher mode ⇒ effective $SU(2)\!\times\!U(1)$ | Gauge structure is *not* assumed; it precipitates from the constraint geometry.                               |
| *Relativistic kinematics* | Dispersion $\omega^{2}=c^{2}k^{2}+m^{2}$ for normal modes, operational Lorentz symmetry             | Speed of light arises as a ratio of Fisher tensors; observers must agree on it.                               |
| *Matter & mass*           | Standing‑wave normal modes; entropy curvature λ plays role of $m^{2}$                               | "Particle rest‑energy" is literally *entropy cost* of disturbing the plateau.                                 |
| *Approximate laws*        | Quasi‑conserved charges, Pauli‑like exclusion, nested symmetry breaking                             | Finite Fisher gaps create "laws" that last exponentially long—realistic physics scales without exact tunings. |
| *Ultimate fate*           | Heat‑death colour‑singlet vacuum; only gauge redundancy survives                                    | Predicts a true end‑state, yet explains why rich physics dominates any accessible epoch.                      |

## MEPP Library Installation

The MEPP quantum thermalization library implements the Jaynes/natural-parameter framework:

```bash
# Install dependencies and download library files
pip install numpy matplotlib scipy tqdm

# Download library directly (for Google Colab)
curl -O https://raw.githubusercontent.com/lawrennd/the-inaccessible-game/main/mepp.py
curl -O https://raw.githubusercontent.com/lawrennd/the-inaccessible-game/main/__init__.py
```

### Quick Start

```python
from mepp import MEPPSimulator
import numpy as np

# Create a 4-qubit MEPP simulator with natural parameter tracking
simulator = MEPPSimulator(n_qubits=4, max_support_size=3, d=2)

# Run two-stage thermalization: Stage A (dephasing) → Stage B (isolation)
final_state, _ = simulator.simulate_evolution(
    n_steps=30,          # Total evolution steps
    block_size=8,        # Random gates per block
    sigma_alpha=1.0,     # Gate strength
    dephasing_steps=10   # Stage A duration
)

# Visualize entropy evolution and Fisher spectrum
simulator.plot_results()

# Access natural parameters and Fisher analysis
theta_soft_history = simulator.theta_soft_history
fisher_eigenvals = simulator.fisher_eigenval_history
```

See `mepp_demo.ipynb` for comprehensive demonstrations including:
- Bell pair initialization and two-stage thermalization
- Natural parameter evolution in Jaynes coordinates  
- Fisher matrix eigenspectrum analysis
- Charge classification (hard vs soft constraints)
- SEA dynamics verification: θ̇ = -G_∥θ

## Theoretical Framework

### Jaynes/Natural-Parameter Foundation

The framework operates in *canonical coordinates* θ_j = log p_j - ψ where:

- *Natural Parameters*: Jaynes Lagrange multipliers for MaxEnt constraints
- *Hard Constraints*: Fixed single-site marginals (exactly conserved)
- *Soft Constraints*: Correlation parameters (free to evolve under SEA)
- *Fisher Matrix*: G_ij = curvature of entropy landscape
- *SEA Evolution*: θ̇ = -G_∥θ (steepest entropy ascent in natural coordinates)

### Three-Stage Evolution

1. *Stage A (Dephasing)*: Pure quantum coherences rapidly decohere via random phase gates
2. *Stage B (Isolation)*: Correlations evolve under MaxEnt constraint, approaching thermal equilibrium  
3. *Plateaus*: Quasi-conserved charges (small Fisher eigenvalues) create long-lived quasi-equilibrium states

### Emergent Physical Structures

- *Charges*: Fisher eigenvectors with λ_k < λ_cut (automatic gauge symmetry detection)
- *Particles*: Normal mode oscillations of natural parameters with effective mass λ_k
- *Gauge Theory*: Gauss law emerges from constraint geometry, not fundamental assumptions
- *Spacetime*: Dispersion relations ω² = c²k² + m² arise from Fisher tensor ratios

## Project Structure

### Core Implementation
- `mepp.py`: Main MEPP simulator with Jaynes/natural-parameter framework
- `__init__.py`: Package initialization and exports
- `mepp_demo.ipynb`: Comprehensive demonstration notebook

### Documentation
- `cip/cip0005.md`: Core CIP documenting MEPP thermalization with SEA dynamics
- `backlog/features/`: Detailed task specifications for advanced extensions
- `the-inaccessible-game.tex`: Theoretical foundation paper

### Advanced Features (Backlog)
- *Natural Parameter Framework*: Full Jaynes coordinate implementation
- *Charge Classification*: Automatic hard/soft constraint identification
- *SEA Verification*: Direct validation of θ̇ = -G_∥θ dynamics  
- *Spectrum Flow Analysis*: Time × log λ_k heatmaps showing symmetry breaking cascades

## Broader Significance

1. *Unifies dynamical, statistical, and information‑geometric viewpoints*
   – SEA flow = Fisher‑metric gradient → links Jaynes‑MaxEnt constraints, RG flow and thermodynamic time in one equation.

2. *Shows how hierarchy and fine structure can be *self‑organized*
   – Sloppy Fisher spectrum is *automatic* in large systems; no external "fundamental constants" are needed to separate scales.

3. *Gives a laboratory for testing foundational questions*
   * Which ingredients are minimal for fermionic statistics?
   * How do curvature and gravity appear if you let the Fisher metric back‑react?
   * Can adding higher‑order soft constraints reproduce the full Standard Model?

4. *Offers a new simulation pathway*
   – All dynamics expressed as CPTP maps and matrix exponentials; no sign‑problem, no Monte‑Carlo weights—ideal for tensor‑network or GPU implementation.

5. *Reframes "laws of physics" as long‑lived information regularities*
   – Conservation laws and symmetries are not immutable givens but plateaux whose lifetime depends on system size and observation scale.

## Computational Features

### Natural Parameter Analysis
```python
# Track evolution in Jaynes canonical coordinates
theta_soft = simulator.natural_params(rho)
fisher_matrix = simulator.fisher_matrix_soft(rho)

# Verify SEA dynamics: θ̇ = -G_∥θ
sea_error = simulator.verify_sea_dynamics(rho_before, rho_after, dt)
```

### Automatic Charge Classification  
```python
# Identify conserved quantities from Fisher eigenspectrum
charge_info = simulator.classify_charges(fisher_eigenvals, lambda_cut=1e-6)
hard_charges = charge_info['hard_charges']  # Exactly conserved (λ ≈ 0)
soft_charges = charge_info['soft_charges']  # Quasi-conserved (λ < λ_cut)
```

### Spectrum Flow Visualization
```python
# Visualize symmetry breaking as time × log λ_k heatmaps
fig, ax = simulator.create_spectrum_flow_heatmap()
band_gaps = simulator.analyze_band_gaps(gap_threshold=2.0)
```

## Dependencies

Core requirements:
- `numpy`: Numerical linear algebra and matrix operations
- `matplotlib`: Visualization and plotting  
- `scipy`: Linear algebra (eigenvalue decomposition, matrix exponentials)
- `tqdm`: Progress tracking for long simulations

## Research Context

This implementation demonstrates that:

- *Complex gauge theories emerge automatically* from simple entropy maximization
- *Particle physics arises organically* as normal modes of information geometry
- *Cosmological evolution sequences* (dephasing → confinement → electroweak plateau → heat death) follow naturally from Fisher eigenvalue hierarchy
- *Relativistic dispersion relations* emerge from Fisher tensor structure without assuming spacetime

### In short
The framework turns *information conservation* into a generative engine for gauge fields, spacetime symmetries, particle spectra, and cosmological history—suggesting that what we call "fundamental physics" might just be the long, slow echo of the steepest possible rise of entropy.

## Key References

- Beretta, G. P. (2019). The fourth law of thermodynamics: steepest entropy ascent. [arXiv:1908.05768](https://arxiv.org/abs/1908.05768) [cond-mat.stat-mech]. *Philosophical Transactions of the Royal Society A*, 378, 20190168 (2020)
- Surace, J. (2023). A Theory of Inaccessible Information. [arXiv:2305.05734](https://arxiv.org/abs/2305.05734) [quant-ph]. *Quantum*, 8, 1464 (2024)
- Lawrence, N. D. (2025). The Inaccessible Game. *draft paper* (see `the-inaccessible-game.tex`)
- Jaynes, E. T. (1957). Information theory and statistical mechanics. *Physical Review*, 106(4), 620-630
- Caticha, A. (2011). Entropic inference and the foundations of physics. *Monograph commissioned by the 11th Brazilian Meeting on Bayesian Statistics*

---

*"It is IT from it ..."* — The emergence of physics from pure information
