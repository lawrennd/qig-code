---
id: 2025-12-07_notebooks-natural-parameter-dynamics
title: Refactor notebooks to use natural parameter (θ) dynamics instead of density matrix (ρ) space
status: proposed
priority: high
created: 2025-12-07
owner: null
dependencies: []
---

# Task: Refactor notebooks to use natural parameter dynamics

## Description

The `boring_game_dynamics.ipynb` and `entropy_time_paths.ipynb` notebooks currently work in density matrix space (ρ) with manual Euler integration. This misses the whole point of the inaccessible game, which is formulated in natural parameter space (θ).

The qig package provides:
- `QuantumExponentialFamily` with `pair_basis=True` for entangled pairs
- `InaccessibleGameDynamics` for constrained dynamics: θ̇ = -Π_∥ G θ
- `set_time_mode('entropy')` for entropy time parametrization
- `get_bell_state_parameters(epsilon)` for regularized Bell state parameters
- Fisher information G(θ), constraint gradient a = ∇C, etc.

## Current State (Wrong)

The notebooks currently:
1. Construct ρ directly (Bell state density matrix)
2. Compute ∇H = -(log ρ + I) in ρ-space
3. Do manual Euler integration: ρ_new = ρ + dt * grad
4. Project back to valid density matrices

This is **classical** steepest ascent in ρ-space, not the quantum inaccessible game.

## Desired State (Correct)

The notebooks should:
1. Use `QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)`
2. Get initial θ via `exp_family.get_bell_state_parameters(epsilon)`
3. Use `InaccessibleGameDynamics(exp_family)` for the dynamics
4. Call `dynamics.set_time_mode('entropy')` for entropy time
5. Integrate via `dynamics.integrate(theta_0, t_span)` or `solve_constrained_maxent()`

## Key Concepts to Demonstrate

### boring_game_dynamics.ipynb
- Show that from LME origin, constraint gradient a ≈ 0
- Therefore Π_∥ ≈ I (constraint is automatically satisfied)
- θ-space dynamics shows the "boring" game clearly

### entropy_time_paths.ipynb  
- Different regularizations (σ) → different θ starting points
- Entropy time: dH/dt = 1 by construction
- L'Hôpital limit as θ → boundary
- Multiple paths sharing the same pure-state limit

## Implementation

### Use QuantumExponentialFamily

```python
from qig.exponential_family import QuantumExponentialFamily

# For single pair of qutrits
exp_family = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)

# Get Bell state parameters
theta_bell = exp_family.get_bell_state_parameters(epsilon=0.01)

# Fisher metric
G = exp_family.fisher_information(theta)

# Constraint and gradient
C, grad_C = exp_family.marginal_entropy_constraint(theta)
```

### Use InaccessibleGameDynamics

```python
from qig.dynamics import InaccessibleGameDynamics

dynamics = InaccessibleGameDynamics(exp_family)
dynamics.set_time_mode('entropy')  # For entropy time

# Integrate
result = dynamics.integrate(theta_0, t_span=(0, 10), n_points=100)

# Or use gradient descent solver
result = dynamics.solve_constrained_maxent(
    theta_init, n_steps=1000, dt=0.001,
    use_entropy_time=True
)
```

## Acceptance Criteria

- [ ] Notebooks work in θ-space, not ρ-space
- [ ] Use `QuantumExponentialFamily` with `pair_basis=True`
- [ ] Use `InaccessibleGameDynamics` for constrained dynamics
- [ ] Demonstrate constraint gradient a = ∇C
- [ ] Show Π_∥ projection clearly
- [ ] Entropy time parametrization via `set_time_mode('entropy')`
- [ ] Results show the same physics but in the correct formulation

## Related

- Previous task (2025-12-07_refactor-notebooks-to-use-qig-api) only replaced utilities
- CIP-0001: Package structure
- `qig/exponential_family.py`: QuantumExponentialFamily class
- `qig/dynamics.py`: InaccessibleGameDynamics class

## Progress Updates

### 2025-12-07
Task created. Previous refactoring missed the point - replaced utilities but kept ρ-space dynamics. Need to properly use θ-space formulation.
