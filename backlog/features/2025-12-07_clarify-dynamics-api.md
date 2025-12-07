---
id: 2025-12-07_clarify-dynamics-api
title: Clarify InaccessibleGameDynamics API - when to use which method
status: proposed
priority: medium
created: 2025-12-07
owner: null
dependencies: []
---

# Task: Clarify InaccessibleGameDynamics API

## Description

The `InaccessibleGameDynamics` class has three methods for computing/integrating the constrained dynamics θ̇ = -Π_∥ G θ:

1. `flow(t, theta)` - computes θ̇ at a single point
2. `integrate(theta_0, t_span)` - uses scipy `solve_ivp` with `flow()`
3. `solve_constrained_maxent(theta_init, ...)` - gradient descent with Newton projection

The relationship between these methods and when to use each is not immediately clear from the API.

## Current State

### flow()
- Computes θ̇ = -Π_∥ G θ using projection matrix Π_∥ = I - aa^T/||a||²
- Handles entropy time by dividing by entropy production rate
- Low-level building block

### integrate()
- Wraps `scipy.integrate.solve_ivp` with `flow()` as the RHS
- Simple but can be numerically unstable (constraint drift)
- No projection back onto constraint manifold

### solve_constrained_maxent()
- Uses Lagrange multiplier formulation: F = F_unc - νa
- Newton projection onto constraint manifold every N steps
- Convergence checking, diagnostic output
- Supports `use_entropy_time=True`
- More stable for constrained optimisation

## Proposed Clarification

### Option A: Better docstrings
Add clear guidance in docstrings about when to use each method:
- `flow()`: For understanding the dynamics, custom integrators
- `integrate()`: Quick exploration (may drift off constraint)
- `solve_constrained_maxent()`: **Recommended** for actual computation

### Option B: Deprecate or rename
Consider whether `integrate()` should be:
- Deprecated in favour of `solve_constrained_maxent()`
- Renamed to `integrate_unconstrained()` to make clear it doesn't project
- Updated to use Newton projection like `solve_constrained_maxent()`

### Option C: Unified interface
Create a single `solve()` method with options:
```python
result = dynamics.solve(
    theta_0,
    method='constrained_maxent',  # or 'ivp', 'euler'
    use_entropy_time=True,
    project=True
)
```

## Acceptance Criteria

- [ ] Clear documentation of when to use each method
- [ ] API makes the recommended approach obvious
- [ ] Examples in docstrings showing typical usage

## Related

- Task: 2025-12-07_notebooks-natural-parameter-dynamics (notebooks should use `solve_constrained_maxent()`)
- `qig/dynamics.py`: InaccessibleGameDynamics class

## Progress Updates

### 2025-12-07
Task created. Current API has three methods with unclear relationship.
