---
id: 2025-12-07_clarify-dynamics-api
title: Clarify InaccessibleGameDynamics API - when to use which method
status: completed
priority: high
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

## Proposed Solution: Option C - Unified interface with cleanup

**Problem**: `integrate()` looks like the obvious choice but it's broken for constrained problems (no projection → constraint drift). This is a trap for users.

### New unified API

```python
result = dynamics.solve(
    theta_0,
    t_end=10.0,           # or n_steps for gradient descent
    method='gradient',     # 'gradient' (recommended) or 'ivp' (unstable)
    entropy_time=True,
    project=True,
    project_every=10
)
```

### Cleanup of broken/confusing methods

| Method | Action |
|--------|--------|
| `solve_constrained_maxent()` | Rename to `solve()` or keep as implementation |
| `integrate()` | **Remove** or make private `_integrate_ivp()` |
| `flow()` | Keep as `_flow()` for internal use, or expose for advanced users |

### Why remove integrate()?
- Uses `solve_ivp` without constraint projection
- Looks like the "right" way to integrate but isn't
- Constraint drifts, giving wrong answers silently
- If someone needs raw IVP, they can use scipy directly with `flow()`

## Acceptance Criteria

- [ ] Single `solve()` method as the primary API
- [ ] `integrate()` removed or clearly marked as internal/deprecated
- [ ] `flow()` available for advanced use but not the main interface
- [ ] Clear error message if constraint drift detected
- [ ] Examples in docstrings showing typical usage

## Related

- Task: 2025-12-07_notebooks-natural-parameter-dynamics (notebooks should use `solve_constrained_maxent()`)
- `qig/dynamics.py`: InaccessibleGameDynamics class

## Progress Updates

### 2025-12-07
Task created. Current API has three methods with unclear relationship.

Decision: Go with Option C (unified interface). `integrate()` is essentially broken for constrained problems - it looks right but drifts off the constraint silently. Need to clean up rather than just document the trap.

### 2025-12-07 (completed)
Implemented Option C:
- Added `solve()` as the primary API (wraps `solve_constrained_maxent()`)
- Deprecated `integrate()` with `DeprecationWarning`
- Improved docstrings with usage examples
- Added `'theta'` alias in results for backward compatibility
- Updated `integrate_with_monitoring()` to use `solve()` internally
- All 343 tests pass
