---
id: 2025-12-07_refactor-notebooks-to-use-qig-api
title: Refactor boring_game_dynamics and entropy_time_paths notebooks to use qig API
status: proposed
priority: medium
created: 2025-12-07
owner: null
dependencies: []
---

# Task: Refactor notebooks to use qig API

## Description

The `boring_game_dynamics.ipynb` and `entropy_time_paths.ipynb` notebooks currently re-implement much of the functionality already available in the qig package. This results in:

- Code duplication
- Potential inconsistencies with the tested qig implementations
- Missed opportunity to showcase the qig API
- Longer, harder-to-maintain notebooks

## Current State

### boring_game_dynamics.ipynb re-implements:
- Bell state construction (manual `psi_bell` loop)
- `partial_trace_B()` function  
- `entropy()` function
- `constraint_C()` function
- `mutual_information()` function
- Manual Euler gradient flow loop

### entropy_time_paths.ipynb re-implements:
- All of the above, plus:
- Steepest ascent computation
- BKM kernel construction
- Entropy time scaling
- Backward trajectory tracing

## Proposed Changes

### Replace with qig API:

| Manual Code | qig API Replacement |
|-------------|---------------------|
| Bell state construction | `qig.create_lme_state(2, d)` or `bell_state_density_matrix(d)` |
| `partial_trace_B()` | `qig.partial_trace(rho, dims, keep)` |
| `entropy()` | `qig.von_neumann_entropy(rho)` |
| Per-subsystem entropy loops | `qig.marginal_entropies(rho, dims)` |
| Mutual information | `exp_family.mutual_information(theta)` |
| Gradient flow loop | `InaccessibleGameDynamics.integrate()` |
| Entropy time scaling | `dynamics.set_time_mode('entropy')` |
| Constraint computation | `exp_family.marginal_entropy_constraint(theta)` |

### Use QuantumExponentialFamily with pair_basis=True:
- `get_bell_state_parameters(epsilon)` for regularized Bell states
- `fisher_information(theta)` for BKM metric
- `rho_from_theta(theta)` for density matrices

### Use InaccessibleGameDynamics:
- `integrate(theta_0, t_span, n_points)` for trajectories
- `solve_constrained_maxent()` for gradient descent
- `set_time_mode('entropy')` for entropy time

## Acceptance Criteria

- [ ] Both notebooks use qig imports instead of manual implementations
- [ ] Results remain identical (validate with numerical comparison)
- [ ] Notebooks are shorter and cleaner
- [ ] Notebooks serve as good API examples for users
- [ ] All cells execute successfully
- [ ] Notebook tests pass

## Implementation Notes

### Validation-First Approach

Before replacing any code cell, first validate that the qig API produces identical results:

1. **For each code cell**:
   - Add a new cell immediately after that computes the same result using qig API
   - Compare outputs numerically (use `np.allclose()` with appropriate tolerances)
   - Print both results side-by-side for visual confirmation
   - Only after validation passes, replace the original cell

2. **Validation template**:
   ```python
   # Original result
   result_manual = ...  # existing code
   
   # qig API result  
   result_qig = ...  # new qig call
   
   # Validate
   assert np.allclose(result_manual, result_qig, rtol=1e-10), \
       f"Mismatch: manual={result_manual}, qig={result_qig}"
   print(f"✓ Validated: {description}")
   ```

3. **Tolerance guidance**:
   - Entropy calculations: `rtol=1e-12` (should be very close)
   - Density matrices: `rtol=1e-12` 
   - Gradient flows: `rtol=1e-6` (numerical integration has inherent differences)
   - Fisher information: `rtol=1e-10`

### Implementation Steps

1. Start by importing qig at the top of each notebook
2. For each utility function (entropy, partial_trace, etc.):
   - Add validation cell comparing manual vs qig
   - After all validations pass, replace manual code
3. For gradient flows, consider whether to keep some manual code for pedagogical clarity
4. The "boring" vs "interesting" game distinction could be demonstrated using `exp_family.marginal_entropy_constraint()` to show when constraint gradient ≈ 0

### Discrepancy Handling

If validation reveals discrepancies:
- Document the difference in the notebook (could be educational!)
- Investigate whether manual code or qig has a bug
- If qig is more accurate, note why in comments
- If manual code was intentionally simplified, explain the trade-off

## Related

- Both notebooks are in `examples/`
- qig API docs in `docs/`
- Core utilities in `qig/core.py`
- Exponential family in `qig/exponential_family.py`
- Dynamics in `qig/dynamics.py`

## Progress Updates

### 2025-12-07
Task created with Proposed status. Analysis shows significant code duplication between notebooks and qig API.
