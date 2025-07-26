---
title: "MEPPSimulator Stage 3 Integration and API Updates"
id: "2025-07-26_mepp-stage3-integration"
status: "proposed"
priority: "high"
created: "2025-07-26"
updated: "2025-07-26"
owner: "Neil"
dependencies: ["2025-07-26_adaptive-sea-integrator", "2025-07-26_stage3-simulation-loop"]
category: "features"
related_cip: "CIP-0007"
---

# Task: MEPPSimulator Stage 3 Integration and API Updates

## Description

Integrate the Stage 3 plateau simulation capabilities into the main MEPPSimulator class, updating the public API to support three-stage evolution while maintaining backward compatibility with existing Stage 1-2 workflows.

## Acceptance Criteria

### 1. MEPPSimulator Class Updates
- [ ] Add `enable_stage3=False` parameter to `__init__()`
- [ ] Update `simulate_evolution()` to optionally call `simulate_stage3()`
- [ ] Add Stage 3 configuration parameters: `tau_max`, `entropy_tolerance`
- [ ] Extend data storage arrays for Stage 3 results

### 2. Public API Extensions
- [ ] Add `run_three_stage_evolution()` convenience method
- [ ] Add `get_stage3_results()` for accessing plateau data
- [ ] Add `plot_complete_evolution()` including lapse function
- [ ] Maintain existing two-stage API unchanged for compatibility

### 3. Configuration and Defaults
- [ ] Sensible defaults: `tau_max=1e3`, `entropy_tolerance=1e-3`
- [ ] Stage 3 threshold: `STALL_THRESHOLD=1e-4` for plateau detection
- [ ] Configurable Fisher matrix computation frequency
- [ ] Optional Stage 3 disable flag for performance

### 4. Data Management
- [ ] Extend existing history arrays: `tau_history`, `lapse_history`
- [ ] Add Stage 3 specific arrays: `min_fisher_eig`, `real_time_history`
- [ ] Proper array concatenation between stages
- [ ] Memory-efficient storage for long plateau simulations

## Implementation Notes

### Backward Compatibility Strategy
- All existing code continues to work unchanged
- Stage 3 is opt-in via `enable_stage3=True` parameter
- Default behavior remains two-stage evolution
- Legacy plotting functions work with or without Stage 3 data

### API Design Principles
- Clear method names indicating stage coverage
- Consistent parameter naming across stages
- Proper documentation with usage examples
- Type hints for all new methods and parameters

### Performance Considerations
- Stage 3 can be computationally expensive (long plateau evolution)
- Optional early termination based on convergence criteria
- Configurable logging frequency to balance detail vs. performance
- Memory usage warnings for very long simulations

## Testing Strategy

### Unit Tests
- Test Stage 3 enable/disable functionality
- Verify backward compatibility with existing code
- Test parameter validation and error handling

### Integration Tests
- Complete three-stage evolution for reference systems
- Performance benchmarking against two-stage baseline
- Memory usage analysis for long plateau simulations

### API Validation Tests
- Test all new public methods with various parameter combinations
- Verify data consistency across stage boundaries
- Test plotting functions with incomplete Stage 3 data

## Dependencies

- *Adaptive SEA Integrator*: Core entropy-time evolution algorithm
- *Stage 3 Simulation Loop*: Main plateau evolution orchestration
- *Lapse Function Visualization*: Extended plotting capabilities
- *CIP-0005*: Base MEPPSimulator infrastructure

## Related Tasks

- `2025-07-26_adaptive-sea-integrator`: Provides entropy-time stepping
- `2025-07-26_stage3-simulation-loop`: Main Stage 3 implementation
- `2025-07-26_lapse-function-visualization`: Enhanced plotting
- `2025-07-25_fisher-matrix-analysis`: Fisher eigenvalue computation

## Usage Examples

### Basic Three-Stage Evolution
```python
# Enable Stage 3 in simulator configuration
sim = MEPPSimulator(d=3, n_qubits=4, enable_stage3=True)

# Run complete three-stage evolution
results = sim.run_three_stage_evolution(
    dephasing_steps=50,
    isolation_steps=100, 
    tau_max=1000,
    entropy_tolerance=1e-3
)

# Plot complete evolution including lapse function
sim.plot_complete_evolution()
```

### Legacy Compatibility
```python
# Existing code continues to work unchanged
sim = MEPPSimulator(d=3, n_qubits=4)  # Stage 3 disabled by default
results = sim.simulate_evolution(n_steps=150)  # Two-stage evolution
sim.plot_entropy_evolution()  # Original plotting
```

## Success Metrics

### Integration Quality
- Zero breaking changes to existing Stage 1-2 API
- Clean three-stage evolution with smooth data transitions
- Professional documentation with clear examples

### Performance
- Stage 3 adds <20% overhead when disabled
- Reasonable memory usage for plateau simulations
- Configurable performance vs. detail trade-offs

## References

- CIP-0007: Stage 3 Plateau/Stalling Regime Implementation
- CIP-0005: Base MEPP framework and two-stage evolution
- Python API design best practices for scientific computing 