---
title: "Stage 3 Simulation Loop for Long Plateau Evolution"
id: "2025-07-26_stage3-simulation-loop"
status: "proposed"
priority: "high"
created: "2025-07-26"
updated: "2025-07-26"
owner: "Neil"
dependencies: ["2025-07-26_adaptive-sea-integrator", "2025-07-25_plateau-detection"]
category: "features"
related_cip: "CIP-0007"
---

# Task: Stage 3 Simulation Loop for Long Plateau Evolution

## Description

Implement the main `simulate_stage3()` method that orchestrates the long plateau evolution after Stages 1-2 (dephasing and isolation). This loop continues adaptive SEA evolution until entropy approaches S_max or maximum entropy-time τ_max is reached.

## Acceptance Criteria

### 1. Stage 3 Main Loop
- [ ] Implement `simulate_stage3(tau_max=1e3, entropy_tolerance=1e-3)`
- [ ] Continue from end state of Stage 2 isolation
- [ ] Use adaptive SEA integrator for entropy-time evolution
- [ ] Stop when `(S_max - S_current) < entropy_tolerance` or `τ > τ_max`

### 2. Data Logging and Diagnostics
- [ ] Store `tau_history[]` - entropy-time progression
- [ ] Store `lapse_history[]` - real-time dilation factor dt/dτ
- [ ] Store `min_fisher_eig[]` - slowest Fisher eigenvalue λ_min
- [ ] Store `entropy_history[]` - entropy evolution during plateau
- [ ] Store `real_time_history[]` - accumulated real-time t

### 3. Plateau Detection Integration
- [ ] Use automatic plateau detection to mark plateau onset
- [ ] Flag when Fisher eigenvalue drops below `STALL_THRESHOLD = 1e-4`
- [ ] Record plateau duration in both entropy-time and real-time
- [ ] Store plateau characteristics for analysis

### 4. Progress Monitoring
- [ ] Add progress bar with entropy approach to S_max
- [ ] Periodic logging of key metrics (every 100 τ steps)
- [ ] Convergence diagnostics and early stopping conditions
- [ ] Memory usage monitoring for large system runs

## Implementation Notes

### Integration with Existing Framework
- Called after `simulate_isolation()` in main evolution pipeline
- Inherits state from Stage 2: density matrix, natural parameters, Fisher metrics
- Maintains backward compatibility with existing plotting and analysis tools
- Extends data arrays rather than replacing Stage 1-2 results

### Stage 3 Physics
- Long plateau regime where entropy barely changes but τ continues
- Information stalling due to near-zero Fisher eigenvalues
- Time dilation: real-time t slows while entropy-time τ proceeds normally
- Quasi-symmetry emergence in sloppy parameter directions

### Performance Optimizations
- Adaptive sampling: dense logging during transitions, sparse during plateaus
- Fisher matrix caching between similar parameter states
- Early convergence detection to avoid unnecessary computation
- Configurable logging frequency for large system runs

## Testing Strategy

### Unit Tests
- Verify Stage 3 correctly inherits Stage 2 final state
- Test convergence criteria with known endpoint states
- Validate data logging and array extension functionality

### Integration Tests
- Complete 3-stage evolution for 2-qubit system (known behavior)
- 3-qutrit system plateau emergence and duration analysis
- Larger system (6+ qubits) scaling and memory performance

### Physics Validation Tests
- Entropy conservation: verify total entropy remains constant
- Fisher eigenvalue evolution: check λ_min decreases during plateau
- Lapse function analysis: verify dt/dτ increases as expected
- Compare plateau duration with analytical estimates

## Dependencies

- *Adaptive SEA Integrator*: Core entropy-time evolution algorithm
- *Plateau Detection*: Automatic identification of plateau regimes
- *Fisher Matrix Analysis*: Eigenvalue monitoring and stall detection
- *CIP-0005*: Base MEPPSimulator framework and Stage 1-2 implementation

## Related Tasks

- `2025-07-26_adaptive-sea-integrator`: Provides core adaptive stepping
- `2025-07-25_plateau-detection`: Automatic plateau identification
- `2025-07-26_lapse-function-visualization`: Plotting for Stage 3 results
- `2025-07-25_fisher-matrix-analysis`: Fisher eigenvalue tracking

## Success Metrics

### Quantitative Targets
- Successful plateau detection for d=3 qutrit systems
- Lapse function dt/dτ increases by factor >10³ during plateau
- Entropy approaches within 1e-3 nats of theoretical S_max
- Stage 3 duration >80% of total simulation time for long plateau systems

### Qualitative Indicators
- Clear visual separation between Stages 2 (isolation) and 3 (plateau)
- Smooth entropy-time evolution without numerical artifacts
- Physically reasonable Fisher eigenvalue decay patterns
- Stable integration without divergence or stalling

## References

- CIP-0007: Stage 3 Plateau/Stalling Regime Implementation
- CIP-0005: Two-stage MEPP framework foundation
- Beretta (2020): SEA dynamics and entropy-time formalism 