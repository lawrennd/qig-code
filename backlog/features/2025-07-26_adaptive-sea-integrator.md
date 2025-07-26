---
title: "Adaptive SEA Integrator for Entropy-Time Evolution"
id: "2025-07-26_adaptive-sea-integrator"
status: "proposed"
priority: "high"
created: "2025-07-26"
updated: "2025-07-26"
owner: "Neil"
dependencies: ["CIP-0005", "2025-07-25_fisher-matrix-analysis"]
category: "features"
related_cip: "CIP-0007"
---

# Task: Adaptive SEA Integrator for Entropy-Time Evolution

## Description

Implement the core adaptive Runge-Kutta-Fehlberg integrator that switches from fixed time steps to entropy-time τ with adaptive stepping. This is the central algorithm for Stage 3 plateau simulation where Fisher norm shrinks and time dilation effects emerge.

## Acceptance Criteria

### 1. Entropy-Time RHS Function
- [ ] Implement `sea_rhs_tau(theta)` returning `G^(-1) ∇S`
- [ ] Use Fisher matrix G(θ) for metric-aware gradient
- [ ] Handle numerical stability when Fisher matrix becomes singular
- [ ] Return SEA flow direction in natural parameter coordinates

### 2. Adaptive RK45 Integrator  
- [ ] Replace fixed `sea_evolution_step` with `sea_step_tau(dt_tau)`
- [ ] Implement Runge-Kutta-Fehlberg with automatic step size control
- [ ] Accumulate real-time via `dt = ||∇S||²_{G^(-1)} dτ`
- [ ] Track both entropy-time τ and real-time t histories

### 3. Fisher Norm Monitoring
- [ ] Compute Fisher norm `||∇S||²_{G^(-1)}` at each step  
- [ ] Detect when norm drops below `FISHER_NORM_THRESHOLD = 1e-6`
- [ ] Switch to entropy-time mode when norm becomes small
- [ ] Store lapse function `lapse_history = dt/dτ` for analysis

### 4. Integration Controls
- [ ] Configurable tolerance for RK45 error control
- [ ] Maximum/minimum step size limits for stability
- [ ] Convergence criteria for entropy approach to S_max
- [ ] Timeout protection via `τ_max` parameter

## Implementation Notes

### Technical Approach
- Use scipy.integrate.solve_ivp with RK45 method for robust adaptive stepping
- Fisher matrix computation delegated to existing `compute_fisher()` utility
- Entropy gradient calculation leverages existing `sea_generator()` infrastructure
- Real-time accumulation maintains physical interpretation of lapse function

### Physics Insights
- Entropy-time τ provides natural clock where dS/dτ = 1 always
- Real-time t slows down (large dt/dτ) when Fisher eigenvalues become small
- This implements Beretta's lapse function formalism for non-equilibrium thermodynamics
- Stage 3 plateau emerges when λ_min << 1 causing information stalling

### Performance Considerations
- Fisher matrix eigendecomposition can be expensive for large systems
- Consider sparse eigenvalue methods if only λ_min is needed
- Cache Fisher matrix between steps if parameters change slowly
- Use vectorized operations for gradient and matrix computations

## Testing Strategy

### Unit Tests
- Verify RK45 integrator conserves trace and Hermiticity
- Test Fisher norm calculation against finite difference
- Validate entropy-time accumulation dS/dτ ≈ 1

### Integration Tests  
- Run small system (4 qubits) through complete Stage 3 evolution
- Verify lapse function dt/dτ increases during plateau regime
- Check entropy approaches S_max within specified tolerance

### Performance Tests
- Benchmark against fixed time-step evolution for speed comparison
- Memory usage analysis for large system scaling
- Convergence analysis for different tolerance settings

## Dependencies

- *CIP-0005*: Base MEPPSimulator infrastructure and sea_generator
- *Fisher Matrix Analysis*: compute_fisher() and eigenvalue utilities
- *Plateau Detection*: Integration with automatic plateau flagging

## Related Tasks

- `2025-07-25_fisher-matrix-analysis`: Provides Fisher matrix computation
- `2025-07-25_plateau-detection`: Uses results for automatic plateau identification
- `2025-07-26_stage3-simulation-loop`: Integrates adaptive stepping into Stage 3

## References

- CIP-0007: Stage 3 Plateau/Stalling Regime Implementation
- Beretta (2020): Fourth law of thermodynamics - lapse function theory
- Numerical Recipes: Runge-Kutta-Fehlberg adaptive integration 