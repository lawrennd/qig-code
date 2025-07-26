---
title: "Lapse Function and Fisher Eigenvalue Visualization"
id: "2025-07-26_lapse-function-visualization"
status: "proposed"
priority: "medium"
created: "2025-07-26"
updated: "2025-07-26"
owner: "Neil"
dependencies: ["2025-07-26_stage3-simulation-loop"]
category: "features"
related_cip: "CIP-0007"
---

# Task: Lapse Function and Fisher Eigenvalue Visualization

## Description

Extend the existing plotting infrastructure to visualize Stage 3 plateau dynamics, including the lapse function dt/dτ that shows time dilation effects and Fisher eigenvalue evolution that reveals information horizons and stalling mechanisms.

## Acceptance Criteria

### 1. Lapse Function Plotting
- [ ] Add `plot_lapse_function()` method to visualization module
- [ ] Semi-log plot of dt/dτ vs real-time t showing time dilation
- [ ] Annotate plateau onset where lapse function diverges
- [ ] Include theoretical lapse = 1/(Fisher norm) for comparison

### 2. Fisher Eigenvalue Traces
- [ ] Add `plot_fisher_spectrum()` for eigenvalue evolution
- [ ] Track λ_min(t) - the slowest mode governing plateau duration
- [ ] Show full eigenvalue spectrum with color-coded evolution
- [ ] Mark STALL_THRESHOLD crossing for plateau detection

### 3. Enhanced Multi-Panel Figure
- [ ] Extend existing entropy plots to include Stage 3 data
- [ ] New bottom panel: lapse function (semilog scale)
- [ ] New right panel: Fisher eigenvalue trace (log scale)
- [ ] Consistent time axis across all panels (real-time t)

### 4. Stage Annotation and Formatting
- [ ] Clear visual separation between Stages 1, 2, and 3
- [ ] Color-coded background regions for each stage
- [ ] Stage transition markers with timing information
- [ ] Professional formatting for AISTATS Figure 3 generation

## Implementation Notes

### Integration with Existing Plotting
- Extends current `plot_entropy_evolution()` infrastructure
- Maintains backward compatibility with Stage 1-2 only simulations
- Uses consistent styling and color schemes
- Integrates with existing figure saving and formatting options

### Scientific Visualization Best Practices
- Clear axis labels with physical units (nats, time steps, eigenvalues)
- Legends explaining all plot elements and color coding
- Error bars or confidence intervals where appropriate
- Publication-quality figure sizing and font choices

### Performance Considerations
- Efficient plotting for large datasets (Stage 3 can be very long)
- Optional data decimation for visualization without losing key features
- Caching of heavy computations like eigenvalue decompositions
- Interactive plotting options for exploration vs. static figures for papers

## Testing Strategy

### Visual Validation Tests
- Compare lapse function plots against known analytical behavior
- Verify Fisher eigenvalue traces show expected hierarchy emergence
- Test plot formatting with various system sizes and evolution lengths

### Integration Tests
- Ensure plots work with incomplete data (Stage 3 not run)
- Test backward compatibility with existing plotting calls
- Validate figure saving and export functionality

## Dependencies

- *Stage 3 Simulation Loop*: Provides lapse_history and min_fisher_eig data
- *Fisher Matrix Analysis*: Eigenvalue computation and tracking
- *Existing Plotting Infrastructure*: Base entropy evolution plots

## Related Tasks

- `2025-07-26_stage3-simulation-loop`: Generates data for visualization
- `2025-07-25_fisher-matrix-analysis`: Provides eigenvalue data
- *AISTATS Paper*: Figure 3 showing plateau dynamics

## Physics Interpretation Features

### Lapse Function Analysis
- Visual demonstration of Beretta's entropy-time vs. real-time separation
- Clear evidence of time dilation during information stalling regime
- Quantitative measure of "how slow" the system becomes during plateau

### Fisher Eigenvalue Hierarchy
- Identification of sloppy vs. stiff parameter directions
- Evolution of information geometry during thermalization
- Visual proof of approaching information horizons (λ_min → 0)

## Success Metrics

### Figure Quality
- Publication-ready figures suitable for AISTATS submission
- Clear visual demonstration of Stage 3 plateau physics
- Professional appearance with proper scaling and annotations

### Scientific Value
- Lapse function clearly shows >10³ time dilation during plateau
- Fisher eigenvalue hierarchy emerges visibly during evolution
- Stage transitions are clearly demarcated and interpretable

## References

- CIP-0007: Stage 3 Plateau/Stalling Regime Implementation
- Beretta (2020): Lapse function formalism for non-equilibrium thermodynamics
- AISTATS submission requirements: Figure quality and formatting standards 