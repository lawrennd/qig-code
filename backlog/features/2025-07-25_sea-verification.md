---
title: "SEA Dynamics Verification in Natural Coordinates"
id: "2025-07-25_sea-verification"
status: "proposed"
priority: "high"
created: "2025-07-25"
updated: "2025-07-25"
owner: "Neil"
dependencies: ["CIP-0005", "2025-07-25_jaynes-natural-parameters"]
category: "features"
---

# Task: SEA Dynamics Verification in Natural Coordinates

## Description

Implement numerical verification of Steepest-Entropy-Ascent (SEA) dynamics in natural parameter coordinates: θ̇ = -G_∥θ. This provides direct validation that the MEPP simulation follows the theoretical SEA evolution law in canonical coordinates.

## Acceptance Criteria

### 1. SEA Evolution Equation
- [ ] Implement `verify_sea_dynamics()` method for θ̇ = -G_∥θ verification
- [ ] Compute finite-difference approximation: θ̇ ≈ (θ_next - θ_now) / Δt
- [ ] Calculate theoretical RHS: -G_∥θ using Fisher matrix in soft coordinates
- [ ] Track SEA error norm: ||θ̇ - (-G_∥θ)|| throughout simulation

### 2. Fisher Matrix in Natural Coordinates
- [ ] Implement Fisher matrix computation in soft parameter space
- [ ] Project Fisher matrix: G_soft = P_soft @ G_full @ P_soft.T
- [ ] Handle numerical stability for small eigenvalues
- [ ] Verify positive semi-definiteness and symmetry

### 3. Time-Scale Analysis
- [ ] Track SEA compliance across different time scales (block vs step level)
- [ ] Identify where/when SEA approximation breaks down
- [ ] Correlate SEA errors with entropy plateaus and Fisher eigenvalues
- [ ] Test different time discretization schemes

### 4. Theoretical Validation
- [ ] Compare with analytical SEA solutions for simple cases
- [ ] Verify that SEA flow increases entropy monotonically
- [ ] Check that fixed marginals remain exactly preserved
- [ ] Validate connection to steepest entropy ascent principle

## Implementation Notes

### Core SEA Verification
```python
def verify_sea_dynamics(self, rho_before, rho_after, dt_block):
    """Verify θ̇ = -G_∥θ numerically."""
    # Compute natural parameters before and after evolution step
    theta_before = self.natural_params(rho_before)
    theta_after = self.natural_params(rho_after)
    
    # Finite difference approximation of θ̇
    dtheta_dt = (theta_after - theta_before) / dt_block
    
    # Theoretical RHS: -G_∥θ
    G_soft = self.fisher_matrix_soft(rho_before)
    theoretical_rhs = -G_soft @ theta_before
    
    # Compute SEA error
    sea_error = np.linalg.norm(dtheta_dt - theoretical_rhs)
    
    # Store for analysis
    self.sea_error_history.append(sea_error)
    self.sea_dtheta_history.append(dtheta_dt)
    self.sea_theoretical_history.append(theoretical_rhs)
    
    return {
        'error': sea_error,
        'measured_dtheta': dtheta_dt,
        'theoretical_dtheta': theoretical_rhs,
        'relative_error': sea_error / (np.linalg.norm(theoretical_rhs) + 1e-12)
    }

def fisher_matrix_soft(self, rho):
    """Compute Fisher matrix in soft parameter coordinates."""
    # Full Fisher matrix in probability space
    p = np.diag(rho).real
    p = p / p.sum()
    G_full = np.diag(1.0 / (p + 1e-12)) - 1.0  # Add regularization
    
    # Project to soft coordinates
    G_soft = self.P_soft @ G_full @ self.P_soft.T
    
    return G_soft
```

### Advanced Analysis
```python
def analyze_sea_compliance(self):
    """Analyze SEA compliance across different metrics."""
    errors = np.array(self.sea_error_history)
    
    analysis = {
        'mean_error': np.mean(errors),
        'max_error': np.max(errors),
        'error_trend': np.polyfit(range(len(errors)), errors, 1)[0],
        'compliance_ratio': np.sum(errors < 1e-6) / len(errors),
        'error_vs_entropy': np.corrcoef(errors, self.entropy_history[:len(errors)])[0,1]
    }
    
    return analysis

def sea_breakdown_analysis(self):
    """Identify when and why SEA approximation fails."""
    breakdowns = []
    
    for i, error in enumerate(self.sea_error_history):
        if error > 10 * np.mean(self.sea_error_history):  # Significant deviation
            breakdown = {
                'step': i,
                'error': error,
                'entropy': self.entropy_history[i] if i < len(self.entropy_history) else None,
                'fisher_condition': self._compute_fisher_condition_number(i),
                'possible_cause': self._diagnose_sea_failure(i)
            }
            breakdowns.append(breakdown)
    
    return breakdowns

def _diagnose_sea_failure(self, step_index):
    """Diagnose likely cause of SEA approximation failure."""
    if step_index < len(self.fisher_eigenval_history):
        eigenvals = self.fisher_eigenval_history[step_index]
        min_eigval = np.min(eigenvals)
        
        if min_eigval < 1e-10:
            return "Near-singular Fisher matrix (approaching conserved charge)"
        elif step_index in self.plateau_events:
            return "Plateau region with slow dynamics"
        else:
            return "Large discretization error or numerical instability"
    
    return "Insufficient data for diagnosis"
```

### Time Scale Analysis
```python
def multi_scale_sea_verification(self, scales=[1, 5, 10]):
    """Test SEA compliance at multiple time scales."""
    results = {}
    
    for scale in scales:
        if len(self.theta_soft_history) > scale:
            # Coarse-grain time evolution
            theta_coarse = self.theta_soft_history[::scale]
            dt_coarse = scale * self.dt_per_step
            
            # Verify SEA at coarse scale
            sea_errors_coarse = []
            for i in range(len(theta_coarse) - 1):
                dtheta = (theta_coarse[i+1] - theta_coarse[i]) / dt_coarse
                
                # Use Fisher matrix at midpoint
                mid_rho = self.rho_history[i*scale + scale//2]
                G_soft = self.fisher_matrix_soft(mid_rho)
                theoretical = -G_soft @ theta_coarse[i]
                
                error = np.linalg.norm(dtheta - theoretical)
                sea_errors_coarse.append(error)
            
            results[f'scale_{scale}'] = {
                'mean_error': np.mean(sea_errors_coarse),
                'compliance_ratio': np.sum(np.array(sea_errors_coarse) < 1e-6) / len(sea_errors_coarse)
            }
    
    return results
```

## Physics Insights

### What We Learn
1. *SEA Validity*: Direct test of steepest-entropy-ascent theoretical foundation
2. *Discretization Effects*: How finite time steps affect continuous SEA dynamics
3. *Singular Limits*: Behavior near conserved charges where Fisher matrix becomes singular
4. *Multi-Scale Dynamics*: SEA compliance at different temporal resolutions

### Expected Results
- Good SEA compliance during smooth evolution phases
- Increased errors near entropy plateaus and singular Fisher matrices
- Better compliance at coarser time scales due to reduced discretization errors
- Correlation between SEA errors and approach to conserved charges

## Testing Strategy

### Unit Tests
- Test with known analytical SEA solutions (simple 2-level systems)
- Verify Fisher matrix properties (symmetry, positive semi-definiteness)
- Check natural parameter computation accuracy

### Convergence Tests
- Test SEA compliance as time step size decreases
- Verify that errors scale appropriately with discretization
- Check stability of verification algorithm

### Physics Validation
- Confirm entropy increase during SEA-compliant evolution
- Verify marginal conservation throughout SEA flow
- Compare with independent SEA implementations

## Diagnostic Capabilities

### Real-Time Monitoring
- Track SEA compliance during simulation
- Flag significant deviations for investigation
- Correlate errors with other simulation metrics

### Post-Analysis
- Generate SEA compliance reports
- Identify systematic patterns in SEA violations
- Provide recommendations for parameter tuning

## Visualization

### Error Evolution Plots
- SEA error vs simulation time
- Correlation plots: error vs entropy, Fisher eigenvalues
- Multi-scale compliance comparison

### Vector Field Analysis
- Measured vs theoretical θ̇ vector fields
- Phase space trajectories in natural parameter space
- Streamlines of SEA flow

## Related

- CIP: 0005 (MEPP Thermalization)
- Theory: Steepest-Entropy-Ascent principle, Fisher information geometry
- Files: `mepp.py`, `mepp_demo.ipynb`
- Dependencies: Jaynes natural parameters, Fisher matrix computation

## Progress Updates

### 2025-07-25
Task created with Proposed status. SEA verification framework designed with comprehensive error analysis and multi-scale validation. 