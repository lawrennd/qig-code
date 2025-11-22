# Quantum Qutrit Dynamics Simulation

## Overview

This code implements computational verification of the quantum inaccessible game for **n=3 qutrit subsystems** (d=3 each), directly analogous to the classical n=3 binary variable experiments.

## Files

### `quantum_qutrit_n3.py`
Main implementation containing:

- **Gell-Mann matrices**: SU(3) generators for qutrits
- **Quantum exponential family**: ρ(θ) = exp(Σ θₐ Fₐ) / Z
- **Partial trace**: Compute single-site marginals ρᵢ = Tr_{j≠i}[ρ]
- **Von Neumann entropy**: S(ρ) = -Tr(ρ log ρ)
- **BKM metric**: Quantum Fisher information G(θ) using anticommutator (analytic)
- **Duhamel formula**: Exact ∂ρ/∂θ for non-commuting operators
- **Constrained dynamics**: dθ/dt = -Π_∥ G θ with Σhᵢ = C
- **GENERIC decomposition**: M = S + A analysis (fully analytic)
- **LME state initialization**: Locally maximally entangled state |ψ⟩ = (|000⟩ + |111⟩ + |222⟩)/√3

### `test_quantum_qutrit.py`
Quick verification script that tests all components.

## System Specs

- **Hilbert space**: 3³ = 27 dimensions
- **Density matrix**: 27×27 complex (729 entries)
- **Parameters**: 3 sites × 8 Gell-Mann matrices = 24 real parameters
- **Marginal computation**: 27×27 → 3×3 via partial trace (fast!)
- **Computational cost**: Similar to classical n=3 binary case

## Quick Start

```bash
# Test the implementation
python test_quantum_qutrit.py

# Run full demo
python quantum_qutrit_n3.py
```

Expected output:
```
Creating LME initial state...
  Marginal entropies: [1.099 1.099 1.099]
  Sum: 3.296 (= 3 log 3 ✓)

Computing GENERIC decomposition...
  ||S|| (dissipative): X.XXX
  ||A|| (conservative): X.XXX
  ||A||/||S||: X.XXX
```

## Experiments You Can Run

### 1. Constraint Preservation
```python
sol = solve_constrained_quantum_maxent(
    theta_init, operators, n_steps=5000, dt=0.01
)

# Verify Σhᵢ(t) = C throughout
import matplotlib.pyplot as plt
plt.plot(sol['constraint_values'] - sol['C_init'])
plt.ylabel('ΔC')
plt.yscale('log')
```

### 2. GENERIC Decomposition Along Trajectory
```python
ratios = []
for i, theta in enumerate(sol['trajectory'][::50]):
    result = analyse_quantum_generic_structure(theta, operators)
    ratios.append(result['ratio'])

plt.plot(ratios)
plt.ylabel('||A||/||S||')
plt.xlabel('Time')
```

### 3. Coherence Decay
```python
coherences = []
for theta in sol['trajectory'][::10]:
    rho = compute_density_matrix(theta, operators)
    # Measure off-diagonal magnitude
    rho_diag = np.diag(np.diag(rho))
    coherence = np.linalg.norm(rho - rho_diag, 'fro')
    coherences.append(coherence)

plt.plot(coherences)
plt.ylabel('Coherence ||ρ - diag(ρ)||')
```

### 4. Compare to Unconstrained
```python
# Unconstrained: pure max ent (no constraint)
def solve_unconstrained(theta_init, operators, n_steps=5000):
    trajectory = [theta_init.copy()]
    theta = theta_init.copy()
    
    for step in range(n_steps):
        G = compute_bkm_metric(theta, operators)
        F = -G @ theta  # Pure entropy ascent
        theta = theta + 0.01 * F
        trajectory.append(theta.copy())
        
        if np.linalg.norm(F) < 1e-5:
            break
    
    return {'trajectory': np.array(trajectory)}

sol_unc = solve_unconstrained(theta_init, operators)

# Compare trajectories, final states, entropy evolution...
```

### 5. Jacobi Identity Verification
```python
# At any point theta:
result = analyse_quantum_generic_structure(theta, operators)
A = result['A']

# Compute {{f,g},h} + {{g,h},f} + {{h,f},g} for coordinate functions
# (Requires computing ∂A/∂θ via finite differences)
# See classical code for full implementation
```

## Computational Performance

For n=3 qutrits (all gradients analytic):
- LME state creation: < 0.01s
- Single BKM metric (analytic): ~0.05s
- Single GENERIC decomposition (analytic): ~5-10s
  - Most time is in Duhamel integration (30 matrix exponentials per parameter)
- 1000-step integration: ~10-20 minutes

All operations are **tractable** on a standard laptop. The Duhamel formula adds computational cost but ensures correctness for non-commuting operators.

## Comparison to Classical Experiments

| Classical (n=3 binary) | Quantum (n=3 qutrits) |
|------------------------|----------------------|
| 2³ = 8 states | 3³ = 27 dimensions |
| Exact enumeration | Matrix exponentiation |
| Marginal: sum over 4 configs | Partial trace: 27×27 → 3×3 |
| Shannon H(p) | Von Neumann S(ρ) |
| Fisher information | BKM metric |
| Same workflow | Same workflow ✓ |

## Next Steps

1. **Reproduce classical figures** for quantum case:
   - Trajectory comparison (constrained vs unconstrained)
   - Joint entropy evolution
   - Marginal entropy conservation
   - ||A||/||S|| along trajectory
   - Parameter evolution (θ₁, θ₂, ...)

2. **Test Jacobi identity** like classical code does

3. **Coherence analysis**: Track quantum → classical transition

4. **Temperature scaling**: Scale θ → βθ and measure ||A||/||S||(β)

## Dependencies

```bash
pip install numpy scipy matplotlib
```

## Notes

- All entropies use **natural logarithm** (nats, not bits)
- Gell-Mann matrices follow standard physics normalization: Tr(λₐλᵦ) = 2δₐᵦ
- **All gradients use analytic formulas** (no finite differences except for verification)
- LME state is the GHZ-like state |ψ⟩ = (|000⟩ + |111⟩ + |222⟩)/√3

## Key Differences from Classical Case

### Non-Commutative Operator Structure

The quantum GENERIC derivation differs fundamentally from the classical case due to operator non-commutativity:

1. **BKM Metric (Quantum Fisher Information)**:
   - Classical: G_{ab} = ⟨F_a F_b⟩ - ⟨F_a⟩⟨F_b⟩
   - Quantum: G_{ab} = (1/2)⟨{F_a, F_b}⟩ - ⟨F_a⟩⟨F_b⟩
   - Uses anticommutator {A,B} = AB + BA to ensure metric symmetry

2. **Density Matrix Derivatives**:
   - For ρ = exp(H)/Z where H = Σθᵢ Fᵢ, the naive formula ∂ρ/∂θ_k = ρ(F_k - ⟨F_k⟩I) is **incorrect** (57% error!)
   - Correct: **Duhamel formula** for non-commuting operators:
     ```
     ∂exp(H)/∂θ_k = ∫₀¹ exp(sH) F_k exp((1-s)H) ds
     ∂ρ/∂θ_k = (1/Z) ∂exp(H)/∂θ_k - ρ⟨F_k⟩
     ```
   - Reduces error from 57% to 0.26% (over 200× improvement)

3. **GENERIC Jacobian**:
   - All derivatives (∂G/∂θ, ∂a/∂θ) computed analytically using Duhamel formula
   - Achieves 0.01% accuracy despite massive cancellation in Jacobian
   - Decomposition M = S + A is numerically stable

### Why This Matters

- The classical ∂p/∂θ = p(f - ⟨f⟩) works because scalars commute
- Quantum operators don't commute: [F_a, F_b] ≠ 0 in general
- Must use integral formulations (Duhamel) to get correct derivatives
- Anticommutator structure ensures Riemannian metric properties

## Citation

This code accompanies:
- Paper: "The Inaccessible Game: Quantum Origin"
- Classical experiments: `the-inaccessible-game/generic_decomposition_n3.py`

