<a href="https://colab.research.google.com/github/lawrennd/qig-code/blob/main/examples/multi_pair_regularisation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Multi-Pair Origin Regularisation

### Neil D. Lawrence

### December 2025

This tutorial demonstrates the regularisation machinery for multi-pair quantum systems
at the inaccessible game origin. We explore how different choices of the regularisation
matrix σ affect the physics and computational efficiency.

```python
# Auto-install QIG package if not available
import os

try:
    import qig
except ImportError:
    print("📦 Installing QIG package...")
    %pip install -q git+https://github.com/lawrennd/qig-code.git
    import qig
    print("✓ QIG package installed!")
```

## The North Pole Analogy

The Local Maximum Entropy (LME) origin—a product of Bell states—is like a
**coordinate singularity at the north pole** of a sphere:

- **Many meridians, one pole**: Just as infinitely many lines of longitude
  converge at the north pole, infinitely many distinct trajectories through
  state space converge at the LME origin.

- **Different σ = Different histories**: The regularisation matrix σ encodes
  the "direction of approach" to this singularity. Different choices of σ
  represent different physical histories that all share the same pure-state limit.

- **Isotropic σ = I/D is "boring"**: The maximally symmetric choice hides the
  rich structure of possible departure directions.

```python
import numpy as np
import matplotlib.pyplot as plt
from qig.exponential_family import QuantumExponentialFamily
from qig.pair_operators import product_of_bell_states, bell_state
```

## Setup: Multi-Pair Bell States

Let's create a system of multiple entangled pairs. For n pairs of qutrits (d=3),
the total Hilbert space dimension is D = (d²)ⁿ = 9ⁿ.

```python
# System parameters
d = 3  # Local dimension (qutrit)
n_pairs = 2  # Number of entangled pairs
D = d ** (2 * n_pairs)  # Total Hilbert space dimension

print(f"System: {n_pairs} qutrit pairs")
print(f"Local dimension: d = {d}")
print(f"Hilbert space dimension: D = {D}")

# Create the exponential family with pair basis
# n_pairs + pair_basis=True uses su(d²) operators acting on each pair
qef = QuantumExponentialFamily(n_pairs=n_pairs, d=d, pair_basis=True)

print(f"Number of natural parameters: {qef.n_params}")
```

## The Pure Origin: Product of Bell States

The LME origin is a product of maximally entangled Bell states:

```python
# The default Bell state |Φ₀⟩ = (1/√d) Σⱼ |jj⟩
psi_origin = product_of_bell_states(n_pairs, d)
rho_pure = np.outer(psi_origin, psi_origin.conj())

print(f"Origin state: product of {n_pairs} Bell states")
print(f"State vector shape: {psi_origin.shape}")
print(f"Density matrix shape: {rho_pure.shape}")
print(f"Purity: Tr(ρ²) = {np.trace(rho_pure @ rho_pure).real:.6f}")
```

## Regularisation: The Direction of Approach

To define natural parameters θ, we need to regularise the pure state:

$$\rho_\varepsilon = (1 - \varepsilon) |\Psi\rangle\langle\Psi| + \varepsilon \sigma$$

where σ is a valid density matrix (Hermitian, PSD, unit trace).

### Option 1: Isotropic σ = I/D (Default)

The maximally symmetric choice. Computationally optimal but physically "boring":

```python
# Isotropic regularisation (default)
epsilon = 1e-6
theta_isotropic = qef.get_bell_state_parameters(epsilon=epsilon)

print(f"Isotropic regularisation (σ = I/D)")
print(f"  ε = {epsilon:.0e}")
print(f"  ||θ|| = {np.linalg.norm(theta_isotropic):.4f}")
print(f"  θ range: [{theta_isotropic.min():.4f}, {theta_isotropic.max():.4f}]")

# Verify the density matrix
rho_from_theta = qef.rho_from_theta(theta_isotropic)
print(f"  Reconstructed purity: {np.trace(rho_from_theta @ rho_from_theta).real:.6f}")
```

### Option 2: Product σ = σ₁ ⊗ σ₂ ⊗ ... ⊗ σₙ (Efficient)

Each pair gets its own regularisation direction, but pairs remain uncorrelated.
This preserves O(n × d⁶) efficiency while allowing anisotropic regularisation.

```python
# Create anisotropic per-pair regularisation
# Each σ_k is a d² × d² density matrix for pair k
np.random.seed(42)

sigma_per_pair = []
for k in range(n_pairs):
    # Create a random PSD matrix
    A = np.random.randn(d**2, d**2) + 1j * np.random.randn(d**2, d**2)
    sigma_k = A @ A.conj().T
    sigma_k = sigma_k / np.trace(sigma_k)  # Normalise to unit trace
    sigma_per_pair.append(sigma_k)
    print(f"Pair {k+1} σ: trace = {np.trace(sigma_k).real:.6f}, "
          f"rank = {np.linalg.matrix_rank(sigma_k, tol=1e-10)}")

# Get natural parameters with product σ
theta_product = qef.get_bell_state_parameters(
    epsilon=epsilon,
    sigma_per_pair=sigma_per_pair
)

print(f"\nProduct regularisation (σ = σ₁⊗σ₂)")
print(f"  ||θ|| = {np.linalg.norm(theta_product):.4f}")
print(f"  θ range: [{theta_product.min():.4f}, {theta_product.max():.4f}]")

# Compare with isotropic
theta_diff = np.linalg.norm(theta_product - theta_isotropic)
print(f"  ||θ_product - θ_isotropic|| = {theta_diff:.4f}")
```

### Option 3: General σ (Full Flexibility, O(D³) Cost)

For studying correlated noise or entangled regularisation, you can provide
any valid D × D density matrix:

```python
# Create a general (potentially entangled) σ
A = np.random.randn(D, D) + 1j * np.random.randn(D, D)
sigma_general = A @ A.conj().T
sigma_general = sigma_general / np.trace(sigma_general)

# Check σ structure
structure = qef.detect_sigma_structure(sigma_general)
print(f"General σ structure: {structure}")

# Get natural parameters (will be slower for large D)
theta_general = qef.get_bell_state_parameters(
    epsilon=epsilon,
    sigma=sigma_general
)

print(f"\nGeneral regularisation")
print(f"  ||θ|| = {np.linalg.norm(theta_general):.4f}")
print(f"  θ range: [{theta_general.min():.4f}, {theta_general.max():.4f}]")
```

## Different Origins: Using bell_indices

The standard Bell state `|Φ₀⟩ = (1/√d) Σⱼ |jj⟩` is just one of d orthogonal
maximally entangled states. You can choose a different origin by specifying
which Bell state (k = 0, 1, ..., d-1) to use for each pair.

```python
# Different Bell states for a single pair
print("Bell states for d=3:")
for k in range(d):
    psi_k = bell_state(d, k)
    # Show the non-zero components
    nonzero = np.where(np.abs(psi_k) > 1e-10)[0]
    print(f"  |Φ_{k}⟩: non-zero at indices {nonzero}")

# Create a multi-pair state with different Bell states per pair
bell_indices = [0, 1]  # First pair: |Φ₀⟩, Second pair: |Φ₁⟩
psi_mixed = product_of_bell_states(n_pairs, d, bell_indices=bell_indices)

print(f"\nProduct state with bell_indices = {bell_indices}")
print(f"  State norm: {np.linalg.norm(psi_mixed):.6f}")

# Get parameters for this different origin
theta_mixed_origin = qef.get_bell_state_parameters(
    epsilon=epsilon,
    bell_indices=bell_indices
)

print(f"  ||θ|| = {np.linalg.norm(theta_mixed_origin):.4f}")
```

## Block-Diagonal Fisher Information

For product states with product or isotropic σ, the Fisher information
matrix is **block-diagonal**: pairs don't couple in the metric.

```python
# Compute Fisher information using both methods
G_full = qef.fisher_information(theta_isotropic)
G_block = qef.fisher_information_product(theta_isotropic)

print("Fisher Information Comparison")
print(f"  Full computation shape: {G_full.shape}")
print(f"  Block computation shape: {G_block.shape}")

# Check they match
diff = np.linalg.norm(G_full - G_block) / np.linalg.norm(G_full)
print(f"  Relative difference: {diff:.2e}")

# Visualise the block structure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im0 = axes[0].imshow(np.abs(G_full), cmap='viridis')
axes[0].set_title('Full Fisher Information |G|')
axes[0].set_xlabel('Parameter index')
axes[0].set_ylabel('Parameter index')
plt.colorbar(im0, ax=axes[0])

# Show the block-diagonal structure
im1 = axes[1].imshow(np.abs(G_block), cmap='viridis')
axes[1].set_title('Block-Diagonal Fisher Information |G|')
axes[1].set_xlabel('Parameter index')
axes[1].set_ylabel('Parameter index')
plt.colorbar(im1, ax=axes[1])

# Add lines to show block boundaries
n_params_per_pair = d**4 - 1
for ax in axes:
    for i in range(1, n_pairs):
        ax.axhline(i * n_params_per_pair - 0.5, color='red', linewidth=1, alpha=0.5)
        ax.axvline(i * n_params_per_pair - 0.5, color='red', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig('fisher_block_structure.png', dpi=150)
plt.show()
```

## Performance: O(n × d⁶) vs O(D³)

The efficiency gains become dramatic as n increases:

```python
import time

def benchmark_fisher(n_pairs, d, n_trials=3):
    """Benchmark Fisher information computation."""
    D = d ** (2 * n_pairs)
    qef = QuantumExponentialFamily(n_pairs=n_pairs, d=d, pair_basis=True)
    
    # Get reference theta
    theta = qef.get_bell_state_parameters(epsilon=1e-6)
    
    # Time full computation
    times_full = []
    for _ in range(n_trials):
        start = time.perf_counter()
        G_full = qef.fisher_information(theta)
        times_full.append(time.perf_counter() - start)
    
    # Time block computation
    times_block = []
    for _ in range(n_trials):
        start = time.perf_counter()
        G_block = qef.fisher_information_product(theta)
        times_block.append(time.perf_counter() - start)
    
    return {
        'n_pairs': n_pairs,
        'D': D,
        'n_params': qef.n_params,
        'time_full': np.median(times_full),
        'time_block': np.median(times_block),
        'speedup': np.median(times_full) / np.median(times_block)
    }

# Run benchmarks
print("Fisher Information Benchmarks (d=3 qutrits)")
print("-" * 60)
print(f"{'n_pairs':<8} {'D':<8} {'params':<8} {'Full (s)':<12} {'Block (s)':<12} {'Speedup':<8}")
print("-" * 60)

results = []
for n in [1, 2, 3]:  # Don't go higher without --slow flag
    result = benchmark_fisher(n, d=3, n_trials=3)
    results.append(result)
    print(f"{result['n_pairs']:<8} {result['D']:<8} {result['n_params']:<8} "
          f"{result['time_full']:<12.4f} {result['time_block']:<12.4f} "
          f"{result['speedup']:<8.1f}x")
```

## When to Use What

| σ Choice | Use Case | Efficiency |
|----------|----------|------------|
| `σ = I/D` (default) | Symmetric baseline, "boring" dynamics | **O(n × d⁶)** |
| `sigma_per_pair=[σ₁,...,σₙ]` | Independent per-pair directions | **O(n × d⁶)** |
| `sigma=⟨entangled⟩` | Correlated noise, built-in correlations | O(D³) |

**Key insight**: The choice of σ is a **physics decision**, not just computational:

- **Product σ** asks: "What happens if pairs depart the origin independently?"
  Correlations emerge through constraint dynamics.
  
- **Entangled σ** asks: "What if the perturbation itself couples pairs?"
  Correlations are built in from the start.

```python
# Summary visualisation: different θ directions for different σ
fig, ax = plt.subplots(figsize=(10, 6))

# Plot first few θ components for different σ choices
n_show = min(20, len(theta_isotropic))
x = np.arange(n_show)
width = 0.25

ax.bar(x - width, theta_isotropic[:n_show], width, label='Isotropic (I/D)', alpha=0.8)
ax.bar(x, theta_product[:n_show], width, label='Product (σ₁⊗σ₂)', alpha=0.8)
ax.bar(x + width, theta_general[:n_show], width, label='General σ', alpha=0.8)

ax.set_xlabel('Parameter index')
ax.set_ylabel('θ value')
ax.set_title('Natural Parameters for Different Regularisations')
ax.legend()
ax.axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('regularisation_comparison.png', dpi=150)
plt.show()
```

## Summary

The CIP-0008 machinery provides:

1. **Flexible regularisation**: Any valid σ, with structure detection
2. **Efficient paths**: O(n × d⁶) for isotropic and product σ
3. **Different origins**: `bell_indices` for exploring alternative pure states
4. **Block-diagonal Fisher**: `fisher_information_product` exploits product structure
5. **Physics interpretation**: σ encodes the "direction of approach" to the origin

The key trade-off is between computational efficiency (product σ) and the ability
to model pre-existing inter-pair correlations (entangled σ).
