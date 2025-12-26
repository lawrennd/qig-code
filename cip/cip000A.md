---
author: "Neil Lawrence"
created: "2025-12-20"
id: "000A"
last_updated: "2025-12-26"
status: implemented
tags:
- cip
- duhamel
- frechet-derivative
- matrix-exponential
- numerical-methods
- higham
title: "Block-Matrix Method for Fréchet Derivatives (Higham's Trick)"
---

# CIP-000A: Block-Matrix Method for Fréchet Derivatives (Higham's Trick)

## Summary

Implement Nick Higham's block-matrix identity for computing Fréchet derivatives of matrix exponentials, providing an elegant alternative to the current spectral method. This allows computing the Duhamel integral

$$\frac{\partial \rho}{\partial \theta_i} = \int_0^1 e^{(1-s)K} F_i \, e^{sK} \, ds$$

via a single $2n \times 2n$ matrix exponential instead of eigendecomposition + kernel application.

This CIP is now implemented in `qig/duhamel.py` and wired into
`QuantumExponentialFamily.rho_derivative(method='duhamel_block')`.

We also extend the same block idea to compute **2nd and 3rd Fréchet derivatives**
of `expm` (3×3 and 4×4 block matrices) and use those to compute:
- the **Hessian of** \( \psi(\theta)=\log \mathrm{tr}\,e^{K(\theta)} \) (2nd cumulant / BKM Fisher metric),
- the **3rd cumulant contraction** \( (\nabla G)[\theta] \) on small systems (validation tool).

## Motivation

### The Current Situation

The codebase currently has three methods for computing Kubo-Mori derivatives:

1. **Quadrature** (`method='duhamel'`): Trapezoid rule with 50 points
   - Pros: Simple, works for any H
   - Cons: Expensive (50 matrix exponentials), moderate accuracy (~1e-5)

2. **Spectral** (`method='duhamel_spectral'`): Eigendecomposition with closed-form kernel
   - Pros: Exact (up to diagonalization error), avoids numerical integration
   - Cons: Requires eigendecomposition, kernel application, basis transforms

3. **SLD** (`method='sld'`): Two-point trapezoid approximation
   - Pros: Fast (only 2 matrix exponentials)
   - Cons: Lower accuracy

### The Insight from Nick Higham

When you write the Duhamel formula as an integral:

$$\frac{\partial}{\partial \theta_i} e^{K} = \int_0^1 e^{(1-s)K} F_i \, e^{sK} \, ds$$

it *looks* like you need numerical quadrature—potentially expensive and inaccurate, especially when $K$ has eigenvalues with large imaginary parts where the integrand oscillates.

**Higham's block-matrix identity sidesteps this entirely.** Form the $2n \times 2n$ matrix:

$$\begin{pmatrix} K & F_i \\ 0 & K \end{pmatrix}$$

and compute a **single** matrix exponential. The $(1,2)$ block of the result **is** the Fréchet derivative—the integral has been "compiled away" into the exponential itself.

### Why This Matters for QIG

For quantum exponential families, computing the gradient of $\psi(\theta) = \log \mathrm{tr}\, e^{K(\theta)}$ requires Fréchet derivatives for each component $\theta_i$. 

**Current cost:**
- Spectral: One eigendecomposition + $n$ kernel applications + transforms
- Quadrature: $50n$ matrix exponentials

**Block-matrix cost:**
- $n$ calls to $2n \times 2n$ `expm` (or one $(n+1) \times n$ block matrix for all directions)

**Key advantages:**
1. **Numerical stability**: Leverages highly-optimized `expm` (Padé + scaling & squaring) instead of explicit eigendecomposition
2. **Robustness**: No issues with near-degenerate eigenvalues or ill-conditioned eigenvector matrices
3. **Elegance**: The integral form is useful for *analysis*, but the block trick is better for *computation*
4. **Extensibility**: Can stack multiple directions simultaneously

This is exactly the kind of insight Nick Higham valued: recognizing that a problem which looks like it requires numerical integration can be recast as a single matrix computation with known, controllable error.

## Detailed Description

### Mathematical Foundation

The Fréchet derivative of the matrix exponential at $K$ in direction $F$ is:

$$D\exp_K[F] = \int_0^1 e^{(1-s)K} F e^{sK} \, ds$$

**Higham's block-matrix identity** states:

$$\exp\begin{pmatrix} K & F \\ 0 & K \end{pmatrix} = \begin{pmatrix} e^K & D\exp_K[F] \\ 0 & e^K \end{pmatrix}$$

**Proof sketch:** Let $M(t) = \exp(t \begin{pmatrix} K & F \\ 0 & K \end{pmatrix})$. Then:
- $M(0) = I$
- $M'(t) = \begin{pmatrix} K & F \\ 0 & K \end{pmatrix} M(t)$

Writing $M(t) = \begin{pmatrix} A(t) & B(t) \\ 0 & C(t) \end{pmatrix}$, we get:
- $A'(t) = K A(t)$, so $A(t) = e^{tK}$
- $C'(t) = K C(t)$, so $C(t) = e^{tK}$
- $B'(t) = K B(t) + F A(t) = K B(t) + F e^{tK}$

The last equation is solved by $B(t) = \int_0^t e^{(t-s)K} F e^{sK} ds$, giving $B(1) = D\exp_K[F]$.

### Multiple Directions

For the full gradient, we can stack directions:

$$\exp \begin{pmatrix} K & F_1 & F_2 & \cdots & F_n \\ 0 & K & 0 & \cdots & 0 \\ 0 & 0 & K & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & K \end{pmatrix}$$

The first row's off-diagonal blocks give all Fréchet derivatives simultaneously. However, this is $(n+1) \times n$ block matrix (dimension $(n+1)n \times (n+1)n$), which becomes expensive for large $n$.

**Trade-off analysis needed**: When is one big block exponential better than $n$ separate $2n \times 2n$ exponentials?

### Implementation Approach

Add a new method `duhamel_derivative_block` to `qig/duhamel.py`:

```python
def duhamel_derivative_block(
    rho: np.ndarray,
    H: np.ndarray,
    F_centered: np.ndarray,
) -> np.ndarray:
    """
    Compute ∂ρ/∂θ using Higham's block-matrix identity.
    
    Forms the 2n×2n matrix [[H, F_centered], [0, H]] and computes
    a single matrix exponential. The (1,2) block is the Fréchet
    derivative, equivalent to the Duhamel integral.
    
    This is Nick Higham's elegant trick: the integral is "compiled
    away" into the exponential itself, giving the same result as
    duhamel_derivative_spectral but via a different computational path.
    
    References
    ----------
    - Higham, N. J. (2008). Functions of Matrices: Theory and Computation.
      SIAM. Chapter 10.
    - Al-Mohy, A. H., & Higham, N. J. (2009). Computing the Fréchet 
      derivative of the matrix exponential, with an application to 
      condition number estimation. SIAM J. Matrix Anal. Appl., 30(4), 
      1639-1657.
    """
    n = H.shape[0]
    
    # Build 2n × 2n block matrix
    block = np.block([
        [H, F_centered],
        [np.zeros((n, n)), H]
    ])
    
    # Single matrix exponential
    exp_block = expm(block)
    
    # Extract (1,2) block - this is the Fréchet derivative
    drho = exp_block[:n, n:]
    
    return drho
```

### Comparison with Existing Methods

| Method | Cost | Accuracy | Robustness | Use Case |
|--------|------|----------|------------|----------|
| **Quadrature** | $50n$ `expm` calls | ~1e-5 | High | Reference/validation |
| **Spectral** | 1 `eigh` + kernel | Machine precision | Moderate (eigendecomp) | Current default |
| **Block** | 1 `expm` ($2n \times 2n$) | Machine precision | High (no eigendecomp) | **Proposed** |
| **SLD** | 2 `expm` calls | ~1e-3 | High | Fast approximation |

**When to use block method:**
- Small to medium $n$ (qubits, qutrits: $n \leq 100$)
- Ill-conditioned $H$ where eigendecomposition is unstable
- When you want a single, well-tested numerical routine (`expm`)
- Educational/demonstration purposes (elegant formulation)

**When spectral is better:**
- Very large $n$ where $2n \times 2n$ exponential is expensive
- When eigendecomposition is already available
- When computing many derivatives (can reuse eigendecomposition)

## Implementation Plan

### Phase 1: Core Implementation

1. **Add `duhamel_derivative_block` to `qig/duhamel.py`** ✅
   - Implement the $2n \times 2n$ block-matrix method
   - Include comprehensive docstring with mathematical explanation
   - Add references to Higham's work

2. **Wire into `QuantumExponentialFamily.rho_derivative`** ✅
   - Add `method='duhamel_block'` option
   - Ensure consistent API with existing methods

3. **Unit tests** ✅
   - Added `tests/test_block_frechet.py`
   - Cross-validates `duhamel_block` vs `duhamel_spectral`
   - Validates Hessian(ψ) via block-2 matches `fisher_information()` (single qubit)
   - Validates 3rd cumulant contraction via block-3 matches FD on small systems

### Phase 2: Performance Analysis

4. **Benchmark suite**
   - Compare wall-clock time: block vs spectral vs quadrature
   - Test various matrix sizes: $n = 4, 9, 16, 25, 64, 100$
   - Measure accuracy vs finite differences
   - Test conditioning: well-conditioned vs near-degenerate eigenvalues

5. **Documentation updates**
   - Update `docs/source/api/duhamel.rst` with block-matrix method description
   - Add comparison table showing when to use each method
   - Create `docs/duhamel_methods_comparison.md` with detailed analysis
   - Add references to Higham's work in theory docs
   - Update user guide with method selection guidance

### Phase 3: Extensions (Optional)

6. **Multi-directional block method**
   - Implement stacked version for computing all $n$ derivatives at once
   - Benchmark: one $(n+1)n \times (n+1)n$ exponential vs $n$ separate $2n \times 2n$
   - Determine crossover point

8. **Higher-order cumulants via higher-order Fréchet derivatives** ✅ (partial)
   - Implemented 2nd and 3rd Fréchet derivatives of `expm` via 3×3 / 4×4 block matrices
   - Exposed as validation utilities for Hessian(ψ) and 3rd cumulant contraction on small systems

7. **Condition number estimation**
   - Implement Higham's condition number estimator for $\exp(K)$
   - Use block method to estimate sensitivity of $\psi(\theta)$ to perturbations
   - Add diagnostic warnings when derivatives are ill-conditioned

## Backward Compatibility

**No breaking changes:**
- New method is purely additive (`method='duhamel_block'`)
- All existing methods continue to work
- Default behavior unchanged (currently `method='duhamel'`)

**Future consideration:**
- After validation, might make `'duhamel_block'` or `'duhamel_spectral'` the new default
- Would require deprecation cycle for `'duhamel'` (quadrature) as default

## Testing Strategy

### Unit Tests

1. **Correctness tests** (`tests/test_duhamel.py`):
   ```python
   def test_block_vs_spectral():
       """Block and spectral methods should agree to machine precision."""
       # Test on qubit, qutrit, 2-qubit systems
       # Compare ||block - spectral|| < 1e-12
   
   def test_block_vs_finite_differences():
       """Block method should match finite differences."""
       # Use eps=1e-8, expect error ~1e-8
   
   def test_block_hermiticity():
       """Block method should preserve Hermiticity exactly."""
       # Check ||drho - drho†|| < 1e-14
   
   def test_block_ill_conditioned():
       """Block method should handle near-degenerate eigenvalues."""
       # Create H with eigenvalues differing by ~1e-10
       # Verify block method still accurate when spectral might struggle
   ```

2. **Integration tests** (`tests/test_exponential_family.py`):
   ```python
   def test_gradient_with_block_method():
       """Full gradient computation using block method."""
       # Compute ∇ψ(θ) using block method
       # Compare with existing methods
   ```

3. **Regression tests** (`tests/test_pair_exponential_family.py`):
   ```python
   def test_all_derivative_methods_agree():
       """All four methods should give consistent results."""
       # quadrature, spectral, block, sld
       # Check pairwise agreement within expected tolerances
   ```

### Performance Tests

4. **Benchmark suite** (`tests/benchmark_duhamel.py`):
   ```python
   @pytest.mark.benchmark
   def test_benchmark_derivative_methods():
       """Compare wall-clock time for different methods."""
       # Sizes: n = 4, 9, 16, 25, 64, 100
       # Report: time, accuracy, memory
   ```

### Documentation Tests

5. **Example notebook** (`examples/duhamel_methods_comparison.ipynb`):
   - Visual comparison of all methods
   - Convergence plots
   - Performance vs accuracy trade-offs
   - When to use which method

## Related Requirements

This CIP enhances the computational infrastructure for:

- **CIP-0009**: Hamiltonian extraction (uses Kubo-Mori derivatives)
- **CIP-0006**: GENERIC decomposition (relies on accurate derivatives)
- **Backlog task 2025-12-08**: BCH-Duhamel implementation (provides alternative to spectral method)

Specifically, it provides:

1. **Numerical robustness**: Alternative when eigendecomposition is ill-conditioned
2. **Pedagogical clarity**: Elegant formulation connecting integrals to matrix functions
3. **Performance options**: Users can choose best method for their problem size and conditioning
4. **Validation**: Independent implementation for cross-checking spectral method

## Implementation Status

- [ ] Phase 1: Core implementation
  - [ ] Add `duhamel_derivative_block` to `qig/duhamel.py`
  - [ ] Wire into `QuantumExponentialFamily.rho_derivative` with `method='duhamel_block'`
  - [ ] Unit tests: correctness vs finite differences
  - [ ] Unit tests: cross-validation with spectral method
  - [ ] Unit tests: Hermiticity preservation
  - [ ] Unit tests: ill-conditioned matrices
- [ ] Phase 2: Performance analysis and documentation
  - [ ] Benchmark suite comparing all methods across matrix sizes
  - [ ] Update `docs/source/api/duhamel.rst` with block-matrix method
  - [ ] Create `docs/duhamel_methods_comparison.md` with detailed comparison
  - [ ] Add method selection guidance to user guide
  - [ ] Add performance comparison table to documentation
- [ ] Phase 3: Extensions (optional)
  - [ ] Multi-directional block method (all derivatives at once)
  - [ ] Condition number estimation using block method
  - [ ] Example notebook demonstrating all methods

## References

### Nick Higham's Work

- **Higham, N. J. (2008).** *Functions of Matrices: Theory and Computation.* SIAM.
  - Chapter 10: The Fréchet Derivative
  - Section 10.4: The Exponential Function

- **Al-Mohy, A. H., & Higham, N. J. (2009).** Computing the Fréchet derivative of the matrix exponential, with an application to condition number estimation. *SIAM Journal on Matrix Analysis and Applications*, 30(4), 1639-1657.
  - Describes the block-matrix method in detail
  - Includes condition number estimation algorithms

- **Higham, N. J., & Al-Mohy, A. H. (2010).** Computing matrix functions. *Acta Numerica*, 19, 159-208.
  - Survey of matrix function computation
  - Section on Fréchet derivatives

### Implementation References

- **SciPy `expm`**: Uses Padé approximation with scaling and squaring (based on Al-Mohy & Higham, 2009)
- **MATLAB `expm`**: Reference implementation of Higham's algorithms

### Existing Code

- `qig/duhamel.py`: Current quadrature and spectral implementations
- `qig/exponential_family.py`: `rho_derivative` method that will be extended
- `tests/test_pair_exponential_family.py`: Existing derivative validation tests
- `docs/source/api/duhamel.rst`: API documentation to be updated

### Related CIPs

- **CIP-0009**: Hamiltonian extraction from antisymmetric flow
- **CIP-0006**: GENERIC decomposition framework
- **Backlog 2025-12-08**: BCH-Duhamel implementation and validation

