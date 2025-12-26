# Duhamel Derivative Methods: Comparison and Selection Guide

This document provides a detailed comparison of the methods available in `qig` for computing Kubo-Mori (Duhamel) derivatives of quantum exponential families.

## The Mathematical Problem

For a quantum exponential family with density matrix:

$$\rho(\theta) = \exp\left(\sum_a \theta_a F_a - \psi(\theta) I\right)$$

we need to compute derivatives:

$$\frac{\partial \rho}{\partial \theta_a} = \int_0^1 e^{(1-s)H} (F_a - \langle F_a \rangle I) e^{sH} \, ds$$

where $H = \sum_a \theta_a F_a - \psi(\theta) I$.

This **Duhamel integral** (also called the Wilcox formula or Dalecki-Krein exponential formula) is the exact derivative that respects operator ordering and preserves Hermiticity.

## Available Methods

### Method 1: Quadrature (`method='duhamel'`)

**Approach:** Directly evaluate the integral using trapezoid or Simpson's rule.

**Implementation:**
```python
# qig/duhamel.py: duhamel_derivative()
s_vals = np.linspace(0, 1, n_points)  # Default: 50 points
for s in s_vals:
    integrand = expm(s * H) @ F_centered @ expm((1-s) * H)
    drho += weight * integrand
```

**Pros:**
- Simple and transparent
- Works for any Hermitian $H$
- Accuracy controlled by number of quadrature points
- Useful as reference implementation

**Cons:**
- Expensive: 50 matrix exponentials for default accuracy (~1e-5)
- Still only achieves moderate accuracy
- Inefficient for repeated evaluations

**Use case:** Validation and cross-checking other methods

**Cost:** $O(50 \cdot n^3)$ for 50-point quadrature

---

### Method 2: Spectral (`method='duhamel_spectral'` or `'duhamel_bch'`)

**Approach:** Use eigendecomposition of $H$ to evaluate the Fréchet derivative analytically.

**Mathematical foundation:**

In the eigenbasis of $H = U \text{diag}(\lambda) U^\dagger$, the Fréchet derivative has closed form:

$$(D\exp_H[F])_{ij} = \begin{cases}
e^{\lambda_i} F_{ii}, & i = j \\
F_{ij} \frac{e^{\lambda_i} - e^{\lambda_j}}{\lambda_i - \lambda_j}, & i \neq j
\end{cases}$$

This can be written as a matrix function $K_\rho = f(\text{ad}_H)$ where $f(z) = (e^z - 1)/z$.

**Implementation:**
```python
# qig/duhamel.py: duhamel_derivative_spectral()
evals, U = eigh(H)  # Diagonalize H
X_tilde = U.T @ F_centered @ U  # Transform to eigenbasis
K = (np.exp(lam_i) - np.exp(lam_j)) / (lam_i - lam_j)  # Kernel
d_rho_tilde = K * X_tilde
drho = U @ d_rho_tilde @ U.T  # Transform back
```

**Pros:**
- Exact (up to diagonalization error, ~machine precision)
- Avoids explicit numerical integration
- Aligns with Lie-algebraic/BCH structure in theory
- Fast for small to medium systems

**Cons:**
- Requires eigendecomposition (can be expensive for large $n$)
- May be unstable for ill-conditioned eigenvector matrices
- Sensitive to near-degenerate eigenvalues

**Use case:** Default method for well-conditioned small to medium systems

**Cost:** $O(n^3)$ for eigendecomposition + $O(n^2)$ for kernel application

---

### Method 3: Block-Matrix (`method='duhamel_block'`) ✨ **NEW**

**Approach:** Use Higham's block-matrix identity to compute the Fréchet derivative via a single $2n \times 2n$ matrix exponential.

**The Insight:**

The Duhamel integral *looks* like it requires numerical quadrature. But Higham's block-matrix identity "compiles away" the integral:

$$\exp\begin{pmatrix} H & F \\ 0 & H \end{pmatrix} = \begin{pmatrix} e^H & \int_0^1 e^{(1-s)H} F e^{sH} ds \\ 0 & e^H \end{pmatrix}$$

The (1,2) block **is** the Fréchet derivative!

**Implementation:**
```python
# qig/duhamel.py: duhamel_derivative_block()
n = H.shape[0]
block = np.block([[H, F_centered],
                  [np.zeros((n,n)), H]])
exp_block = expm(block)
drho = exp_block[:n, n:]  # Extract (1,2) block
```

**Pros:**
- Single well-tested routine: one call to `expm` (Padé + scaling & squaring)
- No eigendecomposition: more robust for ill-conditioned $H$
- Machine precision accuracy
- Elegant: the integral form is useful for *analysis*, but the algebraic identity is better for *computation*
- Extensible: can compute higher-order Fréchet derivatives with larger blocks

**Cons:**
- Requires $2n \times 2n$ exponential (vs $n \times n$ eigendecomposition)
- Less efficient than spectral when eigendecomposition is cheap and well-conditioned
- Cost grows with matrix size

**Use case:** 
- Ill-conditioned $H$ where eigendecomposition is unstable
- When you want maximum numerical robustness
- Educational/demonstration purposes (elegant formulation)
- Small to medium systems ($n \leq 100$)

**Cost:** $O((2n)^3) = O(8n^3)$ for $2n \times 2n$ matrix exponential

---

### Method 4: SLD (`method='sld'`)

**Approach:** Symmetric Logarithmic Derivative - two-point trapezoid approximation.

**Implementation:**
```python
# Equivalent to n_points=2 in quadrature method
drho = 0.5 * (F_centered @ rho + rho @ F_centered)
```

**Pros:**
- Very fast (no matrix exponentials!)
- Simple closed form
- Useful for quick approximate gradients

**Cons:**
- Lower accuracy (~1e-3 typical error)
- Only valid as approximation

**Use case:** Fast approximate gradients when high accuracy not needed

**Cost:** $O(n^2)$

---

## Comparison Table

| Method | Cost | Accuracy | Robustness | Memory | Best For |
|--------|------|----------|------------|--------|----------|
| **Quadrature** | $50 \times O(n^3)$ | ~10⁻⁵ | High | $O(n^2)$ | Validation |
| **Spectral** | $O(n^3)$ | Machine ε | Moderate | $O(n^2)$ | Well-conditioned, default |
| **Block** | $O(8n^3)$ | Machine ε | **High** | $O(4n^2)$ | **Ill-conditioned** |
| **SLD** | $O(n^2)$ | ~10⁻³ | High | $O(n^2)$ | Fast approximation |

## When to Use Each Method

### Use Spectral When:
- $H$ is well-conditioned (eigenvalues well-separated)
- You need machine-precision accuracy
- You're computing many derivatives (can reuse eigendecomposition)
- $n$ is small to medium (up to ~1000)
- You want to align with BCH/Lie-algebraic theory

### Use Block-Matrix When:
- $H$ is ill-conditioned (near-degenerate eigenvalues)
- Eigendecomposition is numerically unstable
- You want maximum numerical robustness
- $n$ is small ($n \leq 100$)
- You're computing higher-order derivatives (Hessian, 3rd cumulants)
- You want a single, well-tested computational routine

### Use Quadrature When:
- You need a reference implementation for validation
- You're debugging other methods
- You want to control accuracy via number of points
- Computational cost is not a concern

### Use SLD When:
- You only need approximate gradients
- Speed is critical and accuracy can be sacrificed
- You're doing optimization where exact gradients aren't required

## Numerical Validation

All methods are cross-validated in the test suite:

- `tests/test_pair_exponential_family.py::TestRhoDerivativeNumerical`
  - Tests quadrature and spectral methods
  - Validates against finite differences
  - Confirms convergence properties

- `tests/test_block_frechet.py::TestBlockFrechet`
  - Tests block-matrix method
  - Cross-validates block vs spectral (should agree to ~1e-12)
  - Tests higher-order Fréchet derivatives

## Example Usage

```python
from qig.exponential_family import QuantumExponentialFamily
import numpy as np

# Create a single-qubit system
exp_fam = QuantumExponentialFamily(n_sites=1, d=2)
theta = np.array([0.3, 0.5, 0.2])

# Compute derivative using different methods
drho_spectral = exp_fam.rho_derivative(theta, a=0, method='duhamel_spectral')
drho_block = exp_fam.rho_derivative(theta, a=0, method='duhamel_block')
drho_quadrature = exp_fam.rho_derivative(theta, a=0, method='duhamel', n_points=100)
drho_sld = exp_fam.rho_derivative(theta, a=0, method='sld')

# They should all agree (within their accuracy)
print("Spectral vs Block:", np.max(np.abs(drho_spectral - drho_block)))
# Output: ~1e-14 (machine precision)
```

## References

### Nick Higham's Work on Block-Matrix Method

- **Higham, N. J. (2008).** *Functions of Matrices: Theory and Computation.* SIAM.
  - Chapter 10: The Fréchet Derivative
  - Section 10.4: The Exponential Function

- **Al-Mohy, A. H., & Higham, N. J. (2009).** Computing the Fréchet derivative of the matrix exponential, with an application to condition number estimation. *SIAM Journal on Matrix Analysis and Applications*, 30(4), 1639-1657.

- **Higham, N. J., & Al-Mohy, A. H. (2010).** Computing matrix functions. *Acta Numerica*, 19, 159-208.

### Implementation Details

- **SciPy `scipy.linalg.expm`**: Uses Al-Mohy & Higham's (2009) Padé approximation with scaling and squaring
- **SciPy `scipy.linalg.eigh`**: LAPACK-based eigendecomposition for Hermitian matrices

### Related Documentation

- **CIP-000A**: Block-Matrix Method for Fréchet Derivatives (implementation plan)
- **CIP-0009**: Hamiltonian Extraction (uses Kubo-Mori derivatives)
- **Backlog 2025-12-08**: BCH-Duhamel Implementation (led to spectral method)

## Performance Considerations

### Memory Usage

- **Spectral**: Stores eigenvectors ($n \times n$) + working arrays
- **Block**: Requires $2n \times 2n$ block matrix (4× memory)
- **Quadrature**: Minimal extra memory (stores intermediate exponentials)

### Parallelization

- **Spectral**: Eigendecomposition can use BLAS/LAPACK parallelism
- **Block**: `expm` can use parallel linear algebra
- **Quadrature**: Loop over $s$ values can be parallelized

### Cache Efficiency

For small $n$ (qubits, qutrits), all methods fit in L1/L2 cache. For large $n$:
- **Spectral**: Good locality after eigendecomposition
- **Block**: $2n \times 2n$ matrix may stress cache
- **Quadrature**: Each exponential has good locality

## Future Work

- **Benchmark suite**: Systematic comparison across matrix sizes (CIP-000A Phase 2)
- **Multi-directional block**: Compute all $n$ derivatives simultaneously
- **Condition number estimation**: Use block method for sensitivity analysis
- **GPU acceleration**: Offload large matrix exponentials to GPU

