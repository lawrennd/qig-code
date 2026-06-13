---
author: Neil Lawrence
created: '2026-06-13'
id: 000B
last_updated: '2026-06-13'
status: Implemented
tags:
- cip
- loewner-kernel
- kubo-mori
- fisher-information
- hamiltonian-emergence
- core-api
related_requirements:
- "0001"
- "0003"
title: Loewner Kernel and Hamiltonian Emergence Experiments
---

# CIP-000B: Loewner Kernel and Hamiltonian Emergence Experiments

## Status

- [x] Proposed
- [x] Accepted
- [x] Implemented
- [ ] Closed

## Summary

Expose the Loewner (Kubo-Mori / BKM) divided-difference kernel as a public API
function `qig.core.loewner_kernel`, and add a companion paper-experiment notebook
`examples/hamiltonian_emergence_experiments.ipynb` that validates the core
structural claims of the Gibbs-locked linearised dynamics from
`the-inaccessible-game-hamiltonian.tex`.

This CIP was implemented retroactively in commit `4a4a5cc` (2026-06-13).

## Motivation

The Loewner / Kubo-Mori kernel is the central object linking the exponential-family
geometry of quantum states to the BKM Fisher information metric and to the structure
of Kubo-Mori (Duhamel) derivatives. Until now it was computed internally inside
`qig/duhamel.py` (via eigendecomposition) but was not exposed as a standalone utility.

Making it public serves two purposes:

1. **Accessibility for analysis**: Other modules and notebooks that need the kernel
   directly (e.g. to compute the Loewner map as an operator, study its spectral
   properties, or compute the BKM Fisher metric) do not have to reimplement the
   divided-difference formula.

2. **Paper validation**: The Hamiltonian-emergence paper makes several structural
   claims about how the linearised dynamics near a Gibbs state decompose. A dedicated
   notebook that exercises these claims numerically both validates the theory and serves
   as a reproducible companion to the paper.

## Detailed Description

### The Loewner Kernel

For a background density matrix \(\rho_0\) with eigenvalues
\(\lambda_1, \ldots, \lambda_D\) and eigenvectors forming the columns of \(U\),
the Loewner (BKM / Kubo-Mori) kernel matrix is defined in the eigenbasis as:

$$C_{ij} = \begin{cases}
  \dfrac{\lambda_i - \lambda_j}{\log \lambda_i - \log \lambda_j} & \lambda_i \neq \lambda_j \\[6pt]
  \lambda_i & \lambda_i = \lambda_j
\end{cases}$$

The L'Hôpital limit gives the diagonal entries. For near-degenerate pairs
(\(|\lambda_i - \lambda_j| < \text{tol}\)) the arithmetic mean is used as a
numerically stable fallback.

The Loewner map acts on perturbations \(\delta K\) of the modular generator \(K\)
as:

$$\bigl(J_{\rho_0}(\delta K)\bigr)_{ij} = C_{ij} \cdot (\delta K)_{ij}$$

(in the eigenbasis of \(\rho_0\)), giving the first-order perturbation of \(\rho\).

**LME limit**: When all \(\lambda_i \to 1/D\), every entry
\(C_{ij} \to 1/D\), and the induced Fisher metric degenerates to \(D \cdot \mathrm{Id}\)
(flat geometry at the Lindblad-Master-Equation fixed point).

### API

```python
def loewner_kernel(
    rho0: np.ndarray, tol: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (C, vals, vecs) where C is the kernel matrix in the eigenbasis of rho0,
    vals are the eigenvalues, and vecs are the eigenvectors as columns.
    """
```

Exported from `qig.__init__` so it is accessible as `qig.loewner_kernel`.

### Hamiltonian Emergence Experiments Notebook

`examples/hamiltonian_emergence_experiments.ipynb` implements five numerical
experiments for the Gibbs-locked linearised dynamics:

| Experiment | Claim validated |
|---|---|
| **Exp 1 – Iso-marginal tangency** | Matched-index off-diagonal modes leak into marginals; doubly off-diagonal modes do not |
| **Exp 2 – Mode decoupling** | \([J_{\rho_0}, i[K_0, \cdot]] = 0\) in the eigenbasis (machine zero); nonzero after local unitary rotation |
| **Exp 3 – Frame covariance** | Bohr gap spectra are invariant under \(U = U_A \otimes U_B\); degeneracy in \(H_A\) exposes frame ambiguity |
| **Exp 4 – Loewner → Fisher bridge** | Kernel eigenvalue fan collapses to \(1/D\) as \(\beta\delta \to 0\) (LME limit), confirming flat Fisher limit |
| **Exp 5 – \(\mu_0\) as resolution floor** | Two-gap Fisher information shows \(\Delta\omega_\text{min} \sim \mu_0\); relative floor diverges near LME |

## Implementation Plan

(Retrospective — all steps completed in commit `4a4a5cc`, 2026-06-13.)

1. **Add `loewner_kernel` to `qig/core.py`** ✅
   - Implement the divided-difference formula with near-degeneracy fallback
   - Full docstring explaining the mathematical definition, LME limit, and usage pattern
   - Add to `__all__`

2. **Export from `qig/__init__.py`** ✅
   - Make accessible as `qig.loewner_kernel`

3. **Test suite** ✅
   - Add `TestLoewnerKernel` to `tests/test_core_utilities.py`
   - Tests: shape, symmetry, diagonal entries (L'Hôpital), divided-difference formula,
     LME limit, near-degeneracy fallback, positive semi-definiteness, Loewner map application

4. **Companion notebook** ✅
   - `examples/hamiltonian_emergence_experiments.ipynb` (39 cells)
   - Five experiments validating Gibbs-locked linearised dynamics claims

## Backward Compatibility

Purely additive: `loewner_kernel` is a new public function. No existing APIs changed.

## Testing Strategy

- **Unit tests** in `tests/test_core_utilities.py` (`TestLoewnerKernel` suite, 225 lines):
  - Shapes and data types
  - Symmetry: \(C_{ij} = C_{ji}\)
  - Diagonal entries equal eigenvalues
  - Divided-difference formula matches direct computation
  - LME limit: all entries → \(1/D\)
  - Near-degeneracy: arithmetic-mean fallback triggered when \(|\lambda_i - \lambda_j| < \text{tol}\)
  - Positive semi-definiteness of \(C\)
  - Loewner map application returns correct shape and value

## Related Requirements

- **REQ-0001** (entangled systems): The notebook exercises two-qubit and qutrit-pair
  Gibbs states, directly validating entangled-system structural claims.
- **REQ-0003** (numerical stability and transparency): The near-degeneracy fallback
  and the LME-limit test directly address numerical transparency requirements.

## Implementation Status

- [x] Add `loewner_kernel` to `qig/core.py` with full docstring (commit `4a4a5cc`)
- [x] Export from `qig/__init__.py` (commit `4a4a5cc`)
- [x] `TestLoewnerKernel` suite in `tests/test_core_utilities.py` (commit `4a4a5cc`)
- [x] `examples/hamiltonian_emergence_experiments.ipynb` — five experiments (commit `4a4a5cc`)

## References

- **Code**: `qig/core.py` — `loewner_kernel` function
- **Notebook**: `examples/hamiltonian_emergence_experiments.ipynb`
- **Tests**: `tests/test_core_utilities.py` — `TestLoewnerKernel`
- **Paper**: `the-inaccessible-game-hamiltonian.tex` — Gibbs-locked linearised dynamics
- **Related CIPs**:
  - CIP-000A: Block-matrix Fréchet derivatives (uses the same kernel internally)
  - CIP-0009: Hamiltonian extraction from antisymmetric GENERIC flow
