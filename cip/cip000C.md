---
author: Neil Lawrence
created: '2026-06-13'
id: 000C
last_updated: '2026-06-13'
status: Implemented
tags:
- cip
- gibbs-lock
- hamiltonian-emergence
- loewner-kernel
- bohr-gaps
- mu0-inference
- core-api
related_requirements:
- "0001"
- "0003"
title: Gibbs-Lock API and mu_0 Inference
---

# CIP-000C: Gibbs-Lock API and mu_0 Inference

## Status

- [x] Proposed
- [x] Accepted
- [x] Implemented
- [ ] Closed

## Summary

Create a named `GibbsLockedFrame` class in a new `qig/gibbs_lock.py` module that
encapsulates the Gibbs-locked background $K_0 = \beta H$, exposes its spectral
geometry (Bohr gaps, Loewner kernel, iso-marginal test), and provides an
`infer_mu0` function that fits the uniform decay rate $\mu_0$ from a simulated
trajectory. This gives both papers a concrete, testable API for the central object
around which their analysis is organised.

## Motivation

The Hamiltonian paper (`the-inaccessible-game-hamiltonian.tex`) and the origin
paper (`the-inaccessible-game-origin.tex`) both organise their analysis around
Gibbs-locked frames, yet this concept has no named representation in the codebase.
Specific gaps:

1. **No `GibbsLockedFrame` class.** The combination $K_0 = \beta H$ (Gibbs state,
   eigenbasis, Bohr gaps) is assembled ad hoc in each notebook.

2. **$\mu_0$ is a free parameter.** The uniform decay rate $\mu_0$ that appears in
   $(\dot{\delta\rho})_{ij} = (i\beta\Delta\epsilon_{ij} - \mu_0)\delta\rho_{ij}$
   has no inference routine connecting it to the constrained dynamics solver in
   `qig/dynamics.py`.

3. **Iso-marginal test is implicit.** There is no function that checks whether a
   perturbation $\delta K$ is tangent to the iso-marginal surface, even though
   this is the key class of perturbations studied in the Hamiltonian paper.

CIP-0007 is closed as superseded; its `qig/symbolic/lme_exact.py` provides exact
LME eigenvalues and constraint gradients that will serve as validation targets for
the new module's tests.

## Detailed Description

### `qig/gibbs_lock.py` — new module

```python
class GibbsLockedFrame:
    """
    Encapsulates a Gibbs-locked background K_0 = beta * H.

    A Gibbs-locked frame is a fixed point of the linearised reversible dynamics:
    because [K_0, H] = 0 by construction, the commutator term i[delta_K, K_0]
    generates pure phase rotation in the eigenbasis of H without shifting K_0.

    Parameters
    ----------
    H : array, shape (D, D)
        Traceless Hermitian generator.
    beta : float
        Inverse temperature / coupling strength.
    dims : list of int, optional
        Subsystem dimensions for partial-trace operations (default: treat as
        single system with dims=[D]).
    """

    def __init__(self, H, beta, dims=None): ...

    @property
    def rho0(self) -> np.ndarray:
        """Background Gibbs state rho0 = exp(-beta*H) / Z."""

    def bohr_gaps(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (eigenvalues, gap_matrix) where gap_matrix[i,j] = eps_i - eps_j.
        Eigenvalues are those of H sorted ascending.
        """

    def loewner_kernel(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loewner divided-difference kernel at rho0.
        Thin wrapper around qig.core.loewner_kernel(self.rho0).
        Returns (C, vals, vecs).
        """

    def is_iso_marginal(self, delta_K: np.ndarray, tol: float = 1e-10) -> bool:
        """
        True if delta_K is tangent to the iso-marginal surface to first order.
        Checks tr_{!=k}(J_{rho0}(delta_K)) < tol for every subsystem k.
        Requires dims to be set.
        """

    def linearised_flow(self, delta_K: np.ndarray, mu0: float, t: float
                        ) -> np.ndarray:
        """
        Analytical element-wise solution of the linearised GENERIC equation:
            d/dt delta_K_{ij} = (i * beta * Delta_eps_{ij} - mu0) * delta_K_{ij}
        Returns delta_K evolved to time t.
        """
```

```python
def infer_mu0(
    trajectory: dict,
    frame: GibbsLockedFrame,
    tol: float = 1e-3,
) -> float:
    """
    Estimate mu_0 from an off-diagonal decay trajectory.

    Fits exp(-mu0 * t) to the magnitude of off-diagonal elements of
    delta_rho(t) = rho(t) - rho0, averaging over all off-diagonal (i,j)
    that are above `tol` at t=0.  The oscillatory phase factor is removed
    before fitting.

    Parameters
    ----------
    trajectory : dict
        Output of InaccessibleGameDynamics.solve() or solve_constrained_maxent().
        Must contain keys 'times' and 'rho_trajectory'.
    frame : GibbsLockedFrame
        The background Gibbs-locked frame.

    Returns
    -------
    mu0 : float
        Fitted uniform decay rate.
    """
```

### Validation strategy

Tests in `tests/test_gibbs_lock.py` will verify:

- `GibbsLockedFrame.rho0` matches `expm(-beta * H) / trace(expm(-beta * H))`.
- `bohr_gaps()` returns gaps consistent with `H`'s eigenvalues.
- `loewner_kernel()` output agrees with `qig.core.loewner_kernel(frame.rho0)`.
- `is_iso_marginal` correctly identifies doubly off-diagonal modes as iso-marginal
  and matched-index modes as non-iso-marginal (reproducing Experiment 1 of
  `hamiltonian_emergence_experiments.ipynb`).
- `linearised_flow` matches a direct numerical integration via `scipy.linalg.expm`
  applied element-wise.
- `infer_mu0` recovers a known $\mu_0$ from a synthetic exponential decay to within
  5%.
- At the LME limit ($\beta\delta \ll 1$), `loewner_kernel()` entries collapse toward
  $1/D$, consistent with `qig/symbolic/lme_exact.py` exact values.

## Implementation Plan

1. Create `qig/gibbs_lock.py` with `GibbsLockedFrame` and `infer_mu0`.
2. Add `tests/test_gibbs_lock.py` with the validation tests listed above.
3. Export `GibbsLockedFrame` and `infer_mu0` from `qig/__init__.py`.
4. Add a short API stub `docs/source/api/gibbs_lock.rst` (one-liner `automodule`).
5. Update `docs/source/api/index.rst` to include the new stub.

## Backward Compatibility

Entirely additive. No existing module is modified except `qig/__init__.py`
(new exports) and the docs index.

## Implementation Status

- [x] `qig/gibbs_lock.py` written and passing lint
- [x] `tests/test_gibbs_lock.py` all tests pass (37/37)
- [x] `qig/__init__.py` updated with new exports
- [x] `docs/source/api/gibbs_lock.rst` created
- [x] `docs/source/api/index.rst` updated
- [ ] Commit referencing CIP-000C

## References

- `the-inaccessible-game-hamiltonian.tex` — defines Gibbs-locked frame,
  Bohr gaps, iso-marginal perturbations, uniform decay rate $\mu_0$.
- `qig/core.py::loewner_kernel` — provides the divided-difference kernel.
- `qig/symbolic/lme_exact.py` — exact LME eigenvalues for validation.
- CIP-0009 — implemented `effective_hamiltonian_coefficients`; the Gibbs-lock
  frame is the natural context in which to call it.
- CIP-000D — depends on this CIP; uses `GibbsLockedFrame` in the end-to-end
  companion notebook.
