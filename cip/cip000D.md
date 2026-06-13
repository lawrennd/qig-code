---
author: Neil Lawrence
created: '2026-06-13'
id: 000D
last_updated: '2026-06-13'
status: Implemented
tags:
- cip
- hamiltonian-emergence
- companion-notebook
- gibbs-lock
- generic-decomposition
- documentation
related_requirements:
- "0001"
- "0003"
- "0005"
title: Hamiltonian Paper End-to-End Companion
---

# CIP-000D: Hamiltonian Paper End-to-End Companion

## Status

- [x] Proposed
- [x] Accepted
- [x] Implemented
- [ ] Closed

## Summary

Add a clean end-to-end companion notebook `examples/gibbs_lock_hamiltonian_extraction.ipynb`
that traces the full chain described in `the-inaccessible-game-hamiltonian.tex`:
Gibbs state → Loewner kernel → linearised GENERIC decomposition → effective
Hamiltonian extraction → verification and $\mu_0$ inference. Also finalise
the Sphinx theory page for Hamiltonian extraction (CIP-0009 step 5 was only
partially completed in commit `89c5ed1`).

## Motivation

Two open deliverables were left when CIP-0009 was closed:

- **Step 3**: End-to-end example notebook using LME qutrits.
- **Step 5**: Update Sphinx docs with a Hamiltonian-extraction theory section and
  cross-references.

Additionally, the existing `hamiltonian_emergence_experiments.ipynb` (CIP-000B)
validates structural *claims* of the paper through isolated experiments, but does
not walk a reader through the complete narrative chain from a Gibbs state to an
extracted Hamiltonian. A companion notebook that follows the paper section by
section would make the code substantially more useful as a reproducible research
artefact.

This CIP depends on CIP-000C, which provides the `GibbsLockedFrame` and `infer_mu0`
API needed to make the notebook concise.

## Detailed Description

### `examples/gibbs_lock_hamiltonian_extraction.ipynb`

A four-section notebook corresponding to the main sections of the Hamiltonian paper:

**Section 1 — Setup: Gibbs-locked qutrit pair**

```python
from qig import GibbsLockedFrame
from qig.pair_operators import build_joint_hamiltonian

H = build_joint_hamiltonian(delta=0.5)   # two-qutrit H with gap delta
beta = 2.0
frame = GibbsLockedFrame(H, beta, dims=[3, 3])

print("rho0 eigenvalues:", np.linalg.eigvalsh(frame.rho0))
eps, gaps = frame.bohr_gaps()
print("Bohr gaps (selected):", gaps[gaps > 0].min(), "...", gaps.max())
```

Shows the Gibbs state construction, Bohr gaps, and confirms Gibbs-lock:
`[K0, H] == 0` to machine precision.

**Section 2 — Loewner kernel and iso-marginal tangency**

Links directly to `hamiltonian_emergence_experiments.ipynb` Experiments 1 and 2.
Uses `frame.loewner_kernel()` and `frame.is_iso_marginal()` to classify
perturbation modes. Shows that doubly off-diagonal modes are iso-marginal while
matched-index modes are not.

**Section 3 — GENERIC decomposition and Hamiltonian extraction**

```python
from qig.exponential_family import MatrixExponentialFamily
from qig.generic_decomposition import GENERICDecomposer, effective_hamiltonian_operator

exp_fam = MatrixExponentialFamily(d=3, n_pairs=1)
# Construct theta corresponding to frame.rho0 ...
decomposer = GENERICDecomposer(exp_fam)
decomposer.compute_all(theta_star)

H_eff = decomposer.results['H_eff']
```

Verifies $\dot\rho_\text{rev} = -i[H_\text{eff}, \rho]$ using
`verify_antisymmetric_flow_equals_commutator`. Shows agreement to the precision
documented in CIP-0009.

**Section 4 — $\mu_0$ inference and resolution floor**

```python
from qig import infer_mu0

result = dynamics.solve(theta_star, n_steps=300, dt=0.02)
mu0_fit = infer_mu0(result, frame)
print(f"Inferred mu_0 = {mu0_fit:.4f}")
```

Links to `hamiltonian_emergence_experiments.ipynb` Experiment 5 to show the
connection between the inferred $\mu_0$ and the Fisher-information resolution floor.

### `docs/source/theory/hamiltonian_extraction.rst`

Check the file written by commit `89c5ed1`. If it covers only the extraction step,
extend it with:

- A brief derivation of the Gibbs-lock fixed-point condition $[K_0, H] = 0$.
- The element-wise linearised dynamics formula
  $(\dot{\delta\rho})_{ij} = (i\beta\Delta\epsilon_{ij} - \mu_0)\delta\rho_{ij}$.
- The Loewner-kernel pushforward linking modular-generator and density-matrix
  coordinates.
- Cross-references to `gibbs_lock.py` API and to `hamiltonian_emergence_experiments.ipynb`.

## Implementation Plan

1. Check existing `docs/source/theory/hamiltonian_extraction.rst` content (commit
   `89c5ed1`). Extend where needed.
2. Write `examples/gibbs_lock_hamiltonian_extraction.ipynb` with the four sections
   described above, using `GibbsLockedFrame` from CIP-000C.
3. Confirm the notebook executes end-to-end with `jupyter nbconvert --to notebook
   --execute`.
4. Update `docs/source/index.rst` or `docs/source/theory/index.rst` if the theory
   page is not already linked.
5. Commit referencing CIP-000D.

## Dependencies

- **CIP-000C must be implemented first.** The notebook uses `GibbsLockedFrame`
  and `infer_mu0`.
- CIP-0009 Hamiltonian-extraction functions (`effective_hamiltonian_coefficients`,
  `effective_hamiltonian_operator`, `verify_antisymmetric_flow_equals_commutator`)
  are already present in `qig/generic_decomposition.py`.

## Backward Compatibility

Entirely additive. No existing module is modified.

## Implementation Status

- [x] `docs/source/theory/hamiltonian_extraction.rst` extended with Gibbs-lock section,
      linearised dynamics formula, Loewner-kernel pushforward, and cross-references
- [x] `examples/gibbs_lock_hamiltonian_extraction.ipynb` written and executes
      end-to-end (all four sections: setup, Loewner/iso-marginal, GENERIC/H_eff,
      mu_0 inference)
- [x] Sphinx docs API stub `docs/source/api/gibbs_lock.rst` and `index.rst` updated
      (via CIP-000C)
- [ ] Commit referencing CIP-000D

## References

- `the-inaccessible-game-hamiltonian.tex` — paper whose narrative the notebook
  follows.
- `examples/hamiltonian_emergence_experiments.ipynb` — structural-claim
  validation experiments (CIP-000B); cross-linked from Section 2 and Section 4
  of the new notebook.
- `qig/generic_decomposition.py` — Hamiltonian extraction functions (CIP-0009).
- CIP-000C — provides `GibbsLockedFrame` and `infer_mu0`.
- CIP-0009 — end-to-end notebook was its step 3; this CIP delivers it.
- Commit `89c5ed1` — partial Sphinx docs for Hamiltonian extraction.
