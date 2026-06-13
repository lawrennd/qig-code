---
id: "2026-01-13_two-qutrit-worked-example"
title: "Two-Qutrit Worked Example Notebook"
status: "Completed"
priority: "Medium"
created: "2026-01-13"
last_updated: "2026-01-13"
category: "features"
related_cips: ["0009", "000A"]
---

# Task: Two-Qutrit Worked Example Notebook

## Description

Create a self-contained worked-example notebook
`examples/origin_two_qutrit_worked_example.ipynb` that walks through the
two-qutrit entropy-time trajectory from first principles using the `qig` API.
The notebook is intended as an accessible companion to the more technical
`duhamel_methods_comparison.ipynb` and `hamiltonian_emergence_experiments.ipynb`,
showing a complete end-to-end example of the information-geometric dynamics for
an entangled qutrit pair.

The notebook also includes publication-quality single-panel PDF figures for
entropy bookkeeping (marginal and joint entropies along the trajectory).

## Acceptance Criteria

- [x] `examples/origin_two_qutrit_worked_example.ipynb` created and runs clean
- [x] Smoke test added in `tests/test_notebook.py`
- [x] Added to project README
- [x] Single-panel PDF figure for the two-qutrit entropy trajectory included
- [x] Single-panel entropy bookkeeping PDF figure included
- [x] Notebook cell ids normalised for nbformat compatibility

## Implementation Notes

The notebook reuses `qig.core.create_lme_state`, `MatrixExponentialFamily`, and the
block-matrix Duhamel derivative (CIP-000A default) to trace the entropy-time path.
PDF figure generation uses matplotlib with publication-quality settings.

## Related

- CIP-0009: Hamiltonian extraction from antisymmetric GENERIC flow
- CIP-000A: Block-matrix Fréchet derivatives (default Duhamel method)
- Notebooks: `examples/origin_two_qutrit_worked_example.ipynb`

## Progress Updates

### 2026-01-13
Created `examples/origin_two_qutrit_worked_example.ipynb` with smoke test (commit `35a673d`).
Added single-panel PDF figure for two-qutrit example (commit `bf0d3f1`).
Added single-panel entropy bookkeeping PDF figure (commit `48f868a`).
Normalised notebook cell ids for nbformat compatibility (commit `0c2ec51`).
Task completed.
