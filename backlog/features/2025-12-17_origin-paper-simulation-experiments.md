---
id: "2025-12-17_origin-paper-simulation-experiments"
title: "Origin Paper Simulation Experiments Notebook"
status: "Completed"
priority: "High"
created: "2025-12-17"
last_updated: "2025-12-19"
category: "features"
related_cips: []
---

# Task: Origin Paper Simulation Experiments Notebook

## Description

Add a reproducible notebook `examples/origin_paper_simulation_experiments.ipynb` that
replicates and validates the simulation experiments from the origin paper
(`the-inaccessible-game-origin.tex`). The notebook serves as a living, executable
companion to the paper, demonstrating that the theoretical results are computationally
confirmed by the `qig` implementation.

Also extend `examples/boring_game_dynamics.ipynb` to explain the LME-origin
symmetry reduction that appears in the paper's derivation.

## Acceptance Criteria

- [x] `examples/origin_paper_simulation_experiments.ipynb` created and runs clean
- [x] Notebook added to README under examples
- [x] Notebook added to CI notebook-test suite (`.github/workflows/notebook-tests.yml`)
- [x] Smoke test added in `tests/test_notebook.py`
- [x] LME-origin symmetry reduction explained in `boring_game_dynamics.ipynb`

## Implementation Notes

The notebook was built using the existing `qig` exponential-family and GENERIC
infrastructure. It directly exercises the constrained maximum-entropy production
formulation and validates numerical outputs against paper equations.

## Related

- Paper: `the-inaccessible-game-origin.tex`
- Notebooks: `examples/origin_paper_simulation_experiments.ipynb`,
  `examples/boring_game_dynamics.ipynb`

## Progress Updates

### 2025-12-17
Task started. Created `examples/origin_paper_simulation_experiments.ipynb` (commit `6588993`).
Explained LME-origin symmetry reduction in `boring_game_dynamics.ipynb` (commit `7f7c7de`).

### 2025-12-19
Added notebook to README and CI (commit `578ac91`).
Updated notebook outputs (commit `970b339`).
Task completed.
