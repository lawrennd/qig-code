---
id: 2025-08-03_info-geom-hawkes
status: Proposed
priority: Medium
date_created: "2025-08-03"
owner: neil
---

# Feature: Implement Information-Geometric Hawkes Diagnostic

## Motivation
The new paper subsection proposes using blur-window–controlled Fisher eigenvalues to parameterise a Hawkes-process cascade model. We need reproducible code and results.

## Scope
1. Python module to compute Fisher spectrum from simulation data.
2. Routine to fit multi-scale Hawkes kernels and compare excitation parameters.
3. Jupyter notebook demonstrating pipeline on toy and real datasets.
4. Export key metrics for inclusion in LaTeX.

## Deliverables
- `src/ig_hawkes.py`
- `notebooks/ig_hawkes_demo.ipynb`
- Figure/CSV outputs referenced in manuscript.

## Acceptance Criteria
- [ ] Unit test achieves >90 % coverage for eigenvalue→Hawkes conversion.
- [ ] Notebook reproduces main plot within tolerance.
- [ ] Section in paper updated with final numbers/figure.
