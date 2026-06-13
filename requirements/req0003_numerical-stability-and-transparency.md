---
id: "0003"
title: "Numerical stability and transparency"
status: "Ready"
priority: "High"
created: "2026-01-13"
last_updated: "2026-01-13"
related_tenets:
  - "scientific-rigour"
stakeholders:
  - "researchers"
  - "maintainers"
tags:
  - "numerics"
  - "stability"
---

# REQ-0003: Numerical stability and transparency

## Description
Core numerical routines must use stable defaults for ill-conditioned problems and must expose the information needed to interpret results (e.g., solver choice, residuals, condition numbers, tolerances).

The goal is to avoid “looks plausible” outputs from numerically ill-posed computations.

## Acceptance Criteria
- [ ] Routines with potentially singular/ill-conditioned linear algebra return diagnostics (at least residual + conditioning + method/solver label).
- [ ] Default solvers are chosen to behave sensibly on rank-deficient problems (e.g., least-squares minimum-norm solutions).
- [ ] Test tolerances reflect realistic numerical precision rather than aspirational machine-precision bounds.
