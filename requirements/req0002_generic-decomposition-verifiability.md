---
id: "0002"
title: "GENERIC decomposition and Hamiltonian extraction are verifiable"
status: "Ready"
priority: "High"
created: "2026-01-13"
last_updated: "2026-01-13"
related_tenets:
  - "scientific-rigour"
  - "separation-of-concerns"
stakeholders:
  - "researchers"
  - "maintainers"
tags:
  - "generic"
  - "hamiltonian"
  - "validation"
---

# REQ-0002: GENERIC decomposition and Hamiltonian extraction are verifiable

## Description
The project must provide a way to compute the GENERIC decomposition (symmetric/antisymmetric split of the flow Jacobian) and to map the antisymmetric sector to an effective Hamiltonian in a chosen operator basis, with explicit diagnostics that quantify how well the mapping holds.

Where strong identities do not hold in general, the implementation and tests should document the weaker, correct statements (e.g., best-fit residuals).

## Acceptance Criteria
- [ ] There is a library API for obtaining \(S\) and \(A\) (or equivalents) from the dynamics/Jacobian computation.
- [ ] There is a routine that produces an effective Hamiltonian representation (basis coefficients + operator) and returns solver diagnostics (residual, conditioning, method).
- [ ] Tests/examples quantify the mismatch when assumptions fail, rather than asserting false equalities.
