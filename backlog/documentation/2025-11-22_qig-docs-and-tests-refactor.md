---
id: "2025-11-22_qig-docs-and-tests-refactor"
title: "Refactor quantum game documentation and test narratives"
status: "Closed"
priority: "Medium"
created: "2025-11-22"
last_updated: "2025-12-01"
owner: "Neil D. Lawrence"
github_issue: ""
dependencies: ""
tags:
- backlog
- documentation
- testing
---

# Task: Refactor quantum game documentation and test narratives

## Description

As the quantum inaccessible game code is refactored into the `qig` package and
numerical gradients are replaced by analytic/AD-based implementations, the
documentation and test narratives need to be updated for coherence and clarity.

Currently:

- Several documents explicitly describe Fisher/BKM metric and higher cumulants
  as being computed via finite differences:
  - `IMPLEMENTATION_SUMMARY.md`
  - `IMPLEMENTATION_NOTES.md`
  - `README_quantum_simulation.md`
- The main test suite (`test_inaccessible_game.py`) contains comments and
  expectations that are tied to the current finite-difference implementations
  (e.g. for `compute_jacobian` and constraint tangency).

This task is to bring the documentation and tests into line with the new
structure and mathematical implementation, as part of a wider documentation
tidy-up and refactor for the quantum game.

## Acceptance Criteria

- [ ] All references to “finite differences” for core quantities (Fisher/BKM
      metric, constraint gradients, Jacobian) in the main documentation files
      are updated to describe the new implementation.
- [ ] `IMPLEMENTATION_SUMMARY.md` and `IMPLEMENTATION_NOTES.md` clearly separate:
      - theoretical definitions;
      - library implementations in `qig.*`; and
      - validation/testing strategy.
- [ ] `README_quantum_simulation.md` (and any similar guides) show up-to-date
      import paths (`qig.*`) and a minimal, accurate description of how the
      metric and GENERIC decomposition are computed.
- [ ] Comments/docstrings in `test_inaccessible_game.py` accurately reflect the
      refactored API and do not refer to obsolete finite-difference details.
- [ ] The documentation structure for the quantum game (READMEs, summaries,
      notes) is internally consistent and references CIP-0001 and the relevant
      backlog tasks where appropriate.

## Implementation Notes

- Coordinate with:
  - CIP-0001 (codebase consolidation and documentation).
  - Backlog:
    - `2025-11-22_remove-numerical-gradients`
    - `2025-11-22_full-jacobian-and-third-order-cumulants`
- Start from the current state of:
  - `IMPLEMENTATION_SUMMARY.md` and `IMPLEMENTATION_NOTES.md`;
  - `README_VALIDATION.md`, `README_quantum_simulation.md`;
  - `test_inaccessible_game.py` docstrings and comments.
- Once the analytic/AD-based implementations for gradients and Jacobians are in
  place, update the prose to:
  - describe the new methods at a high level; and
  - avoid over-emphasising implementation details that may change again.

## Related

- CIPs:
  - 0001 (Consolidate and document quantum inaccessible game validation code).
- Backlog:
  - `2025-11-22_remove-numerical-gradients`
  - `2025-11-22_full-jacobian-and-third-order-cumulants`

## Progress Updates

### 2025-11-22

Task created with Proposed status. This item captures the documentation/test
tidy-up required once the numerical gradient and full-Jacobian tasks are
implemented, and ties it into the broader `qig` refactor.

### 2025-12-01 - CLOSED

**Reason**: Most referenced documentation files don't exist, and existing docs are already accurate.

**Findings**:
- ❌ `IMPLEMENTATION_SUMMARY.md` - **does not exist**
- ❌ `IMPLEMENTATION_NOTES.md` - **does not exist**  
- ✅ `README_quantum_simulation.md` - Already states "**All gradients use analytic formulas**"
- ✅ `README_VALIDATION.md` - Exists and is accurate
- ✅ Test documentation - Already updated in CIP-0004 test suite rewrite

**Remaining Issue**: One reference to finite differences for Jacobi identity (README line 126)

**Decision**: Task as written is obsolete. The documentation is already mostly correct. If the single FD reference needs fixing, it can be done as a minor update without a full backlog task. CIP-0005 (Sphinx documentation) will be the proper venue for comprehensive documentation improvements.


