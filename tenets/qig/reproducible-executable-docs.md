---
id: "reproducible-executable-docs"
title: "Reproducible, executable documentation"
status: "Active"
created: "2026-01-13"
last_reviewed: "2026-01-13"
review_frequency: "Quarterly"
tags: ["tenet", "qig"]
---

# Project Tenet: Reproducible, executable documentation

## Description
The codebase is part of a research pipeline. Results should be reproducible from version-controlled code, tests, and examples. Where randomness is used, it should be controlled (seeded) and documented.

Documentation should stay synchronized with the implementation. Prefer “executable docs” (tests + examples + notebooks) over prose that can drift.

## Quote
*"If we can’t rerun it, we don’t really have it."*

## Examples
- Keep example scripts/notebooks in `examples/` runnable against the current API.
- Ensure tests and examples are deterministic (explicit seeding where appropriate).
- Update narrative docs when refactors change recommended methods or APIs.

## Counter-examples
- Notebooks that only run after undocumented manual steps.
- Tests that depend on uncontrolled RNG state.
- Docs that describe removed modules or obsolete implementation details.

## Conflicts
- Potential conflict with **minimal-boilerplate**.
- Resolution: keep reproducibility scaffolding lightweight, but don’t remove it if it protects rerun-ability.
