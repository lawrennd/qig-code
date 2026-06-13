---
id: "separation-of-concerns"
title: "Separation of concerns: core library, tests, and research artifacts"
status: "Active"
created: "2026-01-13"
last_reviewed: "2026-01-13"
review_frequency: "Quarterly"
tags: ["tenet", "qig"]
---

# Project Tenet: Separation of concerns

## Description
Core mathematical functionality should live in the `qig/` package with a stable API. Tests should validate that API and its mathematical properties. Notebooks and scripts should demonstrate workflows and produce figures, but should not be the only place where core logic lives.

This keeps the research code maintainable while still supporting experimentation.

## Quote
*"Experiments can be messy; the library can’t."*

## Examples
- Put operator-basis construction and structure constants in `qig/`, not in notebooks.
- Keep notebook cells focused on using the API, not reimplementing it.
- Use CIPs for design choices and backlog tasks for execution, keeping code changes scoped.

## Counter-examples
- Duplicating core algorithms inside notebooks “because it was quicker”.
- Having only notebooks demonstrate correctness, with no tests.
- Mixing plotting/CLI behavior directly into library functions.

## Conflicts
- Potential conflict with **rapid-prototyping**.
- Resolution: prototype in notebooks/scripts, then migrate stabilized logic into `qig/` with tests.
