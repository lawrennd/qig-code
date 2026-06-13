---
id: "entanglement-first-class"
title: "Entanglement is first-class"
status: "Active"
created: "2026-01-13"
last_reviewed: "2026-01-13"
review_frequency: "Quarterly"
tags: ["tenet", "qig"]
---

# Project Tenet: Entanglement is first-class

## Description
If the theory and paper claims involve entangled states (e.g., LME origins, Bell states, pair-basis \(su(d^2)\)), the codebase must represent and manipulate those states natively. We avoid “local-only” constructions when they cannot express the phenomena under study.

This tenet is about aligning representational capacity with scientific scope.

## Quote
*"Don’t study entanglement with tools that can’t entangle."*

## Examples
- Use pair operator bases (\(su(d^2)\)) when modelling maximally entangled pairs.
- Include validations that distinguish separable vs entangled regimes (e.g., mutual information behavior).
- Ensure examples and tests cover entangled cases, not just local/separable ones.

## Counter-examples
- Implementations that implicitly constrain the state space to separable families while discussing entanglement.
- Tests that pass only in commuting/local regimes but are presented as general.
- Claims of “Bell-state” behavior without a representable Bell state in the model.

## Conflicts
- Potential conflict with **computational-cost-minimization**.
- Resolution: provide scalable approximations, but keep correctness-first reference paths for small systems.
