---
title: "Refactor Relativity Emergence into EPI Section"
date: 2025-08-03
status: "proposed"
priority: "High"
owner: "Neil"
dependencies: ["cip0006", "2025-08-03_m5-plateau-stage"]
---

# Task: Refactor Relativity Emergence into EPI Section

## Description
Move all relativity-emergence commentary and related mathematical narrative from the current *Plateau & Fisher geometry* portion of the draft into the existing *Extreme Physical Information (EPI)* section.  Introduce the proxy-energy functional there and show how 
SEA ⇒ proxy energy extremum ⇒ EPI ⇒ metric ⇒ Lorentz symmetry.

## Motivation
A single Information→Energy→Relativity story placed inside the EPI section is cleaner than the current duplicate treatment split between Stage-3 discussion and EPI comparison tables.  It keeps Stage-3 focused on constants & quasi-symmetries, while EPI becomes the natural bridge to continuum physics.

## Acceptance Criteria
1. **Comment Relocation**
   - [ ] Locate all relativity/Lorentz TODO blocks (grep `Lorentz`, `Relativity`, etc.).
   - [ ] Cut or copy them into the EPI section (around lines ≈ 1820–1890).
   - [ ] Delete or replace original blocks with forward pointers.
2. **EPI Section Upgrade**
   - [ ] Insert subsection “Relativity from EPI / Proxy Energy”.
   - [ ] Derive metric role of Fisher block `g_{μν}`.
   - [ ] Show entropy-time gauge fixing `c=1` ⇒ Lorentz-invariant action.
   - [ ] Summarise “Fisher isotropy ⇒ Minkowski patch ⇒ Lorentz group”.
3. **Proxy-Energy Introduction**
   - [ ] Define `K = 𝓗 − C` early in EPI and call it proxy energy.
   - [ ] Explain SEA minimises `K` ↔ EPI extremum.
4. **Cleanup & Cross-references**
   - [ ] Update all TODO comments to new locations.
   - [ ] Stage-3 section now references EPI for relativity.
5. **Narrative Flow**
   - [ ] EPI section reads smoothly, ~2-3 paragraphs, links to Stage-3 constants.

## Implementation Notes
*Search hits*: lines 1760–1830, 2007–2035, 2890–2940, 3110+ in `the-inaccessible-game.tex`.

## Progress Updates
### 2025-08-03
Task created after agreement to consolidate relativity under EPI narrative.

## Success Metrics
- Relativity content exists only in EPI section.
- Stage-3 plateau section free of long relativity blocks.
- Proxy energy functional clearly introduced and linked to SEA and EPI.
- Lorentz symmetry derivation concise and rigorously referenced.
