---
id: 2025-08-03_draft-cleanup-meta-content
status: proposed
priority: High
owner: Neil
created: 2025-08-03
---

# Task: Clean Up Draft Meta-Content (Comments, TODOs, CIPs)

This task collects a set of editorial clean-up actions identified during a triage pass over the current LaTeX draft.  None of these affect the scientific content; they are purely organisational/structural and aimed at improving readability for collaborators and reviewers.

## Description
The draft currently contains numerous inline comments, obsolete TODO tags, redundant derivations and other author-side artefacts that are "loud" in the compiled PDF and distract from the main argument.  Consolidating or relocating these items will slim the document and clarify the storyline.

## Acceptance Criteria
- All CIP/M-x tags are migrated to a single external tracker file (or GitHub project board).
- Long explainer comment blocks (> 20 lines) are moved to `AUTHOR_NOTES.md`, leaving only a 1-line reference marker in the TeX.
- A single canonical SEA equation macro (`\SEAeq`) is defined and referenced everywhere; alternate notations are removed.
- Duplicate entropy-clock derivations are pruned; only the boxed derivation in §3 remains, with forward references elsewhere.
- Remaining inline comments follow the house-style rules:
  - One line per TODO/REF/FIG
  - Third-person voice, no chatty author tags (e.g. `TK:`)
- Proto-figures that are not ready for publication are either promoted to real figures or moved out of the main text.
- Literature placeholders use stub citations (e.g. `\citep{SuraceInacc2024?}`) so the paragraph scans; outstanding citation tasks are logged in the tracker.
- PDF compiles cleanly with no orphan comment blocks or outdated outline markers.

## Checklist of Sub-Tasks
- [ ]  SWEEP CIP tags → “Task tracker” file; keep only short `% TODO:` slugs in draft.
- [ ]  MOVE long comment essays to `AUTHOR_NOTES.md`.
- [ ]  ADD `\newcommand{\SEAeq}{…}` and replace all manual SEA equations.
- [ ]  DELETE redundant entropy-clock proofs; add cross-references.
- [ ]  GREP for `TK:`/`Neil:` and convert to `% TODO:` or relocate.
- [ ]  HANDLE proto-figures: promote or move.
- [ ]  REPLACE literature placeholders with stub `\citep{…}`; log real-citation tasks.
- [ ]  RUN `latexdiff` / compile to ensure no stray `% TODO` lines appear in PDF.

## Related
- CIP-0006 (structure)
- CIP-0007 (appendix reshuffle)

## Progress Updates
- 2025-08-03: Task created after triage sheet review.
