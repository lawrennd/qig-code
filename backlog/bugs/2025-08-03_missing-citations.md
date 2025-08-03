---
id: 2025-08-03_missing-citations
status: Proposed
priority: Medium
date_created: "2025-08-03"
owner: neil
---

# Task: Resolve Missing Citations and Duplicate Labels

## Description
The latest LaTeX build generated 13 undefined citations, 3 unresolved references, and one duplicate label. These warnings risk reader confusion and bibliography errors in the final PDF.

### Missing citation keys
Amari-information16, Jaynes-maxent57, Nadakuditi-spectra12, Cover-thomas06, Prokopenko-efficiency23, Felice-fisher16, Sornette-dragon04, Bouchaud-clustered05, Laub-spectral21, Ay-information17, Maes-nonequilibrium13, Goswami-flashcrash2021, Amari-information00 (verify duplicates).

### Other issues
* Undefined refs: `sec:epi-bridge`, `sec:EW_kinetic`, `tab:KK_vs_IG`
* Duplicate label: `sec:multiscale-cascades`

## Acceptance Criteria
- [ ] All listed citation keys either appear in `the-inaccessible-game.bib` **or** the `\cite{}` commands are removed/revised.
- [ ] Undefined `\ref`/`\label` pairs resolved; no duplicate labels.
- [ ] `latexmk -pdf` runs cleanly with **zero** undefined references/citations.

## Implementation Notes
1. Search `.tex` for each key and provide or fix BibTeX entries.
2. Audit section/table labels, ensure uniqueness.
3. Commit fixes and re-compile.
