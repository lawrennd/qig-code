---
title: "Complete arXiv Paper Restructuring per CIP-0006"
date: 2025-08-01
status: "proposed"
priority: "High"
owner: "Neil"
dependencies: ["cip0006"]
---

# Task: Complete arXiv Paper Restructuring per CIP-0006

## Description

Complete the remaining work on restructuring the arXiv paper draft (`the-inaccessible-game.tex`) according to the comprehensive plan outlined in CIP-0006. Major progress has been made on the mathematical framework and electroweak analysis, but several critical components remain to be implemented.

## Background

CIP-0006 outlines a major restructuring of the arXiv paper to align with the Jaynes/natural-parameter MEPP framework. Recent commits (2025-08-01) have made significant progress:

- ✅ Enhanced SEA dynamics with entropy time τ and clock-gauge choice
- ✅ Comprehensive electroweak analysis with information-geometric framework
- ✅ Cleaned up Kaluza-Klein digression
- ✅ Added formal mathematical framework with Fisher quadratic Hamiltonian

However, several key structural changes and content integrations remain incomplete.

## Acceptance Criteria

- [ ] **M-1: Split Information Isolation Section** - Separate axioms from dynamics clearly
- [ ] **M-2: Create Information Geometry Preliminaries Box** - Move detailed derivations to appendix
- [ ] **M-3: Stage 1 - Dephasing Subsection** - Define block channel and dephasing process
- [ ] **M-4: Stage 2 - Isolation & SEA Subsection** - Complete charge classification connection
- [ ] **M-5: Stage 3 - Long Plateau & Quasi-Symmetries** - Add Lorentz dispersion relation
- [ ] **M-6: Beyond Plateau - Spatial Metrics & Particles** - Complete particle modes presentation
- [ ] **M-7: Streamline Mathematical Proofs** - Move long proofs to appendices
- [ ] **M-8: Update Abstract & Introduction** - Reflect new structure and framework
- [ ] **M-0.4: Figure Placeholders** - Add entropy curve and Fisher eigenvalue charts
- [ ] **d=3 Optimality Argument** - Provide principled selection for qutrit choice
- [ ] **arXiv Compliance** - Verify format and ML relevance
- [ ] **Double-Blind Preparation** - Remove self-identifying content

## Implementation Notes

### High Priority Items (Must Complete First)

1. **M-1: Structural Split** - This is foundational and affects the entire paper flow
2. **M-3: Dephasing Stage** - Critical for establishing the two-stage narrative
3. **d=3 Optimality** - Addresses reviewer concerns about ad-hoc choices
4. **Figure Integration** - Required for arXiv submission

### Medium Priority Items

1. **M-2: Information Geometry Box** - Improves readability without changing core content
2. **M-7: Proof Streamlining** - Essential for page limit compliance
3. **M-8: Abstract Update** - Critical for submission but can be done last

### Low Priority Items

1. **M-6: Particle Modes** - Speculative content that can be moved to appendix if needed
2. **Double-Blind Preparation** - Can be done as final step before submission

### Technical Approach

1. **Work in Phases**: Complete M-1 through M-4 first, then polish with M-7 and M-8
2. **Maintain Mathematical Rigor**: Keep completed mathematical framework intact
3. **Preserve Recent Progress**: Don't lose the enhanced electroweak analysis
4. **Use TODO Comments**: Leverage existing TODO markers in the tex file

## Dependencies

- **CIP-0006**: Provides the comprehensive plan and requirements
- **CIP-0007**: Stage 3 plateau simulation for required figures
- **MEPP Simulator**: Results for entropy plateau visualization
- **Bibliography**: Beretta, Surace, and other key references already added

## Related

- **CIP-0006**: Update arXiv Paper Draft to Jaynes/Natural-Parameter Framework
- **CIP-0007**: Extend MEPPSimulator to Cover Stage 3 Plateau/Stalling Regime
- **Backlog Task**: 2025-07-26_lapse-function-visualization (figure generation)

## Progress Updates

### 2025-08-01
Task created to track remaining work after major progress on mathematical framework and electroweak analysis. Current completion estimate: 60% of CIP-0006 requirements.

### Next Steps
1. Begin with M-1 structural split
2. Implement M-3 dephasing stage content
3. Add figure placeholders and simulation results
4. Complete abstract and introduction updates

## Success Metrics

- [ ] Paper follows coherent dephasing → isolation → plateau narrative
- [ ] Jaynes natural parameters used consistently throughout
- [ ] Fisher eigenvalue hierarchy clearly explains emergence
- [ ] Within arXiv format with comprehensive content
- [ ] Ready for arXiv submission with proper formatting 