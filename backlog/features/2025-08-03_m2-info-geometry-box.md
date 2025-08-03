---
title: "Implement M-2: Create Information Geometry Preliminaries Box"
date: 2025-08-03
status: superseded
superseded_by: CIP-0006
priority: "High"
owner: "Neil"
dependencies: ["cip0006", "2025-08-03_m1-structural-split"]
---

# Task: Implement M-2: Create Information Geometry Preliminaries Box

## Description

Create a concise "Information Geometry Preliminaries" box that summarizes the key mathematical concepts needed for the paper, moving detailed derivations to appendices. This box serves as a bridge between the axiomatic foundation and the dynamical framework.

## Background

The paper contains extensive mathematical derivations that interrupt the narrative flow. M-2 creates a focused summary box that presents the essential information geometry concepts without losing the reader in technical details. This improves readability while maintaining mathematical rigor.

## Acceptance Criteria

### Information Geometry Preliminaries Box
- [ ] Move Submodularity + Noether content to ≤½-page summary
- [ ] Present Fisher metric G_ij = ∂²S/∂θ_i∂θ_j clearly
- [ ] Explain natural parameter geometry θ_j = log p_j - ψ
- [ ] Show constraint projection operators Π_soft
- [ ] Introduce Fisher eigenvalue hierarchy λ_max/λ_min
- [ ] Connect to SEA dynamics θ̇ = -G_∥θ

### Content Organization
- [ ] Place after Axioms section, before Dynamics section
- [ ] Keep to ≤½-page as specified in CIP-0006
- [ ] Focus on concepts, not derivations
- [ ] Provide clear forward references to detailed proofs
- [ ] Maintain mathematical precision without verbosity

### Appendix Integration
- [ ] Move detailed derivations to Appendix A
- [ ] Move Parzygnat proof sketches to Appendix B
- [ ] Ensure cross-references are clear and accurate
- [ ] Maintain logical flow between main text and appendices

## Implementation Notes

### Box Content Structure

**Essential Concepts to Include:**
1. **Natural Parameters**: θ_j = log p_j - ψ as canonical coordinates
2. **Fisher Metric**: G_ij = ∂²S/∂θ_i∂θ_j defines information geometry
3. **Constraint Projection**: Π_soft projects dynamics to constraint surface
4. **Eigenvalue Hierarchy**: λ_max/λ_min ratios drive plateau formation
5. **SEA Dynamics**: θ̇ = -G_∥θ as fundamental equation

**Content to Move to Appendices:**
- Detailed submodularity proofs
- Noether theorem derivations
- Long convex-linearity arguments
- Technical lemma proofs

### Integration Strategy

- **Preserve**: All mathematical content and rigor
- **Condense**: Focus on essential concepts and intuition
- **Reference**: Clear pointers to detailed proofs in appendices
- **Connect**: Link to existing TODO comments in tex file

### Dependencies

- **M-1 Implementation**: Requires structural split to be completed first
- **CIP-0006**: Provides detailed requirements and specifications
- **Existing mathematical framework**: Builds on established SEA and Fisher geometry

## Related

- **CIP-0006**: Update arXiv Paper Draft to Jaynes/Natural-Parameter Framework
- **Backlog Task**: 2025-08-03_m1-structural-split (prerequisite)
- **Backlog Task**: 2025-08-03_m7-streamline-proofs (related proof organization)

## Progress Updates

### 2025-08-03
Task created to implement the information geometry preliminaries box. This task improves paper readability while maintaining mathematical rigor.

### Implementation Priority
1. **Content identification** - determine what goes in box vs appendices
2. **Box creation** - write concise summary of essential concepts
3. **Appendix reorganization** - move detailed derivations appropriately
4. **Cross-references** - ensure clear connections between main text and appendices

## Success Metrics

- [ ] Information geometry box is ≤½-page as specified
- [ ] Essential concepts clearly presented without technical overload
- [ ] Detailed derivations properly moved to appendices
- [ ] Clear cross-references between main text and appendices
- [ ] Maintains mathematical rigor while improving readability
- [ ] Serves as effective bridge between axioms and dynamics 