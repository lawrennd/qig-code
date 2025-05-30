---
id: "2025-05-30_exercise-reordering"
title: "Fix Exercise Logical Dependency Order (6 before 5)"
status: "Proposed"
priority: "High"
created: "2025-05-30"
last_updated: "2025-05-30"
owner: "Neil Lawrence"
github_issue: ""
dependencies: ""
tags:
- backlog
- infrastructure
- critical
- exercises
---

# Task: Fix Exercise Logical Dependency Order

## Description

Exercise 5 (Hierarchical Partition) uses geometric concepts (Fisher metrics, distance between variables, parameter space geometry) that are not established until Exercise 6 (Distinguishability). This creates a logical dependency problem where students encounter undefined concepts.

**Current Problematic Sequence:**
- Exercise 5: Uses "distance between variables," Fisher metrics, geometric structure  
- Exercise 6: *Establishes* distinguishability, Fisher metrics, geometric foundations

**Required Solution:**
Reorder exercises so Exercise 6 comes before Exercise 5, providing proper geometric foundations.

## Acceptance Criteria

- [ ] Swap the order of Exercise 5 and Exercise 6 in the main document
- [ ] Update all cross-references between exercises
- [ ] Add forward reference in Exercise 6: "The geometric structure developed here will be crucial for understanding effective field theory emergence in the next exercise"
- [ ] Verify Exercise 5 text makes sense with geometric foundations established
- [ ] Update exercise numbering consistently throughout

## Implementation Notes

**Benefits of reordering:**
- **Logical flow**: Each exercise builds only on established concepts
- **Mathematical rigor**: Geometric concepts properly defined before use  
- **Conceptual clarity**: Students understand what "distance" means in information theory
- **Theoretical coherence**: Clear progression from discrete → geometric → field theoretic

**Changes needed:**
- Exercise 6 becomes Exercise 5 (Distinguishability)
- Exercise 5 becomes Exercise 6 (Hierarchical Partition)
- Exercise 7 remains Exercise 7 but may need updates based on new flow

## Related

- Task: 2025-05-30_time-conversion-addition (depends on this reordering)
- Task: 2025-05-30_exercise7-rework (may be affected by this change)
- Original analysis in: suggested-improvements.md

## Progress Updates

### 2025-05-30
Task created. This is identified as the most critical structural fix needed for the exercise sequence. 