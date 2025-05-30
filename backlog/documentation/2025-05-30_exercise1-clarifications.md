---
id: "2025-05-30_exercise1-clarifications"
title: "Add Clarifications to Exercise 1"
status: "Completed"
priority: "Medium"
created: "2025-05-30"
last_updated: "2025-05-30"
owner: "Neil Lawrence"
github_issue: ""
dependencies: ""
tags:
- backlog
- documentation
- exercises
- clarifications
- minor
---

# Task: Add Clarifications to Exercise 1

## Description

Exercise 1 is well-formulated but needs minor clarifications to improve student understanding. The main issues are explaining what "conserved" means and noting partition independence.

**Current Status:** Well-formulated, needs only minor improvements.

## Acceptance Criteria

- [ ] Clarify meaning of "conserved" in task statement
- [ ] Add note about partition independence
- [ ] Consider adding simple verification example
- [ ] Ensure mathematical precision throughout

## Implementation Notes

**Clarification 1 - Meaning of "conserved":**
```latex
Show that for this system the sum of the joint entropy S(Z) and the multi information I(Z) 
is also conserved (i.e., remains constant as variables transition between latent and active partitions).
```

**Clarification 2 - Partition independence:**
```latex
Note that this conservation law holds regardless of how the N variables are partitioned 
between latent M and active X.
```

**Optional Enhancement - Simple example:**
```latex
Verify this conservation law with a simple example: Take N=2 with one latent variable 
(S(m₁) = log 2) and one active variable with probabilities p₁ = 3/4, p₂ = 1/4.
```

## Related

- Original analysis in: suggested-improvements.md
- Foundation for all subsequent exercises
- Mathematical precision important for theoretical coherence

## Progress Updates

### 2025-05-30
Task created. This exercise is already in good shape - just needs minor clarifications for pedagogical improvement. 