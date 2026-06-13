---
id: "0005"
title: "Documentation and examples remain synchronized with the code"
status: "Ready"
priority: "Medium"
created: "2026-01-13"
last_updated: "2026-01-13"
related_tenets:
  - "reproducible-executable-docs"
  - "separation-of-concerns"
stakeholders:
  - "researchers"
  - "new-contributors"
tags:
  - "documentation"
  - "examples"
---

# REQ-0005: Documentation and examples remain synchronized with the code

## Description
The documentation and example artifacts (scripts/notebooks) must track the actual implementation, so that new readers can reproduce and understand the intended workflows without reverse-engineering internal APIs.

## Acceptance Criteria
- [ ] Examples in `examples/` reflect the current public API (imports, arguments, recommended methods).
- [ ] Narrative docs and backlog documentation tasks avoid claims that conflict with the current implementation.
- [ ] At least one lightweight check exists to catch gross drift in key example notebooks/scripts.
