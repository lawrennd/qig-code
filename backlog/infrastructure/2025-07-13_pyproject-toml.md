---
id: "2025-07-13_pyproject-toml"
title: "Create pyproject.toml for Python project dependencies"
status: "Proposed"
priority: "High"
created: "2025-07-13"
last_updated: "2025-07-13"
owner: "Neil Lawrence"
github_issue: ""
dependencies: ""
tags:
- backlog
- infrastructure
- packaging
- dependencies
---

# Task: Create pyproject.toml for Python project dependencies

## Description

Set up a `pyproject.toml` file for the inaccessible game project to standardize Python dependency management and enable modern packaging workflows. This will include all existing dependencies and add `quimb` for advanced quantum tensor network operations.

## Acceptance Criteria

- [ ] A `pyproject.toml` file exists at the project root
- [ ] All current dependencies are listed in the file
- [ ] The `quimb` package is included as a dependency
- [ ] The file is compatible with standard Python build tools (e.g., pip, poetry, hatch)
- [ ] Documentation is updated to reference the new dependency management approach

## Implementation Notes

- Review all current dependencies (from requirements.txt, notebooks, and scripts)
- Add `quimb` (https://quimb.readthedocs.io/) to the dependencies section
- Choose a build backend (e.g., `hatchling`, `poetry`, or `setuptools`)
- Ensure compatibility with virtual environments and CI workflows
- Update README and developer docs as needed

## Related
- CIP: 0001
- Documentation: https://packaging.python.org/en/latest/tutorials/packaging-projects/

## Progress Updates

### 2025-07-13
Task proposed to modernize dependency management and enable use of quimb for quantum tensor operations. 