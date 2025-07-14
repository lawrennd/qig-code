---
id: "2025-07-13_pyproject-toml"
title: "Create pyproject.toml for Python project dependencies"
status: "Completed"
priority: "High"
created: "2025-07-13"
last_updated: "2025-01-27"
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

- [x] A `pyproject.toml` file exists at the project root
- [x] All current dependencies are listed in the file
- [x] The `quimb` package is included as a dependency
- [x] The file is compatible with standard Python build tools (e.g., pip, poetry, hatch)
- [x] Documentation is updated to reference the new dependency management approach

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

### 2025-01-27
Task completed! Created comprehensive `pyproject.toml` file with:
- All current dependencies (numpy, scipy, matplotlib, jupyter, quimb)
- Optional dependency groups for development, documentation, and research
- Modern build system configuration using hatchling
- Code quality tools configuration (black, isort, mypy, pytest, ruff)
- Proper project metadata and classifiers for scientific research
- Support for Python 3.8+ compatibility 