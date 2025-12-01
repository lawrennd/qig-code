---
id: "2025-12-01_consolidate-root-python-scripts"
title: "Consolidate and clean up Python scripts in project root"
status: "Proposed"
priority: "Medium"
created: "2025-12-01"
last_updated: "2025-12-01"
owner: "Neil D. Lawrence"
github_issue: ""
dependencies: ""
tags:
- backlog
- infrastructure
- cleanup
- refactoring
---

# Task: Consolidate and Clean Up Python Scripts in Project Root

## Description

The project root directory contains 9 Python scripts (totaling ~90KB), many of which may be legacy code now superseded by the `qig` package. These scripts should be reviewed, consolidated, and either moved to appropriate locations or removed.

## Current Inventory

### Scripts Currently Used in CI/CD
- ✅ `run_all_migrated_experiments.py` (11KB) - Used in `.github/workflows/tests.yml`
- ✅ `validate_phase3_entanglement.py` (3.6KB) - Used in `.github/workflows/tests.yml`

### Scripts Requiring Review
- ❓ `advanced_analysis.py` (15KB) - Advanced GENERIC analysis (may be legacy)
- ❓ `inaccessible_game_quantum.py` (23KB) - Standalone validation script (likely legacy, was in README_VALIDATION.md)
- ❓ `quantum_qutrit_n3.py` (12KB) - Standalone qutrit implementation (likely legacy, was in README_quantum_simulation.md)
- ❓ `run_qutrit_experiment.py` (7.2KB) - Qutrit experiment runner (may be redundant with qig package)
- ❓ `run_qutrit_quick.py` (4.5KB) - Quick qutrit runner (may be redundant with qig package)
- ❓ `validate_qutrit_optimality.py` (16KB) - Qutrit optimality validation (purpose unclear)

### Keep as-is
- ✅ `setup.py` (253B) - Minimal setup for backward compatibility (points to pyproject.toml)

## Analysis Required

For each script, determine:
1. **Is it used?** Check imports in other code, references in workflows, or external dependencies
2. **Is it superseded?** Does the `qig` package now provide this functionality?
3. **Is it legacy?** Was it part of pre-migration standalone code?
4. **Should it be kept?** Is it a useful example, validation, or test script?

## Proposed Actions

### Option A: Move to Examples
If scripts demonstrate useful workflows:
- Create `examples/scripts/` directory
- Move useful demonstration scripts there
- Update with modern `qig` package imports
- Add brief README explaining each script

### Option B: Move to Tests
If scripts are validation/testing tools:
- Refactor into proper pytest tests in `tests/`
- Remove standalone script versions
- Ensure CI/CD workflows reference new test locations

### Option C: Delete
If scripts are truly legacy and superseded:
- Verify no external dependencies
- Document what `qig` module/function replaces each script
- Delete the script

### Option D: Keep in Root (sparingly)
Only for scripts that:
- Are actively used in CI/CD
- Serve as top-level entry points
- Have no better home

## Acceptance Criteria

- [ ] All 9 Python scripts reviewed and categorized
- [ ] Legacy scripts identified and either refactored or removed
- [ ] Useful demonstration scripts moved to `examples/scripts/` with updated imports
- [ ] Validation scripts converted to proper tests in `tests/` or verified as still needed
- [ ] CI/CD workflows updated if script locations change
- [ ] Documentation updated to reflect new script locations
- [ ] Root directory contains only essential scripts (likely just those used in CI/CD + setup.py)
- [ ] All removed scripts' functionality confirmed available in `qig` package or proper tests

## Implementation Notes

### Step 1: Audit Each Script
For each script, document:
- What it does
- Whether it imports from `qig` or reimplements functionality
- Whether it's referenced elsewhere in the codebase
- Last modified date (all currently Nov 30 19:10)

### Step 2: Create Migration Plan
Create a mapping:
```
script_name.py → [keep in root | move to examples/ | move to tests/ | delete]
```

### Step 3: Update Dependencies
- Update `.github/workflows/tests.yml` if script paths change
- Update any documentation referencing scripts
- Ensure imports still work after moves

### Step 4: Commit Changes
Use surgical git adds to commit changes in logical groups:
```bash
git add examples/scripts/  # New examples
git commit -m "Move demonstration scripts to examples/scripts/"

git add tests/  # New tests
git commit -m "Convert validation scripts to proper pytest tests"

git add <deleted_scripts>  # Deletions
git commit -m "Remove legacy scripts superseded by qig package"
```

## Related

- CIP-0001: Code consolidation and module structure
- CIP-0002: Migration from standalone scripts to qig package
- Recent cleanup: Removed README_VALIDATION.md and README_quantum_simulation.md (2025-12-01)

## Progress Updates

### 2025-12-01

Task created following cleanup of redundant Markdown documentation. Many of these scripts were documented in the now-deleted README_VALIDATION.md and README_quantum_simulation.md files, suggesting they are legacy code from the pre-qig-package era.

