---
id: 2025-12-07_fix-docstring-rst-formatting
title: Fix docstring RST formatting for Sphinx documentation
status: completed
priority: medium
created: 2025-12-07
owner: null
dependencies: []
---

# Task: Fix docstring RST formatting for Sphinx documentation

## Description

The Sphinx documentation build produces 63 warnings, mostly from RST formatting issues in docstrings. The main problem is using `|Φ⟩` notation which RST interprets as substitution references.

## Affected Files

### pair_operators.py (most warnings)
- `bell_state`: lines 3, 5 - "Inline substitution_reference start-string without end-string"
- `bell_state_density_matrix`: line 3 - same issue
- `product_of_bell_states`: line 3 - same issue  
- `gell_mann_generators`: line 7 - "Undefined substitution referenced: k⟩⟨j, j⟩⟨k"

### symbolic/lme_exact.py
- `permutation_matrix`: lines 3, 5 - substitution reference issues

### development/notebooks.rst
- Line 2: Duplicate "Open in Colab" target names

## Root Cause

RST interprets `|text|` as a substitution reference. When docstrings contain:
- `|Φ⟩` (ket notation)
- `|j⟩⟨k|` (bra-ket notation)
- `|00⟩ + |11⟩` (Bell states)

These are parsed as substitution references rather than mathematical notation.

## Solution Options

### Option A: Escape pipe characters
Replace `|Φ⟩` with `\|Φ⟩` or use raw strings.

### Option B: Use math mode
Replace `|Φ⟩` with `:math:`|\Phi\rangle`` for proper LaTeX rendering.

### Option C: Use Unicode alternatives  
Replace `|` with similar Unicode characters like `│` (box drawing) or `❘` (light vertical bar).

**Recommended: Option B** - Use math mode for proper rendering in Sphinx.

## Implementation

### For ket notation
```python
# Before
"""
Returns the Bell state |Φ⟩ = (1/√d) Σ |ii⟩
"""

# After  
"""
Returns the Bell state :math:`|\Phi\\rangle = \\frac{1}{\\sqrt{d}} \\sum_i |ii\\rangle`
"""
```

### For inline references
```python
# Before
|j⟩⟨k| + |k⟩⟨j|

# After
:math:`|j\\rangle\\langle k| + |k\\rangle\\langle j|`
```

## Acceptance Criteria

- [ ] Sphinx build produces 0 errors
- [ ] Sphinx build warnings reduced to <10 (some may be acceptable)
- [ ] Mathematical notation renders correctly in HTML docs
- [ ] Docstrings remain readable in plain text (e.g., `help()`)

## Files to Update

1. `qig/pair_operators.py` - bell_state, bell_state_density_matrix, product_of_bell_states, gell_mann_generators
2. `qig/symbolic/lme_exact.py` - permutation_matrix
3. `docs/source/development/notebooks.rst` - fix duplicate target names

## Related

- CIP-0005: Sphinx Documentation and Read the Docs Integration
- Phase 4 of CIP-0005 includes "Improve docstrings"

## Progress Updates

### 2025-12-07
Task created. 63 warnings in Sphinx build, mostly from RST substitution reference issues in docstrings using ket notation.

### 2025-12-07 (continued)
Partial fix committed:
- ✅ `qig/pair_operators.py` - Fixed ket notation with double backticks
- ✅ `qig/symbolic/lme_exact.py` - Fixed permutation_matrix docstring
- ✅ `docs/source/development/notebooks.rst` - Fixed |θ| notation

Remaining warnings (78 total):
- ~50 "duplicate object description" - Sphinx config issue (API documented twice)
- ~20 docstring formatting in `exponential_family.py` (blank lines needed)
- A few remaining substitution issues

Next steps:
- Fix exponential_family.py docstring formatting
- Update Sphinx config to avoid duplicate API docs

### 2025-12-07 (completed)
Sphinx warnings reduced from 132 to 10:

- ✅ Removed duplicate automodule directives from api/index.rst (~90 warnings)
- ✅ Removed duplicate autofunction directives from api/symbolic.rst (~8 warnings)
- ✅ Fixed docstring RST formatting in exponential_family.py
- ✅ Fixed ket notation in lme_exact.py with double backticks
- ✅ Fixed duplicate "Open in Colab" targets in notebooks.rst
- ✅ Fixed *.ipynb glob in notebook_output_filtering.rst

Remaining 10 warnings: ValidationCheck dataclass attributes (known Sphinx/dataclass edge case, acceptable)
