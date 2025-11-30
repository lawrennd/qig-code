# Notebook Test Workflow Fixes

## Summary

Fixed the notebook test workflow to properly locate and test Jupyter notebooks in the project.

## Issues Fixed

### 1. **Incorrect Notebook Path**
- **Problem**: Test was looking for `generate-paper-figures.ipynb` in project root
- **Solution**: Updated to `examples/generate-origin-paper-figures.ipynb`

### 2. **Notebook Discovery**
- **Problem**: `--all` flag only searched project root
- **Solution**: Now searches both project root and `examples/` directory

### 3. **Output Validation**
- **Problem**: Test expected `✅ PASSED` markers that don't exist in figure generation notebooks
- **Solution**: Made output validation optional via `require_pass_markers` parameter
  - Figure generation notebooks: validation disabled
  - Test notebooks: validation enabled (looks for PASSED/FAILED markers)

### 4. **Pytest Integration**
- **Problem**: Test could only be run standalone, not via pytest
- **Solution**: Added `test_default_notebook()` function for pytest compatibility
  - Properly decorated with `@slow_test` marker
  - Can be run with `pytest tests/test_notebook.py`
  - Can be skipped with `pytest -m "not slow"`

## Changes Made

### `tests/test_notebook.py`

1. **Updated imports**:
   ```python
   try:
       import pytest
       HAS_PYTEST = True
   except ImportError:
       pytest = None
       HAS_PYTEST = False
   ```

2. **Added slow test decorator**:
   ```python
   def slow_test(func):
       """Mark test as slow if pytest is available."""
       if HAS_PYTEST:
           return pytest.mark.slow(func)
       return func
   ```

3. **Updated notebook discovery logic**:
   - Default notebook: `examples/generate-origin-paper-figures.ipynb`
   - `--all` searches both root and `examples/` directory

4. **Enhanced output validation**:
   - Added `require_pass_markers` parameter
   - Checks for execution errors in all cases
   - Optionally checks for PASSED/FAILED markers

5. **Added pytest-compatible test function**:
   ```python
   @slow_test
   def test_default_notebook():
       # Pytest-compatible test implementation
   ```

### `TESTING.md`

Added comprehensive notebook testing documentation:
- How to run notebook tests via pytest
- How to run notebook tests standalone
- Configuration via environment variables
- Examples of different test modes

## Usage

### Via pytest (recommended)

```bash
# Run notebook test
pytest tests/test_notebook.py::test_default_notebook -v

# Skip slow tests in regular runs
pytest -m "not slow"

# Run only slow tests
pytest -m slow
```

### Standalone execution

```bash
# Test default notebook
python tests/test_notebook.py

# Test specific notebook
python tests/test_notebook.py examples/generate-origin-paper-figures.ipynb

# Test all notebooks
python tests/test_notebook.py --all
```

### With configuration

```bash
# Quick test mode
DYNAMICS_POINTS=5 python tests/test_notebook.py

# Custom configuration
QUTRIT_DIM=4 N_PAIRS=2 python tests/test_notebook.py
```

## Verification

All checks pass:
- ✅ Notebook exists at correct location
- ✅ Test script properly configured
- ✅ Pytest can discover test
- ✅ Slow marker properly applied
- ✅ Standalone execution works
- ✅ No linter errors

## Testing

To verify the fixes work:

```bash
# Verify test discovery
pytest tests/test_notebook.py --collect-only

# Verify slow marker
pytest tests/test_notebook.py -v --collect-only -m slow

# Verify notebook path
python -c "from pathlib import Path; print(Path('examples/generate-origin-paper-figures.ipynb').exists())"
```

## Benefits

1. **Flexible execution**: Can run via pytest or standalone
2. **Better organization**: Slow tests can be skipped in regular runs
3. **Clear validation**: Different validation modes for different notebook types
4. **Easy discovery**: Works with standard pytest test discovery
5. **Well documented**: Updated TESTING.md with clear examples

## Future Enhancements

Potential improvements for future work:
1. Add timeout configuration for long-running notebooks
2. Add parallel notebook execution
3. Add notebook output caching
4. Add CI/CD integration for notebook tests
5. Add more granular validation criteria

