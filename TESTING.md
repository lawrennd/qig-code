# Testing Documentation

This document describes the testing infrastructure for the QIG codebase.

## Overview

The QIG project has multiple testing layers:
1. **Unit Tests**: `tests/test_*.py` - Core functionality tests
2. **Notebook Tests**: `tests/test_notebook.py` - Jupyter notebook validation
3. **Integration Tests**: Various validation scripts
4. **GitHub Actions CI/CD**: Automated testing workflows

## Notebook Testing

### Available Notebooks

- `examples/generate-origin-paper-figures.ipynb` - Paper figure generation and validation

### Running Notebook Tests

#### Option 1: Via pytest (recommended)

```bash
# Run the notebook test (marked as slow)
pytest tests/test_notebook.py::test_default_notebook -v

# Skip slow tests in regular test runs
pytest -m "not slow"

# Run only slow tests
pytest -m slow
```

#### Option 2: Direct execution

```bash
# Test default notebook
python tests/test_notebook.py

# Test specific notebook
python tests/test_notebook.py examples/generate-origin-paper-figures.ipynb

# Test all notebooks
python tests/test_notebook.py --all
```

### Notebook Test Configuration

The notebook test script supports environment variables for configuration:

```bash
# With custom configuration
QUTRIT_DIM=4 DYNAMICS_POINTS=10 python tests/test_notebook.py

# Quick test mode
DYNAMICS_POINTS=5 DYNAMICS_T_MAX=0.5 python tests/test_notebook.py
```

See the script header in `tests/test_notebook.py` for all available environment variables.

## CIP-0002 Migration Testing

The migration from LOCAL to PAIR operators is validated through:
1. **Interactive Jupyter Notebook**: `CIP-0002_Migration_Validation.ipynb`
2. **Python Test Script**: `test_notebook.py`
3. **Python Integration Suite**: `run_all_migrated_experiments.py`
4. **GitHub Actions CI/CD**: `.github/workflows/test-migration-validation.yml`

## Running Tests Locally

### Option 1: Run Jupyter Notebook Interactively

```bash
jupyter notebook CIP-0002_Migration_Validation.ipynb
```

Execute cells to validate the migration. The notebook is self-contained with explanations.

### Option 2: Run Notebook Test Script

Execute the notebook programmatically:

```bash
# Default configuration
python test_notebook.py

# Quick test (fewer integration points)
DYNAMICS_POINTS=5 DYNAMICS_T_MAX=0.5 python test_notebook.py

# Custom configuration
QUTRIT_DIM=4 N_PAIRS=2 python test_notebook.py
```

### Option 3: Run Python Integration Suite

```bash
# Comprehensive test (4 experiments, ~30 seconds)
python run_all_migrated_experiments.py

# Focused entanglement validation
python validate_phase3_entanglement.py
```

## Configuration via Environment Variables

The notebook accepts the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `QUTRIT_DIM` | Qutrit dimension (d) | 3 |
| `QUBIT_DIM` | Qubit dimension | 2 |
| `N_PAIRS` | Number of pairs | 1 |
| `DYNAMICS_POINTS` | Integration points | 20 |
| `DYNAMICS_T_MAX` | Max integration time | 2.0 |
| `RANDOM_SEED_1` | Seed for experiment 1 | 42 |
| `RANDOM_SEED_2` | Seed for experiment 2 | 123 |
| `RANDOM_SEED_3` | Seed for experiment 3 | 456 |
| `THETA_SCALE` | Parameter scaling | 0.5 |
| `TOLERANCE` | Numerical tolerance | 1e-6 |

### Examples

**Quick smoke test** (5 integration points, 0.5s simulation):
```bash
DYNAMICS_POINTS=5 DYNAMICS_T_MAX=0.5 python test_notebook.py
```

**Test with 2 qutrit pairs** (more parameters):
```bash
N_PAIRS=2 python test_notebook.py
```

**Test with d=4 (ququarts)**:
```bash
QUTRIT_DIM=4 python test_notebook.py
```

**Stricter tolerance**:
```bash
TOLERANCE=1e-8 python test_notebook.py
```

## CI/CD Workflows

### Automated Testing on GitHub

Three test jobs run automatically on push/PR:

#### 1. **Default Configuration**
- Uses default parameters (d=3, 1 pair, 20 points)
- Full validation (~30-60 seconds)
- Runs on: every push to main/develop

#### 2. **Quick Configuration**
- Uses reduced parameters (5 points, 0.5s)
- Fast smoke test (~10-20 seconds)
- Runs on: every push to main/develop

#### 3. **Python Suite**
- Runs `run_all_migrated_experiments.py`
- Runs `validate_phase3_entanglement.py`
- Validates all migrated scripts
- Runs on: every push to main/develop

#### 4. **Custom Configuration** (Manual Trigger)
- Run from GitHub Actions UI
- Specify custom `QUTRIT_DIM`, `DYNAMICS_POINTS`
- Useful for testing edge cases

### Viewing CI/CD Results

1. Go to repository → **Actions** tab
2. Select workflow run
3. View job logs
4. Download executed notebook artifacts (if failed)

### Manual Workflow Dispatch

Trigger custom tests from GitHub UI:

1. Go to **Actions** → **CIP-0002 Migration Validation**
2. Click **Run workflow**
3. Select branch
4. Enter custom parameters:
   - Qutrit dimension (e.g., 4)
   - Dynamics points (e.g., 10)
5. Click **Run workflow**

## Test Structure

### Experiment 1: Entanglement Validation
- **Tests**: LME state, generic entangled state
- **Expected**: I = 2.197 (maximal), I > 0.1 (generic)
- **Duration**: ~2 seconds

### Experiment 2: Qubit Pair Dynamics
- **Tests**: Constraint preservation, entropy increase
- **Expected**: Violation < 1e-6, ΔH ≥ 0
- **Duration**: ~10-20 seconds (depends on `DYNAMICS_POINTS`)

### Experiment 3: Qutrit vs Qubit
- **Tests**: System comparison, Fisher metric
- **Expected**: Qutrit advantage ~1.16×
- **Duration**: ~5 seconds

### Experiment 4: API Compatibility
- **Tests**: Density matrix, entropies, gradients
- **Expected**: All physically valid
- **Duration**: ~2 seconds

## Success Criteria

All tests must pass:
- ✅ 4/4 experiments passed
- ✅ Notebook executes without errors
- ✅ Python suite completes successfully
- ✅ All scripts produce I > 0 (genuine entanglement)

## Debugging Failed Tests

### Notebook Execution Failed

```bash
# Run with more verbose output
jupyter nbconvert --to notebook --execute --debug \
  CIP-0002_Migration_Validation.ipynb
```

### Numerical Tolerance Issues

Try relaxing tolerance:
```bash
TOLERANCE=1e-5 python test_notebook.py
```

### Memory Issues (Large Systems)

Reduce system size:
```bash
N_PAIRS=1 DYNAMICS_POINTS=10 python test_notebook.py
```

### Check Individual Components

```bash
# Test qig library
python -m pytest tests/

# Test pair operators
python -c "from qig.pair_operators import bell_state; print(bell_state(2))"

# Test quantum_qutrit_n3 wrapper
python -c "import quantum_qutrit_n3 as qq; print(qq.__version__)"
```

## Adding New Tests

### To add a new experiment to the notebook:

1. Add markdown cell with explanation
2. Add code cell with test
3. Print "✅ PASSED" or "❌ FAILED"
4. Update summary cell

### To add new CI/CD job:

Edit `.github/workflows/test-migration-validation.yml`:

```yaml
test-notebook-myconfig:
  name: Test Notebook (My Config)
  runs-on: ubuntu-latest
  steps:
    # ... (copy from existing job)
    - name: Run notebook test (my configuration)
      env:
        MY_VAR: my_value
      run: python test_notebook.py
```

## Performance Benchmarks

Typical execution times (on GitHub Actions, ubuntu-latest):

| Configuration | Time | Notes |
|---------------|------|-------|
| Default (d=3, 20 pts) | ~30-40s | Full validation |
| Quick (d=3, 5 pts) | ~15-20s | Fast smoke test |
| Large (d=3, 2 pairs) | ~60-90s | More parameters |
| Ququart (d=4, 1 pair) | ~40-50s | Larger Hilbert space |

## Related Files

- `CIP-0002_Migration_Validation.ipynb` - Interactive validation notebook
- `test_notebook.py` - Automated notebook test runner
- `run_all_migrated_experiments.py` - Python integration suite
- `validate_phase3_entanglement.py` - Phase 3 validation
- `.github/workflows/test-migration-validation.yml` - CI/CD workflow
- `cip/cip0002.md` - Complete CIP documentation

## Contact

For issues with testing infrastructure, see:
- CIP-0002 documentation: `cip/cip0002.md`
- GitHub Issues: File under "testing" or "CI/CD" labels
