# MEPP Test Suite

This directory contains the comprehensive test suite for the Maximum Entropy Production Principle (MEPP) quantum thermalization simulation.

## Test Structure

### Core Physics Tests (`test_core_physics.py`)
- *Bell Pair Creation*: Tests for qubit and qutrit Bell pair states
- *Entropy Calculations*: Von Neumann entropy and coarse-grained entropy
- *Gell-Mann Matrices*: Mathematical properties of qutrit operators
- *Unitary Gates*: Verification that generated gates are unitary

### Thermalization Tests (`test_thermalization.py`)
- *Dephasing Stage*: Tests entropy increase and trace preservation
- *Isolation Stage*: Tests entropy increase and trace preservation
- *Full Thermalization*: End-to-end thermalization process
- *Thermalization Properties*: Density matrix properties and entropy bounds

### Performance Tests (`test_performance.py`)
- *Performance Benchmarks*: Speed and memory usage tests
- *Scalability Tests*: Large system performance (marked as slow)
- *Tensor Network Tests*: Tensor network usage verification
- *Benchmark Tests*: Automated performance benchmarking

## Running Tests

### Basic Test Execution
```bash
# Run all tests
poetry run pytest tests/ -v

# Run specific test file
poetry run pytest tests/test_core_physics.py -v

# Run specific test class
poetry run pytest tests/test_core_physics.py::TestBellPairCreation -v

# Run specific test method
poetry run pytest tests/test_core_physics.py::TestBellPairCreation::test_qubit_bell_pairs -v
```

### Test Categories
```bash
# Run only unit tests
poetry run pytest tests/ -m unit

# Run only integration tests
poetry run pytest tests/ -m integration

# Run only physics validation tests
poetry run pytest tests/ -m physics

# Run only performance tests
poetry run pytest tests/ -m performance

# Skip slow tests
poetry run pytest tests/ -m "not slow"
```

### Coverage Reports
```bash
# Run with coverage
poetry run pytest tests/ --cov=mepp --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Test Results Summary

### Current Status (Latest Run)
- *Total Tests*: 50
- *Passed*: 45 ✅
- *Failed*: 3 ❌
- *Skipped*: 2 ⏭️
- *Code Coverage*: 38%

### Known Issues
1. *Performance Tests*: Large qutrit systems (6 qudits, d=3) are slow but functional
2. *Qutrit Isolation*: Minor entropy decrease in one test case (investigation needed)

### Test Categories
- *Unit Tests*: Core physics components (Bell pairs, entropy, Gell-Mann matrices)
- *Integration Tests*: Thermalization stages and full process
- *Performance Tests*: Speed and scalability benchmarks
- *Physics Tests*: Mathematical and physical property validation

## Adding New Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Test Markers
Use appropriate markers for test categorization:
```python
@pytest.mark.unit
@pytest.mark.integration
@pytest.mark.physics
@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.benchmark
```

### Fixtures
Use shared fixtures from `conftest.py`:
- `qubit_simulator`: 4-qubit simulator
- `qutrit_simulator`: 4-qutrit simulator
- `mixed_qubit_state`: Mixed qubit state after dephasing
- `mixed_qutrit_state`: Mixed qutrit state after dephasing

## Continuous Integration

The test suite is designed to work with CI/CD pipelines:
- Fast tests run on every commit
- Slow tests run on pull requests
- Coverage reports are generated automatically
- Performance benchmarks track regressions

## Debugging Failed Tests

### Common Issues
1. *Import Errors*: Ensure `sys.path` includes project root
2. *Numerical Precision*: Use appropriate tolerances for floating-point comparisons
3. *Performance Timeouts*: Adjust time limits for different hardware

### Debug Commands
```bash
# Run with detailed output
poetry run pytest tests/ -v -s

# Run single failing test
poetry run pytest tests/test_thermalization.py::TestFullThermalization::test_thermalization_monotonicity -v -s

# Run with maximum verbosity
poetry run pytest tests/ -vvv
``` 