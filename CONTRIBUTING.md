# Contributing to The Inaccessible Game (Quantum Implementation)

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/the-inaccessible-game-orgin.git
   cd the-inaccessible-game-orgin
   ```
3. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install in development mode**:
   ```bash
   pip install -r requirements.txt
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/new-analysis` for new features
- `bugfix/issue-123` for bug fixes
- `docs/improve-readme` for documentation
- `test/add-coverage` for test improvements
- `cip/0003-description` for new CIPs

### 2. Make Your Changes

- Write clean, readable code following PEP 8 style guidelines
- Add docstrings for all functions, classes, and modules
- Include type hints where appropriate
- Keep functions focused and modular

### 3. Add Tests

All new functionality must include tests:

```python
# tests/test_your_module.py
import pytest
import numpy as np

class TestYourFeature:
    def test_basic_functionality(self):
        result = your_function(input_data)
        assert result == expected_output
    
    def test_numerical_precision(self):
        # Test with known values
        assert np.abs(computed - expected) < 1e-10
```

Run tests locally:
```bash
pytest tests/ -v
```

Check coverage:
```bash
pytest tests/ --cov=qig --cov-report=term
```

### 4. Working with Notebooks

If you're contributing example notebooks or modifying existing ones:

**First-time setup**: Install and configure `nbstripout` to automatically clean notebook outputs before committing:

```bash
pip install nbstripout  # Already in requirements.txt
nbstripout --install    # Install git filter
```

This creates a git filter that automatically strips outputs, execution counts, and metadata from notebooks when you commit, preventing:
- Large binary data (images, plots) from bloating the repository
- Merge conflicts from execution counts and timestamps
- Accidentally committing sensitive data in outputs

The filter is already configured in `.gitattributes`:
```
*.ipynb filter=nbstripout
*.ipynb diff=ipynb
```

**Manual stripping**: To manually strip outputs from notebooks:

```bash
nbstripout *.ipynb
# or for specific files:
nbstripout CIP-0002_Migration_Validation.ipynb
```

**Testing notebooks**: Before submitting, ensure your notebook runs cleanly:

```bash
# Run notebook validation test
python test_notebook.py

# Run with custom parameters
DYNAMICS_POINTS=10 python test_notebook.py
```

**Best practices**:
- Clear all outputs before committing (nbstripout does this automatically)
- Test that notebooks run from a fresh kernel
- Include clear markdown explanations
- Keep computational cells efficient
- Use environment variables for parameters (see CONFIG in validation notebook)

### 5. Update Documentation

- Update README.md if adding new features
- Add or update CIPs for architectural changes
- Update TESTING.md for new test procedures
- Add docstrings following NumPy style:
  ```python
  def compute_fisher_information(theta, operators):
      """
      Compute quantum Fisher information (BKM metric).
      
      Parameters
      ----------
      theta : np.ndarray, shape (n_params,)
          Natural parameters of exponential family
      operators : list of np.ndarray
          Hermitian operators forming the basis
      
      Returns
      -------
      G : np.ndarray, shape (n_params, n_params)
          Fisher information matrix (symmetric, positive semidefinite)
      
      Examples
      --------
      >>> from qig.exponential_family import QuantumExponentialFamily
      >>> exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
      >>> theta = np.random.randn(exp_fam.n_params) * 0.1
      >>> G = exp_fam.fisher_information(theta)
      >>> assert G.shape == (exp_fam.n_params, exp_fam.n_params)
      """
  ```

### 6. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add specific-files.py  # Use surgical adds, NOT git add -A
git commit -m "Add feature: brief description

Detailed explanation of:
- What changed
- Why it changed  
- Any breaking changes or migration notes"
```

**Important**: Follow VibeSafe guidelines:
- ✅ Use surgical `git add` for specific files
- ❌ Never use `git add .` or `git add -A`
- ✅ Commit regularly (after planning, before refactoring, after implementation)
- ✅ Reference CIPs/backlog items when relevant

### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title describing the change
- Description of what the PR does
- Reference to any related CIPs or backlog items
- Test results (all 4 validation experiments should pass)

## Project Management (VibeSafe)

This project uses [VibeSafe](https://github.com/lawrennd/vibesafe) for structured project management:

### Code Improvement Plans (CIPs)

For major architectural changes:
1. Create `cip/cipXXXX.md` using the template
2. Document motivation, implementation plan, and status
3. Reference in commits and PRs
4. See [CIP-0002](cip/cip0002.md) for a complete example

### Backlog Tasks

For features and tasks:
1. Create `backlog/category/YYYY-MM-DD_description.md`
2. Use template from `backlog/task_template.md`
3. Track status (Proposed → Ready → In Progress → Completed)
4. Update index with `python backlog/update_index.py`

### Checking Project Status

```bash
./whats-next  # Shows CIPs, backlog, git status, next steps
```

## Code Style Guidelines

### Python Code

- Follow PEP 8 style guide
- Maximum line length: 100 characters
- Use meaningful variable names
- Prefer explicit over implicit

Example:
```python
# Good
def compute_mutual_information(rho: np.ndarray, dims: list) -> float:
    """Compute mutual information I = Σh_i - H."""
    # Implementation
    
# Avoid
def calc_I(r, d):  # Unclear abbreviations
    # Implementation
```

### Quantum-Specific Guidelines

- Verify Hermiticity of density matrices and operators
- Check positivity and normalization (Tr(ρ) = 1)
- Use high-precision methods for quantum derivatives (Duhamel formula)
- Test commutation relations for operator bases
- Validate physical bounds (0 ≤ entropy ≤ log(d))

## Testing Guidelines

### Test Categories

1. **Unit tests**: Test individual functions
2. **Integration tests**: Test module interactions
3. **Numerical validation**: Compare analytic vs finite differences
4. **Physical properties**: Test quantum constraints (Hermiticity, positivity)

### Test Naming

```python
def test_fisher_information_is_hermitian():
    """BKM metric should be Hermitian (symmetric for real)."""
    
def test_density_matrix_is_normalized():
    """Density matrix trace should equal 1."""
    
def test_mutual_information_is_non_negative():
    """Mutual information I ≥ 0 for all states."""
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_pair_exponential_family.py -v

# Notebook validation
python test_notebook.py

# Quick CI/CD test
DYNAMICS_POINTS=5 DYNAMICS_T_MAX=0.5 python test_notebook.py
```

## Pull Request Checklist

Before submitting your PR, ensure:

- [ ] Code follows style guidelines
- [ ] All pytest tests pass (`pytest tests/ -v`)
- [ ] Notebook validation passes (`python test_notebook.py`)
- [ ] New functionality includes tests with numerical validation
- [ ] Documentation is updated (README, TESTING.md, CIPs)
- [ ] Commit messages are clear and reference CIPs/backlog
- [ ] No merge conflicts with main branch
- [ ] Notebooks have outputs stripped (nbstripout)
- [ ] Quantum properties validated (Hermiticity, positivity, normalization)

## Types of Contributions

### Bug Reports

When reporting bugs, include:
- Python version (`python --version`)
- NumPy/SciPy versions
- Minimal reproducible example
- Expected vs actual behavior
- Error traceback
- Numerical precision issues (if applicable)

### Feature Requests

For new features, describe:
- The problem it solves
- Proposed implementation
- Example usage
- Why it fits the project scope
- Relevant CIP if major change

### Code Contributions

Areas where contributions are especially welcome:
- Additional test coverage
- Numerical precision improvements
- Performance optimizations
- Documentation improvements
- Example notebooks
- Migration of remaining legacy scripts

## Code Review Process

1. Maintainer will review your PR within a few days
2. CI/CD tests will run automatically (4 jobs)
3. Address any requested changes
4. Once approved and tests pass, maintainer will merge
5. Your contribution will be acknowledged in release notes

## Questions?

- Open an issue for general questions
- Reference [TESTING.md](TESTING.md) for testing details
- Reference [CIP-0002](cip/cip0002.md) for migration context
- See [README.md](README.md) for project overview

## License

[License information to be added]

---

Thank you for contributing to The Inaccessible Game! 🎯


