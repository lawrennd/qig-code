# QIG - Quantum Information Geometry

[![Tests](https://github.com/lawrennd/qig-code/actions/workflows/tests.yml/badge.svg)](https://github.com/lawrennd/qig-code/actions/workflows/tests.yml)
[![Notebook Tests](https://github.com/lawrennd/qig-code/actions/workflows/notebook-tests.yml/badge.svg)](https://github.com/lawrennd/qig-code/actions/workflows/notebook-tests.yml)
[![Documentation](https://github.com/lawrennd/qig-code/actions/workflows/docs.yml/badge.svg)](https://github.com/lawrennd/qig-code/actions/workflows/docs.yml)
[![Documentation Status](https://readthedocs.org/projects/qig/badge/?version=latest)](https://qig.readthedocs.io/en/latest/?badge=latest)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for quantum information geometry: constrained dynamics in quantum exponential families, featuring the "inaccessible game" framework for studying maximum entropy production with marginal entropy constraints.

## Overview

`qig` implements quantum systems evolving under marginal entropy constraints, revealing connections between:

- **Quantum information geometry** (Fisher information, BKM metric)
- **Maximum entropy production** (GENERIC framework)
- **Entanglement dynamics** (mutual information evolution)
- **Optimal quantum systems** (qutrit optimality)

## Documentation

**Documentation is available at [qig.readthedocs.io](https://qig.readthedocs.io/)**

The documentation includes:
- **Getting Started**: Installation and quick start guide
- **User Guide**: Detailed usage examples and tutorials
- **API Reference**: Complete API documentation with examples
- **Theory**: Mathematical background and derivations
- **Development Guide**: Contributing guidelines and testing documentation

You can also build the documentation locally:
```bash
cd docs
pip install -r requirements.txt
make html
# Open docs/build/html/index.html in your browser
```

## Installation

### From GitHub
```bash
pip install -q git+https://github.com/lawrennd/qig-code.git
```

### Development Installation
```bash
git clone https://github.com/lawrennd/qig-code.git
cd qig-code
pip install -e .
```

### With Development Tools
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from qig.exponential_family import QuantumExponentialFamily
from qig.dynamics import InaccessibleGameDynamics
from qig.core import create_lme_state
import numpy as np

# Create a system with 1 qutrit pair (genuine entanglement!)
exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
dynamics = InaccessibleGameDynamics(exp_fam)

# Create maximally entangled initial state
rho_lme, dims = create_lme_state(n_sites=2, d=3)

# Integrate constrained dynamics
theta_0 = np.random.randn(exp_fam.n_params)
solution = dynamics.solve_constrained_maxent(
    theta_init, 
    n_steps=500,
    dt=0.005,
    convergence_tol=1e-5,
    project=True,
    project_every=10
)

# Check entanglement evolution
I_initial = exp_fam.mutual_information(theta_0)
I_final = exp_fam.mutual_information(solution['theta'][-1])
print(f"Mutual information: {I_initial:.3f} → {I_final:.3f}")
```

## The Inaccessible Game

The package implements quantum systems under *marginal entropy constraints*:

```
Constraint: C(θ) = Σᵢ hᵢ(θ) = constant
Dynamics:   θ̇ = -Π∥(G·θ)
```

where:
- `θ` are natural parameters of a quantum exponential family
- `hᵢ` are marginal von Neumann entropies  
- `G` is the BKM metric (quantum Fisher information)
- `Π∥` projects onto the constraint manifold

### Key Results

1. **Maximum entropy production**: Systems evolve to maximise joint entropy H while preserving marginal entropies
2. **Qutrit optimality**: Qutrits ($d=3$) are optimal under certain resource constraints
3. **GENERIC structure**: Dynamics decompose into dissipative (S) + Hamiltonian (A) parts
4. **Block-diagonal Fisher metric**: Non-interacting pairs enable computational tractability
5. **Exact analytic forms**: For LME states, A and S have closed-form symbolic expressions (no approximations)

## API Documentation

### Core Modules

#### `qig.exponential_family.QuantumExponentialFamily`

Quantum exponential family with pair operators:

```python
exp_fam = QuantumExponentialFamily(
    n_pairs=1,      # Number of entangled pairs
    d=3,            # Local dimension (qubits: d=2, qutrits: d=3)
    pair_basis=True # Use su(d²) generators per pair
)

# Compute quantum properties
rho = exp_fam.density_matrix(theta)          # Density matrix
G = exp_fam.fisher_information(theta)        # BKM metric
H = exp_fam.entropy(theta)                   # von Neumann entropy
I = exp_fam.mutual_information(theta)        # Mutual information
h = exp_fam.marginal_entropies(theta)        # Marginal entropies
```

#### `qig.dynamics.InaccessibleGameDynamics`

Constrained dynamics integration:

```python
dynamics = InaccessibleGameDynamics(exp_fam)

solution = dynamics.integrate(
    theta_0,           # Initial parameters
    (0, 5.0),         # Time span
    n_points=100,     # Number of points
    method='RK45'     # Integration method
)

# Solution contains:
# - theta: parameter trajectory
# - H: entropy evolution
# - constraint: constraint satisfaction
# - time: time points
# - success: integration status
```

#### `qig.pair_operators`

Operator basis generation.

```python
from qig.pair_operators import bell_state, gell_mann_generators

# Qubit Bell state
rho_bell = bell_state(d=2)

# su(9) generators for qutrit pairs
su9_ops = gell_mann_generators(d=3)  # 80 generators
```

#### `qig.duhamel`

Quantum derivatives using Duhamel's formula.

```python
from qig.duhamel import duhamel_derivative

# Precise derivatives for quantum exponential families
dH_dtheta = duhamel_derivative(rho, drho_dtheta, order=10)
```

#### `qig.symbolic`

Symbolic computation for GENERIC decomposition of *qutrit* pairs.

**Parameterisation**: The code uses the quantum exponential family with Gell-Mann matrices as sufficient statistics.

```
ρ(θ) = exp(K - ψ(θ)·I)   where   K = Σₐ θₐ Fₐ
```

The sufficient statistics `Fₐ` are tensor products of Gell-Mann matrices `{λᵢ}`:
- **Local**: `λᵢ ⊗ I` and `I ⊗ λⱼ` (16 generators)
- **Entangling**: `λᵢ ⊗ λⱼ` (64 generators)

giving 80 generators spanning su(9) for a qutrit pair.

*The exactness trick*: For locally maximally entangled (LME) states, the 9×9 matrix exponential `exp(K)` decomposes into smaller blocks:

```
9×9 → 3×3 + 2×2 + 1×1×4
```

This happens because LME states live in the 3D subspace `{|00⟩, |11⟩, |22⟩}`. The 3×3 and 2×2 blocks have eigenvalues from *quadratic* (not cubic) equations, enabling exact symbolic computation.

**20 block-preserving generators** maintain this structure: 4 local diagonal (`λ₃⊗I`, `I⊗λ₃`, `λ₈⊗I`, `I⊗λ₈`) plus 16 entangling (`λᵢ⊗λⱼ` for i,j ∈ {1,2,3}).

```python
from qig.symbolic.lme_exact import (
    exact_exp_K_lme,
    exact_constraint_lme,
    block_preserving_generators,
    numeric_lme_blocks_from_theta,  # Bridge to numeric representation
)
import sympy as sp

# 20 generators that preserve LME block structure
generators, names = block_preserving_generators()

# Natural parameters θ as coefficients of Gell-Mann tensor products
a = sp.Symbol('a', real=True)  # coefficient of λ₃⊗I (local)
c = sp.Symbol('c', real=True)  # coefficient of λ₁⊗λ₁ (entangling)
theta = {'λ3⊗I': a, 'λ1⊗λ1': c}

# Exact exp(K)
exp_K = exact_exp_K_lme(theta)

# Exact constraint C = h₁ + h₂ (sum of marginal entropies)
C = exact_constraint_lme(theta)

# Bridge numeric θ to symbolic block structure
from qig.exponential_family import QuantumExponentialFamily
qef = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
theta_numeric = qef.get_bell_state_parameters(log_epsilon=-20)
blocks = numeric_lme_blocks_from_theta(theta_numeric, qef.operators)
# blocks['ent_3x3'] is the 3×3 entangled block of K(θ)
```

Key features:
- **Exact exp(K)** via block decomposition
- The decomposition avoids the need for Taylor approximation in the LME dynamics
- Analytic forms for antisymmetric (A) and symmetric (S) parts of GENERIC
- **Numeric-symbolic bridge** via `numeric_lme_blocks_from_theta` connects exponential family θ to block parameters
- See [symbolic computation docs](https://qig.readthedocs.io/en/latest/theory/symbolic_computation.html) for details

## Testing

The test suite is organised into categories with pytest markers:

```bash
# Run standard tests (excludes slow and integration tests)
pytest

# Run all tests including slow ones
pytest -m "slow"

# Run integration tests (notebooks)
pytest -m "integration"

# Run everything
pytest -m ""

# Run with coverage
pytest --cov=qig --cov-report=html
```

### Test Categories

- **Standard tests** (~340 tests, ~3 min): Core functionality, unit tests, numerical validation
- **Slow tests** (~35 tests): Computationally intensive tests (e.g., qutrit pair finite-difference Jacobians)
- **Integration tests**: Notebook smoke tests
- **Skipped tests**: Tests for future CIP-0007 symbolic features (skip gracefully)

The default `pytest` command runs standard tests only, which is suitable for CI/CD and development workflows.

## Examples

See the `examples/` directory:

- **`generate-origin-paper-figures.ipynb`**: Interactive demonstration notebook with validation experiments [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lawrennd/qig-code/blob/main/examples/generate-origin-paper-figures.ipynb)

- **`symbolic_verification_experiments.ipynb`**: Verification of key theoretical predictions (qutrit optimality, constraint linearization, structural identity ν=-1) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lawrennd/qig-code/blob/main/examples/symbolic_verification_experiments.ipynb)

- **`lme_numeric_symbolic_bridge.ipynb`**: Bridge between numeric exponential family and symbolic LME decomposition, showing block structure, eigenvalue analysis, and natural parameter scaling [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lawrennd/qig-code/blob/main/examples/lme_numeric_symbolic_bridge.ipynb)

- **`entropy_time_analysis.ipynb`**: Analysis of entropy time reparameterisation and its relationship to Fisher information geometry, showing time dilation effects near entropy extrema [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lawrennd/qig-code/blob/main/examples/entropy_time_analysis.ipynb)

- **`entropy_time_paths.ipynb`**: Exploration of different paths from the LME origin using entropy time, demonstrating the "north pole" analogy, isotropic vs anisotropic regularisation, L'Hôpital-style limits, and how different σ choices reveal a rich family of trajectories sharing the same asymptotic boundary [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lawrennd/qig-code/blob/main/examples/entropy_time_paths.ipynb)

- **`boring_game_dynamics.ipynb`**: Analysis of why the inaccessible game becomes "boring" from the LME origin when constrained and unconstrained dynamics coincide because the marginal entropy constraint is automatically satisfied along the gradient flow [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lawrennd/qig-code/blob/main/examples/boring_game_dynamics.ipynb)

- **`multi_pair_regularisation.ipynb`**: Tutorial demonstrating CIP-0008 regularisation machinery for multi-pair systems—the "north pole" analogy, isotropic/product/general σ choices, different origins via `bell_indices`, and block-diagonal Fisher information with performance benchmarks [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lawrennd/qig-code/blob/main/examples/multi_pair_regularisation.ipynb)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Lawrence-origin25,
  title={The Origin of the Inaccessible Game},
  author={Lawrence, Neil D.},
  journal={arXiv preprint},
  year={2025},
  note={In preparation}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and install
git clone https://github.com/lawrennd/qig-code.git
cd qig-code
pip install -e ".[dev]"

# Set up pre-commit hooks
nbstripout --install

# Run tests
pytest tests/
```

## Links

- **Documentation**: [README](https://github.com/lawrennd/qig-code#readme)
- **Issues**: [GitHub Issues](https://github.com/lawrennd/qig-code/issues)

## Contact

For questions or collaboration:
- GitHub Issues: [qig-code/issues](https://github.com/lawrennd/qig-code/issues)
- GitHub ID @lawrennd
