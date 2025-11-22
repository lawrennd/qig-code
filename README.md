# The Inaccessible Game (Quantum Implementation)

[![CI/CD Tests](https://github.com/lawrennd/the-inaccessible-game-orgin/actions/workflows/test-migration-validation.yml/badge.svg)](https://github.com/lawrennd/the-inaccessible-game-orgin/actions/workflows/test-migration-validation.yml)
[![Migration Status](https://img.shields.io/badge/CIP--0002-COMPLETED-success)](cip/cip0002.md)
[![Entanglement](https://img.shields.io/badge/Entanglement-GENUINE-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.11+-blue)]()
[![License](https://img.shields.io/badge/License-TBD-lightgrey)]()

A Python implementation of the quantum inaccessible game: a constrained information geometry framework for studying maximum entropy production in quantum systems with marginal entropy constraints.

> **Note**: The CI/CD badge above automatically updates based on GitHub Actions workflow status. If you've forked this repository, update the badge URL in the README to point to your fork.

## 🎯 Overview

This codebase implements the quantum version of the "inaccessible game" described in the paper *"The Inaccessible Game: Constrained Information Geometry for Quantum Systems"*. The game studies the evolution of quantum states under the constraint that marginal entropies remain constant, revealing connections between:

- **Quantum information geometry** (Fisher information, BKM metric)
- **Maximum entropy production** (GENERIC framework)
- **Entanglement dynamics** (mutual information evolution)
- **Optimal quantum systems** (qutrit optimality)

### Key Features

- ✅ **Genuine entanglement**: Creates and evolves maximally entangled pairs (Bell states, qutrit LME states)
- ✅ **Pair operators**: Uses su(d²) generators for qubit/qutrit pairs (not just local operators)
- ✅ **Constrained dynamics**: Marginal entropy constraint preserved to machine precision
- ✅ **Validated implementation**: Comprehensive test suite with 4 validation experiments
- ✅ **Interactive notebooks**: Jupyter notebooks with explanations and visualizations

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/the-inaccessible-game-orgin.git
cd the-inaccessible-game-orgin

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
pytest tests/
python test_notebook.py
```

### Development Setup

For contributors, set up notebook filtering to automatically strip outputs:

```bash
# Install nbstripout (included in requirements.txt)
pip install nbstripout

# Install git filter (one-time setup)
nbstripout --install
```

This automatically strips notebook outputs, execution counts, and metadata on commit, preventing merge conflicts and keeping the repository clean.

See [CONTRIBUTING.md](CONTRIBUTING.md) for complete development guidelines.

### Basic Usage

```python
from qig.exponential_family import QuantumExponentialFamily
from qig.dynamics import InaccessibleGameDynamics
from qig.core import create_lme_state

# Create a system with 1 qutrit pair (genuine entanglement!)
exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
dynamics = InaccessibleGameDynamics(exp_fam)

# Create maximally entangled initial state
rho_lme, dims = create_lme_state(n_sites=2, d=3)

# Integrate constrained dynamics
import numpy as np
theta_0 = np.random.randn(exp_fam.n_params) * 0.1
solution = dynamics.integrate(theta_0, (0, 5.0), n_points=100)

# Check entanglement evolution
I_initial = exp_fam.mutual_information(theta_0)
I_final = exp_fam.mutual_information(solution['theta'][-1])
print(f"Mutual information: {I_initial:.3f} → {I_final:.3f}")
```

### Run Validation Experiments

```bash
# Run all 4 validation experiments (interactive notebook)
jupyter notebook CIP-0002_Migration_Validation.ipynb

# Or run programmatically
python test_notebook.py

# Run comprehensive Python suite
python run_all_migrated_experiments.py
```

## 📂 Project Structure

```
the-inaccessible-game-orgin/
├── qig/                                    # Core quantum information geometry library
│   ├── exponential_family.py              # Quantum exponential family (pair operators)
│   ├── dynamics.py                         # Constrained dynamics integration
│   ├── core.py                             # Basic quantum operations
│   ├── pair_operators.py                   # su(d²) generators for entangled pairs
│   └── duhamel.py                          # High-precision quantum derivatives
│
├── CIP-0002_Migration_Validation.ipynb    # Interactive validation notebook
├── run_all_migrated_experiments.py        # Unified validation suite
├── test_notebook.py                        # Automated notebook testing
│
├── quantum_qutrit_n3.py                    # Backward-compatible wrapper (migrated)
├── inaccessible_game_quantum.py           # Core game implementation
├── validate_qutrit_optimality.py          # Qutrit optimality tests
├── advanced_analysis.py                    # GENERIC decomposition, time parametrizations
├── run_qutrit_experiment.py               # Full qutrit experiments
├── run_qutrit_quick.py                     # Quick qutrit tests
│
├── tests/                                  # Test suite
│   ├── test_pair_exponential_family.py    # Pair operator tests
│   ├── test_pair_numerical_validation.py  # Numerical gradient validation
│   ├── test_jacobian_analytic.py          # Analytic Jacobian tests
│   └── ...                                 # Additional test files
│
├── cip/                                    # Code Improvement Plans
│   ├── cip0001.md                         # Consolidation & documentation
│   └── cip0002.md                         # LOCAL → PAIR migration (COMPLETED)
│
├── backlog/                                # Task tracking
├── .github/workflows/                      # CI/CD pipelines
├── TESTING.md                              # Testing guide
└── README.md                               # This file
```

## 🔬 The Quantum Inaccessible Game

### What is it?

The inaccessible game studies quantum systems evolving under a **marginal entropy constraint**:

```
Constraint: C(θ) = Σᵢ hᵢ(θ) = constant
Dynamics:   θ̇ = -Π∥(G·θ)
```

where:
- `θ` are natural parameters of a quantum exponential family
- `hᵢ` are marginal von Neumann entropies
- `G` is the BKM metric (quantum Fisher information)
- `Π∥` projects onto the constraint manifold

### Why "Inaccessible"?

The constraint makes the **joint entropy H(θ) inaccessible** to direct control. The system evolves to maximize H while keeping marginal entropies fixed, revealing:

- Maximum entropy production (dH/dt ≥ 0)
- Entanglement-entropy tradeoffs
- GENERIC structure (dissipative + Hamiltonian)
- Qutrit optimality under resource constraints

## ⚠️ Critical Migration: CIP-0002 (November 2025)

### The Problem

**Original implementation used LOCAL operators** (Pauli σ_x ⊗ I, Gell-Mann λ_a ⊗ I ⊗ I):
- ❌ Could only create **separable states** (mutual information I = 0 always)
- ❌ Contradicted paper's claim of "locally maximally entangled initial states"
- ❌ Could not represent Bell states or LME states

### The Solution

**Migrated to PAIR operators** (su(4) for qubits, su(9) for qutrits):
- ✅ Can create **genuine entanglement** (I > 0)
- ✅ Achieves maximal entanglement: I = 2log(d)
- ✅ Paper consistency restored
- ✅ All 7 legacy scripts migrated and tested

### Impact

| Metric | Before (LOCAL) | After (PAIR) | Change |
|--------|----------------|--------------|--------|
| **Qubit params** | 6 (3×2 sites) | 15 (su(4)) | 2.5× |
| **Qutrit params** | 24 (8×3 sites) | 80 (su(9)) | 3.3× |
| **Max I (qubits)** | 0.000 | **1.386** | **∞** |
| **Max I (qutrits)** | 0.000 | **2.197** | **∞** |
| **Entanglement** | ❌ Impossible | ✅ Genuine | **Qualitative** |

**See [CIP-0002](cip/cip0002.md) for complete migration documentation.**

## 🧪 Testing

### Local Testing

```bash
# Run pytest suite
pytest tests/

# Quick smoke test
DYNAMICS_POINTS=5 DYNAMICS_T_MAX=0.5 python test_notebook.py

# Full validation
python test_notebook.py

# All migrated experiments
python run_all_migrated_experiments.py
```

### CI/CD

[![CI/CD Status](https://github.com/lawrennd/the-inaccessible-game-orgin/actions/workflows/test-migration-validation.yml/badge.svg)](https://github.com/lawrennd/the-inaccessible-game-orgin/actions/workflows/test-migration-validation.yml)

GitHub Actions workflow **automatically runs** on every push/PR with 4 test jobs:

1. **Default Config** - Full validation (20 integration points, d=3)
   - Tests: 4/4 experiments (entanglement, dynamics, comparison, API)
   - Runtime: ~30-40 seconds
   
2. **Quick Config** - Fast smoke test (5 points, 0.5s)
   - Same tests with reduced parameters
   - Runtime: ~15-20 seconds
   
3. **Python Suite** - All migrated scripts
   - `run_all_migrated_experiments.py`
   - `validate_phase3_entanglement.py`
   - Runtime: ~30 seconds
   
4. **Custom Config** - Manual workflow dispatch
   - Specify custom `QUTRIT_DIM`, `DYNAMICS_POINTS`
   - Useful for edge case testing

**View Results**: Repository → Actions → "CIP-0002 Migration Validation"

See [TESTING.md](TESTING.md) for complete testing guide.

### Parameterized Testing

Use environment variables to customize tests:

```bash
# Test with 2 qutrit pairs
N_PAIRS=2 python test_notebook.py

# Test with ququarts (d=4)
QUTRIT_DIM=4 python test_notebook.py

# Stricter numerical tolerance
TOLERANCE=1e-8 python test_notebook.py
```

## 📚 Documentation

### Core Documentation

- **[TESTING.md](TESTING.md)**: Complete testing guide with examples
- **[CIP-0001](cip/cip0001.md)**: Code consolidation & documentation
- **[CIP-0002](cip/cip0002.md)**: LOCAL → PAIR migration (critical!)
- **[Backlog](backlog/)**: Task tracking and feature planning

### Interactive Notebooks

- **[CIP-0002_Migration_Validation.ipynb](CIP-0002_Migration_Validation.ipynb)**: 
  - 4 experiments validating the migration
  - Markdown explanations between code
  - Parameterizable via environment variables

- **[quantum_qutrit_experiments.ipynb](quantum_qutrit_experiments.ipynb)**:
  - Qutrit system experiments
  - Visualization and analysis

### Key Modules

#### `qig.exponential_family.QuantumExponentialFamily`
Quantum exponential family with pair operators:
```python
exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
rho = exp_fam.density_matrix(theta)
G = exp_fam.fisher_information(theta)
I = exp_fam.mutual_information(theta)
```

#### `qig.dynamics.InaccessibleGameDynamics`
Constrained dynamics integration:
```python
dynamics = InaccessibleGameDynamics(exp_fam)
solution = dynamics.integrate(theta_0, (0, 5.0), n_points=100)
# solution contains: theta, H, constraint, time, success
```

#### `qig.pair_operators`
Operator basis generation:
```python
from qig.pair_operators import bell_state, gell_mann_generators
rho_bell = bell_state(d=2)  # Qubit Bell state
su9_ops = gell_mann_generators(d=3)  # 80 su(9) generators
```

## 🎓 Paper & Theory

### Key Results

1. **Maximum entropy production**: Systems evolve to maximize joint entropy H while preserving marginal entropies
2. **Qutrit optimality**: Qutrits (d=3) are optimal under certain resource constraints
3. **GENERIC structure**: Dynamics decompose into dissipative (symmetric) + Hamiltonian (antisymmetric) parts
4. **Block-diagonal Fisher metric**: Non-interacting pairs → sparse structure → computational tractability

### Theoretical Framework

The quantum exponential family:
```
ρ(θ) = exp(Σₐ θₐFₐ) / Z(θ)
```

where `Fₐ` are su(d²) generators for each pair.

**BKM Metric** (quantum Fisher information):
```
Gₐᵦ = Tr[(∂ρ/∂θₐ)(∂log ρ/∂θᵦ)]
```

**Marginal entropy constraint**:
```
C(θ) = Σᵢ hᵢ(θ) = constant
hᵢ = -Tr[ρᵢ log ρᵢ]
```

**Projected dynamics**:
```
θ̇ = -Π∥(G·θ) = -(I - a(aᵀGa)⁻¹aᵀG)·G·θ
```

where `a = ∇C` is the constraint gradient.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Quick Start for Contributors

1. Fork and clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. **Set up notebook filtering**: `nbstripout --install` (prevents merge conflicts)
4. Create a feature branch: `git checkout -b feature/your-feature`
5. Make changes and add tests
6. Run test suite: `pytest tests/ && python test_notebook.py`
7. Commit with clear messages (reference CIPs/backlog when applicable)
8. Push and create pull request

**Important**: Use surgical `git add` for specific files, not `git add -A`. See [CONTRIBUTING.md](CONTRIBUTING.md) for VibeSafe workflow guidelines.

### Project Management (VibeSafe)

This project uses [VibeSafe](https://github.com/lawrennd/vibesafe) for structured project management:
- **Tenets**: `tenets/` - Guiding principles
- **Backlog**: `backlog/` - Task tracking
- **CIPs**: `cip/` - Code improvement plans (see [CIP-0002](cip/cip0002.md) for example)
- **AI-Requirements**: `ai-requirements/` - Requirements framework

Run `./whats-next` to see current project status, pending tasks, and next steps.

## 📊 Validation Results

Current migration validation (November 2025):

```
✅✅✅ ALL EXPERIMENTS PASSED ✅✅✅

Experiment 1: Entanglement Validation
  LME state: I = 2.197 (100% maximal) ✓
  Generic state: I = 0.560 (genuine) ✓

Experiment 2: Qubit Pair Dynamics
  Constraint violation: 6.55e-09 (< 1e-6) ✓
  Entropy increase: ΔH = 0.030 ≥ 0 ✓
  Entanglement maintained ✓

Experiment 3: Qutrit vs Qubit
  Qutrit advantage: 1.156× ✓

Experiment 4: API Compatibility
  All tests passed ✓

Time: ~30 seconds
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{lawrence2025inaccessible,
  title={The Inaccessible Game: Constrained Information Geometry for Quantum Systems},
  author={Lawrence, Neil D.},
  journal={TBD},
  year={2025}
}
```

## 📄 License

[License information to be added]

## 🙏 Acknowledgments

- VibeSafe project management framework
- NumPy/SciPy for numerical computation
- Jupyter for interactive notebooks
- pytest for testing infrastructure

## 📬 Contact

For questions about the code or paper:
- GitHub Issues: [Link to issues]
- Email: [Contact email]

---

**Status**: ✅ CIP-0002 Migration Complete (November 2025)  
**Tests**: 4/4 Passing  
**Entanglement**: Genuine (I = 2.197 for qutrits)
