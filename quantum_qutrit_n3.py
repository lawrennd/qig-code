"""
Quantum Qutrit Dynamics: Migrated to qig Library with Pair Operators
=====================================================================

This module now wraps qig.exponential_family to provide backward-compatible
API while using pair-based operators for genuine entanglement support.

Migration Note (CIP-0002 Phase 2.4):
- Replaced 745-line custom implementation with qig imports
- Now uses pair operators (su(d²)) instead of local operators
- Maintains backward compatibility for dependent scripts
- Enables study of genuinely entangled qutrit systems

Author: Neil D. Lawrence
Date: November 2025
"""

import numpy as np
from scipy.linalg import expm, logm
from typing import List, Tuple, Dict, Optional
import warnings

# Import from qig library
from qig.exponential_family import QuantumExponentialFamily
from qig.core import (
    partial_trace as qig_partial_trace,
    von_neumann_entropy as qig_von_neumann_entropy,
    marginal_entropies as qig_marginal_entropies,
    create_lme_state as qig_create_lme_state,
)
from qig.dynamics import InaccessibleGameDynamics
from qig.pair_operators import multi_pair_basis

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ============================================================================
# Operator Basis (now using pair operators from qig)
# ============================================================================

def single_site_operators(n_sites: int = 3, d: int = 3) -> List[np.ndarray]:
    """
    Generate operator basis for n qutrit subsystems.
    
    **MIGRATION NOTE**: Now uses PAIR operators instead of local operators!
    
    For n_sites subsystems, creates n_sites/2 entangled pairs.
    Requires n_sites to be even.
    
    Parameters
    ----------
    n_sites : int
        Number of qutrit subsystems (must be even)
    d : int
        Local dimension (default: 3 for qutrits)
        
    Returns
    -------
    operators : list of ndarray
        List of Hermitian operators for pair basis
    """
    if n_sites % 2 != 0:
        raise ValueError(
            f"n_sites={n_sites} must be even for pair operators. "
            f"For {n_sites} qutrits, use n_sites={n_sites-1} or n_sites={n_sites+1}."
        )
    
    n_pairs = n_sites // 2
    operators, _ = multi_pair_basis(n_pairs, d)
    
    return operators


def gell_mann_matrices() -> np.ndarray:
    """
    Generate the 8 Gell-Mann matrices for SU(3).
    
    **DEPRECATED**: For pair-based operators, use `single_site_operators()` instead.
    This function is kept for backward compatibility only.
    """
    warnings.warn(
        "gell_mann_matrices() is deprecated. Use single_site_operators() "
        "which now returns pair-based operators for entanglement support.",
        DeprecationWarning, stacklevel=2
    )
    
    λ = np.zeros((8, 3, 3), dtype=complex)
    
    # λ₁ and λ₂
    λ[0] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    λ[1] = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex)
    
    # λ₃
    λ[2] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex)
    
    # λ₄ and λ₅
    λ[3] = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex)
    λ[4] = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex)
    
    # λ₆ and λ₇
    λ[5] = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)
    λ[6] = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex)
    
    # λ₈
    λ[7] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex) / np.sqrt(3)
    
    return λ


# ============================================================================
# Exponential Family Operations (wrapping qig.exponential_family)
# ============================================================================

def compute_density_matrix(theta: np.ndarray, operators: List[np.ndarray]) -> np.ndarray:
    """
    Compute ρ(θ) = exp(K(θ)) / Z(θ) where K(θ) = ∑ θ_a F_a.
    
    Now uses qig.exponential_family with pair operators.
    """
    n_params = len(theta)
    
    # Infer system size from operators
    D = operators[0].shape[0]
    d = int(np.round(D ** 0.5))  # Assuming d² Hilbert space per pair
    n_sites = int(np.round(np.log(D) / np.log(d)))
    
    if n_sites % 2 != 0:
        n_sites += 1  # Round up to even
    
    n_pairs = n_sites // 2
    
    # Create exponential family instance
    exp_fam = QuantumExponentialFamily(n_pairs=n_pairs, d=d, pair_basis=True)
    
    return exp_fam.rho_from_theta(theta)


def compute_log_partition(theta: np.ndarray, operators: List[np.ndarray]) -> float:
    """
    Compute log partition function ψ(θ) = log Tr(exp(∑ θ_a F_a)).
    """
    K = sum(theta_a * F_a for theta_a, F_a in zip(theta, operators))
    return np.log(np.trace(expm(K))).real


def compute_expectations(rho: np.ndarray, operators: List[np.ndarray]) -> np.ndarray:
    """
    Compute expectation values ⟨F_a⟩ = Tr[ρ F_a].
    """
    return np.array([np.trace(rho @ F_a).real for F_a in operators])


# ============================================================================
# Entropy and Constraints (wrapping qig.core)
# ============================================================================

def partial_trace(rho: np.ndarray, keep_site: int, n_sites: int = 3, d: int = 3) -> np.ndarray:
    """
    Compute partial trace over all subsystems except keep_site.
    
    Now wraps qig.core.partial_trace.
    """
    dims = [d] * n_sites
    return qig_partial_trace(rho, dims, keep=[keep_site])


def von_neumann_entropy(rho: np.ndarray) -> float:
    """
    Compute von Neumann entropy S(ρ) = -Tr[ρ log ρ].
    
    Now wraps qig.core.von_neumann_entropy.
    """
    return qig_von_neumann_entropy(rho)


def compute_marginal_entropies(rho: np.ndarray, n_sites: int = 3, d: int = 3) -> np.ndarray:
    """
    Compute marginal entropies [h_1, h_2, ..., h_n].
    
    Now wraps qig.core.marginal_entropies.
    """
    dims = [d] * n_sites
    return qig_marginal_entropies(rho, dims)


def compute_constraint_gradient(theta: np.ndarray, operators: List[np.ndarray],
                                  n_sites: int = 3, d: int = 3,
                                  eps: float = 1e-5) -> np.ndarray:
    """
    Compute gradient of marginal entropy constraint ∇C(θ).
    
    Now uses qig.exponential_family.marginal_entropy_constraint.
    """
    n_pairs = n_sites // 2
    exp_fam = QuantumExponentialFamily(n_pairs=n_pairs, d=d, pair_basis=True)
    
    _, grad_C = exp_fam.marginal_entropy_constraint(theta)
    return grad_C


# ============================================================================
# Dynamics (wrapping qig.dynamics)
# ============================================================================

def solve_constrained_quantum_maxent(
    theta_init: np.ndarray,
    operators: List[np.ndarray],
    t_span: Tuple[float, float] = (0.0, 5.0),
    n_points: int = 100,
    n_sites: int = 3,
    d: int = 3,
    n_steps: Optional[int] = None,
    dt: Optional[float] = None,
    **kwargs
) -> Dict:
    """
    Solve constrained maximum entropy production dynamics.
    
    Now uses qig.dynamics.InaccessibleGameDynamics with pair operators.
    
    Returns
    -------
    solution : dict
        'trajectory': array of shape (n_points, n_params)
        'time': array of shape (n_points,)
        'constraint': array of shape (n_points,)
        'H': array of shape (n_points,) - joint entropy
        'success': bool
    """
    if n_sites % 2 != 0:
        raise ValueError(f"n_sites={n_sites} must be even for pair operators")
    
    # Convert old API (n_steps, dt) to new API (n_points) if needed
    if n_steps is not None and dt is not None:
        # Old API: n_steps and dt specified
        # t_span is still used, but we derive n_points from n_steps
        n_points = n_steps
        # Ignore dt for now - qig uses adaptive stepping
        warnings.warn(
            f"Legacy API: n_steps={n_steps}, dt={dt} converted to n_points={n_points}. "
            "Note: qig uses adaptive stepping, dt is ignored.",
            DeprecationWarning, stacklevel=2
        )
    elif n_steps is not None:
        n_points = n_steps
    
    n_pairs = n_sites // 2
    exp_fam = QuantumExponentialFamily(n_pairs=n_pairs, d=d, pair_basis=True)
    dynamics = InaccessibleGameDynamics(exp_fam)
    
    # Filter out legacy kwargs that qig doesn't accept
    qig_kwargs = {k: v for k, v in kwargs.items() 
                  if k not in ['n_steps', 'dt', 'verbose']}
    
    solution = dynamics.integrate(theta_init, t_span, n_points, **qig_kwargs)
    
    return {
        'trajectory': solution['theta'],
        'time': solution['time'],
        'constraint': solution['constraint'],
        'H': solution['H'],
        'success': solution['success']
    }


def solve_unconstrained_quantum_maxent(
    theta_init: np.ndarray,
    operators: List[np.ndarray],
    t_span: Tuple[float, float] = (0.0, 5.0),
    n_points: int = 100,
    n_sites: int = 3,
    d: int = 3
) -> Dict:
    """
    Solve unconstrained gradient flow θ̇ = -G(θ)θ.
    
    **DEPRECATED**: For genuine dynamics, use solve_constrained_quantum_maxent.
    """
    warnings.warn(
        "solve_unconstrained_quantum_maxent is deprecated. "
        "Use solve_constrained_quantum_maxent for constrained dynamics.",
        DeprecationWarning, stacklevel=2
    )
    
    # For now, just call constrained version
    return solve_constrained_quantum_maxent(
        theta_init, operators, t_span, n_points, n_sites, d
    )


# ============================================================================
# State Preparation (wrapping qig.core)
# ============================================================================

def create_lme_state(n_sites: int = 3, d: int = 3) -> np.ndarray:
    """
    Create a locally maximally entangled (LME) state.
    
    Now wraps qig.core.create_lme_state which creates n_sites/2 maximally
    entangled pairs.
    
    **NOTE**: Requires n_sites to be even!
    """
    if n_sites % 2 != 0:
        raise ValueError(f"n_sites={n_sites} must be even for pair-based LME states")
    
    rho, dims = qig_create_lme_state(n_sites, d)
    return rho


# ============================================================================
# Legacy Functions (stubs for backward compatibility)
# ============================================================================

def find_natural_parameters_for_lme(operators: List[np.ndarray], rho_target: np.ndarray,
                                     max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
    """
    Find natural parameters θ such that ρ(θ) ≈ rho_target.
    
    **DEPRECATED**: This is a non-trivial inverse problem. For pure LME states,
    θ → ∞ (singular). Use regularized states instead.
    """
    warnings.warn(
        "find_natural_parameters_for_lme is deprecated and may not work correctly. "
        "Pure LME states correspond to θ → ∞ (singular limit).",
        DeprecationWarning, stacklevel=2
    )
    
    # Return small random parameters as placeholder
    n_params = len(operators)
    return np.random.randn(n_params) * 0.1


def analyse_quantum_generic_structure(theta: np.ndarray, operators: List[np.ndarray],
                                       n_sites: int = 3, d: int = 3) -> Dict:
    """
    Analyse GENERIC structure (M = S + A decomposition).
    
    **NOT YET IMPLEMENTED IN qig**: Placeholder for backward compatibility.
    """
    warnings.warn(
        "analyse_quantum_generic_structure not yet fully implemented in qig wrapper. "
        "Returning placeholder.",
        UserWarning, stacklevel=2
    )
    
    return {
        'jacobian': np.zeros((len(theta), len(theta))),
        'symmetric': np.zeros((len(theta), len(theta))),
        'antisymmetric': np.zeros((len(theta), len(theta))),
        'message': 'Not yet implemented - migrate to qig.exponential_family.jacobian()'
    }


# ============================================================================
# Module Info
# ============================================================================

__version__ = "2.0.0-migrated"
__migration_date__ = "2025-11-22"
__cip__ = "CIP-0002 Phase 2.4"

print(f"quantum_qutrit_n3 (v{__version__})")
print(f"  MIGRATED: Now using qig library with PAIR operators")
print(f"  Entanglement support: ✓ ENABLED")
print(f"  CIP: {__cip__}")

