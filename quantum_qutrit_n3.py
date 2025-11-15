"""
Quantum Qutrit Dynamics: Clean Exponential Family Implementation
==================================================================

This implementation uses quantum exponential family formulas throughout.

For ρ = exp(Σθ_a F_a)/Z, all quantities are derived from the cumulant
generating function ψ(θ) = log Z(θ):
  - First cumulants:  κ_a = ∂ψ/∂θ_a = ⟨F_a⟩
  - Second cumulants: κ_ab = ∂²ψ/∂θ_a∂θ_b (Kubo-Mori metric / BKM metric)
  - Third cumulants:  κ_abc = ∂³ψ/∂θ_a∂θ_b∂θ_c

Key point: For quantum systems, cumulants equal derivatives of ψ, BUT
the simple covariance ⟨F_a F_b⟩ - ⟨F_a⟩⟨F_b⟩ does NOT equal ∂²ψ/∂θ_a∂θ_b!

Instead, we must use the Kubo-Mori metric (quantum Fisher information):
  G_ab = ∂²ψ/∂θ_a∂θ_b = Tr[∂ρ/∂θ_b · F_a]
  
where ∂ρ/∂θ is computed using the Duhamel formula for non-commuting operators.

Author: Neil Lawrence
Date: November 2025
"""

import numpy as np
from scipy.linalg import expm, logm
from typing import List, Tuple, Dict
import warnings

# Suppress overflow warnings from expm
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ============================================================================
# Gell-Mann Matrices (SU(3) Generators)
# ============================================================================

def gell_mann_matrices() -> np.ndarray:
    """
    Generate the 8 Gell-Mann matrices for SU(3).
    
    These are the traceless Hermitian generators of SU(3),
    analogous to Pauli matrices for SU(2).
    
    Returns
    -------
    matrices : ndarray, shape (8, 3, 3)
        The 8 Gell-Mann matrices
    """
    λ = np.zeros((8, 3, 3), dtype=complex)
    
    # λ₁ and λ₂ (like σ_x and σ_y for first two levels)
    λ[0] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    λ[1] = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex)
    
    # λ₃ (like σ_z for first two levels)
    λ[2] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex)
    
    # λ₄ and λ₅ (mixing first and third levels)
    λ[3] = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex)
    λ[4] = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex)
    
    # λ₆ and λ₇ (mixing second and third levels)
    λ[5] = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)
    λ[6] = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex)
    
    # λ₈ (diagonal, analogous to hypercharge)
    λ[7] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex) / np.sqrt(3)
    
    return λ


def single_site_operators(n_sites: int = 3) -> List[np.ndarray]:
    """
    Generate local single-site operators for n qutrit subsystems.
    
    Parameters
    ----------
    n_sites : int
        Number of qutrit subsystems
        
    Returns
    -------
    operators : list of ndarray
        List of n × 8 = 24 Hermitian operators (for n=3)
    """
    d = 3  # Qutrit dimension
    D = d ** n_sites  # Total Hilbert space dimension
    gm = gell_mann_matrices()
    operators = []
    
    identity = np.eye(d, dtype=complex)
    
    for site in range(n_sites):
        for k in range(8):  # 8 Gell-Mann matrices
            # Build F_site,k = I ⊗ ... ⊗ λ_k ⊗ ... ⊗ I
            op_list = [identity] * n_sites
            op_list[site] = gm[k]
            
            # Tensor product
            F = op_list[0]
            for op in op_list[1:]:
                F = np.kron(F, op)
            
            operators.append(F)
    
    return operators


# ============================================================================
# Quantum Exponential Family
# ============================================================================

def compute_density_matrix(theta: np.ndarray, operators: List[np.ndarray]) -> np.ndarray:
    """
    Compute ρ(θ) = exp(Σθ_a F_a) / Z.
    
    Parameters
    ----------
    theta : ndarray
        Natural parameters
    operators : list of ndarray
        Hermitian generators
        
    Returns
    -------
    rho : ndarray
        Normalized density matrix
    """
    H = sum(theta[i] * operators[i] for i in range(len(theta)))
    exp_H = expm(H)
    Z = np.trace(exp_H)
    return exp_H / Z


def compute_log_partition(theta: np.ndarray, operators: List[np.ndarray]) -> float:
    """
    Compute ψ(θ) = log Z(θ) where Z = tr[exp(Σθ_a F_a)].
    
    This is the cumulant generating function.
    """
    H = sum(theta[i] * operators[i] for i in range(len(theta)))
    Z = np.real(np.trace(expm(H)))
    return np.log(Z)


def compute_expectations(rho: np.ndarray, operators: List[np.ndarray]) -> np.ndarray:
    """
    Compute ⟨F_a⟩ = tr[ρ F_a] for all operators.
    
    These are the first cumulants: κ_a = ∂ψ/∂θ_a.
    """
    return np.array([np.real(np.trace(rho @ F)) for F in operators])


def compute_drho_dtheta(theta: np.ndarray, operators: List[np.ndarray], k: int,
                        n_integration_points: int = 100) -> np.ndarray:
    """
    Compute ∂ρ/∂θ_k using the Duhamel formula for matrix exponentials.
    
    For ρ = exp(H)/Z where H = Σθ_i F_i:
        ∂exp(H)/∂θ_k = ∫₀¹ exp(sH) F_k exp((1-s)H) ds
        ∂ρ/∂θ_k = (1/Z) ∂exp(H)/∂θ_k - ρ⟨F_k⟩
    
    This is the exact formula for non-commuting operators.
    
    Parameters
    ----------
    theta : ndarray
        Natural parameters
    operators : list of ndarray
        Hermitian generators
    k : int
        Index of parameter to differentiate
    n_integration_points : int
        Number of points for Duhamel integration (default: 100)
        
    Returns
    -------
    drho : ndarray
        ∂ρ/∂θ_k
    """
    H = sum(theta[i] * operators[i] for i in range(len(theta)))
    exp_H = expm(H)
    Z = np.real(np.trace(exp_H))
    rho = exp_H / Z
    E_k = np.real(np.trace(rho @ operators[k]))
    
    # Duhamel integral: ∫₀¹ exp(sH) F_k exp((1-s)H) ds
    # Using Simpson's rule for better accuracy
    s_values = np.linspace(0, 1, n_integration_points)
    ds = 1.0 / (n_integration_points - 1) if n_integration_points > 1 else 1.0
    
    integrand_values = []
    for s in s_values:
        exp_sH = expm(s * H)
        exp_1msH = expm((1 - s) * H)
        integrand_values.append(exp_sH @ operators[k] @ exp_1msH)
    
    # Simpson's rule
    if n_integration_points > 1:
        integral = integrand_values[0] + integrand_values[-1]
        for i in range(1, n_integration_points - 1):
            weight = 4 if i % 2 == 1 else 2
            integral += weight * integrand_values[i]
        integral *= ds / 3
    else:
        integral = integrand_values[0]
    
    drho = (1 / Z) * integral - rho * E_k
    return drho


def compute_covariance_matrix(theta: np.ndarray, operators: List[np.ndarray],
                              n_integration_points: int = 100) -> np.ndarray:
    """
    Compute the Kubo-Mori metric (BKM metric / quantum Fisher information).
    
    For quantum exponential families:
        G_ab = ∂²ψ/∂θ_a∂θ_b = Tr[∂ρ/∂θ_b · F_a]
    
    This is the second cumulant tensor and requires computing ∂ρ/∂θ using
    the Duhamel formula because operators don't commute.
    
    Note: The simple covariance ⟨F_a F_b⟩ - ⟨F_a⟩⟨F_b⟩ is NOT equal to
    ∂²ψ/∂θ_a∂θ_b for quantum systems!
    
    Parameters
    ----------
    theta : ndarray
        Natural parameters
    operators : list of ndarray
        Hermitian generators
    n_integration_points : int
        Number of points for Duhamel integration
        
    Returns
    -------
    G : ndarray, shape (n, n)
        Kubo-Mori metric (symmetric positive definite)
    """
    n = len(operators)
    G = np.zeros((n, n))
    
    for b in range(n):
        drho_b = compute_drho_dtheta(theta, operators, b, n_integration_points)
        for a in range(n):
            G[a, b] = np.real(np.trace(drho_b @ operators[a]))
    
    return G


def compute_third_cumulants(theta: np.ndarray, operators: List[np.ndarray],
                            n_integration_points: int = 100, eps: float = 1e-6) -> np.ndarray:
    """
    Compute third-order cumulants κ_abc = ∂³ψ/∂θ_a∂θ_b∂θ_c = ∂G_ab/∂θ_c.
    
    Since we have G_ab = ∂²ψ/∂θ_a∂θ_b computed analytically (via Duhamel),
    we compute the third cumulant by taking finite differences of G with
    respect to θ_c. This is more accurate than triple finite differences on ψ.
    
    Parameters
    ----------
    theta : ndarray
        Natural parameters
    operators : list of ndarray
        Hermitian generators
    n_integration_points : int
        Number of integration points for Duhamel (default: 100)
    eps : float
        Finite difference step size (default: 1e-6)
        
    Returns
    -------
    T : ndarray, shape (n, n, n)
        Third cumulant tensor (totally symmetric)
    """
    n = len(operators)
    T = np.zeros((n, n, n))
    
    # Compute κ_abc = ∂G_ab/∂θ_c via finite differences
    # Only compute unique elements due to total symmetry
    for c in range(n):
        # Compute G at θ + ε e_c
        theta_p = theta.copy()
        theta_p[c] += eps
        G_p = compute_covariance_matrix(theta_p, operators, n_integration_points)
        
        # Compute G at θ - ε e_c
        theta_m = theta.copy()
        theta_m[c] -= eps
        G_m = compute_covariance_matrix(theta_m, operators, n_integration_points)
        
        # ∂G_ab/∂θ_c = (G_ab(θ+ε) - G_ab(θ-ε)) / (2ε)
        dG_dc = (G_p - G_m) / (2 * eps)
        
        # Fill T[a, b, c] for all a, b
        for a in range(n):
            for b in range(n):
                kappa = dG_dc[a, b]
                
                # Fill all permutations (totally symmetric by Schwarz's theorem)
                for perm in [(a,b,c), (a,c,b), (b,a,c), (b,c,a), (c,a,b), (c,b,a)]:
                    T[perm] = kappa
    
    return T


# ============================================================================
# Marginal Entropies
# ============================================================================

def partial_trace(rho: np.ndarray, keep_site: int, n_sites: int = 3, d: int = 3) -> np.ndarray:
    """
    Compute partial trace to get single-site marginal.
    
    Traces out all sites except keep_site.
    """
    # Reshape to separate subsystems
    shape = [d] * n_sites
    rho_tensor = rho.reshape(shape + shape)
    
    # Trace over all sites except keep_site
    axes_to_trace = []
    for site in range(n_sites):
        if site != keep_site:
            axes_to_trace.append((site, n_sites + site))
    
    # Perform traces
    result = rho_tensor
    for ax_pair in sorted(axes_to_trace, reverse=True):
        ax1, ax2 = ax_pair
        # Adjust indices after previous traces
        offset = sum(1 for pair in axes_to_trace if pair[0] < ax_pair[0])
        result = np.trace(result, axis1=ax1-offset, axis2=ax2-offset-len(axes_to_trace)+offset)
    
    # Manual implementation for n=3, d=3
    # More explicit and reliable
    rho_marginal = np.zeros((d, d), dtype=complex)
    
    # Sum over all basis states of the OTHER two sites
    other_sites = [s for s in range(n_sites) if s != keep_site]
    
    for i in range(d):  # kept site state i
        for j in range(d):  # kept site state j
            for k1 in range(d):  # first other site
                for k2 in range(d):  # second other site
                    # Build full basis state indices
                    idx_bra = [0] * n_sites
                    idx_ket = [0] * n_sites
                    idx_bra[keep_site] = i
                    idx_ket[keep_site] = j
                    idx_bra[other_sites[0]] = k1
                    idx_ket[other_sites[0]] = k1
                    idx_bra[other_sites[1]] = k2
                    idx_ket[other_sites[1]] = k2
                    
                    # Convert to flat indices
                    flat_bra = sum(idx_bra[s] * (d ** (n_sites - 1 - s)) for s in range(n_sites))
                    flat_ket = sum(idx_ket[s] * (d ** (n_sites - 1 - s)) for s in range(n_sites))
                    
                    rho_marginal[i, j] += rho[flat_bra, flat_ket]
    
    return rho_marginal


def von_neumann_entropy(rho: np.ndarray) -> float:
    """
    Compute S(ρ) = -tr(ρ log ρ).
    """
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]  # Remove numerical zeros
    return -np.sum(eigvals * np.log(eigvals))


def compute_marginal_entropies(rho: np.ndarray, n_sites: int = 3) -> np.ndarray:
    """
    Compute marginal von Neumann entropies [h_1, h_2, h_3].
    """
    h = np.zeros(n_sites)
    for i in range(n_sites):
        rho_i = partial_trace(rho, keep_site=i, n_sites=n_sites)
        h[i] = von_neumann_entropy(rho_i)
    return h


# ============================================================================
# Constrained Dynamics
# ============================================================================

def compute_constraint_gradient(theta: np.ndarray, operators: List[np.ndarray],
                                n_sites: int = 3) -> np.ndarray:
    """
    Compute ∇(Σh_i) where h_i are marginal entropies.
    
    Uses chain rule: ∂h_i/∂θ_a = -tr[(log ρ_i + I) ∂ρ_i/∂θ_a]
    
    For exponential families: ∂ρ_i/∂θ_a = ρ_i F_{i,a} - ρ_i ⟨F_a⟩
    where ρ_i is the marginal and F_{i,a} is F_a with others traced out.
    """
    rho = compute_density_matrix(theta, operators)
    E = compute_expectations(rho, operators)
    n_params = len(theta)
    grad = np.zeros(n_params)
    
    for a in range(n_params):
        for i in range(n_sites):
            # Get marginal
            rho_i = partial_trace(rho, keep_site=i, n_sites=n_sites)
            
            # Compute ∂ρ_i/∂θ_a using finite differences (more reliable)
            eps = 1e-7
            theta_p = theta.copy()
            theta_p[a] += eps
            rho_p = compute_density_matrix(theta_p, operators)
            rho_i_p = partial_trace(rho_p, keep_site=i, n_sites=n_sites)
            
            drho_i = (rho_i_p - rho_i) / eps
            
            # Compute log(ρ_i)
            eigvals, eigvecs = np.linalg.eigh(rho_i)
            eigvals_safe = np.maximum(eigvals, 1e-12)
            log_rho_i = eigvecs @ np.diag(np.log(eigvals_safe)) @ eigvecs.conj().T
            
            # ∂h_i/∂θ_a
            d = rho_i.shape[0]
            grad[a] += -np.real(np.trace((log_rho_i + np.eye(d, dtype=complex)) @ drho_i))
    
    return grad


def constrained_flow_step(theta: np.ndarray, operators: List[np.ndarray],
                          n_sites: int = 3, dt: float = 0.01) -> np.ndarray:
    """
    One step of constrained steepest entropy ascent:
        dθ/dt = -Π_∥(θ) G(θ) θ
    
    where Π_∥ projects onto constraint tangent space.
    """
    rho = compute_density_matrix(theta, operators)
    E = compute_expectations(rho, operators)
    G = compute_covariance_matrix(theta, operators)
    a = compute_constraint_gradient(theta, operators, n_sites)
    
    # Projection operator
    if np.linalg.norm(a) > 1e-10:
        Pi_parallel = np.eye(len(theta)) - np.outer(a, a) / np.dot(a, a)
    else:
        Pi_parallel = np.eye(len(theta))
    
    # Constrained flow
    F = -Pi_parallel @ G @ theta
    
    return theta + dt * F


def solve_constrained_quantum_maxent(
    theta_init: np.ndarray,
    operators: List[np.ndarray],
    n_sites: int = 3,
    n_steps: int = 5000,
    dt: float = 0.005,
    convergence_tol: float = 1e-5,
    verbose: bool = True
) -> Dict:
    """
    Integrate constrained quantum maximum entropy dynamics.
    
    Dynamics: dθ/dt = -Π_∥(θ) G(θ) θ
    
    where Π_∥ projects onto tangent space of constraint Σ h_i = C.
    
    Parameters
    ----------
    theta_init : ndarray
        Initial natural parameters
    operators : list
        Hermitian generators
    n_sites : int
        Number of subsystems
    n_steps : int
        Maximum integration steps
    dt : float
        Time step
    convergence_tol : float
        Stop when ||F(θ)|| < tol
    verbose : bool
        Print progress
        
    Returns
    -------
    solution : dict
        trajectory, flow_norms, constraint_values, converged, C_init
    """
    d = 3  # Qutrit dimension
    trajectory = [theta_init.copy()]
    flow_norms = []
    constraint_values = []
    theta = theta_init.copy()
    
    # Initial constraint value
    rho_init = compute_density_matrix(theta_init, operators)
    h_init = []
    for i in range(n_sites):
        rho_i = partial_trace(rho_init, keep_site=i, n_sites=n_sites, d=d)
        h_init.append(von_neumann_entropy(rho_i))
    C_init = np.sum(h_init)
    constraint_values.append(C_init)
    
    if verbose:
        print(f"Initial constraint value: C = {C_init:.6f}")
        print(f"  Individual marginals: h = {h_init}")
    
    for step in range(n_steps):
        # Compute flow at current point
        try:
            G = compute_covariance_matrix(theta, operators)
            a = compute_constraint_gradient(theta, operators, n_sites)
            
            # Unconstrained flow: -Gθ (entropy ascent)
            F_unc = -G @ theta
            
            # Lagrange multiplier for tangency
            nu = np.dot(F_unc, a) / (np.dot(a, a) + 1e-10)
            
            # Constrained flow
            F = F_unc - nu * a
            
            # Integration step
            theta = theta + dt * F
            
            # Track
            flow_norm = np.linalg.norm(F)
            flow_norms.append(flow_norm)
            trajectory.append(theta.copy())
            
            # Constraint value
            rho_current = compute_density_matrix(theta, operators)
            h_current = []
            for i in range(n_sites):
                rho_i = partial_trace(rho_current, keep_site=i, n_sites=n_sites, d=d)
                h_current.append(von_neumann_entropy(rho_i))
            C_current = np.sum(h_current)
            constraint_values.append(C_current)
            
            # Verbose output
            if verbose and step % 100 == 0:
                print(f"Step {step:4d}: ||F|| = {flow_norm:.6e}, "
                      f"ΔC = {abs(C_current - C_init):.8e}")
            
            # Convergence check
            if flow_norm < convergence_tol:
                if verbose:
                    print(f"\nConverged at step {step}")
                return {
                    'trajectory': np.array(trajectory),
                    'flow_norms': np.array(flow_norms),
                    'constraint_values': np.array(constraint_values),
                    'converged': True,
                    'C_init': C_init,
                    'n_steps': step + 1
                }
                
        except Exception as e:
            print(f"Error at step {step}: {e}")
            break
    
    if verbose:
        print(f"\nReached maximum steps ({n_steps})")
    
    return {
        'trajectory': np.array(trajectory),
        'flow_norms': np.array(flow_norms),
        'constraint_values': np.array(constraint_values),
        'converged': False,
        'C_init': C_init,
        'n_steps': n_steps
    }


def solve_unconstrained_quantum_maxent(
    theta_init: np.ndarray,
    operators: List[np.ndarray],
    n_sites: int = 3,
    n_steps: int = 5000,
    dt: float = 0.005,
    convergence_tol: float = 1e-5,
    verbose: bool = True
) -> Dict:
    """
    Integrate unconstrained quantum maximum entropy dynamics.
    
    Dynamics: dθ/dt = -G(θ) θ
    
    Pure steepest entropy ascent without constraints.
    
    Parameters
    ----------
    theta_init : ndarray
        Initial natural parameters
    operators : list
        Hermitian generators
    n_sites : int
        Number of subsystems
    n_steps : int
        Maximum integration steps
    dt : float
        Time step
    convergence_tol : float
        Stop when ||F(θ)|| < tol
    verbose : bool
        Print progress
        
    Returns
    -------
    solution : dict
        trajectory, flow_norms, constraint_values, converged
    """
    d = 3  # Qutrit dimension
    trajectory = [theta_init.copy()]
    flow_norms = []
    constraint_values = []
    theta = theta_init.copy()
    
    # Initial constraint value
    rho_init = compute_density_matrix(theta_init, operators)
    h_init = []
    for i in range(n_sites):
        rho_i = partial_trace(rho_init, keep_site=i, n_sites=n_sites, d=d)
        h_init.append(von_neumann_entropy(rho_i))
    C_init = np.sum(h_init)
    constraint_values.append(C_init)
    
    if verbose:
        print(f"Initial marginal entropy sum: C = {C_init:.6f}")
    
    for step in range(n_steps):
        # Compute flow at current point
        try:
            G = compute_covariance_matrix(theta, operators)
            
            # Unconstrained flow: -Gθ (entropy ascent)
            F = -G @ theta
            
            # Integration step
            theta = theta + dt * F
            
            # Track
            flow_norm = np.linalg.norm(F)
            flow_norms.append(flow_norm)
            trajectory.append(theta.copy())
            
            # Constraint value (will drift)
            rho_current = compute_density_matrix(theta, operators)
            h_current = []
            for i in range(n_sites):
                rho_i = partial_trace(rho_current, keep_site=i, n_sites=n_sites, d=d)
                h_current.append(von_neumann_entropy(rho_i))
            C_current = np.sum(h_current)
            constraint_values.append(C_current)
            
            # Verbose output
            if verbose and step % 100 == 0:
                print(f"Step {step:4d}: ||F|| = {flow_norm:.6e}, "
                      f"C = {C_current:.6f}")
            
            # Convergence check
            if flow_norm < convergence_tol:
                if verbose:
                    print(f"\nConverged at step {step}")
                return {
                    'trajectory': np.array(trajectory),
                    'flow_norms': np.array(flow_norms),
                    'constraint_values': np.array(constraint_values),
                    'converged': True,
                    'n_steps': step + 1
                }
                
        except Exception as e:
            print(f"Error at step {step}: {e}")
            break
    
    if verbose:
        print(f"\nReached maximum steps ({n_steps})")
    
    return {
        'trajectory': np.array(trajectory),
        'flow_norms': np.array(flow_norms),
        'constraint_values': np.array(constraint_values),
        'converged': False,
        'n_steps': n_steps
    }


# ============================================================================
# Utilities
# ============================================================================

def create_lme_state(n_sites: int = 3, d: int = 3) -> np.ndarray:
    """
    Create locally maximally entangled (LME) state.
    
    For qutrits: |ψ⟩ = (|000⟩ + |111⟩ + |222⟩)/√3
    """
    D = d ** n_sites
    psi = np.zeros(D, dtype=complex)
    
    # |000⟩ + |111⟩ + |222⟩
    for i in range(d):
        idx = sum(i * (d ** (n_sites - 1 - site)) for site in range(n_sites))
        psi[idx] = 1.0
    
    psi /= np.sqrt(d)
    rho = np.outer(psi, psi.conj())
    
    return rho


def find_natural_parameters_for_lme(operators: List[np.ndarray], rho_target: np.ndarray,
                                   max_iter: int = 100, lr: float = 0.1) -> np.ndarray:
    """
    Find θ such that ρ(θ) ≈ ρ_target using gradient descent.
    """
    theta = np.zeros(len(operators))
    
    for iteration in range(max_iter):
        rho = compute_density_matrix(theta, operators)
        E_current = compute_expectations(rho, operators)
        E_target = compute_expectations(rho_target, operators)
        
        gradient = E_current - E_target
        theta -= lr * gradient
        
        loss = np.linalg.norm(gradient)
        if iteration % 10 == 0:
            print(f"Iter {iteration}: loss={loss:.4e}")
        
        if loss < 1e-6:
            break
    
    return theta


print("Quantum qutrit module loaded (clean exponential family implementation)")
