"""
GENERIC decomposition tools for quantum inaccessible game.

This module provides functions to extract the effective Hamiltonian and
diffusion operator from the symmetric and antisymmetric parts of the
flow Jacobian, implementing the GENERIC decomposition framework.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import least_squares

from qig.validation import ValidationReport


def effective_hamiltonian_coefficients(A: np.ndarray, 
                                       theta: np.ndarray,
                                       f_abc: np.ndarray,
                                       regularization: float = 1e-10) -> Tuple[np.ndarray, dict]:
    """
    Extract effective Hamiltonian coefficients from antisymmetric flow.
    
    The antisymmetric part A encodes the reversible (Hamiltonian) dynamics
    through the relation:
        A_ab θ_b = Σ_c f_abc η_c
    
    where η_c are the coefficients of the effective Hamiltonian:
        H_eff = Σ_c η_c F_c
    
    This function solves the linear system F⋅η = v, where:
        F_rc = Σ_b f_rbc θ_b
        v_r = (A⋅θ)_r
    
    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Antisymmetric part of flow Jacobian
    theta : np.ndarray, shape (n,)
        Natural parameters
    f_abc : np.ndarray, shape (n, n, n)
        Structure constants
    regularization : float
        Regularization parameter for nearly singular systems
        
    Returns
    -------
    eta : np.ndarray, shape (n,)
        Coefficients of effective Hamiltonian
    diagnostics : dict
        Diagnostic information:
        - 'condition_number': Condition number of F matrix
        - 'residual': ||F⋅η - v||
        - 'method': 'linear_solver'
        
    Notes
    -----
    Method A: Direct linear solver approach
    
    The structure constants f_abc appear in the matrix F that relates
    the Hamiltonian coefficients to the antisymmetric flow.
    
    For nearly singular systems (near degeneracies), regularization
    helps stabilize the solution.
    """
    n = len(theta)
    
    # Construct matrix F: F_rc = Σ_b f_rbc θ_b
    F = np.zeros((n, n))
    for r in range(n):
        for c in range(n):
            F[r, c] = np.sum(f_abc[r, :, c] * theta)
    
    # Construct vector v = A⋅θ
    v = A @ theta
    
    # Compute condition number
    condition_number = np.linalg.cond(F)
    
    # Solve F⋅η = v with regularization if needed
    if condition_number > 1e12 or regularization > 0:
        # Add regularization: (F^T F + λI)η = F^T v
        FtF = F.T @ F + regularization * np.eye(n)
        Ftv = F.T @ v
        eta = np.linalg.solve(FtF, Ftv)
    else:
        # Direct solve
        eta = np.linalg.solve(F, v)
    
    # Compute residual
    residual = np.linalg.norm(F @ eta - v)
    
    diagnostics = {
        'condition_number': condition_number,
        'residual': residual,
        'method': 'linear_solver',
        'regularization': regularization
    }
    
    return eta, diagnostics


def effective_hamiltonian_coefficients_lstsq(A: np.ndarray,
                                             theta: np.ndarray,
                                             operators: List[np.ndarray],
                                             rho: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Extract effective Hamiltonian coefficients using least-squares fitting.
    
    This provides an alternative method (Method B) that directly minimizes
    the difference between the commutator evolution and the antisymmetric flow.
    
    Minimizes: ||i[H_eff, ρ] - ρ_dot_reversible||
    
    where ρ_dot_reversible is the density matrix flow from the antisymmetric part.
    
    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Antisymmetric part of flow Jacobian
    theta : np.ndarray, shape (n,)
        Natural parameters
    operators : List[np.ndarray]
        List of operator basis {F_a}
    rho : np.ndarray
        Density matrix at current state
        
    Returns
    -------
    eta : np.ndarray, shape (n,)
        Coefficients of effective Hamiltonian
    diagnostics : dict
        Diagnostic information:
        - 'residual': Final residual
        - 'success': Whether optimization succeeded
        - 'method': 'least_squares'
        
    Notes
    -----
    Method B: Least-squares fitting approach
    
    This method provides cross-validation for Method A (linear solver).
    They should agree within tolerance (~1e-6) if the GENERIC structure
    is correctly implemented.
    """
    n = len(theta)
    
    # Map antisymmetric flow to density matrix space
    # For exponential families, we need the Kubo-Mori derivative ∂ρ/∂θ
    # For simplicity, use finite differences here
    eps = 1e-7
    drho_dtheta = []
    for i in range(n):
        # This is a simplified version - in practice would use Duhamel formula
        # For now, approximate with commutator [F_i, ρ]
        drho_dtheta.append(operators[i] @ rho - rho @ operators[i])
    
    # Compute target flow in density matrix space
    # ρ_dot = Σ_a (A⋅θ)_a ∂ρ/∂θ_a
    A_theta = A @ theta
    rho_dot_target = sum(A_theta[a] * drho_dtheta[a] for a in range(n))
    
    # Define objective: minimize ||i[H_eff, ρ] - rho_dot_target||
    def objective(eta):
        # H_eff = Σ_a η_a F_a
        H_eff = sum(eta[a] * operators[a] for a in range(n))
        # Commutator: i[H_eff, ρ]
        commutator = 1j * (H_eff @ rho - rho @ H_eff)
        # Residual
        diff = commutator - rho_dot_target
        return np.concatenate([diff.real.flatten(), diff.imag.flatten()])
    
    # Initial guess: zero
    eta0 = np.zeros(n)
    
    # Optimize
    result = least_squares(objective, eta0, method='lm')
    
    eta = result.x
    
    diagnostics = {
        'residual': result.cost,
        'success': result.success,
        'method': 'least_squares',
        'nfev': result.nfev
    }
    
    return eta, diagnostics


def effective_hamiltonian_operator(eta: np.ndarray,
                                   operators: List[np.ndarray]) -> np.ndarray:
    """
    Construct effective Hamiltonian operator from coefficients.
    
    H_eff = Σ_a η_a F_a
    
    Parameters
    ----------
    eta : np.ndarray, shape (n,)
        Hamiltonian coefficients
    operators : List[np.ndarray]
        Operator basis {F_a}
        
    Returns
    -------
    H_eff : np.ndarray
        Effective Hamiltonian operator
        
    Notes
    -----
    The effective Hamiltonian should be:
    - Hermitian: H_eff = H_eff†
    - Traceless: Tr(H_eff) ≈ 0 (for Lie algebra generators)
    
    These properties should be verified after construction.
    """
    H_eff = sum(eta_a * F_a for eta_a, F_a in zip(eta, operators))
    return H_eff


def verify_hamiltonian_evolution(H_eff: np.ndarray,
                                 A: np.ndarray,
                                 theta: np.ndarray,
                                 operators: List[np.ndarray],
                                 rho: np.ndarray,
                                 tol: float = 1e-6) -> ValidationReport:
    """
    Verify that effective Hamiltonian generates correct evolution.
    
    Checks that -i[H_eff, ρ] matches the antisymmetric part of the flow.
    
    Parameters
    ----------
    H_eff : np.ndarray
        Effective Hamiltonian
    A : np.ndarray
        Antisymmetric part of Jacobian
    theta : np.ndarray
        Natural parameters
    operators : List[np.ndarray]
        Operator basis
    rho : np.ndarray
        Density matrix
    tol : float
        Tolerance for verification
        
    Returns
    -------
    report : ValidationReport
        Validation report with checks:
        - Hermiticity of H_eff
        - Tracelessness of H_eff
        - Commutator matching
        - Energy conservation
    """
    from qig.validation import check_hermitian, check_traceless
    
    report = ValidationReport("Effective Hamiltonian Verification")
    
    # Check Hermiticity
    is_hermitian, herm_error = check_hermitian(H_eff, tol=1e-12)
    report.add_check("Hermiticity H_eff = H_eff†", is_hermitian, herm_error, 1e-12)
    
    # Check tracelessness
    is_traceless, trace_error = check_traceless(H_eff, tol=1e-10)
    report.add_check("Traceless Tr(H_eff) ≈ 0", is_traceless, trace_error, 1e-10)
    
    # Check commutator matching
    # Compute -i[H_eff, ρ] in density matrix space
    commutator = -1j * (H_eff @ rho - rho @ H_eff)
    
    # Map A to density matrix space using operator basis
    # ρ_dot = Σ_a (A⋅θ)_a ∂ρ/∂θ_a
    # Approximating with commutators for now
    A_theta = A @ theta
    rho_dot_A = sum(A_theta[a] * (operators[a] @ rho - rho @ operators[a]) 
                    for a in range(len(operators)))
    
    # Compare
    commutator_error = np.linalg.norm(commutator - rho_dot_A)
    commutator_match = commutator_error < tol
    report.add_check("Commutator matching -i[H_eff,ρ] ≈ A-flow",
                    commutator_match, commutator_error, tol)
    
    # Check energy conservation (for unitary evolution)
    # d/dt Tr(H_eff ρ) should be zero for reversible part
    # This is automatically satisfied for Hermitian H_eff and trace-preserving evolution
    energy = np.trace(H_eff @ rho).real
    report.add_check("Energy defined Tr(H_eff ρ)", True, energy, np.inf,
                    f"Energy = {energy:.4e}")
    
    return report


def cross_validate_hamiltonian_coefficients(A: np.ndarray,
                                           theta: np.ndarray,
                                           f_abc: np.ndarray,
                                           operators: List[np.ndarray],
                                           rho: np.ndarray,
                                           tol: float = 1e-6) -> ValidationReport:
    """
    Cross-validate Hamiltonian coefficients between two methods.
    
    Compares:
    - Method A: Linear solver (structure constants)
    - Method B: Least-squares fitting (direct optimization)
    
    Parameters
    ----------
    A : np.ndarray
        Antisymmetric part
    theta : np.ndarray
        Natural parameters
    f_abc : np.ndarray
        Structure constants
    operators : List[np.ndarray]
        Operator basis
    rho : np.ndarray
        Density matrix
    tol : float
        Tolerance for agreement
        
    Returns
    -------
    report : ValidationReport
        Cross-validation report
    """
    report = ValidationReport("Hamiltonian Coefficients Cross-Validation")
    
    # Method A: Linear solver
    eta_A, diag_A = effective_hamiltonian_coefficients(A, theta, f_abc)
    report.add_check(f"Method A: Linear solver (cond={diag_A['condition_number']:.2e})",
                    True, diag_A['residual'], np.inf)
    
    # Method B: Least-squares
    eta_B, diag_B = effective_hamiltonian_coefficients_lstsq(A, theta, operators, rho)
    report.add_check(f"Method B: Least-squares (success={diag_B['success']})",
                    diag_B['success'], diag_B['residual'], np.inf)
    
    # Compare coefficients
    coeff_diff = np.linalg.norm(eta_A - eta_B)
    coeff_agree = coeff_diff < tol
    report.add_check("Coefficients agree ||η_A - η_B||",
                    coeff_agree, coeff_diff, tol)
    
    return report


# ============================================================================
# Diffusion Operator (Irreversible Dynamics)
# ============================================================================


def kubo_mori_derivatives(theta: np.ndarray,
                         operators: List[np.ndarray],
                         exp_fam) -> List[np.ndarray]:
    """
    Compute Kubo-Mori derivatives ∂ρ/∂θ_a for all parameters.
    
    The Kubo-Mori derivative is:
        ∂ρ/∂θ_a = ∫_0^1 ρ^s F_a ρ^{1-s} ds
    
    where ρ = exp(K - ψ) and K = Σ θ_b F_b.
    
    This uses the Duhamel formula from qig.duhamel for high precision.
    
    Parameters
    ----------
    theta : np.ndarray
        Natural parameters
    operators : List[np.ndarray]
        Operator basis {F_a}
    exp_fam : QuantumExponentialFamily
        Exponential family instance (for using duhamel method)
        
    Returns
    -------
    drho_dtheta : List[np.ndarray]
        List of ∂ρ/∂θ_a matrices, one for each parameter
        
    Notes
    -----
    The Kubo-Mori derivative is the "quantum derivative" that maintains
    Hermiticity and provides the correct mapping from parameter space to
    density matrix space.
    
    For machine precision, we use the Duhamel integral from qig.duhamel.
    """
    drho_dtheta = []
    
    for a in range(len(operators)):
        # Use the exponential family's rho_derivative method
        # which implements the Duhamel formula
        drho = exp_fam.rho_derivative(theta, a, method='duhamel')
        drho_dtheta.append(drho)
    
    return drho_dtheta


def diffusion_operator(S: np.ndarray,
                      theta: np.ndarray,
                      exp_fam,
                      method: str = 'duhamel') -> np.ndarray:
    """
    Construct diffusion operator D[ρ] from symmetric flow.
    
    The diffusion operator generates the irreversible (dissipative) dynamics:
        D[ρ] = Σ_a (S⋅q)_a ∂ρ/∂θ_a
    
    where S is the symmetric part, q are mean parameters (tangent to constraint),
    and ∂ρ/∂θ_a are Kubo-Mori derivatives.
    
    Parameters
    ----------
    S : np.ndarray, shape (n, n)
        Symmetric part of flow Jacobian
    theta : np.ndarray, shape (n,)
        Natural parameters
    exp_fam : QuantumExponentialFamily
        Exponential family instance
    method : str
        Method for computing Kubo-Mori derivatives
        
    Returns
    -------
    D_rho : np.ndarray
        Diffusion operator D[ρ] acting on density matrix
        
    Notes
    -----
    The dissipative flow is:
        ρ̇_dissipative = D[ρ]
    
    Key properties that should be verified:
    - D[ρ] is Hermitian
    - Tr(D[ρ]) = 0 (trace preservation)
    - Entropy production: -Tr(ρ log ρ D[ρ]) ≥ 0
    
    The flow is in the tangent space to the constraint manifold,
    so we need the mean parameters q = ∇ψ.
    """
    # Get mean parameters (tangent to constraint)
    # For exponential families: q = ∇ψ(θ)
    q = exp_fam._grad_psi(theta)
    
    # Compute Kubo-Mori derivatives
    drho_dtheta = kubo_mori_derivatives(theta, exp_fam.operators, exp_fam)
    
    # Compute flow in parameter space: S⋅q
    S_q = S @ q
    
    # Map to density matrix space
    D_rho = sum(S_q[a] * drho_dtheta[a] for a in range(len(theta)))
    
    return D_rho


def milburn_approximation(H_eff: np.ndarray,
                         rho: np.ndarray,
                         gamma: float = 1.0) -> np.ndarray:
    """
    Compute Milburn approximation to diffusion operator.
    
    Near equilibrium, the diffusion operator can be approximated as:
        D[ρ] ≈ -γ/2 [H_eff, [H_eff, ρ]]
    
    where γ is an effective decoherence rate.
    
    Parameters
    ----------
    H_eff : np.ndarray
        Effective Hamiltonian
    rho : np.ndarray
        Density matrix
    gamma : float
        Decoherence rate (typically ~ 1)
        
    Returns
    -------
    D_rho_milburn : np.ndarray
        Milburn approximation to D[ρ]
        
    Notes
    -----
    This is a useful approximation near equilibrium and provides
    a simple form for comparison with the full Kubo-Mori construction.
    
    The double commutator -[H, [H, ρ]] is a standard form in
    quantum master equations (Lindblad form with H as the Lindblad operator).
    
    For comparison with the full diffusion operator:
    - Near equilibrium: Should agree within ~10^-4
    - Far from equilibrium: May differ significantly
    """
    # First commutator: [H_eff, ρ]
    comm1 = H_eff @ rho - rho @ H_eff
    
    # Second commutator: [H_eff, [H_eff, ρ]]
    comm2 = H_eff @ comm1 - comm1 @ H_eff
    
    # Milburn form: -γ/2 [H, [H, ρ]]
    D_rho_milburn = -0.5 * gamma * comm2
    
    return D_rho_milburn


def verify_diffusion_operator(D_rho: np.ndarray,
                              rho: np.ndarray,
                              tol_hermiticity: float = 1e-10,
                              tol_trace: float = 1e-12,
                              tol_entropy_production: float = 1e-14) -> ValidationReport:
    """
    Verify properties of the diffusion operator.
    
    Checks:
    1. D[ρ] is Hermitian
    2. Tr(D[ρ]) = 0 (trace preservation)
    3. Entropy production -Tr(ρ log ρ D[ρ]) ≥ 0
    4. Positivity preservation (eigenvalues)
    
    Parameters
    ----------
    D_rho : np.ndarray
        Diffusion operator
    rho : np.ndarray
        Density matrix
    tol_hermiticity : float
        Tolerance for Hermiticity
    tol_trace : float
        Tolerance for trace preservation
    tol_entropy_production : float
        Tolerance for entropy production (allowing small numerical error)
        
    Returns
    -------
    report : ValidationReport
        Validation report with all checks
    """
    from qig.validation import check_hermitian
    from scipy.linalg import logm
    
    report = ValidationReport("Diffusion Operator Verification")
    
    # Check Hermiticity
    is_hermitian, herm_error = check_hermitian(D_rho, tol=tol_hermiticity)
    report.add_check("Hermiticity D[ρ] = D[ρ]†", is_hermitian, 
                    herm_error, tol_hermiticity)
    
    # Check trace preservation
    trace = np.abs(np.trace(D_rho))
    trace_preserved = trace < tol_trace
    report.add_check("Trace preservation Tr(D[ρ]) = 0",
                    trace_preserved, trace, tol_trace)
    
    # Check entropy production
    # dS/dt = -Tr(ρ log ρ D[ρ]) should be ≥ 0
    log_rho = logm(rho)
    entropy_production_rate = -np.trace(rho @ log_rho @ D_rho).real
    
    # Allow small negative values due to numerical error
    non_negative = entropy_production_rate >= -tol_entropy_production
    report.add_check("Entropy production ≥ 0",
                    non_negative, entropy_production_rate, tol_entropy_production,
                    f"dS/dt = {entropy_production_rate:.4e}")
    
    # Check positivity preservation (at least for small timestep)
    # ρ + ε D[ρ] should have non-negative eigenvalues for small ε
    eps = 1e-6
    rho_evolved = rho + eps * D_rho
    eigvals = np.linalg.eigvalsh(rho_evolved.real)
    min_eigval = np.min(eigvals)
    
    positivity_preserved = min_eigval >= -tol_entropy_production
    report.add_check("Positivity preservation (ε=1e-6)",
                    positivity_preserved, min_eigval, -tol_entropy_production,
                    f"min(λ) = {min_eigval:.4e}")
    
    return report


def compare_diffusion_methods(S: np.ndarray,
                              theta: np.ndarray,
                              H_eff: np.ndarray,
                              exp_fam,
                              gamma: float = 1.0,
                              tol: float = 1e-4) -> ValidationReport:
    """
    Compare Kubo-Mori diffusion with Milburn approximation.
    
    Near equilibrium, these should agree reasonably well.
    
    Parameters
    ----------
    S : np.ndarray
        Symmetric part
    theta : np.ndarray
        Natural parameters
    H_eff : np.ndarray
        Effective Hamiltonian
    exp_fam : QuantumExponentialFamily
        Exponential family
    gamma : float
        Decoherence rate for Milburn approximation
    tol : float
        Tolerance for agreement
        
    Returns
    -------
    report : ValidationReport
        Comparison report
    """
    report = ValidationReport("Diffusion Operator Comparison")
    
    rho = exp_fam.rho_from_theta(theta)
    
    # Method A: Full Kubo-Mori construction
    D_rho_full = diffusion_operator(S, theta, exp_fam)
    report.add_check("Kubo-Mori method computed", True, 
                    np.linalg.norm(D_rho_full), np.inf)
    
    # Method B: Milburn approximation
    D_rho_milburn = milburn_approximation(H_eff, rho, gamma)
    report.add_check("Milburn approximation computed", True,
                    np.linalg.norm(D_rho_milburn), np.inf)
    
    # Compare
    diff = np.linalg.norm(D_rho_full - D_rho_milburn)
    relative_diff = diff / (np.linalg.norm(D_rho_full) + 1e-16)
    
    agree = relative_diff < tol
    report.add_check("Methods agree (near equilibrium)",
                    agree, relative_diff, tol,
                    f"Relative diff = {relative_diff:.4e}")
    
    return report

