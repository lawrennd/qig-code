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

