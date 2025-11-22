"""
Duhamel formula for quantum exponential family derivatives.

For ρ = exp(H) where H = ∑ θ_a F_a - ψ(θ)I, the exact derivative is:

    ∂ρ/∂θ_a = ∫₀¹ exp(sH) (F_a - ⟨F_a⟩I) exp((1-s)H) ds

This is the Duhamel (or Dalecki-Krein exponential) formula. The SLD is just
the trapezoid rule approximation (average of s=0 and s=1 endpoints).

QUANTUM DERIVATIVE PRINCIPLE: This is the EXACT formula that respects
operator ordering and preserves Hermiticity.
"""

import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad_vec


def duhamel_derivative(
    rho: np.ndarray,
    H: np.ndarray, 
    F_centered: np.ndarray,
    n_points: int = 50
) -> np.ndarray:
    """
    Compute ∂ρ/∂θ using the Duhamel formula with numerical integration.
    
    ∂ρ/∂θ = ∫₀¹ exp(sH) F_centered exp((1-s)H) ds
    
    where F_centered = F - ⟨F⟩I.
    
    Parameters
    ----------
    rho : ndarray, shape (D, D)
        Density matrix
    H : ndarray, shape (D, D)  
        Hamiltonian H = ∑ θ_a F_a - ψ(θ)I
    F_centered : ndarray, shape (D, D)
        Centered operator F - ⟨F⟩I
    n_points : int
        Number of quadrature points for integration
        
    Returns
    -------
    drho : ndarray, shape (D, D)
        Derivative ∂ρ/∂θ (exactly Hermitian)
        
    Notes
    -----
    - Uses trapezoid rule for integration
    - For n_points=2, recovers the SLD formula
    - Higher n_points gives better accuracy
    """
    D = rho.shape[0]
    
    # Trapezoid rule
    s_vals = np.linspace(0, 1, n_points)
    ds = s_vals[1] - s_vals[0]
    
    drho = np.zeros((D, D), dtype=complex)
    
    # Precompute matrix exponentials for efficiency
    exp_sH = {}
    exp_1msH = {}
    
    for s in s_vals:
        exp_sH[s] = expm(s * H)
        exp_1msH[s] = expm((1 - s) * H)
    
    # Trapezoid rule integration
    for i, s in enumerate(s_vals):
        integrand = exp_sH[s] @ F_centered @ exp_1msH[s]
        
        if i == 0 or i == len(s_vals) - 1:
            # Endpoints get weight 0.5
            drho += 0.5 * ds * integrand
        else:
            # Interior points get weight 1
            drho += ds * integrand
    
    return drho


def duhamel_derivative_simpson(
    rho: np.ndarray,
    H: np.ndarray, 
    F_centered: np.ndarray,
    n_points: int = 51  # Must be odd for Simpson's rule
) -> np.ndarray:
    """
    Compute ∂ρ/∂θ using Duhamel formula with Simpson's rule.
    
    More accurate than trapezoid rule for smooth integrands.
    
    Parameters
    ----------
    rho : ndarray, shape (D, D)
        Density matrix
    H : ndarray, shape (D, D)
        Hamiltonian
    F_centered : ndarray, shape (D, D)
        Centered operator
    n_points : int, odd
        Number of quadrature points (must be odd)
        
    Returns
    -------
    drho : ndarray, shape (D, D)
        Derivative ∂ρ/∂θ
    """
    if n_points % 2 == 0:
        n_points += 1  # Ensure odd
    
    D = rho.shape[0]
    s_vals = np.linspace(0, 1, n_points)
    h = s_vals[1] - s_vals[0]
    
    drho = np.zeros((D, D), dtype=complex)
    
    # Precompute exponentials
    exp_sH = [expm(s * H) for s in s_vals]
    exp_1msH = [expm((1 - s) * H) for s in s_vals]
    
    # Simpson's rule: (h/3)[f0 + 4f1 + 2f2 + 4f3 + 2f4 + ... + 4fn-1 + fn]
    for i in range(n_points):
        integrand = exp_sH[i] @ F_centered @ exp_1msH[i]
        
        if i == 0 or i == n_points - 1:
            weight = h / 3
        elif i % 2 == 1:
            weight = 4 * h / 3
        else:
            weight = 2 * h / 3
        
        drho += weight * integrand
    
    return drho


def compute_H_from_theta(operators: list, theta: np.ndarray) -> tuple:
    """
    Compute H = K - ψ(θ)I where K = ∑ θ_a F_a.
    
    For the exponential family ρ = exp(H), we have:
    - K = ∑ θ_a F_a (the "bare" Hamiltonian)
    - ψ(θ) = log Tr[exp(K)] (log partition function)
    - H = K - ψ(θ)I (normalized so exp(H) is normalized)
    
    Parameters
    ----------
    operators : list
        Basis operators {F_a}
    theta : ndarray
        Natural parameters
        
    Returns
    -------
    H : ndarray
        Hamiltonian H = K - ψ I
    K : ndarray
        Bare Hamiltonian K = ∑ θ_a F_a
    psi : float
        Log partition function ψ = log Tr[exp(K)]
    """
    # Compute K = ∑ θ_a F_a
    D = operators[0].shape[0]
    K = np.zeros((D, D), dtype=complex)
    for theta_a, F_a in zip(theta, operators):
        K += theta_a * F_a
    
    # Compute ψ = log Tr[exp(K)]
    exp_K = expm(K)
    Z = np.trace(exp_K)
    psi = np.log(Z).real
    
    # H = K - ψ I (normalized Hamiltonian)
    H = K - psi * np.eye(D, dtype=complex)
    
    return H, K, psi


def test_duhamel_convergence():
    """Test convergence of Duhamel integration."""
    from qig.exponential_family import QuantumExponentialFamily
    
    print("=" * 70)
    print("TESTING DUHAMEL FORMULA CONVERGENCE")
    print("=" * 70)
    
    exp_family = QuantumExponentialFamily(n_sites=1, d=2)
    theta = np.array([0.3, 0.5, 0.2])
    
    rho = exp_family.rho_from_theta(theta)
    H, K, psi = compute_H_from_theta(exp_family.operators, theta)
    
    # Verify H is correct: exp(H) should equal ρ
    rho_check = expm(H)
    print(f"\nVerify exp(H) = ρ:")
    print(f"  Max error: {np.max(np.abs(rho_check - rho)):.6e}")
    print(f"  (Should be machine precision)")
    
    # Also verify: ρ = exp(K)/Z
    exp_K = expm(K)
    Z = np.trace(exp_K)
    rho_from_K = exp_K / Z
    print(f"\nVerify ρ = exp(K)/Z:")
    print(f"  Max error: {np.max(np.abs(rho_from_K - rho)):.6e}")
    
    # Test ∂ρ/∂θ_X
    a = 0
    F_a = exp_family.operators[a]
    mean_Fa = np.trace(rho @ F_a).real
    F_centered = F_a - mean_Fa * np.eye(2, dtype=complex)
    
    # Ground truth: finite differences
    eps = 1e-8
    theta_plus = theta.copy()
    theta_plus[a] += eps
    rho_plus = exp_family.rho_from_theta(theta_plus)
    
    theta_minus = theta.copy()
    theta_minus[a] -= eps
    rho_minus = exp_family.rho_from_theta(theta_minus)
    
    drho_fd = (rho_plus - rho_minus) / (2 * eps)
    
    print(f"\n∂ρ/∂θ_X convergence:")
    print(f"{'n_points':<10} {'Error':<15} {'Hermiticity':<15}")
    print("-" * 40)
    
    for n_points in [2, 5, 10, 20, 50, 100]:
        drho_duhamel = duhamel_derivative(rho, H, F_centered, n_points)
        
        err = np.max(np.abs(drho_duhamel - drho_fd))
        herm_err = np.max(np.abs(drho_duhamel - drho_duhamel.conj().T))
        
        print(f"{n_points:<10} {err:<15.6e} {herm_err:<15.6e}")
    
    # Also test SLD (which is n_points=2)
    drho_sld = exp_family.rho_derivative(theta, a)
    err_sld = np.max(np.abs(drho_sld - drho_fd))
    print(f"\nSLD formula error: {err_sld:.6e}")
    print(f"  (Should match n_points=2)")


if __name__ == "__main__":
    test_duhamel_convergence()

