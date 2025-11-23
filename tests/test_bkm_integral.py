"""
Test BKM metric using the direct integral definition.

The Bogoliubov-Kubo-Mori (BKM) metric is defined as:
    G_ab = ∫_0^1 Tr[ρ^s F_a ρ^{1-s} F_b] ds

where F_a are the (centred) sufficient statistics.

This module computes the BKM metric using:
1. Direct numerical integration of the integral definition
2. The spectral formula currently implemented
3. Finite-difference Hessian of ψ(θ)

to identify where the discrepancy arises.
"""

import numpy as np
from scipy.linalg import expm, logm, fractional_matrix_power
from scipy.integrate import quad_vec

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def bkm_metric_integral(rho, F_a, F_b, n_quad=50):
    """
    Compute BKM metric element G_ab using direct numerical integration.
    
    G_ab = ∫_0^1 Tr[ρ^s F_a ρ^{1-s} F_b] ds
    
    Parameters
    ----------
    rho : ndarray
        Density matrix
    F_a, F_b : ndarray
        Operators (should be centred: Tr[ρ F] = 0)
    n_quad : int
        Number of quadrature points
    
    Returns
    -------
    float
        BKM metric element G_ab
    """
    def integrand(s):
        # Compute ρ^s and ρ^{1-s}
        rho_s = fractional_matrix_power(rho, s)
        rho_1ms = fractional_matrix_power(rho, 1 - s)
        
        # Compute Tr[ρ^s F_a ρ^{1-s} F_b]
        product = rho_s @ F_a @ rho_1ms @ F_b
        return np.trace(product).real
    
    # Numerical integration
    s_vals = np.linspace(0, 1, n_quad)
    integrand_vals = np.array([integrand(s) for s in s_vals])
    
    # Trapezoidal rule
    G_ab = np.trapz(integrand_vals, s_vals)
    return G_ab


def test_single_qubit_x_y():
    """
    Test BKM metric for single qubit with X and Y operators.
    """
    # Parameter point
    theta_x = 0.3
    theta_y = 0.5
    
    # Construct density matrix
    K = theta_x * X + theta_y * Y
    rho_unnorm = expm(K)
    Z = np.trace(rho_unnorm)
    rho = rho_unnorm / Z
    
    print(f"Density matrix ρ:")
    print(rho)
    print(f"Tr(ρ) = {np.trace(rho):.6f}")
    print(f"Eigenvalues: {np.linalg.eigvalsh(rho)}")
    
    # Centre the operators
    mean_X = np.trace(rho @ X).real
    mean_Y = np.trace(rho @ Y).real
    
    F_X = X - mean_X * I
    F_Y = Y - mean_Y * I
    
    print(f"\nCentred operators:")
    print(f"⟨X⟩ = {mean_X:.6f}, ⟨Y⟩ = {mean_Y:.6f}")
    print(f"Tr(ρ F_X) = {np.trace(rho @ F_X):.6e}")
    print(f"Tr(ρ F_Y) = {np.trace(rho @ F_Y):.6e}")
    
    # Compute BKM metric using integral definition
    print(f"\n{'='*60}")
    print("BKM metric via integral definition:")
    print(f"{'='*60}")
    
    G_XX_integral = bkm_metric_integral(rho, F_X, F_X, n_quad=100)
    G_XY_integral = bkm_metric_integral(rho, F_X, F_Y, n_quad=100)
    G_YX_integral = bkm_metric_integral(rho, F_Y, F_X, n_quad=100)
    G_YY_integral = bkm_metric_integral(rho, F_Y, F_Y, n_quad=100)
    
    G_integral = np.array([
        [G_XX_integral, G_XY_integral],
        [G_YX_integral, G_YY_integral]
    ])
    
    print(G_integral)
    print(f"Symmetry check: |G_XY - G_YX| = {abs(G_XY_integral - G_YX_integral):.6e}")
    
    # Compute using spectral formula (current implementation)
    print(f"\n{'='*60}")
    print("BKM metric via spectral formula:")
    print(f"{'='*60}")
    
    from scipy.linalg import eigh
    
    eigvals, U = eigh(rho)
    eps_p = 1e-14
    p = np.clip(np.real(eigvals), eps_p, None)
    
    # Transform centred operators to eigenbasis
    F_X_eig = U.conj().T @ F_X @ U
    F_Y_eig = U.conj().T @ F_Y @ U
    
    print(f"Eigenvalues p: {p}")
    print(f"F_X in eigenbasis:\n{F_X_eig}")
    print(f"F_Y in eigenbasis:\n{F_Y_eig}")
    
    # BKM kernel
    p_i = p[:, None]
    p_j = p[None, :]
    diff = p_i - p_j
    log_diff = np.log(p_i) - np.log(p_j)
    
    k = np.zeros_like(diff)
    off_diag = np.abs(diff) > 1e-14
    k[off_diag] = diff[off_diag] / log_diff[off_diag]
    diag_mask = np.eye(len(p), dtype=bool)
    k[diag_mask] = p
    
    print(f"\nBKM kernel k:\n{k}")
    
    # Assemble metric
    G_XX_spectral = np.sum(k * F_X_eig * F_X_eig.T.conj()).real
    G_XY_spectral = np.sum(k * F_X_eig * F_Y_eig.T.conj()).real
    G_YX_spectral = np.sum(k * F_Y_eig * F_X_eig.T.conj()).real
    G_YY_spectral = np.sum(k * F_Y_eig * F_Y_eig.T.conj()).real
    
    G_spectral = np.array([
        [G_XX_spectral, G_XY_spectral],
        [G_YX_spectral, G_YY_spectral]
    ])
    
    print(G_spectral)
    
    # Compute finite-difference Hessian
    print(f"\n{'='*60}")
    print("Hessian via finite differences:")
    print(f"{'='*60}")
    
    def log_partition(theta_x, theta_y):
        K = theta_x * X + theta_y * Y
        return np.log(np.trace(expm(K))).real
    
    eps = 1e-5
    
    psi_pp = log_partition(theta_x + eps, theta_y + eps)
    psi_pm = log_partition(theta_x + eps, theta_y - eps)
    psi_mp = log_partition(theta_x - eps, theta_y + eps)
    psi_mm = log_partition(theta_x - eps, theta_y - eps)
    
    G_XX_fd = (log_partition(theta_x + eps, theta_y) - 2*log_partition(theta_x, theta_y) + log_partition(theta_x - eps, theta_y)) / eps**2
    G_YY_fd = (log_partition(theta_x, theta_y + eps) - 2*log_partition(theta_x, theta_y) + log_partition(theta_x, theta_y - eps)) / eps**2
    G_XY_fd = (psi_pp - psi_pm - psi_mp + psi_mm) / (4 * eps**2)
    
    G_fd = np.array([
        [G_XX_fd, G_XY_fd],
        [G_XY_fd, G_YY_fd]
    ])
    
    print(G_fd)
    
    # Compare all three
    print(f"\n{'='*60}")
    print("COMPARISON:")
    print(f"{'='*60}")
    
    print(f"\nIntegral vs Spectral:")
    diff_IS = G_integral - G_spectral
    print(diff_IS)
    print(f"Max abs diff: {np.max(np.abs(diff_IS)):.6e}")
    print(f"Rel error: {np.max(np.abs(diff_IS)) / np.max(np.abs(G_integral)):.6e}")
    
    print(f"\nIntegral vs Finite-diff:")
    diff_IF = G_integral - G_fd
    print(diff_IF)
    print(f"Max abs diff: {np.max(np.abs(diff_IF)):.6e}")
    print(f"Rel error: {np.max(np.abs(diff_IF)) / np.max(np.abs(G_integral)):.6e}")
    
    print(f"\nSpectral vs Finite-diff:")
    diff_SF = G_spectral - G_fd
    print(diff_SF)
    print(f"Max abs diff: {np.max(np.abs(diff_SF)):.6e}")
    print(f"Rel error: {np.max(np.abs(diff_SF)) / np.max(np.abs(G_fd)):.6e}")
    
    # Check which one is correct
    print(f"\n{'='*60}")
    print("CONCLUSION:")
    print(f"{'='*60}")
    
    if np.max(np.abs(diff_IF)) < 1e-4:
        print("✅ Integral definition matches finite-difference Hessian")
        print("   → The integral definition is the correct ground truth")
        
        if np.max(np.abs(diff_IS)) > 1e-4:
            print("❌ Spectral formula does NOT match integral definition")
            print("   → The spectral implementation has a bug")
    else:
        print("⚠️  Integral and finite-difference don't match")
        print("   → Need to investigate further")


if __name__ == "__main__":
    test_single_qubit_x_y()

