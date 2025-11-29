"""
Test different variants of the spectral BKM formula.

The BKM metric integral is:
    G_ab = ∫_0^1 Tr[ρ^s F_a ρ^{1-s} F_b] ds

In the eigenbasis of ρ = ∑_i p_i |i⟩⟨i|, this should become:
    G_ab = ∑_{i,j} c(p_i, p_j) * ⟨i|F_a|j⟩ * ⟨j|F_b|i⟩

where c(p_i, p_j) is the Morozova-Chentsov function for the BKM metric.

The question is: what is the correct form of c(p_i, p_j) and how do we
assemble the sum?

This module tests several variants to find the correct one.
"""

import numpy as np
from scipy.linalg import expm, eigh, fractional_matrix_power

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)


def bkm_metric_integral(rho, F_a, F_b, n_quad=100):
    """Ground truth: direct numerical integration."""
    def integrand(s):
        rho_s = fractional_matrix_power(rho, s)
        rho_1ms = fractional_matrix_power(rho, 1 - s)
        product = rho_s @ F_a @ rho_1ms @ F_b
        return np.trace(product).real
    
    s_vals = np.linspace(0, 1, n_quad)
    integrand_vals = np.array([integrand(s) for s in s_vals])
    return np.trapz(integrand_vals, s_vals)


def spectral_variant_1_current(rho, F_a, F_b):
    """
    Current implementation in qig/exponential_family.py:
        G_ab = ∑_{i,j} k(p_i, p_j) * A_a[i,j] * conj(A_b[j,i])
    where k(p_i, p_j) = (p_i - p_j)/(log p_i - log p_j)
    """
    eigvals, U = eigh(rho)
    p = np.clip(np.real(eigvals), 1e-14, None)
    
    # Transform to eigenbasis
    A_a = U.conj().T @ F_a @ U
    A_b = U.conj().T @ F_b @ U
    
    # Kernel: k(p_i, p_j) = (p_i - p_j)/(log p_i - log p_j)
    p_i = p[:, None]
    p_j = p[None, :]
    diff = p_i - p_j
    log_diff = np.log(p_i) - np.log(p_j)
    
    k = np.zeros_like(diff)
    off_diag = np.abs(diff) > 1e-14
    k[off_diag] = diff[off_diag] / log_diff[off_diag]
    diag_mask = np.eye(len(p), dtype=bool)
    k[diag_mask] = p
    
    # Current formula: A_a[i,j] * conj(A_b[j,i])
    G_ab = np.sum(k * A_a * A_b.T.conj()).real
    return G_ab


def spectral_variant_2_symmetric_kernel(rho, F_a, F_b):
    """
    Variant 2: Use symmetric kernel
        c(p_i, p_j) = (p_i + p_j) / 2 * (log p_i - log p_j) / (p_i - p_j)
                    = (p_i + p_j) / (2 * (p_i - p_j) / (log p_i - log p_j))
    """
    eigvals, U = eigh(rho)
    p = np.clip(np.real(eigvals), 1e-14, None)
    
    A_a = U.conj().T @ F_a @ U
    A_b = U.conj().T @ F_b @ U
    
    p_i = p[:, None]
    p_j = p[None, :]
    diff = p_i - p_j
    log_diff = np.log(p_i) - np.log(p_j)
    
    c = np.zeros_like(diff)
    off_diag = np.abs(diff) > 1e-14
    c[off_diag] = ((p_i + p_j) / 2 * log_diff / diff)[off_diag]
    diag_mask = np.eye(len(p), dtype=bool)
    c[diag_mask] = p
    
    G_ab = np.sum(c * A_a * A_b.T.conj()).real
    return G_ab


def spectral_variant_3_rld_kernel(rho, F_a, F_b):
    """
    Variant 3: RLD (Right Logarithmic Derivative) kernel
        c(p_i, p_j) = 2 * p_i * p_j / (p_i + p_j)
    with diagonal limit c(p, p) = p
    """
    eigvals, U = eigh(rho)
    p = np.clip(np.real(eigvals), 1e-14, None)
    
    A_a = U.conj().T @ F_a @ U
    A_b = U.conj().T @ F_b @ U
    
    p_i = p[:, None]
    p_j = p[None, :]
    
    # c must be 2D array (n x n) to match off_diag boolean mask
    c = np.zeros((len(p), len(p)))
    off_diag = np.abs(p_i - p_j) > 1e-14
    c[off_diag] = (2 * p_i * p_j / (p_i + p_j))[off_diag]
    diag_mask = np.eye(len(p), dtype=bool)
    c[diag_mask] = p
    
    G_ab = np.sum(c * A_a * A_b.T.conj()).real
    return G_ab


def spectral_variant_4_correct_assembly(rho, F_a, F_b):
    """
    Variant 4: Correct assembly with proper Hermitian conjugation
        G_ab = ∑_{i,j} c(p_i, p_j) * A_a[i,j] * conj(A_b[i,j])
    
    Note: This is different from A_a[i,j] * conj(A_b[j,i])!
    """
    eigvals, U = eigh(rho)
    p = np.clip(np.real(eigvals), 1e-14, None)
    
    A_a = U.conj().T @ F_a @ U
    A_b = U.conj().T @ F_b @ U
    
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
    
    # Correct assembly: A_a[i,j] * conj(A_b[i,j])
    G_ab = np.sum(k * A_a * np.conj(A_b)).real
    return G_ab


def spectral_variant_5_integral_kernel(rho, F_a, F_b):
    """
    Variant 5: Compute the kernel by evaluating the integral
        c(p_i, p_j) = ∫_0^1 p_i^s p_j^{1-s} ds
    
    For p_i ≠ p_j:
        c(p_i, p_j) = (p_i - p_j) / (log p_i - log p_j)
    For p_i = p_j = p:
        c(p, p) = p
    
    This is the same as variant 1, but let's verify the formula.
    """
    eigvals, U = eigh(rho)
    p = np.clip(np.real(eigvals), 1e-14, None)
    
    A_a = U.conj().T @ F_a @ U
    A_b = U.conj().T @ F_b @ U
    
    # Compute kernel by integration
    n = len(p)
    c = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if abs(p[i] - p[j]) < 1e-14:
                c[i, j] = p[i]
            else:
                # Integrate p_i^s * p_j^{1-s} from 0 to 1
                # = (p_i - p_j) / (log p_i - log p_j)
                c[i, j] = (p[i] - p[j]) / (np.log(p[i]) - np.log(p[j]))
    
    # Assembly: sum over i,j of c[i,j] * A_a[i,j] * conj(A_b[i,j])
    G_ab = np.sum(c * A_a * np.conj(A_b)).real
    return G_ab


def test_all_variants():
    """Test all spectral variants against the integral definition."""
    # Parameter point
    theta_x = 0.3
    theta_y = 0.5
    
    # Construct density matrix
    K = theta_x * X + theta_y * Y
    rho_unnorm = expm(K)
    Z = np.trace(rho_unnorm)
    rho = rho_unnorm / Z
    
    # Centre operators
    mean_X = np.trace(rho @ X).real
    mean_Y = np.trace(rho @ Y).real
    F_X = X - mean_X * I
    F_Y = Y - mean_Y * I
    
    # Ground truth: integral definition
    G_XX_integral = bkm_metric_integral(rho, F_X, F_X)
    G_XY_integral = bkm_metric_integral(rho, F_X, F_Y)
    G_YY_integral = bkm_metric_integral(rho, F_Y, F_Y)
    
    print("="*70)
    print("GROUND TRUTH (Integral Definition):")
    print("="*70)
    G_integral = np.array([
        [G_XX_integral, G_XY_integral],
        [G_XY_integral, G_YY_integral]
    ])
    print(G_integral)
    print()
    
    # Test variants
    variants = [
        ("Variant 1: Current implementation", spectral_variant_1_current),
        ("Variant 2: Symmetric kernel", spectral_variant_2_symmetric_kernel),
        ("Variant 3: RLD kernel", spectral_variant_3_rld_kernel),
        ("Variant 4: Correct assembly (A*conj(B))", spectral_variant_4_correct_assembly),
        ("Variant 5: Integral kernel", spectral_variant_5_integral_kernel),
    ]
    
    for name, func in variants:
        print("="*70)
        print(name)
        print("="*70)
        
        G_XX = func(rho, F_X, F_X)
        G_XY = func(rho, F_X, F_Y)
        G_YY = func(rho, F_Y, F_Y)
        
        G = np.array([
            [G_XX, G_XY],
            [G_XY, G_YY]
        ])
        
        print(G)
        
        diff = G - G_integral
        max_err = np.max(np.abs(diff))
        rel_err = max_err / np.max(np.abs(G_integral))
        
        print(f"\nDifference from integral:")
        print(diff)
        print(f"Max absolute error: {max_err:.6e}")
        print(f"Relative error: {rel_err:.6e}")
        
        if rel_err < 1e-4:
            print("✅ MATCH!")
        else:
            print("❌ MISMATCH")
        print()


if __name__ == "__main__":
    test_all_variants()

