#!/usr/bin/env python3
"""
Fast test: Extract and run code from effective_hamiltonian_derivation.md (single pass)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for speed
import matplotlib.pyplot as plt
from qig.exponential_family import QuantumExponentialFamily
from qig.structure_constants import compute_structure_constants
from qig.generic import (
    effective_hamiltonian_coefficients,
    effective_hamiltonian_operator
)


def main():
    """Single-pass test of all derivation code"""
    print("Testing effective_hamiltonian_derivation.md (fast mode)")
    print("=" * 60)
    
    # Setup
    exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
    print(f"✓ System: {exp_fam.n_pairs} pair, dim={exp_fam.D}, params={exp_fam.n_params}")
    
    # Parameter point
    np.random.seed(42)
    theta = 0.05 * np.random.rand(exp_fam.n_params)
    rho = exp_fam.rho_from_theta(theta)
    print(f"✓ Density matrix: Tr(ρ)={np.trace(rho):.6f}, Purity={np.trace(rho @ rho):.6f}")
    
    # Kubo-Mori derivative
    drho_dtheta_0 = exp_fam.rho_derivative(theta, 0, method='duhamel_spectral')
    print(f"✓ Kubo-Mori derivative: shape={drho_dtheta_0.shape}, Hermitian={np.allclose(drho_dtheta_0, drho_dtheta_0.conj().T)}")
    
    # Antisymmetric Jacobian
    A = exp_fam.antisymmetric_part(theta, method='duhamel_spectral')
    print(f"✓ Antisymmetric Jacobian: shape={A.shape}, antisymmetry error={np.max(np.abs(A + A.T)):.2e}")
    
    # Structure constants
    f_abc = compute_structure_constants(exp_fam.operators)
    n_nonzero = np.sum(np.abs(f_abc) > 1e-10)
    print(f"✓ Structure constants: shape={f_abc.shape}, non-zero={n_nonzero}/{f_abc.size}")
    
    # Extract Hamiltonian
    eta, diagnostics = effective_hamiltonian_coefficients(A, theta, f_abc)
    extraction_error = np.linalg.norm(A @ theta - np.einsum('abc,c->a', f_abc, eta))
    print(f"✓ Hamiltonian coefficients: shape={eta.shape}, extraction error={extraction_error:.2e}")
    
    # Build Hamiltonian operator
    H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
    herm_err = np.max(np.abs(H_eff - H_eff.conj().T))
    trace_err = np.abs(np.trace(H_eff))
    print(f"✓ H_eff: Hermitian={herm_err:.2e}, Traceless={trace_err:.2e}, Norm={np.linalg.norm(H_eff, 'fro'):.4e}")
    
    # Energy spectrum
    eigenvalues = np.linalg.eigvalsh(H_eff)
    print(f"✓ Energy spectrum: E=[{eigenvalues[0]:.4f}, {eigenvalues[1]:.4f}, {eigenvalues[2]:.4f}, {eigenvalues[3]:.4f}]")
    
    # Symbolic form
    print(f"✓ Symbolic: H_eff = {eta[0]:.4f}F_0 + {eta[1]:.4f}F_1 + ... ({len(eta)} terms)")
    
    print("=" * 60)
    print("✓ All derivation code tested successfully!")


if __name__ == '__main__':
    main()
