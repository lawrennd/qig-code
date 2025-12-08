#!/usr/bin/env python3
"""
Test script for effective_hamiltonian_derivation.md

Extracts and runs all code examples from the markdown document to verify they work.
"""
import numpy as np
import matplotlib.pyplot as plt
from qig.exponential_family import QuantumExponentialFamily
from qig.structure_constants import compute_structure_constants
from qig.generic import (
    effective_hamiltonian_coefficients,
    effective_hamiltonian_operator
)


def section_1_2_setup():
    """Section 1.2: Code Setup"""
    print("\n" + "=" * 70)
    print("SECTION 1.2: Code Setup")
    print("=" * 70)
    
    # Create a simple qubit-pair system with Lie-closed basis
    exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
    
    print(f"System dimension: {exp_fam.D}")
    print(f"Number of parameters: {exp_fam.n_params}")
    print(f"Basis operators: {len(exp_fam.operators)}")
    
    return exp_fam


def section_2_3_kubo_mori(exp_fam):
    """Section 2.3: Computing the Kubo-Mori Derivatives"""
    print("\n" + "=" * 70)
    print("SECTION 2.3: Computing the Kubo-Mori Derivatives")
    print("=" * 70)
    
    # Choose a parameter point
    np.random.seed(42)
    theta = 0.05 * np.random.rand(exp_fam.n_params)
    
    # Get the density matrix
    rho = exp_fam.rho_from_theta(theta)
    print(f"Density matrix shape: {rho.shape}")
    print(f"Tr(ρ) = {np.trace(rho):.6f}")
    print(f"Is Hermitian: {np.allclose(rho, rho.conj().T)}")
    
    # Compute Kubo-Mori derivative for first parameter
    drho_dtheta_0 = exp_fam.rho_derivative(theta, 0, method='duhamel_spectral')
    print(f"\n∂ρ/∂θ_0 shape: {drho_dtheta_0.shape}")
    print(f"Is Hermitian: {np.allclose(drho_dtheta_0, drho_dtheta_0.conj().T)}")
    print(f"Tr(∂ρ/∂θ_0) = {np.trace(drho_dtheta_0):.2e}")  # Should be ~0
    
    return theta, rho


def section_3_2_antisymmetric(exp_fam, theta):
    """Section 3.2: Computing the Antisymmetric Jacobian"""
    print("\n" + "=" * 70)
    print("SECTION 3.2: Computing the Antisymmetric Jacobian")
    print("=" * 70)
    
    # Compute the antisymmetric part of the flow Jacobian
    A = exp_fam.antisymmetric_part(theta, method='duhamel_spectral')
    
    print(f"Antisymmetric Jacobian shape: {A.shape}")
    print(f"Max symmetry error: {np.max(np.abs(A + A.T)):.2e}")
    
    # Visualize the structure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # A matrix
    im1 = ax1.imshow(A, cmap='RdBu', vmin=-np.max(np.abs(A)), vmax=np.max(np.abs(A)))
    ax1.set_title('Antisymmetric Jacobian A')
    ax1.set_xlabel('Parameter index b')
    ax1.set_ylabel('Parameter index a')
    plt.colorbar(im1, ax=ax1)
    
    # Antisymmetry check
    antisym_check = A + A.T
    im2 = ax2.imshow(antisym_check, cmap='viridis')
    ax2.set_title('Antisymmetry Check: A + Aᵀ (should be ~0)')
    ax2.set_xlabel('Parameter index b')
    ax2.set_ylabel('Parameter index a')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('/Users/neil/lawrennd/qig-code/examples/antisymmetric_jacobian.png', dpi=150)
    print("Saved figure: antisymmetric_jacobian.png")
    plt.close()
    
    return A


def section_4_2_structure_constants(exp_fam):
    """Section 4.2: Computing Structure Constants"""
    print("\n" + "=" * 70)
    print("SECTION 4.2: Computing Structure Constants")
    print("=" * 70)
    
    # Compute structure constants for our basis
    f_abc = compute_structure_constants(exp_fam.operators)
    
    print(f"Structure constants shape: {f_abc.shape}")
    print(f"Antisymmetry check (should be ~0): {np.max(np.abs(f_abc + f_abc.swapaxes(0,1))):.2e}")
    
    # How many non-zero entries?
    n_nonzero = np.sum(np.abs(f_abc) > 1e-10)
    n_total = f_abc.size
    print(f"Non-zero entries: {n_nonzero} / {n_total} ({100*n_nonzero/n_total:.1f}%)")
    
    return f_abc


def section_5_2_extraction(A, theta, f_abc):
    """Section 5.2: Code Implementation"""
    print("\n" + "=" * 70)
    print("SECTION 5.2: Extracting Hamiltonian Coefficients")
    print("=" * 70)
    
    # Extract Hamiltonian coefficients
    eta, diagnostics = effective_hamiltonian_coefficients(A, theta, f_abc)
    
    print(f"Hamiltonian coefficients η shape: {eta.shape}")
    print(f"Solution residual: {diagnostics['residual']:.2e}")
    print(f"Condition number: {diagnostics['condition_number']:.2e}")
    
    # Verify the extraction formula
    lhs = A @ theta  # Left-hand side
    rhs = np.einsum('abc,c->a', f_abc, eta)  # Right-hand side
    
    extraction_error = np.linalg.norm(lhs - rhs)
    print(f"\nExtraction formula error: {extraction_error:.2e}")
    print(f"  ||A @ θ||: {np.linalg.norm(lhs):.4e}")
    print(f"  ||f @ η||: {np.linalg.norm(rhs):.4e}")
    
    return eta


def section_6_2_build_hamiltonian(eta, exp_fam):
    """Section 6.2: Code Implementation"""
    print("\n" + "=" * 70)
    print("SECTION 6.2: Building the Effective Hamiltonian Operator")
    print("=" * 70)
    
    # Build the effective Hamiltonian operator
    H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
    
    print(f"H_eff shape: {H_eff.shape}")
    print(f"Is Hermitian: {np.allclose(H_eff, H_eff.conj().T)}")
    print(f"Max Hermiticity error: {np.max(np.abs(H_eff - H_eff.conj().T)):.2e}")
    print(f"Trace: {np.trace(H_eff):.2e}")
    print(f"Frobenius norm: {np.linalg.norm(H_eff, 'fro'):.4e}")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Real part
    im1 = ax1.imshow(np.real(H_eff), cmap='RdBu', 
                      vmin=-np.max(np.abs(H_eff)), vmax=np.max(np.abs(H_eff)))
    ax1.set_title('Re(H_eff)')
    plt.colorbar(im1, ax=ax1)
    
    # Imaginary part
    im2 = ax2.imshow(np.imag(H_eff), cmap='RdBu',
                      vmin=-np.max(np.abs(H_eff)), vmax=np.max(np.abs(H_eff)))
    ax2.set_title('Im(H_eff)')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('/Users/neil/lawrennd/qig-code/examples/effective_hamiltonian.png', dpi=150)
    print("Saved figure: effective_hamiltonian.png")
    plt.close()
    
    return H_eff


def section_6_3_energy_spectrum(H_eff):
    """Section 6.3: Physical Interpretation"""
    print("\n" + "=" * 70)
    print("SECTION 6.3: Physical Interpretation - Energy Spectrum")
    print("=" * 70)
    
    eigenvalues = np.linalg.eigvalsh(H_eff)
    print("Energy eigenvalues:")
    for i, E in enumerate(eigenvalues):
        print(f"  E_{i} = {E:+.6f}")
    
    # Energy gaps
    gaps = np.diff(eigenvalues)
    print(f"\nEnergy gaps: {gaps}")
    print(f"Ground state: E_0 = {eigenvalues[0]:.6f}")
    print(f"Excited state splitting: ΔE = {eigenvalues[-1] - eigenvalues[0]:.6f}")


def section_10_complete_example():
    """Section 10: Complete Working Example"""
    print("\n" + "=" * 70)
    print("SECTION 10: Complete Working Example (Integrated)")
    print("=" * 70)
    
    # 1. Create quantum exponential family
    print("=" * 60)
    print("EFFECTIVE HAMILTONIAN EXTRACTION")
    print("=" * 60)
    
    exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
    print(f"\nSystem: {exp_fam.n_pairs} qubit pair(s)")
    print(f"Hilbert dimension: {exp_fam.D}")
    print(f"Parameters: {exp_fam.n_params}")
    
    # 2. Choose parameter point
    np.random.seed(42)
    theta = 0.05 * np.random.rand(exp_fam.n_params)
    rho = exp_fam.rho_from_theta(theta)
    
    print(f"\nDensity matrix:")
    print(f"  Tr(ρ) = {np.trace(rho):.6f}")
    print(f"  Purity = {np.trace(rho @ rho):.6f}")
    
    # 3. Compute antisymmetric Jacobian
    print(f"\nComputing GENERIC decomposition...")
    A = exp_fam.antisymmetric_part(theta, method='duhamel_spectral')
    print(f"  Antisymmetric Jacobian A: {A.shape}")
    print(f"  Max antisymmetry error: {np.max(np.abs(A + A.T)):.2e}")
    
    # 4. Compute structure constants
    print(f"\nComputing Lie structure constants...")
    f_abc = compute_structure_constants(exp_fam.operators)
    print(f"  Structure constants f_abc: {f_abc.shape}")
    n_nonzero = np.sum(np.abs(f_abc) > 1e-10)
    print(f"  Non-zero entries: {n_nonzero} / {f_abc.size}")
    
    # 5. Extract Hamiltonian coefficients
    print(f"\nExtracting Hamiltonian coefficients...")
    eta, diagnostics = effective_hamiltonian_coefficients(A, theta, f_abc)
    print(f"  Coefficients η: {eta.shape}")
    print(f"  Residual: {diagnostics['residual']:.2e}")
    print(f"  Condition number: {diagnostics['condition_number']:.2e}")
    
    # Verify extraction formula
    lhs = A @ theta
    rhs = np.einsum('abc,c->a', f_abc, eta)
    print(f"  Extraction error: {np.linalg.norm(lhs - rhs):.2e}")
    
    # 6. Build effective Hamiltonian operator
    print(f"\nBuilding effective Hamiltonian H_eff...")
    H_eff = effective_hamiltonian_operator(eta, exp_fam.operators)
    print(f"  H_eff: {H_eff.shape}")
    print(f"  Hermiticity error: {np.max(np.abs(H_eff - H_eff.conj().T)):.2e}")
    print(f"  Trace: {np.trace(H_eff):.2e}")
    print(f"  Norm: {np.linalg.norm(H_eff, 'fro'):.4e}")
    
    # 7. Analyze energy spectrum
    print(f"\nEnergy spectrum:")
    eigenvalues = np.linalg.eigvalsh(H_eff)
    for i, E in enumerate(eigenvalues):
        print(f"  E_{i} = {E:+.6f}")
    
    # 8. Show symbolic form
    print(f"\nSymbolic form:")
    print(f"  H_eff(θ) = Σ_c η_c(θ) F_c")
    print(f"  where η solves: A(θ)θ = f·η")
    print(f"\n  Explicitly:")
    print(f"    H_eff = ", end="")
    for c in range(min(3, len(eta))):
        if c > 0:
            print(" + ", end="")
        print(f"{eta[c]:.4f} F_{c}", end="")
    if len(eta) > 3:
        print(f" + ... ({len(eta)-3} more terms)")
    else:
        print()
    
    print("\n" + "=" * 60)
    print("✓ Extraction complete!")
    print("=" * 60)


def main():
    """Run all code examples from the markdown document"""
    print("=" * 70)
    print("TESTING CODE FROM effective_hamiltonian_derivation.md")
    print("=" * 70)
    print("\nThis script runs all code examples from the markdown document")
    print("to verify they execute correctly.\n")
    
    # Run section by section
    exp_fam = section_1_2_setup()
    theta, rho = section_2_3_kubo_mori(exp_fam)
    A = section_3_2_antisymmetric(exp_fam, theta)
    f_abc = section_4_2_structure_constants(exp_fam)
    eta = section_5_2_extraction(A, theta, f_abc)
    H_eff = section_6_2_build_hamiltonian(eta, exp_fam)
    section_6_3_energy_spectrum(H_eff)
    
    # Run complete integrated example
    section_10_complete_example()
    
    print("\n" + "=" * 70)
    print("✓ ALL CODE EXAMPLES EXECUTED SUCCESSFULLY")
    print("=" * 70)
    print("\nGenerated figures:")
    print("  - examples/antisymmetric_jacobian.png")
    print("  - examples/effective_hamiltonian.png")


if __name__ == '__main__':
    main()
