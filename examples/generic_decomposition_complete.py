"""
Complete GENERIC decomposition example using high-level interface.

This demonstrates the GenericDecomposition class which orchestrates
the full 12-step procedure for extracting reversible (Hamiltonian)
and irreversible (dissipative) dynamics from the quantum inaccessible game.

System: 2-qubit entangled pair with su(4) generators (15 parameters)
"""

import numpy as np
import matplotlib.pyplot as plt

from qig.exponential_family import QuantumExponentialFamily
from qig.generic_decomposition import GenericDecomposition, run_generic_decomposition


def main():
    """Run complete GENERIC decomposition example."""
    
    print("\n" + "="*70)
    print("GENERIC Decomposition Example")
    print("2-Qubit Entangled Pair")
    print("="*70)
    
    # Initialize 2-qubit system with pair basis (su(4) generators)
    print("\n1. Initialize exponential family...")
    exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
    print(f"   System: {exp_fam.n_sites} qubits")
    print(f"   Hilbert space dimension: {exp_fam.D}")
    print(f"   Number of parameters: {exp_fam.n_params}")
    
    # Start near the LME origin
    print("\n2. Choose initial state near LME origin...")
    np.random.seed(42)
    theta = 0.1 * np.random.randn(exp_fam.n_params)
    print(f"   ||θ|| = {np.linalg.norm(theta):.6f}")
    
    # Method 1: Using the convenience function
    print("\n3. Running GENERIC decomposition (using convenience function)...")
    print("   Note: Skipping diffusion operator (expensive)")
    results = run_generic_decomposition(
        theta,
        exp_fam,
        method='duhamel',
        compute_diffusion=False,
        verbose=True,
        print_summary=True
    )
    
    # Extract key results
    print("\n4. Key Results:")
    print(f"   Effective Hamiltonian shape: {results['H_eff'].shape}")
    print(f"   Hamiltonian coefficients: {results['eta'][:5]}... (first 5)")
    print(f"   Antisymmetric norm: {np.linalg.norm(results['A'], 'fro'):.6f}")
    print(f"   Symmetric norm: {np.linalg.norm(results['S'], 'fro'):.6f}")
    
    # Visualize the decomposition
    print("\n5. Creating visualizations...")
    visualize_decomposition(results)
    
    # Method 2: Using the class directly for more control
    print("\n6. Alternative: Using GenericDecomposition class directly...")
    decomp = GenericDecomposition(exp_fam, method='duhamel', compute_diffusion=False)
    results2 = decomp.compute_all(theta, verbose=False)
    decomp.print_summary(detailed=True)
    
    print("\n" + "="*70)
    print("Example Complete!")
    print("="*70)


def visualize_decomposition(results):
    """Create visualizations of GENERIC decomposition."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('GENERIC Decomposition: 2-Qubit Entangled Pair', fontsize=14, fontweight='bold')
    
    # Row 1: Jacobian components
    M = results['M']
    S = results['S']
    A = results['A']
    
    vmax = max(np.abs(M).max(), np.abs(S).max(), np.abs(A).max())
    
    im0 = axes[0, 0].imshow(M, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0, 0].set_title('Jacobian M = ∂F/∂θ')
    axes[0, 0].set_xlabel('θ_j')
    axes[0, 0].set_ylabel('θ_i')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(S, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0, 1].set_title('Symmetric Part S (Dissipative)')
    axes[0, 1].set_xlabel('θ_j')
    axes[0, 1].set_ylabel('θ_i')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(A, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0, 2].set_title('Antisymmetric Part A (Reversible)')
    axes[0, 2].set_xlabel('θ_j')
    axes[0, 2].set_ylabel('θ_i')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Row 2: Operators and coefficients
    H_eff = results['H_eff']
    eta = results['eta']
    
    # Effective Hamiltonian (real and imaginary parts)
    vmax_H = max(np.abs(H_eff.real).max(), np.abs(H_eff.imag).max())
    
    im3 = axes[1, 0].imshow(H_eff.real, cmap='RdBu_r', vmin=-vmax_H, vmax=vmax_H)
    axes[1, 0].set_title('Re(H_eff)')
    axes[1, 0].set_xlabel('j')
    axes[1, 0].set_ylabel('i')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(H_eff.imag, cmap='RdBu_r', vmin=-vmax_H, vmax=vmax_H)
    axes[1, 1].set_title('Im(H_eff)')
    axes[1, 1].set_xlabel('j')
    axes[1, 1].set_ylabel('i')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # Hamiltonian coefficients
    axes[1, 2].bar(range(len(eta)), eta)
    axes[1, 2].set_title('Hamiltonian Coefficients η_a')
    axes[1, 2].set_xlabel('Parameter index a')
    axes[1, 2].set_ylabel('η_a')
    axes[1, 2].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('generic_decomposition_complete.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: generic_decomposition_complete.png")
    
    # Create a second figure for diagnostics
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot checks as a bar chart
    checks = results['diagnostics']['checks']
    check_names = list(checks.keys())
    check_values = [1 if v else 0 for v in checks.values()]
    colors = ['green' if v else 'red' for v in check_values]
    
    ax.barh(check_names, check_values, color=colors, alpha=0.7)
    ax.set_xlim([0, 1.2])
    ax.set_xlabel('Pass (1) / Fail (0)')
    ax.set_title('GENERIC Decomposition Validation Checks')
    ax.axvline(x=1, color='k', linestyle='--', linewidth=1)
    
    # Add text annotations
    for i, (name, value) in enumerate(zip(check_names, check_values)):
        status = "✓ PASS" if value else "✗ FAIL"
        ax.text(1.05, i, status, va='center', fontweight='bold',
                color='green' if value else 'red')
    
    plt.tight_layout()
    plt.savefig('generic_decomposition_diagnostics.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: generic_decomposition_diagnostics.png")


if __name__ == "__main__":
    main()

