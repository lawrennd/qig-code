"""
Demonstration of GENERIC decomposition for quantum systems.

This example shows how to decompose the flow Jacobian M into symmetric
and antisymmetric parts (S and A), and verify the degeneracy conditions
that characterize the GENERIC structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from qig.exponential_family import QuantumExponentialFamily


def demo_two_qubit_decomposition():
    """
    Demonstrate GENERIC decomposition for a 2-qubit system.
    """
    print("="*70)
    print("GENERIC Decomposition Demo: 2-Qubit System")
    print("="*70)
    
    # Initialize exponential family
    exp_fam = QuantumExponentialFamily(n_sites=2, d=2)
    print(f"\nSystem: {exp_fam.n_sites} qubits")
    print(f"Parameters: {exp_fam.n_params}")
    
    # Choose a point on the manifold
    theta = 0.1 * np.random.rand(exp_fam.n_params)
    print(f"\nState parameters θ: {theta}")
    
    # Compute full Jacobian
    print("\n" + "-"*70)
    print("Computing Flow Jacobian M...")
    M = exp_fam.jacobian(theta)
    print(f"M shape: {M.shape}")
    print(f"M norm: {np.linalg.norm(M):.4f}")
    
    # Decompose into symmetric and antisymmetric parts
    print("\n" + "-"*70)
    print("GENERIC Decomposition: M = S + A")
    S = exp_fam.symmetric_part(theta)
    A = exp_fam.antisymmetric_part(theta)
    
    print(f"\nSymmetric part S:")
    print(f"  Shape: {S.shape}")
    print(f"  Norm: {np.linalg.norm(S):.4f}")
    print(f"  Symmetry error ||S - S^T||: {np.max(np.abs(S - S.T)):.2e}")
    
    print(f"\nAntisymmetric part A:")
    print(f"  Shape: {A.shape}")
    print(f"  Norm: {np.linalg.norm(A):.4f}")
    print(f"  Antisymmetry error ||A + A^T||: {np.max(np.abs(A + A.T)):.2e}")
    
    # Verify reconstruction
    reconstruction_error = np.max(np.abs(M - (S + A)))
    print(f"\nReconstruction error ||M - (S+A)||: {reconstruction_error:.2e}")
    
    # Verify degeneracy conditions
    print("\n" + "-"*70)
    print("Verifying GENERIC Degeneracy Conditions...")
    diagnostics = exp_fam.verify_degeneracy_conditions(theta, tol=1e-6)
    
    print(f"\n1. S annihilates constraint gradient:")
    print(f"   ||S @ ∇C||: {diagnostics['S_annihilates_constraint']:.2e}")
    print(f"   Passed: {diagnostics['S_annihilates_constraint'] < 1e-6}")
    
    print(f"\n2. A annihilates entropy gradient:")
    print(f"   ||A @ ∇H||: {diagnostics['A_annihilates_entropy_gradient']:.2e}")
    print(f"   Passed: {diagnostics['A_annihilates_entropy_gradient'] < 1e-6}")
    
    print(f"\n3. Entropy production (non-negative):")
    print(f"   θ^T S θ: {diagnostics['entropy_production']:.2e}")
    print(f"   Passed: {diagnostics['entropy_production'] >= -1e-12}")
    
    print(f"\nAll conditions passed: {diagnostics['all_passed']}")
    
    # Visualize eigenvalue structure
    print("\n" + "-"*70)
    print("Eigenvalue Structure...")
    
    eigvals_S = np.linalg.eigvalsh(S)
    eigvals_A_real = np.linalg.eigvals(A).real  # Should be zero for antisymmetric
    eigvals_A_imag = np.linalg.eigvals(A).imag
    
    print(f"\nS eigenvalues (real, symmetric):")
    print(f"  Range: [{np.min(eigvals_S):.4f}, {np.max(eigvals_S):.4f}]")
    print(f"  Negative count: {np.sum(eigvals_S < -1e-10)}")
    print(f"  Positive count: {np.sum(eigvals_S > 1e-10)}")
    print(f"  Near-zero count: {np.sum(np.abs(eigvals_S) < 1e-10)}")
    
    print(f"\nA eigenvalues (purely imaginary for antisymmetric):")
    print(f"  Real parts max: {np.max(np.abs(eigvals_A_real)):.2e} (should be ~0)")
    print(f"  Imag parts range: [{np.min(eigvals_A_imag):.4f}, {np.max(eigvals_A_imag):.4f}]")
    
    return {
        'M': M,
        'S': S,
        'A': A,
        'diagnostics': diagnostics,
        'eigvals_S': eigvals_S,
        'eigvals_A_imag': eigvals_A_imag
    }


def plot_eigenvalue_structure(eigvals_S, eigvals_A_imag):
    """
    Plot eigenvalue structure of S and A.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Symmetric part eigenvalues
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.plot(eigvals_S, 'o-', markersize=8, label='S eigenvalues')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Symmetric Part S (Real Eigenvalues)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Antisymmetric part eigenvalues (imaginary)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.plot(eigvals_A_imag, 'o-', markersize=8, color='orange', label='A eigenvalues (imag)')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Imaginary part')
    ax2.set_title('Antisymmetric Part A (Purely Imaginary Eigenvalues)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Run demonstration
    results = demo_two_qubit_decomposition()
    
    # Plot eigenvalue structure
    print("\n" + "="*70)
    print("Generating eigenvalue structure plots...")
    fig = plot_eigenvalue_structure(
        results['eigvals_S'],
        results['eigvals_A_imag']
    )
    
    # Save or show
    try:
        fig.savefig('generic_decomposition_eigenvalues.png', dpi=150, bbox_inches='tight')
        print("Plot saved to: generic_decomposition_eigenvalues.png")
    except:
        print("Could not save plot. Displaying instead...")
        plt.show()
    
    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)

