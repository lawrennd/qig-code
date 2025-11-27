"""
Test that the third-order cumulant tensor T_abc = ∂³ψ/∂θ_a∂θ_b∂θ_c is totally symmetric.

This is a fundamental sanity check: since ψ(θ) is a scalar function, mixed partial
derivatives must commute regardless of operator non-commutativity.

Updated for CIP-0002: Uses QuantumExponentialFamily from qig library.
"""
import numpy as np
import pytest
from qig.exponential_family import QuantumExponentialFamily
from tests.tolerance_framework import quantum_assert_close


def test_third_cumulant_symmetry():
    """
    Test T_abc = T_bac = T_cab etc. (total symmetry under all permutations).
    
    We verify that the BKM derivative (third cumulant) is symmetric by checking
    that the Hessian of the Fisher information is symmetric in its indices.
    """
    print("=" * 70)
    print("TESTING THIRD-ORDER CUMULANT SYMMETRY")
    print("=" * 70)
    
    # Setup: Use 1 qutrit pair (smaller system for faster testing)
    exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
    n_params = exp_fam.n_params
    
    print(f"System: 1 qutrit pair")
    print(f"Hilbert space dimension: {exp_fam.D}")
    print(f"Number of parameters: {n_params}")
    
    # Test at a non-zero point
    np.random.seed(42)
    theta = 0.1 * np.random.randn(n_params)
    
    # Compute Fisher information at this point
    G = exp_fam.fisher_information(theta)
    print(f"\nFisher information ||G|| = {np.linalg.norm(G):.4f}")
    
    # Test symmetry by computing ∂G_ab/∂θ_c numerically
    # This is the third cumulant T_abc
    eps = 1e-5
    
    # Sample a few indices to test (checking all would be too slow)
    test_indices = [(0, 1, 2), (0, 2, 1), (1, 0, 2), 
                    (2, 0, 1), (1, 2, 0), (2, 1, 0)]
    
    print(f"\nComputing third cumulant for sample indices...")
    print(f"Using finite difference with eps = {eps}")
        
    # Compute T[0,1,2] and its permutations
    a, b, c = 0, 1, 2
    
    def compute_T_abc(a, b, c):
        """Compute T_abc = ∂G_ab/∂θ_c"""
        theta_plus = theta.copy()
        theta_plus[c] += eps
        G_plus = exp_fam.fisher_information(theta_plus)
        
        theta_minus = theta.copy()
        theta_minus[c] -= eps
        G_minus = exp_fam.fisher_information(theta_minus)
        
        return (G_plus[a, b] - G_minus[a, b]) / (2 * eps)
    
    # Compute all six permutations
    T_values = {}
    for perm in test_indices:
        T_values[perm] = compute_T_abc(*perm)
        print(f"  T[{perm[0]},{perm[1]},{perm[2]}] = {T_values[perm]:.6e}")
    
    reference = T_values[(0, 1, 2)]
    
    print(f"\nSymmetry analysis:")
    print(f"  Reference value: {reference:.6e}")
    
    for perm in test_indices[1:]:
        quantum_assert_close(T_values[perm], reference, 'fisher_metric',
                           err_msg=f"Third cumulant permutation T[{perm}] != T[0,1,2]")
    
    print("\n✓ Third cumulant is symmetric (within numerical precision)")
    print("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
