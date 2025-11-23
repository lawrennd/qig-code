"""
Test that the third-order cumulant tensor T_abc = ∂³ψ/∂θ_a∂θ_b∂θ_c is totally symmetric.

This is a fundamental sanity check: since ψ(θ) is a scalar function, mixed partial
derivatives must commute regardless of operator non-commutativity.
"""
import numpy as np
import sys
sys.path.append('/Users/neil/lawrennd/the-inaccessible-game-orgin')

from quantum_qutrit_n3 import (
    single_site_operators,
    compute_bkm_derivative
)

def test_third_cumulant_symmetry():
    """
    Test T_abc = T_bac = T_cab etc. (total symmetry under all permutations).
    
    We compute T_kab = ∂G_ab/∂θ_k using compute_bkm_derivative(k),
    and verify all six permutations match.
    """
    print("=" * 70)
    print("TESTING THIRD-ORDER CUMULANT SYMMETRY")
    print("=" * 70)
    
    # Setup
    n_sites = 3
    operators = single_site_operators(n_sites)
    
    # Random parameters
    np.random.seed(42)
    theta = np.random.randn(len(operators)) * 0.5
    
    # Test several triples (a, b, c)
    test_triples = [
        (0, 0, 0),   # All same
        (0, 1, 2),   # All different
        (0, 0, 1),   # Two same
        (5, 10, 15), # Different operators
        (3, 7, 11),  # Another set
    ]
    
    all_symmetric = True
    max_asymmetry = 0.0
    
    for a, b, c in test_triples:
        print(f"\nTesting triple ({a}, {b}, {c}):")
        
        # Compute all six permutations of T_abc
        # T_abc = ∂G_ab/∂θ_c
        dG_c = compute_bkm_derivative(theta, operators, c)
        T_abc = dG_c[a, b]
        
        # T_acb = ∂G_ac/∂θ_b
        dG_b = compute_bkm_derivative(theta, operators, b)
        T_acb = dG_b[a, c]
        
        # T_bac = ∂G_ba/∂θ_c (should equal T_abc by symmetry of G)
        T_bac = dG_c[b, a]
        
        # T_bca = ∂G_bc/∂θ_a
        dG_a = compute_bkm_derivative(theta, operators, a)
        T_bca = dG_a[b, c]
        
        # T_cab = ∂G_ca/∂θ_b (should equal T_acb by symmetry of G)
        T_cab = dG_b[c, a]
        
        # T_cba = ∂G_cb/∂θ_a (should equal T_bca by symmetry of G)
        T_cba = dG_a[c, b]
        
        # Collect all six values
        values = np.array([T_abc, T_acb, T_bac, T_bca, T_cab, T_cba])
        
        # Check symmetry
        mean_val = np.mean(values)
        std_val = np.std(values)
        max_dev = np.max(np.abs(values - mean_val))
        rel_dev = max_dev / (abs(mean_val) + 1e-12)
        
        print(f"  T_abc = {T_abc:.8e}")
        print(f"  T_acb = {T_acb:.8e}")
        print(f"  T_bac = {T_bac:.8e}")
        print(f"  T_bca = {T_bca:.8e}")
        print(f"  T_cab = {T_cab:.8e}")
        print(f"  T_cba = {T_cba:.8e}")
        print(f"  Mean: {mean_val:.8e}")
        print(f"  Std:  {std_val:.8e}")
        print(f"  Max deviation: {max_dev:.8e}")
        print(f"  Relative deviation: {rel_dev * 100:.4f}%")
        
        # Threshold: should be symmetric to within numerical precision
        if rel_dev > 0.01:  # 1% tolerance
            print(f"  ⚠️  WARNING: Asymmetry exceeds 1%!")
            all_symmetric = False
        else:
            print(f"  ✓ Symmetric within 1%")
        
        max_asymmetry = max(max_asymmetry, rel_dev)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Maximum relative asymmetry: {max_asymmetry * 100:.4f}%")
    
    if all_symmetric:
        print("✓ Third-order cumulant tensor is totally symmetric")
        print("  This confirms the mathematical property T_abc = T_bac = ... ")
        print("  is correctly implemented despite operator non-commutativity.")
    else:
        print("✗ Third-order cumulant shows asymmetry > 1%")
        print("  This suggests an implementation error in compute_bkm_derivative.")
    
    return all_symmetric

if __name__ == "__main__":
    success = test_third_cumulant_symmetry()
    sys.exit(0 if success else 1)

