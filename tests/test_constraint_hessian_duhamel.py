"""
Test constraint Hessian with high-precision Duhamel method.

This should achieve < 1% error using the numerical differentiation
of Duhamel ∂ρ/∂θ for computing ∂²ρ/∂θ_a∂θ_b.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/neil/lawrennd/the-inaccessible-game-orgin')

from qig.exponential_family import QuantumExponentialFamily


def compute_constraint_hessian_fd(exp_family, theta, eps=1e-7):
    """Ground truth: finite differences of ∇C."""
    n = exp_family.n_params
    
    # Get gradient at theta
    _, grad_C = exp_family.marginal_entropy_constraint(theta)
    
    hessian = np.zeros((n, n))
    
    for j in range(n):
        # Compute ∂(∇C)/∂θ_j using finite differences
        theta_plus = theta.copy()
        theta_plus[j] += eps
        _, grad_C_plus = exp_family.marginal_entropy_constraint(theta_plus)
        
        theta_minus = theta.copy()
        theta_minus[j] -= eps
        _, grad_C_minus = exp_family.marginal_entropy_constraint(theta_minus)
        
        hessian[:, j] = (grad_C_plus - grad_C_minus) / (2 * eps)
    
    return hessian


def test_single_qubit_duhamel():
    """Test constraint Hessian with Duhamel method on single qubit."""
    print("=" * 70)
    print("TESTING CONSTRAINT HESSIAN WITH DUHAMEL")
    print("=" * 70)
    
    exp_family = QuantumExponentialFamily(n_sites=1, d=2)
    theta = np.array([0.3, 0.5, 0.2])  # X, Y, Z
    
    # Ground truth
    hessian_fd = compute_constraint_hessian_fd(exp_family, theta)
    fd_norm = np.linalg.norm(hessian_fd)
    
    print(f"\nFinite difference Hessian:")
    print(f"  Norm: {fd_norm:.6e}")
    print(f"  Symmetry error: {np.max(np.abs(hessian_fd - hessian_fd.T)):.6e}")
    
    # Test both methods
    print(f"\n{'Method':<15} {'n_points':<10} {'Max Error':<15} {'Rel Error':<15} {'Time (s)':<10}")
    print("-" * 75)
    
    import time
    
    # SLD method (fast baseline)
    start = time.time()
    hessian_sld = exp_family.constraint_hessian(theta, method='sld')
    time_sld = time.time() - start
    
    diff_sld = hessian_sld - hessian_fd
    max_err_sld = np.max(np.abs(diff_sld))
    rel_err_sld = max_err_sld / fd_norm
    
    print(f"{'SLD (fast)':<15} {'-':<10} {max_err_sld:<15.6e} {rel_err_sld:<15.6e} {time_sld:<10.3f}")
    
    # Duhamel method with different precisions
    for n_points in [20, 50, 100]:
        start = time.time()
        hessian_duhamel = exp_family.constraint_hessian(
            theta, method='duhamel', n_points=n_points, eps=1e-7
        )
        time_duhamel = time.time() - start
        
        diff_duhamel = hessian_duhamel - hessian_fd
        max_err_duhamel = np.max(np.abs(diff_duhamel))
        rel_err_duhamel = max_err_duhamel / fd_norm
        
        status = "✓" if rel_err_duhamel < 0.01 else "✗"
        print(f"{'Duhamel':<15} {n_points:<10} {max_err_duhamel:<15.6e} {rel_err_duhamel:<15.6e} {time_duhamel:<10.3f} {status}")
    
    print(f"\nImprovement: {rel_err_sld / rel_err_duhamel:.1f}× better with Duhamel n=100")


def test_diagonal_case_duhamel():
    """Test on diagonal operators (classical case)."""
    print("\n" + "=" * 70)
    print("TESTING DIAGONAL CASE WITH DUHAMEL")
    print("=" * 70)
    
    exp_family = QuantumExponentialFamily(n_sites=1, d=3)
    
    # Use only diagonal operators (λ3 and λ8 from Gell-Mann)
    theta = np.zeros(exp_family.n_params)
    theta[2] = 0.5  # λ3 (diagonal)
    theta[7] = 0.3  # λ8 (diagonal)
    
    # Ground truth
    hessian_fd = compute_constraint_hessian_fd(exp_family, theta)
    fd_norm = np.linalg.norm(hessian_fd)
    
    print(f"\nFinite difference Hessian norm: {fd_norm:.6e}")
    
    # Test both methods
    hessian_sld = exp_family.constraint_hessian(theta, method='sld')
    hessian_duhamel = exp_family.constraint_hessian(theta, method='duhamel', n_points=100)
    
    rel_err_sld = np.max(np.abs(hessian_sld - hessian_fd)) / fd_norm
    rel_err_duhamel = np.max(np.abs(hessian_duhamel - hessian_fd)) / fd_norm
    
    print(f"\nSLD relative error: {rel_err_sld:.6e}")
    print(f"Duhamel relative error: {rel_err_duhamel:.6e}")
    print(f"Improvement: {rel_err_sld / rel_err_duhamel:.1f}×")
    
    if rel_err_duhamel < 0.01:
        print("✓ Duhamel achieves < 1% error")
    else:
        print(f"✗ Duhamel has {rel_err_duhamel*100:.1f}% error")


if __name__ == "__main__":
    test_single_qubit_duhamel()
    test_diagonal_case_duhamel()

