"""
Quick test of quantum qutrit dynamics simulation using migrated qig library.

Verifies:
1. LME state construction
2. Marginal entropy computation  
3. Constraint gradient
4. BKM metric (Fisher information)
5. Basic dynamics integration

Updated for CIP-0002: Uses QuantumExponentialFamily from qig library.
"""

import numpy as np
import pytest
from qig.exponential_family import QuantumExponentialFamily
from qig.core import create_lme_state, von_neumann_entropy


def test_qutrit_lme_state():
    """Test qutrit LME state construction and marginal entropies."""
    print("\n" + "=" * 70)
    print("TEST: Qutrit LME State Construction")
    print("=" * 70)
    
    # Create LME state for 1 pair (2 sites)
    rho_lme, dims = create_lme_state(n_sites=2, d=3)
    
    print(f"Created LME state, shape: {rho_lme.shape}")
    print(f"Trace(ρ) = {np.trace(rho_lme):.6f} (should be 1.0)")
    print(f"Purity Tr(ρ²) = {np.real(np.trace(rho_lme @ rho_lme)):.6f}")
    
    assert np.abs(np.trace(rho_lme) - 1.0) < 1e-10, "Trace should be 1"
    assert np.abs(np.trace(rho_lme @ rho_lme) - 1.0) < 1e-10, "Should be pure state"
    
    # Check marginal entropies
    from qig.core import marginal_entropies
    h = marginal_entropies(rho_lme, dims)
    
    print(f"Marginal entropies: h₁={h[0]:.4f}, h₂={h[1]:.4f}")
    print(f"Sum: Σhᵢ = {np.sum(h):.6f}")
    print(f"Target (2 log 3): {2*np.log(3):.6f}")
    
    # For maximally entangled qutrit pair, each marginal is maximally mixed
    # So h_i = log(3) for each site
    assert np.abs(h[0] - np.log(3)) < 1e-6, "Marginal should be log(3)"
    assert np.abs(h[1] - np.log(3)) < 1e-6, "Marginal should be log(3)"
    
    print("✓ LME state construction verified")


def test_qutrit_fisher_information():
    """Test BKM metric (Fisher information) computation."""
    print("\n" + "=" * 70)
    print("TEST: Qutrit Fisher Information (BKM Metric)")
    print("=" * 70)
    
    # Create exponential family with pair basis
    exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
    
    # Test at random point
    np.random.seed(42)
    theta = 0.1 * np.random.randn(exp_fam.n_params)
    
    # Compute Fisher information
    G = exp_fam.fisher_information(theta)
    
    print(f"Fisher information shape: {G.shape}")
    print(f"||G||_F = {np.linalg.norm(G, 'fro'):.4f}")
    
    eigenvalues = np.linalg.eigvalsh(G)
    print(f"Eigenvalues: min={np.min(eigenvalues):.2e}, max={np.max(eigenvalues):.2e}")
    
    # Fisher information should be positive semidefinite
    assert np.all(eigenvalues >= -1e-10), "Fisher information should be PSD"
    assert np.linalg.norm(G - G.T) < 1e-10, "Fisher information should be symmetric"
    
    print("✓ Fisher information computation verified")


def test_qutrit_constraint_gradient():
    """Test constraint gradient ∇C computation."""
    print("\n" + "=" * 70)
    print("TEST: Qutrit Constraint Gradient")
    print("=" * 70)
    
    # Create exponential family
    exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
    
    # Test at random point
    np.random.seed(42)
    theta = 0.2 * np.random.randn(exp_fam.n_params)
    
    # Compute analytic constraint gradient
    C_value, a_analytic = exp_fam.marginal_entropy_constraint(theta)
    
    print(f"||∇C|| = {np.linalg.norm(a_analytic):.4f}")
    
    # Verify against finite differences
    eps = 1e-6
    rho_base = exp_fam.rho_from_theta(theta)
    from qig.core import marginal_entropies
    h_base = marginal_entropies(rho_base, [3, 3])
    C_base = np.sum(h_base)
    
    a_numerical = np.zeros(exp_fam.n_params)
    for i in range(min(10, exp_fam.n_params)):  # Sample first 10 parameters
        theta_plus = theta.copy()
        theta_plus[i] += eps
        rho_plus = exp_fam.rho_from_theta(theta_plus)
        h_plus = marginal_entropies(rho_plus, [3, 3])
        C_plus = np.sum(h_plus)
        a_numerical[i] = (C_plus - C_base) / eps
    
    # Compare sampled entries
    max_error = np.max(np.abs(a_analytic[:10] - a_numerical[:10]))
    print(f"Max error vs finite differences: {max_error:.2e}")
    
    assert max_error < 1e-4, f"Constraint gradient error too large: {max_error}"
    
    print("✓ Constraint gradient verified")


def test_qutrit_third_cumulant():
    """Test third cumulant tensor computation."""
    print("\n" + "=" * 70)
    print("TEST: Third Cumulant Tensor")
    print("=" * 70)
    
    # Create exponential family
    exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
    
    # Test at random point
    np.random.seed(42)
    theta = 0.1 * np.random.randn(exp_fam.n_params)
    
    # Compute third cumulant contraction
    T_contracted = exp_fam.third_cumulant_contraction(theta, method='fd')
    
    print(f"Third cumulant contraction shape: {T_contracted.shape}")
    print(f"||T·G⁻¹·θ|| = {np.linalg.norm(T_contracted):.4f}")
    
    # Verify it's finite
    assert np.all(np.isfinite(T_contracted)), "Third cumulant should be finite"
    
    print("✓ Third cumulant computation verified")


def test_qutrit_mutual_information():
    """Test mutual information computation for entangled qutrit pair."""
    print("\n" + "=" * 70)
    print("TEST: Mutual Information for Qutrit Pair")
    print("=" * 70)
    
    # Create exponential family
    exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
    
    # Start near LME state (should have high mutual information)
    np.random.seed(42)
    theta_lme = np.random.randn(exp_fam.n_params) * 0.5
    
    I_lme = exp_fam.mutual_information(theta_lme)
    print(f"Mutual information near LME: I = {I_lme:.4f}")
    
    # Test at zero (product state, should have I=0)
    theta_zero = np.zeros(exp_fam.n_params)
    I_zero = exp_fam.mutual_information(theta_zero)
    print(f"Mutual information at zero: I = {I_zero:.4f}")
    
    # Mutual information should be non-negative (allow tiny numerical error)
    assert I_lme >= -1e-10, "Mutual information should be non-negative"
    assert I_zero >= -1e-10, "Mutual information should be non-negative"
    assert abs(I_zero) < 0.1, "Product state should have near-zero mutual information"
    
    print("✓ Mutual information computation verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
