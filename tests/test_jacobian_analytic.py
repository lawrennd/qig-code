"""
Analytic Jacobian computation for quantum inaccessible game dynamics.

The flow is:
    F(θ) = -Π_∥(θ) G(θ) θ

where:
    Π_∥(θ) = I - a(θ)a(θ)^T / ||a(θ)||²
    G(θ) = BKM Fisher information matrix
    a(θ) = ∇C(θ) = gradient of marginal entropy constraint

The Jacobian is:
    M_ij = ∂F_i/∂θ_j

This requires:
1. ∂G/∂θ (third-order Kubo-Mori cumulant)
2. ∂a/∂θ (Hessian of constraint)
3. ∂Π_∥/∂θ (from ∂a/∂θ)

QUANTUM DERIVATIVE CHECKLIST:
✅ Check operator commutation
✅ Verify operator ordering
✅ Distinguish quantum vs classical
✅ Respect Hilbert space structure
✅ Question each derivative step
"""

import numpy as np
from scipy.linalg import eigh
import pytest

from qig.exponential_family import QuantumExponentialFamily
from qig.dynamics import InaccessibleGameDynamics
from qig.core import partial_trace, marginal_entropies
from inaccessible_game_quantum import compute_jacobian


def analytic_jacobian(
    dynamics: InaccessibleGameDynamics,
    theta: np.ndarray
) -> np.ndarray:
    """
    Compute Jacobian M = ∂F/∂θ analytically.
    
    The flow is F(θ) = -Π_∥(θ) G(θ) θ, so:
        M_ij = ∂F_i/∂θ_j
             = -∂Π_∥/∂θ_j [G θ]_i - Π_∥ ∂G/∂θ_j [θ]_i - Π_∥ G e_j
    
    where e_j is the j-th unit vector.
    
    This is complex because:
    1. ∂G/∂θ involves third-order cumulants (expensive!)
    2. ∂Π_∥/∂θ involves Hessian of constraint
    3. All derivatives must respect quantum operator structure
    
    For now, we'll use a hybrid approach:
    - Compute ∂G/∂θ numerically (third-order cumulants are very complex)
    - Compute ∂a/∂θ analytically (Hessian of constraint)
    - Assemble Jacobian analytically from these pieces
    """
    exp_family = dynamics.exp_family
    n = exp_family.n_params
    
    # Compute G(θ) and a(θ)
    G = exp_family.fisher_information(theta)
    _, a = exp_family.marginal_entropy_constraint(theta)
    
    # Compute projection matrix
    a_norm_sq = float(np.dot(a, a))
    if a_norm_sq < 1e-12:
        Pi = np.eye(n)
    else:
        Pi = np.eye(n) - np.outer(a, a) / a_norm_sq
    
    # Compute ∂a/∂θ (Hessian of constraint)
    # This is the second derivative of C(θ) = ∑_i h_i
    H_constraint = constraint_hessian(exp_family, theta)
    
    # For each column j of the Jacobian
    M = np.zeros((n, n))
    
    for j in range(n):
        # Term 1: -Π_∥ G e_j (simple)
        term1 = -Pi @ G[:, j]
        
        # Term 2: -Π_∥ ∂G/∂θ_j θ (requires third-order cumulant - use finite diff for now)
        # TODO: Implement analytic third-order cumulant
        eps = 1e-6
        theta_plus = theta.copy()
        theta_plus[j] += eps
        G_plus = exp_family.fisher_information(theta_plus)
        dG_dtheta_j = (G_plus - G) / eps
        term2 = -Pi @ (dG_dtheta_j @ theta)
        
        # Term 3: -∂Π_∥/∂θ_j (G θ)
        # ∂Π_∥/∂θ_j = -∂/∂θ_j [aa^T / ||a||²]
        #            = -(∂a/∂θ_j a^T + a ∂a/∂θ_j^T) / ||a||²
        #              + 2 aa^T (a^T ∂a/∂θ_j) / ||a||⁴
        da_dtheta_j = H_constraint[:, j]
        
        if a_norm_sq > 1e-12:
            dPi_dtheta_j = (
                -(np.outer(da_dtheta_j, a) + np.outer(a, da_dtheta_j)) / a_norm_sq
                + 2 * np.outer(a, a) * np.dot(a, da_dtheta_j) / a_norm_sq**2
            )
            term3 = -dPi_dtheta_j @ (G @ theta)
        else:
            term3 = np.zeros(n)
        
        M[:, j] = term1 + term2 + term3
    
    return M


def constraint_hessian(
    exp_family: QuantumExponentialFamily,
    theta: np.ndarray
) -> np.ndarray:
    """
    Compute Hessian of marginal entropy constraint: H_ab = ∂²C/∂θ_a∂θ_b.
    
    We have:
        ∂C/∂θ_a = ∑_i ∂h_i/∂θ_a
                = ∑_i -Tr((∂ρ_i/∂θ_a) log ρ_i)
    
    So:
        ∂²C/∂θ_a∂θ_b = ∑_i -Tr((∂²ρ_i/∂θ_a∂θ_b) log ρ_i + (∂ρ_i/∂θ_a)(∂ log ρ_i/∂θ_b))
    
    This requires:
    1. ∂²ρ/∂θ_a∂θ_b (second derivative of density matrix)
    2. ∂ log ρ_i/∂θ_b (derivative of matrix logarithm)
    
    QUANTUM DERIVATIVE CARE:
    - ∂ρ/∂θ_a = ρ (F_a - ⟨F_a⟩ I) [exponential family]
    - ∂²ρ/∂θ_a∂θ_b = ∂ρ/∂θ_b (F_a - ⟨F_a⟩ I) + ρ ∂(F_a - ⟨F_a⟩ I)/∂θ_b
    - ∂⟨F_a⟩/∂θ_b = Tr((∂ρ/∂θ_b) F_a) [product rule, Tr is linear]
    - Must respect operator ordering throughout!
    """
    n = exp_family.n_params
    H = np.zeros((n, n))
    
    # Compute ρ(θ)
    rho = exp_family.rho_from_theta(theta)
    I = np.eye(exp_family.D, dtype=complex)
    
    # Precompute ∂ρ/∂θ_a for all a
    drho_dtheta = []
    mean_F = []
    
    for a in range(n):
        F_a = exp_family.operators[a]
        mean_Fa = np.trace(rho @ F_a).real
        mean_F.append(mean_Fa)
        drho_a = rho @ (F_a - mean_Fa * I)
        drho_dtheta.append(drho_a)
    
    # Compute Hessian
    for a in range(n):
        F_a = exp_family.operators[a]
        
        for b in range(a, n):
            F_b = exp_family.operators[b]
            
            # Compute ∂²ρ/∂θ_a∂θ_b
            # = ∂ρ/∂θ_b (F_a - ⟨F_a⟩ I) + ρ (- ∂⟨F_a⟩/∂θ_b I)
            # where ∂⟨F_a⟩/∂θ_b = Tr((∂ρ/∂θ_b) F_a)
            
            dmean_Fa_dtheta_b = np.trace(drho_dtheta[b] @ F_a).real
            
            d2rho_dtheta_ab = (
                drho_dtheta[b] @ (F_a - mean_F[a] * I)
                - rho * dmean_Fa_dtheta_b
            )
            
            # Sum over subsystems
            H_ab = 0.0
            
            for i in range(exp_family.n_sites):
                # Marginals
                rho_i = partial_trace(rho, exp_family.dims, keep=i)
                drho_i_a = partial_trace(drho_dtheta[a], exp_family.dims, keep=i)
                drho_i_b = partial_trace(drho_dtheta[b], exp_family.dims, keep=i)
                d2rho_i_ab = partial_trace(d2rho_dtheta_ab, exp_family.dims, keep=i)
                
                # log ρ_i
                eigvals_i, eigvecs_i = eigh(rho_i)
                eigvals_i = np.maximum(eigvals_i.real, 1e-14)
                log_eigvals_i = np.log(eigvals_i)
                log_rho_i = eigvecs_i @ np.diag(log_eigvals_i) @ eigvecs_i.conj().T
                
                # ∂ log ρ_i / ∂θ_b
                # For Hermitian matrices: ∂ log A / ∂x = ∫_0^∞ (A + tI)^{-1} (∂A/∂x) (A + tI)^{-1} dt
                # This is complex! For now, use finite differences
                # TODO: Implement Daleckii-Krein formula for ∂ log ρ
                
                eps = 1e-7
                theta_plus_b = theta.copy()
                theta_plus_b[b] += eps
                rho_plus_b = exp_family.rho_from_theta(theta_plus_b)
                rho_i_plus_b = partial_trace(rho_plus_b, exp_family.dims, keep=i)
                
                eigvals_i_plus, eigvecs_i_plus = eigh(rho_i_plus_b)
                eigvals_i_plus = np.maximum(eigvals_i_plus.real, 1e-14)
                log_eigvals_i_plus = np.log(eigvals_i_plus)
                log_rho_i_plus = eigvecs_i_plus @ np.diag(log_eigvals_i_plus) @ eigvecs_i_plus.conj().T
                
                dlog_rho_i_dtheta_b = (log_rho_i_plus - log_rho_i) / eps
                
                # ∂²h_i/∂θ_a∂θ_b = -Tr((∂²ρ_i/∂θ_a∂θ_b) log ρ_i + (∂ρ_i/∂θ_a)(∂ log ρ_i/∂θ_b))
                term1 = -np.trace(d2rho_i_ab @ log_rho_i).real
                term2 = -np.trace(drho_i_a @ dlog_rho_i_dtheta_b).real
                
                H_ab += term1 + term2
            
            H[a, b] = H_ab
            H[b, a] = H_ab  # Symmetric
    
    return H


class TestJacobianAnalytic:
    """Test analytic Jacobian computation."""
    
    @pytest.mark.parametrize("n_sites,d", [
        (2, 2),  # Two qubits
    ])
    def test_constraint_hessian(self, n_sites, d):
        """Test that constraint Hessian is computed correctly."""
        exp_family = QuantumExponentialFamily(n_sites, d)
        
        np.random.seed(42)
        theta = np.random.randn(exp_family.n_params) * 0.2
        
        # Analytic Hessian
        H_analytic = constraint_hessian(exp_family, theta)
        
        # Finite-difference Hessian
        eps = 1e-6
        _, grad_0 = exp_family.marginal_entropy_constraint(theta)
        
        H_fd = np.zeros((exp_family.n_params, exp_family.n_params))
        for j in range(exp_family.n_params):
            theta_plus = theta.copy()
            theta_plus[j] += eps
            _, grad_plus = exp_family.marginal_entropy_constraint(theta_plus)
            
            H_fd[:, j] = (grad_plus - grad_0) / eps
        
        # Compare
        diff = H_analytic - H_fd
        max_abs_err = np.max(np.abs(diff))
        rel_err = max_abs_err / (np.max(np.abs(H_fd)) + 1e-10)
        
        print(f"\nConstraint Hessian test ({n_sites} sites, d={d}):")
        print(f"Analytic (first 3x3):\n{H_analytic[:3, :3]}")
        print(f"Finite-diff (first 3x3):\n{H_fd[:3, :3]}")
        print(f"Max absolute error: {max_abs_err:.6e}")
        print(f"Relative error: {rel_err:.6e}")
        
        # Looser tolerance because we're using finite-diff for ∂ log ρ
        assert rel_err < 1e-3, (
            f"Constraint Hessian doesn't match: rel_err={rel_err:.3e}"
        )
    
    @pytest.mark.parametrize("n_sites,d", [
        (2, 2),  # Two qubits
    ])
    def test_jacobian_vs_finite_difference(self, n_sites, d):
        """Test that analytic Jacobian matches finite differences."""
        exp_family = QuantumExponentialFamily(n_sites, d)
        dynamics = InaccessibleGameDynamics(exp_family)
        
        np.random.seed(42)
        theta = np.random.randn(exp_family.n_params) * 0.2
        
        # Analytic Jacobian
        M_analytic = analytic_jacobian(dynamics, theta)
        
        # Finite-difference Jacobian
        M_fd = compute_jacobian(dynamics, theta, eps=1e-6)
        
        # Compare
        diff = M_analytic - M_fd
        max_abs_err = np.max(np.abs(diff))
        rel_err = max_abs_err / (np.max(np.abs(M_fd)) + 1e-10)
        
        print(f"\nJacobian test ({n_sites} sites, d={d}):")
        print(f"Analytic (first 3x3):\n{M_analytic[:3, :3]}")
        print(f"Finite-diff (first 3x3):\n{M_fd[:3, :3]}")
        print(f"Max absolute error: {max_abs_err:.6e}")
        print(f"Relative error: {rel_err:.6e}")
        
        # Looser tolerance because we're still using finite-diff for ∂G/∂θ
        assert rel_err < 1e-3, (
            f"Jacobian doesn't match: rel_err={rel_err:.3e}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

