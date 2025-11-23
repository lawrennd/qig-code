"""
Test and implement analytic gradient of marginal entropy constraint.

For a quantum exponential family ρ(θ) = exp(K(θ))/Z(θ) where K(θ) = ∑_a θ_a F_a,
we need to compute:

    ∂C/∂θ_a where C(θ) = ∑_i h_i(θ)

and h_i = -Tr(ρ_i log ρ_i) is the marginal entropy of subsystem i.

The key insight is that for an exponential family:
    ∂ρ/∂θ_a = ρ (F_a - ⟨F_a⟩ I)
where ⟨F_a⟩ = Tr(ρ F_a).

Then:
    ∂h_i/∂θ_a = -Tr((∂ρ_i/∂θ_a) (log ρ_i + I))
              = -Tr((∂ρ_i/∂θ_a) log ρ_i) - Tr(∂ρ_i/∂θ_a)
              = -Tr((∂ρ_i/∂θ_a) log ρ_i)  (since Tr(∂ρ_i/∂θ_a) = 0)

where ∂ρ_i/∂θ_a is the partial trace of ∂ρ/∂θ_a over all subsystems except i.
"""

import numpy as np
from scipy.linalg import expm, logm
import pytest

from qig.core import partial_trace, von_neumann_entropy, marginal_entropies
from qig.exponential_family import QuantumExponentialFamily


def analytic_marginal_entropy_gradient(
    exp_family: QuantumExponentialFamily,
    theta: np.ndarray
) -> tuple[float, np.ndarray]:
    """
    Compute marginal entropy constraint and its gradient analytically.
    
    Parameters
    ----------
    exp_family : QuantumExponentialFamily
        The exponential family
    theta : ndarray
        Natural parameters
    
    Returns
    -------
    C : float
        Constraint value ∑_i h_i
    grad_C : ndarray
        Gradient ∂C/∂θ
    """
    # Compute ρ(θ)
    rho = exp_family.rho_from_theta(theta)
    
    # Compute marginal entropies
    h = marginal_entropies(rho, exp_family.dims)
    C = float(np.sum(h))
    
    # Compute gradient
    n_params = exp_family.n_params
    n_sites = exp_family.n_sites
    grad_C = np.zeros(n_params)
    
    # For each parameter θ_a
    for a in range(n_params):
        F_a = exp_family.operators[a]
        
        # Compute ⟨F_a⟩ = Tr(ρ F_a)
        mean_Fa = np.trace(rho @ F_a).real
        
        # Compute ∂ρ/∂θ_a = ρ (F_a - ⟨F_a⟩ I)
        I = np.eye(exp_family.D, dtype=complex)
        drho_dtheta_a = rho @ (F_a - mean_Fa * I)
        
        # For each subsystem i
        for i in range(n_sites):
            # Compute marginal ρ_i
            rho_i = partial_trace(rho, exp_family.dims, keep=i)
            
            # Compute ∂ρ_i/∂θ_a (partial trace of ∂ρ/∂θ_a)
            drho_i_dtheta_a = partial_trace(drho_dtheta_a, exp_family.dims, keep=i)
            
            # Compute log(ρ_i) with regularisation
            eigvals_i, eigvecs_i = np.linalg.eigh(rho_i)
            eigvals_i = np.maximum(eigvals_i, 1e-14)
            log_eigvals_i = np.log(eigvals_i)
            log_rho_i = eigvecs_i @ np.diag(log_eigvals_i) @ eigvecs_i.conj().T
            
            # Compute ∂h_i/∂θ_a = -Tr((∂ρ_i/∂θ_a) log(ρ_i))
            dh_i_dtheta_a = -np.trace(drho_i_dtheta_a @ log_rho_i).real
            
            grad_C[a] += dh_i_dtheta_a
    
    return C, grad_C


class TestMarginalEntropyGradient:
    """Test analytic vs finite-difference gradient."""
    
    @pytest.mark.parametrize("n_sites,d", [
        (2, 2),  # Two qubits
        (2, 3),  # Two qutrits
        (3, 2),  # Three qubits
    ])
    def test_analytic_vs_finite_difference(self, n_sites, d):
        """Test that analytic gradient matches finite differences."""
        exp_family = QuantumExponentialFamily(n_sites, d)
        
        # Test at multiple parameter points
        np.random.seed(42)
        for trial in range(3):
            theta = np.random.randn(exp_family.n_params) * 0.3
            
            # Analytic gradient
            C_analytic, grad_analytic = analytic_marginal_entropy_gradient(
                exp_family, theta
            )
            
            # Finite-difference gradient
            C_fd, grad_fd = exp_family.marginal_entropy_constraint(theta)
            
            # Compare constraint values
            assert np.abs(C_analytic - C_fd) < 1e-10, (
                f"Constraint values differ: {C_analytic} vs {C_fd}"
            )
            
            # Compare gradients
            diff = grad_analytic - grad_fd
            max_abs_err = np.max(np.abs(diff))
            rel_err = max_abs_err / (np.max(np.abs(grad_fd)) + 1e-10)
            
            if trial == 0:
                print(f"\n{n_sites} sites, d={d} (trial {trial}):")
                print(f"Analytic gradient: {grad_analytic[:3]}...")
                print(f"Finite-diff gradient: {grad_fd[:3]}...")
                print(f"Max absolute error: {max_abs_err:.6e}")
                print(f"Relative error: {rel_err:.6e}")
            
            assert rel_err < 1e-4, (
                f"Trial {trial}: Gradients don't match\n"
                f"Max abs error: {max_abs_err:.3e}\n"
                f"Relative error: {rel_err:.3e}\n"
                f"Analytic: {grad_analytic}\n"
                f"Finite-diff: {grad_fd}"
            )
    
    def test_gradient_via_chain_rule(self):
        """
        Alternative test: verify gradient using chain rule.
        
        We can also compute the gradient as:
            ∂C/∂θ_a = ∑_i ∂h_i/∂θ_a
        
        where we use the fact that h_i is a function of ρ_i, and ρ_i
        is a function of ρ, and ρ is a function of θ.
        """
        exp_family = QuantumExponentialFamily(n_sites=2, d=2)
        theta = np.array([0.1, 0.2, -0.1, 0.3, -0.2, 0.15])
        
        C_analytic, grad_analytic = analytic_marginal_entropy_gradient(
            exp_family, theta
        )
        
        # Verify using numerical differentiation of C(θ)
        eps = 1e-6
        grad_numerical = np.zeros(exp_family.n_params)
        
        for a in range(exp_family.n_params):
            theta_plus = theta.copy()
            theta_plus[a] += eps
            
            rho_plus = exp_family.rho_from_theta(theta_plus)
            h_plus = marginal_entropies(rho_plus, exp_family.dims)
            C_plus = float(np.sum(h_plus))
            
            theta_minus = theta.copy()
            theta_minus[a] -= eps
            
            rho_minus = exp_family.rho_from_theta(theta_minus)
            h_minus = marginal_entropies(rho_minus, exp_family.dims)
            C_minus = float(np.sum(h_minus))
            
            grad_numerical[a] = (C_plus - C_minus) / (2 * eps)
        
        diff = grad_analytic - grad_numerical
        max_abs_err = np.max(np.abs(diff))
        rel_err = max_abs_err / (np.max(np.abs(grad_numerical)) + 1e-10)
        
        print(f"\nChain rule test:")
        print(f"Analytic: {grad_analytic}")
        print(f"Numerical: {grad_numerical}")
        print(f"Max absolute error: {max_abs_err:.6e}")
        print(f"Relative error: {rel_err:.6e}")
        
        assert rel_err < 1e-5, f"Chain rule gradient doesn't match: rel_err={rel_err:.3e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

