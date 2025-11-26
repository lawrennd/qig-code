"""
Finite-difference helpers for numerical validation tests (CIP-0004).

These utilities centralise common finite-difference patterns used to validate
analytical implementations (e.g. Fisher/BKM metric via ψ(θ) = log Z(θ)).
"""

from __future__ import annotations

import numpy as np


def finite_difference_rho_derivative(exp_family, theta: np.ndarray, a: int, eps: float = 1e-6) -> np.ndarray:
    """
    Compute ∂ρ/∂θ_a using central finite differences.

    Parameters
    ----------
    exp_family :
        Exponential family object with method rho_from_theta(theta).
    theta : ndarray, shape (n_params,)
        Natural parameters.
    a : int
        Parameter index.
    eps : float
        Step size for finite differences.
    """
    theta_plus = theta.copy()
    theta_plus[a] += eps
    theta_minus = theta.copy()
    theta_minus[a] -= eps

    rho_plus = exp_family.rho_from_theta(theta_plus)
    rho_minus = exp_family.rho_from_theta(theta_minus)

    return (rho_plus - rho_minus) / (2 * eps)


def finite_difference_fisher(exp_family, theta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute Fisher metric using finite differences of ψ(θ) = log Z(θ).

    This implements a 4-point finite-difference stencil for the Hessian:
        G_{ab}(θ) ≈ ∂²ψ / ∂θ_a ∂θ_b
    where ψ is provided by exp_family.psi().

    Parameters
    ----------
    exp_family :
        Exponential family object with attributes:
        - n_params
        - psi(theta)
    theta : ndarray, shape (n_params,)
        Point in natural-parameter space at which to compute the Hessian.
    eps : float
        Step size for finite differences.

    Returns
    -------
    G_fd : ndarray, shape (n_params, n_params)
        Symmetric finite-difference approximation to the Fisher/BKM metric.
    """
    n = exp_family.n_params
    G_fd = np.zeros((n, n))

    for a in range(n):
        for b in range(a, n):
            theta_pp = theta.copy()
            theta_pp[a] += eps
            theta_pp[b] += eps

            theta_pm = theta.copy()
            theta_pm[a] += eps
            theta_pm[b] -= eps

            theta_mp = theta.copy()
            theta_mp[a] -= eps
            theta_mp[b] += eps

            theta_mm = theta.copy()
            theta_mm[a] -= eps
            theta_mm[b] -= eps

            psi_pp = exp_family.psi(theta_pp)
            psi_pm = exp_family.psi(theta_pm)
            psi_mp = exp_family.psi(theta_mp)
            psi_mm = exp_family.psi(theta_mm)

            d2psi_dadb = (psi_pp - psi_pm - psi_mp + psi_mm) / (4 * eps * eps)

            G_fd[a, b] = d2psi_dadb
            G_fd[b, a] = d2psi_dadb

    return G_fd


def finite_difference_constraint_gradient(exp_family, theta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute ∇C(θ) using finite differences of the marginal entropy constraint.

    Parameters
    ----------
    exp_family :
        Exponential family with marginal_entropy_constraint(theta) -> (C, a).
    theta : ndarray
        Natural parameters.
    eps : float
        Step size for finite differences.
    """
    _, _ = exp_family.marginal_entropy_constraint(theta)
    grad_fd = np.zeros(exp_family.n_params)

    for i in range(exp_family.n_params):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        theta_minus = theta.copy()
        theta_minus[i] -= eps

        C_plus, _ = exp_family.marginal_entropy_constraint(theta_plus)
        C_minus, _ = exp_family.marginal_entropy_constraint(theta_minus)

        grad_fd[i] = (C_plus - C_minus) / (2 * eps)

    return grad_fd


def finite_difference_constraint_hessian(exp_family, theta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute ∇²C(θ) using finite differences of the constraint gradient.

    Parameters
    ----------
    exp_family :
        Exponential family with marginal_entropy_constraint.
    theta : ndarray
        Natural parameters.
    eps : float
        Step size for finite differences.
    """
    H_fd = np.zeros((exp_family.n_params, exp_family.n_params))

    for i in range(exp_family.n_params):
        for j in range(exp_family.n_params):
            theta_pp = theta.copy()
            theta_pp[i] += eps
            theta_pp[j] += eps

            theta_pm = theta.copy()
            theta_pm[i] += eps
            theta_pm[j] -= eps

            theta_mp = theta.copy()
            theta_mp[i] -= eps
            theta_mp[j] += eps

            theta_mm = theta.copy()
            theta_mm[i] -= eps
            theta_mm[j] -= eps

            grad_pp = finite_difference_constraint_gradient(exp_family, theta_pp, eps)
            grad_pm = finite_difference_constraint_gradient(exp_family, theta_pm, eps)
            grad_mp = finite_difference_constraint_gradient(exp_family, theta_mp, eps)
            grad_mm = finite_difference_constraint_gradient(exp_family, theta_mm, eps)

            H_fd[i, j] = (grad_pp[j] - grad_pm[j] - grad_mp[j] + grad_mm[j]) / (4 * eps * eps)

    return H_fd


def finite_difference_jacobian(exp_family, theta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute Jacobian M using finite differences of the flow F(θ).

    Here F(θ) = -Gθ + νa, with:
        G = fisher_information(θ)
        (C, a) = marginal_entropy_constraint(θ, method='duhamel')

    Parameters
    ----------
    exp_family :
        Exponential family with fisher_information and marginal_entropy_constraint.
    theta : ndarray
        Natural parameters.
    eps : float
        Step size for finite differences.
    """
    M_fd = np.zeros((exp_family.n_params, exp_family.n_params))

    def compute_F(theta_val: np.ndarray) -> np.ndarray:
        G = exp_family.fisher_information(theta_val)
        _, a = exp_family.marginal_entropy_constraint(theta_val, method="duhamel")
        Gtheta = G @ theta_val
        nu = np.dot(a, Gtheta) / np.dot(a, a)
        return -Gtheta + nu * a

    for j in range(exp_family.n_params):
        theta_plus = theta.copy()
        theta_plus[j] += eps
        theta_minus = theta.copy()
        theta_minus[j] -= eps

        F_plus = compute_F(theta_plus)
        F_minus = compute_F(theta_minus)

        M_fd[:, j] = (F_plus - F_minus) / (2 * eps)

    return M_fd


