"""
Constrained dynamics and GENERIC-like structure for the quantum inaccessible game.

This module provides the `InaccessibleGameDynamics` class, which implements
constrained steepest-entropy-ascent dynamics in natural-parameter space,
with support for different time parametrisations.
"""

from typing import Tuple, Dict, Any

import numpy as np
from scipy.integrate import solve_ivp

from qig.core import von_neumann_entropy, marginal_entropies
from qig.exponential_family import QuantumExponentialFamily


class InaccessibleGameDynamics:
    """
    Constrained maximum entropy production dynamics.

    Implements: θ̇ = -Π_∥(θ) G(θ) θ
    where Π_∥ projects onto constraint manifold ∑_i h_i = C.
    """

    def __init__(self, exp_family: QuantumExponentialFamily):
        """
        Initialise dynamics for given exponential family.
        """
        self.exp_family = exp_family
        self.constraint_value: float | None = None

        # Time parametrisation: 'affine', 'entropy', or 'real'
        self.time_mode = "affine"

    def set_time_mode(self, mode: str) -> None:
        """
        Set time parametrisation mode.

        Parameters
        ----------
        mode : str
            'affine' : standard ODE time τ
            'entropy' : entropy time t where dH/dt = 1
            'real' : physical time (reserved for unitary part)
        """
        assert mode in ["affine", "entropy", "real"]
        self.time_mode = mode
        print(f"Time mode set to: {mode}")

    def flow(self, t: float, theta: np.ndarray) -> np.ndarray:
        """
        Compute θ̇ = -Π_∥ G θ at given θ.
        """
        # Compute Fisher information G(θ)
        G = self.exp_family.fisher_information(theta)

        # Compute constraint gradient a(θ) = ∇(∑ h_i)
        _, a = self.exp_family.marginal_entropy_constraint(theta)

        # Projection matrix Π_∥ = I - aa^T / ||a||²
        a_norm_sq = float(np.dot(a, a))
        if a_norm_sq < 1e-12:
            # Near endpoint, no constraint
            Pi = np.eye(len(theta))
        else:
            Pi = np.eye(len(theta)) - np.outer(a, a) / a_norm_sq

        # Unconstrained gradient: -G θ
        grad_H = -G @ theta

        # Project onto constraint manifold
        theta_dot = Pi @ grad_H

        # Time reparametrisation for entropy time
        if self.time_mode == "entropy":
            # dH/dτ = θ^T G Π_∥ G θ
            entropy_production = float(theta @ G @ Pi @ G @ theta)
            if entropy_production > 1e-12:
                theta_dot = theta_dot / entropy_production
            # Now dH/dt = 1 by construction

        return theta_dot

    def integrate(
        self, theta_0: np.ndarray, t_span: Tuple[float, float], n_points: int = 100
    ) -> Dict[str, Any]:
        """
        Integrate constrained dynamics from initial condition.
        """
        # Store initial constraint value
        rho_0 = self.exp_family.rho_from_theta(theta_0)
        h_0 = marginal_entropies(rho_0, self.exp_family.dims)
        self.constraint_value = float(np.sum(h_0))

        print(f"Initial constraint C = {self.constraint_value:.6f}")

        # Solve ODE
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(
            self.flow,
            t_span,
            theta_0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )

        if not sol.success:
            print(f"Warning: Integration failed: {sol.message}")

        # Extract trajectories
        time = sol.t
        theta_traj = sol.y.T

        # Compute entropies along trajectory
        H_traj = np.zeros(len(time))
        h_traj = np.zeros((len(time), self.exp_family.n_sites))
        constraint_traj = np.zeros(len(time))

        for i, theta in enumerate(theta_traj):
            rho = self.exp_family.rho_from_theta(theta)
            H_traj[i] = von_neumann_entropy(rho)
            h_traj[i] = marginal_entropies(rho, self.exp_family.dims)
            constraint_traj[i] = np.sum(h_traj[i])

        return {
            "time": time,
            "theta": theta_traj,
            "H": H_traj,
            "h": h_traj,
            "constraint": constraint_traj,
            "success": sol.success,
        }


__all__ = ["InaccessibleGameDynamics"]



