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
    
    Primary API
    -----------
    solve(theta_0, n_steps, dt, ...) : Recommended method for integrating dynamics.
        Uses gradient descent with Newton projection onto constraint manifold.
        Supports entropy time via `entropy_time=True`.
    
    Advanced/Internal
    -----------------
    flow(t, theta) : Compute θ̇ at a single point. For custom integrators.
    
    Example
    -------
    >>> from qig.exponential_family import QuantumExponentialFamily
    >>> from qig.dynamics import InaccessibleGameDynamics
    >>> 
    >>> exp_family = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
    >>> theta_0 = exp_family.get_bell_state_parameters(epsilon=0.01)
    >>> 
    >>> dynamics = InaccessibleGameDynamics(exp_family)
    >>> result = dynamics.solve(theta_0, n_steps=100, dt=0.01)
    >>> 
    >>> print(f"Converged: {result['converged']}")
    >>> print(f"Constraint preserved: dC = {result['constraint_values'][-1] - result['C_init']:.2e}")
    """

    def __init__(self, exp_family: QuantumExponentialFamily, method: str = 'duhamel'):
        """
        Initialise dynamics for given exponential family.
        
        Parameters
        ----------
        exp_family : QuantumExponentialFamily
            The exponential family to integrate dynamics for
        method : str, optional
            Method for computing ∂ρ/∂θ: 'duhamel' (accurate, slow) or 'sld' (fast, ~5% error)
            Default: 'duhamel'
        """
        self.exp_family = exp_family
        self.constraint_value: float | None = None
        self.method = method

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
        Compute θ̇ = -Π_∥ G θ at given θ (advanced/internal use).
        
        This is a low-level method that computes the flow vector at a single point.
        For most use cases, prefer `solve()` which handles integration and 
        constraint projection automatically.
        
        Parameters
        ----------
        t : float
            Time (used for time-dependent problems; ignored for autonomous flow)
        theta : ndarray
            Natural parameters at which to compute the flow
            
        Returns
        -------
        theta_dot : ndarray
            The constrained flow vector θ̇ = -Π_∥ G θ
            
        Notes
        -----
        If `time_mode='entropy'`, the flow is scaled so dH/dt = 1.
        """
        # Compute Fisher information G(θ)
        G = self.exp_family.fisher_information(theta)

        # Compute constraint gradient a(θ) = ∇(∑ h_i)
        _, a = self.exp_family.marginal_entropy_constraint(theta, method=self.method)

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

    def solve(self, theta_0: np.ndarray, n_steps: int = 1000, dt: float = 0.001,
              convergence_tol: float = 1e-6, entropy_time: bool = False,
              project: bool = True, project_every: int = 10,
              verbose: bool = True) -> Dict[str, Any]:
        """
        Solve constrained maximum entropy dynamics (recommended method).
        
        Uses gradient descent with Newton projection onto constraint manifold.
        This is the primary API for integrating the inaccessible game dynamics.
        
        Parameters
        ----------
        theta_0 : ndarray
            Initial natural parameters
        n_steps : int, default=1000
            Maximum number of steps
        dt : float, default=0.001
            Step size (or entropy increment if entropy_time=True)
        convergence_tol : float, default=1e-6
            Stop when ||F|| < convergence_tol
        entropy_time : bool, default=False
            If True, scale steps so dH/dt = 1 (entropy increases by dt per step)
        project : bool, default=True
            Project onto constraint manifold (recommended)
        project_every : int, default=10
            Project every N steps (for efficiency)
        verbose : bool, default=True
            Print progress messages
            
        Returns
        -------
        dict with keys:
            trajectory : ndarray, shape (n_steps+1, n_params)
                Parameter trajectory θ(t)
            flow_norms : ndarray
                ||F(θ)|| at each step (→ 0 at convergence)
            constraint_values : ndarray
                Σᵢ hᵢ at each step (should stay ≈ constant)
            C_init : float
                Initial constraint value
            converged : bool
                Whether convergence was achieved
            n_steps : int
                Number of steps taken
                
        Example
        -------
        >>> result = dynamics.solve(theta_0, n_steps=500, dt=0.01, entropy_time=True)
        >>> print(f"Converged: {result['converged']}")
        """
        return self.solve_constrained_maxent(
            theta_0, n_steps=n_steps, dt=dt, convergence_tol=convergence_tol,
            project=project, project_every=project_every, 
            use_entropy_time=entropy_time
        )
    
    def integrate(
        self, theta_0: np.ndarray, t_span: Tuple[float, float], n_points: int = 100
    ) -> Dict[str, Any]:
        """
        [DEPRECATED] Integrate using scipy solve_ivp.
        
        .. deprecated::
            This method does not project onto the constraint manifold,
            causing constraint drift. Use `solve()` instead.
            
        Warning
        -------
        This method is numerically unstable for constrained problems.
        The constraint C = Σᵢ hᵢ will drift over time, giving incorrect results.
        Use `solve()` which includes Newton projection for stability.
        """
        import warnings
        warnings.warn(
            "integrate() is deprecated and numerically unstable for constrained problems. "
            "Use solve() instead, which includes Newton projection onto the constraint manifold.",
            DeprecationWarning,
            stacklevel=2
        )
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

    def solve_constrained_maxent(self, theta_init: np.ndarray, n_steps: int = 1000,
                                dt: float = 0.001, convergence_tol: float = 1e-6,
                                project: bool = True, project_every: int = 10,
                                use_entropy_time: bool = False) -> Dict[str, Any]:
        """
        Solve constrained maximum entropy dynamics using gradient descent with projection.

        - Uses gradient descent to maximise joint entropy
        - Projects onto constraint manifold at each step
        - Much more stable than ODE integration for constrained optimisation

        Dynamics: dθ/dt = F(θ) = -Π_∥(θ) G(θ) θ
        where Π_∥ projects onto Σᵢ hᵢ(θ) = C

        Parameters
        ----------
        theta_init : array
            Initial parameter vector
        n_steps : int
            Maximum number of gradient steps
        dt : float
            Step size for gradient descent
        convergence_tol : float
            Stop when ||F|| < convergence_tol
        project : bool
            Whether to project onto constraint manifold
        project_every : int
            Project onto constraint manifold every N steps (for efficiency)
        use_entropy_time : bool
            If True, scale step size so that entropy increases by dt per step

        Returns
        -------
        dict with keys:
            trajectory : array, shape (n_steps, n_params)
                Parameter trajectory θ(t)
            flow_norms : array
                ||F(θ)|| at each step
            constraint_values : array
                Σᵢ hᵢ at each step (should be ≈ constant)
            converged : bool
                Whether convergence criterion was met
            C_init : float
                Initial constraint value
        """
        n_params = len(theta_init)
        trajectory = [theta_init.copy()]
        flow_norms = []
        constraint_values = []
        theta = theta_init.copy()

        # Compute initial constraint value using marginal entropies
        rho_init = self.exp_family.rho_from_theta(theta)
        h_init = marginal_entropies(rho_init, self.exp_family.dims)
        C_init = float(np.sum(h_init))

        for step in range(n_steps):
            # Compute constrained flow at current point
            G = self.exp_family.fisher_information(theta)
            _, a = self.exp_family.marginal_entropy_constraint(theta)

            # Unconstrained flow: maximize entropy
            F_unc = -G @ theta

            # Lagrange multiplier for constraint tangency
            # a^T F = 0 => a^T(F_unc - ν a) = 0 => ν = (a^T F_unc)/(a^T a)
            a_norm_sq = np.dot(a, a)
            if a_norm_sq > 1e-12:
                nu = np.dot(F_unc, a) / a_norm_sq
                F = F_unc - nu * a  # Constrained flow
                # Debug: check tangency
                tangency_check = np.dot(a, F)
                if step < 5 or step % 1000 == 0:
                    print(f"Step {step}: ν = {nu:.6e}, tangency = {tangency_check:.6e}")
            else:
                F = F_unc

            # Gradient descent step
            if use_entropy_time:
                # Scale step size for entropy time: dH/dt = 1
                # Entropy production = θ^T G Π_∥ G θ (same as in flow method)
                a_norm_sq = np.dot(a, a)
                if a_norm_sq > 1e-12:
                    Pi = np.eye(len(theta)) - np.outer(a, a) / a_norm_sq
                else:
                    Pi = np.eye(len(theta))
                entropy_production = float(theta @ G @ Pi @ G @ theta)
                if entropy_production > 1e-12:
                    effective_dt = dt / entropy_production
                else:
                    effective_dt = dt
            else:
                effective_dt = dt

            theta_new = theta + effective_dt * F

            # Project back onto constraint manifold (every project_every steps for efficiency)
            if project and (step + 1) % project_every == 0:
                # Simple projection: adjust along constraint gradient direction
                rho_new = self.exp_family.rho_from_theta(theta_new)
                h_new = marginal_entropies(rho_new, self.exp_family.dims)
                C_new = np.sum(h_new)

                # Iterative Newton projection to constraint manifold
                theta_proj = theta_new.copy()
                for proj_iter in range(10):  # Allow up to 10 iterations
                    rho_proj = self.exp_family.rho_from_theta(theta_proj)
                    h_proj = marginal_entropies(rho_proj, self.exp_family.dims)
                    C_proj = np.sum(h_proj)
                    error = C_proj - C_init

                    if abs(error) < 1e-12:  # Tight convergence
                        break

                    # Newton step: θ ← θ - error / ||a||² * a
                    _, a_proj = self.exp_family.marginal_entropy_constraint(theta_proj)
                    a_norm_sq = np.dot(a_proj, a_proj)
                    if a_norm_sq > 1e-15:
                        step_size = error / a_norm_sq
                        theta_proj = theta_proj - step_size * a_proj
                    else:
                        break

                theta_new = theta_proj

            theta = theta_new

            # Track metrics
            flow_norm = np.linalg.norm(F)
            flow_norms.append(flow_norm)
            trajectory.append(theta.copy())

            # Debug: print progress every 1000 steps
            if (step + 1) % 1000 == 0:
                H_current = self.exp_family.von_neumann_entropy(theta)
                print(f"Step {step+1}: ||F|| = {flow_norm:.6e}, H = {H_current:.6f}, ||θ|| = {np.linalg.norm(theta):.6f}")

            # Check constraint preservation
            rho = self.exp_family.rho_from_theta(theta)
            h = marginal_entropies(rho, self.exp_family.dims)
            C_current = np.sum(h)
            constraint_values.append(C_current)

            # Check convergence - stop early if we achieve target accuracy
            if flow_norm < convergence_tol:
                traj = np.array(trajectory)
                return {
                    'trajectory': traj,
                    'theta': traj,  # Alias for backward compatibility
                    'flow_norms': np.array(flow_norms),
                    'constraint_values': np.array(constraint_values),
                    'C_init': C_init,
                    'converged': True,
                    'n_steps': step + 1
                }

            # Also check if we've achieved very tight convergence at any point
            # This handles cases where the algorithm oscillates after convergence
            if len(flow_norms) > 10:
                recent_min = np.min(flow_norms[-20:])  # Check last 20 steps
                if recent_min < convergence_tol:
                    traj = np.array(trajectory)
                    return {
                        'trajectory': traj,
                        'theta': traj,  # Alias for backward compatibility
                        'flow_norms': np.array(flow_norms),
                        'constraint_values': np.array(constraint_values),
                        'C_init': C_init,
                        'converged': True,
                        'n_steps': step + 1
                    }

        # Check for sustained convergence (last 5 steps all below tolerance)
        if len(flow_norms) >= 5:
            last_5_flow_norms = flow_norms[-5:]
            if np.all(np.array(last_5_flow_norms) < convergence_tol):
                traj = np.array(trajectory)
                return {
                    'trajectory': traj,
                    'theta': traj,  # Alias for backward compatibility
                    'flow_norms': np.array(flow_norms),
                    'constraint_values': np.array(constraint_values),
                    'C_init': C_init,
                    'converged': True,
                    'n_steps': n_steps
                }

        traj = np.array(trajectory)
        return {
            'trajectory': traj,
            'theta': traj,  # Alias for backward compatibility
            'flow_norms': np.array(flow_norms),
            'constraint_values': np.array(constraint_values),
            'C_init': C_init,
            'converged': False,
            'n_steps': n_steps
        }

class GenericDynamics(InaccessibleGameDynamics):
    """
    GENERIC-aware dynamics for the quantum inaccessible game.
    
    Extends InaccessibleGameDynamics to track the GENERIC decomposition:
    - Effective Hamiltonian H_eff(θ) from antisymmetric flow
    - Diffusion operator D[ρ](θ) from symmetric flow
    - Entropy production rate
    - GENERIC structure preservation
    """
    
    def __init__(self, exp_family: QuantumExponentialFamily, 
                 structure_constants: np.ndarray = None,
                 method: str = 'duhamel'):
        """
        Initialize GENERIC-aware dynamics.
        
        Parameters
        ----------
        exp_family : QuantumExponentialFamily
            The exponential family
        structure_constants : np.ndarray, optional
            Lie algebra structure constants f_abc
            If None, will be computed from operators
        method : str, optional
            Method for derivatives: 'duhamel' or 'sld'
        """
        super().__init__(exp_family, method)
        
        # Store structure constants
        if structure_constants is None:
            from qig.structure_constants import compute_structure_constants
            self.f_abc = compute_structure_constants(exp_family.operators)
        else:
            self.f_abc = structure_constants
            
        # Storage for GENERIC decomposition along trajectory
        self.H_eff_traj = []
        self.D_rho_traj = []
        self.entropy_production_traj = []
        self.S_norm_traj = []
        self.A_norm_traj = []
        
    def compute_generic_decomposition(self, theta: np.ndarray) -> Dict[str, Any]:
        """
        Compute full GENERIC decomposition at given θ.
        
        Returns
        -------
        dict with keys:
            'M' : Jacobian matrix
            'S' : Symmetric part
            'A' : Antisymmetric part
            'H_eff' : Effective Hamiltonian
            'eta' : Hamiltonian coefficients
            'D_rho' : Diffusion operator (if compute_diffusion=True)
            'entropy_production' : dS/dt
        """
        from qig.generic import (
            effective_hamiltonian_coefficients,
            effective_hamiltonian_operator
        )
        
        # Get Jacobian and decomposition
        M = self.exp_family.jacobian(theta, method=self.method)
        S = self.exp_family.symmetric_part(theta, method=self.method)
        A = self.exp_family.antisymmetric_part(theta, method=self.method)
        
        # Extract effective Hamiltonian
        eta, info = effective_hamiltonian_coefficients(A, theta, self.f_abc)
        H_eff = effective_hamiltonian_operator(eta, self.exp_family.operators)
        
        # Compute entropy production rate
        # dS/dt = -Tr(ρ̇ log ρ) where ρ̇ comes from dissipative part
        # For now, use simplified form: θ^T G Π_∥ G θ
        G = self.exp_family.fisher_information(theta)
        _, a = self.exp_family.marginal_entropy_constraint(theta, method=self.method)
        a_norm_sq = float(np.dot(a, a))
        if a_norm_sq > 1e-12:
            Pi = np.eye(len(theta)) - np.outer(a, a) / a_norm_sq
        else:
            Pi = np.eye(len(theta))
        entropy_production = float(theta @ G @ Pi @ G @ theta)
        
        return {
            'M': M,
            'S': S,
            'A': A,
            'H_eff': H_eff,
            'eta': eta,
            'entropy_production': entropy_production,
            'S_norm': np.linalg.norm(S, 'fro'),
            'A_norm': np.linalg.norm(A, 'fro'),
        }
    
    def integrate_reversible(
        self, theta_0: np.ndarray, t_span: Tuple[float, float], n_points: int = 100
    ) -> Dict[str, Any]:
        """
        Integrate reversible (Hamiltonian) part only: -i[H_eff, ρ].
        
        This integrates the antisymmetric flow in parameter space,
        which corresponds to unitary evolution in density matrix space.
        """
        # Use only the antisymmetric part of the Jacobian
        def reversible_flow(t: float, theta: np.ndarray) -> np.ndarray:
            A = self.exp_family.antisymmetric_part(theta, method=self.method)
            # For reversible flow: θ̇ = A θ (linearized)
            # But we need the full nonlinear flow, so use constraint structure
            G = self.exp_family.fisher_information(theta)
            _, a = self.exp_family.marginal_entropy_constraint(theta, method=self.method)
            
            # Get Lagrange multiplier gradient
            grad_nu = self.exp_family.lagrange_multiplier_gradient(
                theta, method=self.method
            )
            
            # Antisymmetric contribution: a(∇ν)^T - (∇ν)a^T
            # Applied to current point
            reversible_term = np.outer(a, grad_nu) - np.outer(grad_nu, a)
            return reversible_term @ theta
            
        # Integrate
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(
            reversible_flow,
            t_span,
            theta_0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )
        
        return {
            "time": sol.t,
            "theta": sol.y.T,
            "success": sol.success,
        }
    
    def integrate_irreversible(
        self, theta_0: np.ndarray, t_span: Tuple[float, float], n_points: int = 100
    ) -> Dict[str, Any]:
        """
        Integrate irreversible (dissipative) part only: D[ρ].
        
        This integrates the symmetric flow in parameter space,
        which corresponds to entropy-increasing dissipation.
        """
        # Use only the symmetric part
        def irreversible_flow(t: float, theta: np.ndarray) -> np.ndarray:
            G = self.exp_family.fisher_information(theta)
            third_cumulant = self.exp_family.third_cumulant_contraction(theta)
            hessian_C = self.exp_family.constraint_hessian(theta)
            _, a = self.exp_family.marginal_entropy_constraint(theta, method=self.method)
            
            # Symmetric part: -G - (∇G)[θ] + ν∇²C
            # (drops the a(∇ν)^T term which is antisymmetric)
            Gtheta = G @ theta
            
            # Lagrange multiplier
            a_norm_sq = float(np.dot(a, a))
            if a_norm_sq > 1e-12:
                nu = -np.dot(Gtheta, a) / a_norm_sq
            else:
                nu = 0.0
                
            # Symmetric flow
            return -G @ theta - third_cumulant @ theta + nu * (hessian_C @ theta)
            
        # Integrate
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(
            irreversible_flow,
            t_span,
            theta_0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )
        
        return {
            "time": sol.t,
            "theta": sol.y.T,
            "success": sol.success,
        }
    
    def integrate_with_monitoring(
        self, theta_0: np.ndarray, t_span: Tuple[float, float], n_points: int = 100,
        compute_diffusion: bool = False
    ) -> Dict[str, Any]:
        """
        Integrate full dynamics with GENERIC structure monitoring.
        
        Parameters
        ----------
        theta_0 : np.ndarray
            Initial parameters
        t_span : Tuple[float, float]
            Time interval
        n_points : int
            Number of evaluation points
        compute_diffusion : bool
            Whether to compute D[ρ] at each step (expensive!)
            
        Returns
        -------
        dict with trajectory and GENERIC monitoring data
        """
        # First integrate the full dynamics using solve() (stable)
        # Convert t_span to n_steps and dt
        # Note: solve() returns n_steps+1 points (including initial), 
        # so use n_points-1 to match integrate()'s behavior
        t_total = t_span[1] - t_span[0]
        dt = t_total / (n_points - 1) if n_points > 1 else t_total
        result = self.solve(theta_0, n_steps=n_points-1, dt=dt, verbose=False)
        
        # Add compatibility keys
        result['theta'] = result['trajectory']  # Alias for backward compatibility
        result['time'] = np.linspace(t_span[0], t_span[1], len(result['trajectory']))
        # Note: 'converged' means reached fixed point; 'success' means integration completed
        # For compatibility with old integrate() which used scipy's success flag
        result['success'] = True  # solve() always completes (unlike scipy which can fail)
        
        # Compute entropies along trajectory (for compatibility with integrate())
        theta_traj = result['trajectory']
        n_steps_actual = len(theta_traj)
        H_traj = np.zeros(n_steps_actual)
        h_traj = np.zeros((n_steps_actual, self.exp_family.n_sites))
        
        for i, theta in enumerate(theta_traj):
            rho = self.exp_family.rho_from_theta(theta)
            H_traj[i] = von_neumann_entropy(rho)
            h_traj[i] = marginal_entropies(rho, self.exp_family.dims)
        
        result['H'] = H_traj
        result['h'] = h_traj
        result['constraint'] = np.sum(h_traj, axis=1)
        
        # Compute GENERIC decomposition at each point
        theta_traj = result['theta']
        n_steps = len(theta_traj)
        
        # Storage
        H_eff_traj = []
        entropy_production = np.zeros(n_steps)
        S_norms = np.zeros(n_steps)
        A_norms = np.zeros(n_steps)
        D_rho_traj = [] if compute_diffusion else None
        
        print("Computing GENERIC decomposition along trajectory...")
        for i, theta in enumerate(theta_traj):
            if n_steps >= 10 and i % (n_steps // 10) == 0:
                print(f"  Progress: {i}/{n_steps}")
            elif n_steps < 10 and i == 0:
                print(f"  Progress: {i}/{n_steps}")
                
            decomp = self.compute_generic_decomposition(theta)
            H_eff_traj.append(decomp['H_eff'])
            entropy_production[i] = decomp['entropy_production']
            S_norms[i] = decomp['S_norm']
            A_norms[i] = decomp['A_norm']
            
            if compute_diffusion:
                from qig.generic import diffusion_operator
                D_rho = diffusion_operator(
                    decomp['S'], theta, self.exp_family, method=self.method
                )
                D_rho_traj.append(D_rho)
        
        # Add GENERIC data to results
        result['H_eff'] = H_eff_traj
        result['entropy_production'] = entropy_production
        result['S_norm'] = S_norms
        result['A_norm'] = A_norms
        result['cumulative_entropy'] = np.cumsum(entropy_production) * np.mean(np.diff(result['time']))
        
        if compute_diffusion:
            result['D_rho'] = D_rho_traj
            
        return result


__all__ = ["InaccessibleGameDynamics", "GenericDynamics"]



