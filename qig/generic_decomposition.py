"""
High-level interface for GENERIC decomposition.

This module provides a user-friendly orchestration layer that executes
the complete 12-step GENERIC decomposition procedure and provides
comprehensive results, diagnostics, and visualization.
"""

from typing import Dict, Any, Optional, List
import numpy as np

from qig.exponential_family import QuantumExponentialFamily
from qig.structure_constants import compute_structure_constants
from qig.generic import (
    effective_hamiltonian_coefficients,
    effective_hamiltonian_operator,
    diffusion_operator,
)
from qig.core import von_neumann_entropy, marginal_entropies
from qig.validation import ValidationReport


class GenericDecomposition:
    """
    Complete GENERIC decomposition with diagnostics.
    
    This class orchestrates the full 12-step procedure for computing
    the GENERIC decomposition of quantum inaccessible game dynamics:
    
    1. Initial state and density matrix
    2-8. Information geometry (ψ, μ, G, h_i, a, ν, ∇ν, M)
    9. Symmetric/antisymmetric decomposition (S, A)
    10. Effective Hamiltonian extraction (η, H_eff)
    11. Diffusion operator construction (D[ρ])
    12. Comprehensive diagnostics and validation
    
    Examples
    --------
    >>> exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
    >>> decomp = GenericDecomposition(exp_fam)
    >>> theta = np.random.randn(exp_fam.n_params) * 0.1
    >>> results = decomp.compute_all(theta)
    >>> decomp.print_summary()
    """
    
    def __init__(self, exp_fam: QuantumExponentialFamily,
                 method: str = 'duhamel',
                 compute_diffusion: bool = False):
        """
        Initialize GENERIC decomposition.
        
        Parameters
        ----------
        exp_fam : QuantumExponentialFamily
            The exponential family
        method : str, optional
            Method for derivatives: 'duhamel' (accurate) or 'sld' (fast)
        compute_diffusion : bool, optional
            Whether to compute diffusion operator (expensive!)
            Default: False
        """
        self.exp_fam = exp_fam
        self.method = method
        self.compute_diffusion = compute_diffusion
        self.results = {}
        self.diagnostics = {}
        
        # Precompute structure constants
        self.f_abc = compute_structure_constants(exp_fam.operators)
        
    def compute_all(self, theta: np.ndarray,
                   verbose: bool = False) -> Dict[str, Any]:
        """
        Execute complete 12-step GENERIC decomposition.
        
        Parameters
        ----------
        theta : np.ndarray
            Natural parameters
        verbose : bool
            Print progress messages
            
        Returns
        -------
        results : dict
            Complete results with all intermediate computations
        """
        if verbose:
            print("=" * 70)
            print("GENERIC Decomposition - 12-Step Procedure")
            print("=" * 70)
        
        # Step 0: Initial state
        if verbose:
            print("\nStep 0: Initial State")
        self.results['theta'] = theta
        self.results['rho'] = self.exp_fam.rho_from_theta(theta)
        if verbose:
            print(f"  θ: shape {theta.shape}, norm {np.linalg.norm(theta):.6f}")
        
        # Step 1: Cumulant generating function
        if verbose:
            print("\nStep 1: Cumulant Generating Function")
        self.results['psi'] = self.exp_fam.psi(theta)
        if verbose:
            print(f"  ψ(θ) = {self.results['psi']:.6f}")
        
        # Step 2: Mean parameters (expectation parameters)
        if verbose:
            print("\nStep 2: Mean Parameters")
        self.results['mu'] = self.exp_fam._grad_psi(theta)
        if verbose:
            print(f"  μ = ∇ψ: shape {self.results['mu'].shape}")
        
        # Step 3: Fisher information (BKM metric)
        if verbose:
            print("\nStep 3: Fisher Information Metric")
        self.results['G'] = self.exp_fam.fisher_information(theta)
        if verbose:
            cond = np.linalg.cond(self.results['G'])
            print(f"  G: shape {self.results['G'].shape}, cond={cond:.2e}")
        
        # Step 4: Marginal entropies
        if verbose:
            print("\nStep 4: Marginal Entropies")
        rho = self.results['rho']
        self.results['H_joint'] = von_neumann_entropy(rho)
        self.results['h'] = marginal_entropies(rho, self.exp_fam.dims)
        self.results['C'] = np.sum(self.results['h'])
        if verbose:
            print(f"  H(ρ) = {self.results['H_joint']:.6f}")
            print(f"  h_i = {self.results['h']}")
            print(f"  C = Σh_i = {self.results['C']:.6f}")
        
        # Step 5: Constraint gradient
        if verbose:
            print("\nStep 5: Constraint Gradient")
        _, a = self.exp_fam.marginal_entropy_constraint(theta, method=self.method)
        self.results['a'] = a
        self.results['a_norm'] = np.linalg.norm(a)
        if verbose:
            print(f"  a = ∇C: ||a|| = {self.results['a_norm']:.6f}")
        
        # Step 6: Lagrange multiplier
        if verbose:
            print("\nStep 6: Lagrange Multiplier")
        Gtheta = self.results['G'] @ theta
        a_norm_sq = np.dot(a, a)
        if a_norm_sq > 1e-12:
            nu = -np.dot(Gtheta, a) / a_norm_sq
        else:
            nu = 0.0
        self.results['nu'] = nu
        if verbose:
            print(f"  ν = {nu:.6f}")
        
        # Step 7: Lagrange multiplier gradient
        if verbose:
            print("\nStep 7: Lagrange Multiplier Gradient")
        self.results['grad_nu'] = self.exp_fam.lagrange_multiplier_gradient(
            theta, method=self.method
        )
        if verbose:
            print(f"  ∇ν: shape {self.results['grad_nu'].shape}")
        
        # Step 8: Flow Jacobian
        if verbose:
            print("\nStep 8: Flow Jacobian")
        self.results['M'] = self.exp_fam.jacobian(theta, method=self.method)
        if verbose:
            M_norm = np.linalg.norm(self.results['M'], 'fro')
            print(f"  M = ∂F/∂θ: ||M||_F = {M_norm:.6f}")
        
        # Step 9: Symmetric/Antisymmetric Decomposition
        if verbose:
            print("\nStep 9: GENERIC Decomposition M = S + A")
        self.results['S'] = self.exp_fam.symmetric_part(theta, method=self.method)
        self.results['A'] = self.exp_fam.antisymmetric_part(theta, method=self.method)
        if verbose:
            S_norm = np.linalg.norm(self.results['S'], 'fro')
            A_norm = np.linalg.norm(self.results['A'], 'fro')
            print(f"  S (dissipative): ||S||_F = {S_norm:.6f}")
            print(f"  A (reversible):  ||A||_F = {A_norm:.6f}")
        
        # Step 10: Effective Hamiltonian
        if verbose:
            print("\nStep 10: Effective Hamiltonian Extraction")
        eta, info = effective_hamiltonian_coefficients(
            self.results['A'], theta, self.f_abc
        )
        self.results['eta'] = eta
        self.results['eta_info'] = info
        self.results['H_eff'] = effective_hamiltonian_operator(
            eta, self.exp_fam.operators
        )
        if verbose:
            print(f"  η: shape {eta.shape}, ||η|| = {np.linalg.norm(eta):.6f}")
            H_norm = np.linalg.norm(self.results['H_eff'], 'fro')
            print(f"  H_eff: ||H_eff||_F = {H_norm:.6f}")
        
        # Step 11: Diffusion Operator (optional, expensive)
        if self.compute_diffusion:
            if verbose:
                print("\nStep 11: Diffusion Operator (computing...)")
            self.results['D_rho'] = diffusion_operator(
                self.results['S'], theta, self.exp_fam, method=self.method
            )
            if verbose:
                D_norm = np.linalg.norm(self.results['D_rho'], 'fro')
                print(f"  D[ρ]: ||D[ρ]||_F = {D_norm:.6f}")
        else:
            if verbose:
                print("\nStep 11: Diffusion Operator (skipped, expensive)")
            self.results['D_rho'] = None
        
        # Step 12: Diagnostics
        if verbose:
            print("\nStep 12: Diagnostics and Validation")
        self.diagnostics = self._compute_diagnostics()
        self.results['diagnostics'] = self.diagnostics
        
        if verbose:
            print("\n" + "=" * 70)
            print("GENERIC Decomposition Complete")
            print("=" * 70)
        
        return self.results
    
    def _compute_diagnostics(self) -> Dict[str, Any]:
        """
        Compute comprehensive diagnostics and validation.
        
        Returns
        -------
        diagnostics : dict
            Validation results and property checks
        """
        diag = {}
        
        # Check symmetry/antisymmetry
        S = self.results['S']
        A = self.results['A']
        M = self.results['M']
        
        diag['S_symmetry_error'] = np.linalg.norm(S - S.T, 'fro')
        diag['A_antisymmetry_error'] = np.linalg.norm(A + A.T, 'fro')
        diag['M_reconstruction_error'] = np.linalg.norm(M - (S + A), 'fro')
        
        # Check Hermiticity
        H_eff = self.results['H_eff']
        diag['H_eff_hermiticity_error'] = np.linalg.norm(
            H_eff - H_eff.conj().T, 'fro'
        )
        
        # Check tracelessness
        diag['H_eff_trace'] = abs(np.trace(H_eff))
        
        # Check degeneracy conditions
        theta = self.results['theta']
        a = self.results['a']
        G = self.results['G']
        
        Sa = S @ a
        A_Gtheta = A @ (-G @ theta)
        
        diag['degeneracy_S_condition'] = np.linalg.norm(Sa)
        diag['degeneracy_A_condition'] = np.linalg.norm(A_Gtheta)
        
        # Check constraint tangency
        F = -G @ theta + self.results['nu'] * a
        diag['constraint_tangency'] = abs(np.dot(a, F))
        
        # If diffusion operator computed, check its properties
        if self.results['D_rho'] is not None:
            D_rho = self.results['D_rho']
            diag['D_hermiticity_error'] = np.linalg.norm(
                D_rho - D_rho.conj().T, 'fro'
            )
            diag['D_trace'] = abs(np.trace(D_rho))
            
            # Entropy production: -Tr(D[ρ] log ρ)
            rho = self.results['rho']
            # Compute log(ρ) via eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(rho)
            log_rho_matrix = eigvecs @ np.diag(np.log(eigvals + 1e-15)) @ eigvecs.conj().T
            diag['entropy_production'] = -np.real(np.trace(D_rho @ log_rho_matrix))
        
        # Compute pass/fail for each check
        diag['checks'] = {
            'S_symmetric': diag['S_symmetry_error'] < 1e-10,
            'A_antisymmetric': diag['A_antisymmetry_error'] < 1e-10,
            'M_reconstructs': diag['M_reconstruction_error'] < 1e-10,
            'H_eff_hermitian': diag['H_eff_hermiticity_error'] < 1e-10,
            'H_eff_traceless': diag['H_eff_trace'] < 1e-10,
            'degeneracy_S': diag['degeneracy_S_condition'] < 1e-6,
            'degeneracy_A': diag['degeneracy_A_condition'] < 1e-6,
            'tangency': diag['constraint_tangency'] < 1e-8,
        }
        
        if self.results['D_rho'] is not None:
            diag['checks']['D_hermitian'] = diag['D_hermiticity_error'] < 1e-10
            diag['checks']['D_traceless'] = diag['D_trace'] < 1e-10
            diag['checks']['entropy_production_positive'] = diag['entropy_production'] >= -1e-12
        
        diag['all_checks_pass'] = all(diag['checks'].values())
        
        return diag
    
    def print_summary(self, detailed: bool = False):
        """
        Print human-readable summary of results.
        
        Parameters
        ----------
        detailed : bool
            Print detailed diagnostics
        """
        if not self.results:
            print("No results yet. Run compute_all() first.")
            return
        
        print("\n" + "=" * 70)
        print("GENERIC Decomposition Summary")
        print("=" * 70)
        
        print(f"\nState:")
        print(f"  ||θ|| = {np.linalg.norm(self.results['theta']):.6f}")
        print(f"  H(ρ) = {self.results['H_joint']:.6f}")
        print(f"  C = Σh_i = {self.results['C']:.6f}")
        
        print(f"\nInformation Geometry:")
        print(f"  ψ(θ) = {self.results['psi']:.6f}")
        print(f"  ||a|| = {self.results['a_norm']:.6f}")
        print(f"  ν = {self.results['nu']:.6f}")
        
        print(f"\nGENERIC Decomposition:")
        S_norm = np.linalg.norm(self.results['S'], 'fro')
        A_norm = np.linalg.norm(self.results['A'], 'fro')
        print(f"  ||S||_F = {S_norm:.6f} (dissipative)")
        print(f"  ||A||_F = {A_norm:.6f} (reversible)")
        print(f"  ||η|| = {np.linalg.norm(self.results['eta']):.6f}")
        print(f"  ||H_eff||_F = {np.linalg.norm(self.results['H_eff'], 'fro'):.6f}")
        
        if self.results['D_rho'] is not None:
            print(f"  ||D[ρ]||_F = {np.linalg.norm(self.results['D_rho'], 'fro'):.6f}")
        
        print(f"\nDiagnostics:")
        checks = self.diagnostics['checks']
        for key, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {key}")
        
        if self.diagnostics['all_checks_pass']:
            print("\n✓ All validation checks passed!")
        else:
            print("\n✗ Some validation checks failed.")
        
        if detailed:
            print(f"\nDetailed Diagnostics:")
            for key, value in self.diagnostics.items():
                if key not in ['checks', 'all_checks_pass']:
                    print(f"  {key}: {value:.2e}")
        
        print("=" * 70)


def run_generic_decomposition(theta: np.ndarray,
                              exp_fam: QuantumExponentialFamily,
                              method: str = 'duhamel',
                              compute_diffusion: bool = False,
                              verbose: bool = True,
                              print_summary: bool = True) -> Dict[str, Any]:
    """
    Convenience function for complete GENERIC decomposition.
    
    Parameters
    ----------
    theta : np.ndarray
        Natural parameters
    exp_fam : QuantumExponentialFamily
        Exponential family
    method : str, optional
        Derivative method: 'duhamel' or 'sld'
    compute_diffusion : bool, optional
        Whether to compute diffusion operator (expensive!)
    verbose : bool, optional
        Print progress during computation
    print_summary : bool, optional
        Print summary at end
        
    Returns
    -------
    results : dict
        Complete GENERIC decomposition results
        
    Examples
    --------
    >>> exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
    >>> theta = np.zeros(exp_fam.n_params)
    >>> results = run_generic_decomposition(theta, exp_fam)
    """
    decomp = GenericDecomposition(
        exp_fam,
        method=method,
        compute_diffusion=compute_diffusion
    )
    
    results = decomp.compute_all(theta, verbose=verbose)
    
    if print_summary:
        decomp.print_summary()
    
    return results


__all__ = ["GenericDecomposition", "run_generic_decomposition"]

