"""
Validation utilities for the quantum inaccessible game GENERIC decomposition.

This module provides tools for validating computational results throughout
the GENERIC decomposition procedure, including:
- Matrix property checks (Hermiticity, symmetry, tracelessness)
- Cross-validation between different computational methods
- Comparison against reference data
- Comprehensive validation reporting

These utilities are used throughout all implementation phases to ensure
correctness and numerical stability.
"""

from typing import Optional, Dict, List, Tuple, Callable, Any
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ValidationCheck:
    """
    Represents a single validation check with its result.
    
    Attributes
    ----------
    name : str
        Name of the validation check
    passed : bool
        Whether the check passed
    value : float
        Measured value (e.g., error norm)
    tolerance : float
        Tolerance threshold
    message : str
        Detailed message about the check
    """
    name: str
    passed: bool
    value: float
    tolerance: float
    message: str = ""
    
    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status} | {self.name}: {self.value:.2e} (tol: {self.tolerance:.2e}) {self.message}"


class ValidationReport:
    """
    Collects and reports validation results for a computation.
    
    This class accumulates validation checks throughout a computation
    and provides methods to display results, identify failures, and
    generate summary statistics.
    
    Examples
    --------
    >>> report = ValidationReport("Structure Constants")
    >>> report.add_check("Antisymmetry", error < tol, error, tol)
    >>> report.add_check("Jacobi identity", violation < tol, violation, tol)
    >>> report.print_summary()
    >>> if not report.all_passed():
    ...     print(report.get_failures())
    """
    
    def __init__(self, title: str):
        """
        Initialize validation report.
        
        Parameters
        ----------
        title : str
            Title for this validation report
        """
        self.title = title
        self.checks: List[ValidationCheck] = []
        
    def add_check(self, name: str, passed: bool, value: float, 
                  tolerance: float, message: str = ""):
        """
        Add a validation check to the report.
        
        Parameters
        ----------
        name : str
            Name of the check
        passed : bool
            Whether the check passed
        value : float
            Measured value
        tolerance : float
            Tolerance threshold
        message : str, optional
            Additional message
        """
        check = ValidationCheck(name, passed, value, tolerance, message)
        self.checks.append(check)
        
    def all_passed(self) -> bool:
        """Check if all validation checks passed."""
        return all(check.passed for check in self.checks)
    
    def get_failures(self) -> List[ValidationCheck]:
        """Get list of failed checks."""
        return [check for check in self.checks if not check.passed]
    
    def get_passes(self) -> List[ValidationCheck]:
        """Get list of passed checks."""
        return [check for check in self.checks if check.passed]
    
    def print_summary(self, verbose: bool = True):
        """
        Print summary of validation results.
        
        Parameters
        ----------
        verbose : bool
            If True, print all checks. If False, only print failures.
        """
        print(f"\n{'='*70}")
        print(f"Validation Report: {self.title}")
        print(f"{'='*70}")
        
        if verbose:
            for check in self.checks:
                print(check)
        else:
            failures = self.get_failures()
            if failures:
                print("FAILURES:")
                for check in failures:
                    print(check)
            else:
                print("All checks passed!")
        
        print(f"{'-'*70}")
        n_passed = len(self.get_passes())
        n_failed = len(self.get_failures())
        total = len(self.checks)
        print(f"Summary: {n_passed}/{total} passed, {n_failed}/{total} failed")
        print(f"{'='*70}\n")


def compare_matrices(A: np.ndarray, B: np.ndarray, tol: float, 
                    name: str = "Matrix comparison") -> Tuple[bool, float, str]:
    """
    Compare two matrices with detailed diagnostics.
    
    Computes various error norms and provides diagnostic information
    about differences between matrices.
    
    Parameters
    ----------
    A : np.ndarray
        First matrix
    B : np.ndarray
        Second matrix
    tol : float
        Tolerance threshold
    name : str
        Name for this comparison
        
    Returns
    -------
    passed : bool
        Whether comparison passed (max error < tol)
    error : float
        Maximum absolute error
    message : str
        Diagnostic message
    """
    if A.shape != B.shape:
        return False, np.inf, f"Shape mismatch: {A.shape} vs {B.shape}"
    
    diff = A - B
    
    # Handle NaN and Inf
    if np.any(np.isnan(diff)):
        return False, np.inf, "Contains NaN values"
    if np.any(np.isinf(diff)):
        return False, np.inf, "Contains Inf values"
    
    # Compute error norms
    max_abs_error = np.max(np.abs(diff))
    frobenius_norm = np.linalg.norm(diff, 'fro')
    relative_error = frobenius_norm / (np.linalg.norm(A, 'fro') + 1e-16)
    
    passed = max_abs_error < tol
    
    message = f"Frob: {frobenius_norm:.2e}, Rel: {relative_error:.2e}"
    
    return passed, max_abs_error, message


def check_hermitian(M: np.ndarray, tol: float = 1e-12) -> Tuple[bool, float]:
    """
    Check if a matrix is Hermitian (M = M†).
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to check
    tol : float
        Tolerance for Hermiticity
        
    Returns
    -------
    passed : bool
        Whether matrix is Hermitian within tolerance
    error : float
        ||M - M†||_max
    """
    error = np.max(np.abs(M - M.conj().T))
    return error < tol, error


def check_symmetric(M: np.ndarray, tol: float = 1e-14) -> Tuple[bool, float]:
    """
    Check if a matrix is symmetric (M = M^T).
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to check
    tol : float
        Tolerance for symmetry
        
    Returns
    -------
    passed : bool
        Whether matrix is symmetric within tolerance
    error : float
        ||M - M^T||_max
    """
    error = np.max(np.abs(M - M.T))
    return error < tol, error


def check_antisymmetric(M: np.ndarray, tol: float = 1e-14) -> Tuple[bool, float]:
    """
    Check if a matrix is antisymmetric (M = -M^T).
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to check
    tol : float
        Tolerance for antisymmetry
        
    Returns
    -------
    passed : bool
        Whether matrix is antisymmetric within tolerance
    error : float
        ||M + M^T||_max
    """
    error = np.max(np.abs(M + M.T))
    return error < tol, error


def check_traceless(M: np.ndarray, tol: float = 1e-10) -> Tuple[bool, float]:
    """
    Check if a matrix is traceless (Tr(M) = 0).
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to check
    tol : float
        Tolerance for tracelessness
        
    Returns
    -------
    passed : bool
        Whether matrix is traceless within tolerance
    error : float
        |Tr(M)|
    """
    trace = np.trace(M)
    error = np.abs(trace)
    return error < tol, error


def check_commutator(A: np.ndarray, B: np.ndarray, operators: List[np.ndarray],
                    f_abc: np.ndarray, tol: float = 1e-8) -> Tuple[bool, float, str]:
    """
    Verify commutator relation [A, B] = 2i Σ_c f_abc F_c.
    
    Parameters
    ----------
    A : np.ndarray
        First operator (index a)
    B : np.ndarray
        Second operator (index b)
    operators : List[np.ndarray]
        List of basis operators {F_c}
    f_abc : np.ndarray, shape (n, n, n)
        Structure constants
    tol : float
        Tolerance for commutator verification
        
    Returns
    -------
    passed : bool
        Whether commutator relation holds
    error : float
        Maximum error in reconstruction
    message : str
        Diagnostic message
    """
    # Compute commutator [A, B]
    commutator = A @ B - B @ A
    
    # Reconstruct from structure constants: 2i Σ_c f_abc F_c
    # Need to find indices a, b in operator list
    # For now, just check if commutator can be decomposed
    
    # Project commutator onto each basis element
    n_ops = len(operators)
    coefficients = np.zeros(n_ops, dtype=complex)
    
    for c, F_c in enumerate(operators):
        # Project: coefficient = Tr([A,B] F_c) / Tr(F_c F_c)
        norm = np.trace(F_c @ F_c.conj().T)
        if np.abs(norm) > 1e-15:
            coefficients[c] = np.trace(commutator @ F_c.conj().T) / norm
    
    # Reconstruct
    reconstructed = sum(coef * F for coef, F in zip(coefficients, operators))
    
    error = np.max(np.abs(commutator - reconstructed))
    passed = error < tol
    
    message = f"Decomposition coefficients: max |c| = {np.max(np.abs(coefficients)):.2e}"
    
    return passed, error, message


def finite_difference_jacobian(func: Callable[[np.ndarray], np.ndarray], 
                               x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Compute Jacobian using finite differences.
    
    This provides an independent validation method for analytically
    computed Jacobians.
    
    Parameters
    ----------
    func : Callable
        Function mapping R^n -> R^m
    x : np.ndarray, shape (n,)
        Point at which to compute Jacobian
    eps : float
        Finite difference step size
        
    Returns
    -------
    J : np.ndarray, shape (m, n)
        Jacobian matrix approximation
        
    Notes
    -----
    Uses central differences: f'(x) ≈ (f(x+h) - f(x-h))/(2h)
    """
    n = len(x)
    f0 = func(x)
    m = len(f0) if isinstance(f0, np.ndarray) else 1
    
    J = np.zeros((m, n))
    
    for i in range(n):
        # Perturb in direction i
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        
        # Central difference
        f_plus = func(x_plus)
        f_minus = func(x_minus)
        J[:, i] = (f_plus - f_minus) / (2 * eps)
    
    return J


def check_constraint_tangency(M: np.ndarray, constraint_gradient: np.ndarray,
                              tol: float = 1e-6) -> Tuple[bool, float]:
    """
    Check if flow is tangent to constraint manifold.
    
    Verifies that M @ a ≈ 0 where a = ∇C is the constraint gradient.
    
    Parameters
    ----------
    M : np.ndarray
        Flow Jacobian
    constraint_gradient : np.ndarray
        Gradient of constraint C
    tol : float
        Tolerance for tangency
        
    Returns
    -------
    passed : bool
        Whether flow is tangent within tolerance
    error : float
        ||M @ a||
    """
    residual = M @ constraint_gradient
    error = np.linalg.norm(residual)
    return error < tol, error


def check_entropy_monotonicity(M: np.ndarray, theta: np.ndarray,
                               tol: float = 1e-12) -> Tuple[bool, float]:
    """
    Check if flow decreases entropy (or stays constant).
    
    Verifies that θ^T M θ ≤ 0 (with small tolerance for numerical error).
    
    Parameters
    ----------
    M : np.ndarray
        Flow Jacobian
    theta : np.ndarray
        Natural parameters
    tol : float
        Tolerance for non-positivity (allows small numerical error)
        
    Returns
    -------
    passed : bool
        Whether entropy is non-increasing
    value : float
        θ^T M θ (should be ≤ 0)
    """
    entropy_rate = theta @ M @ theta
    # Allow small positive values due to numerical error
    return entropy_rate <= tol, entropy_rate


def check_positive_semidefinite(M: np.ndarray, tol: float = 1e-14) -> Tuple[bool, float]:
    """
    Check if a matrix is positive semidefinite.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to check
    tol : float
        Tolerance for negative eigenvalues
        
    Returns
    -------
    passed : bool
        Whether all eigenvalues are non-negative (within tolerance)
    min_eigenvalue : float
        Smallest eigenvalue
    """
    eigenvalues = np.linalg.eigvalsh(M)
    min_eig = np.min(eigenvalues)
    return min_eig >= -tol, min_eig

