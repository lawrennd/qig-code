"""
Structure constants for Lie algebras.

This module provides functions to compute and verify structure constants
f_abc for Lie algebras, where [F_a, F_b] = 2i Σ_c f_abc F_c.

The structure constants encode the commutation relations of the algebra
and are fundamental for extracting the effective Hamiltonian in the
GENERIC decomposition.
"""

from typing import List, Tuple, Optional
import numpy as np
from qig.validation import ValidationReport, compare_matrices


def compute_structure_constants(operators: List[np.ndarray], 
                                tol: float = 1e-10) -> np.ndarray:
    """
    Compute structure constants f_abc for a list of operators {F_a}.
    
    The structure constants satisfy: [F_a, F_b] = 2i Σ_c f_abc F_c
    
    Parameters
    ----------
    operators : List[np.ndarray]
        List of operators (Hermitian, traceless generators)
    tol : float
        Tolerance for considering entries as zero
        
    Returns
    -------
    f_abc : np.ndarray, shape (n, n, n)
        Structure constants
        
    Notes
    -----
    The normalization convention is: [F_a, F_b] = 2i Σ_c f_abc F_c
    
    For each pair (a,b), we:
    1. Compute commutator [F_a, F_b]
    2. Project onto each basis element F_c
    3. Extract coefficient f_abc
    
    The projection uses: f_abc = Tr([F_a,F_b] F_c) / (2i Tr(F_c F_c))
    """
    n = len(operators)
    
    if n == 0:
        return np.array([])
    
    # Get dimension of operators
    d = operators[0].shape[0]
    
    # Verify all operators have same shape
    for i, F in enumerate(operators):
        if F.shape != (d, d):
            raise ValueError(f"Operator {i} has shape {F.shape}, expected ({d}, {d})")
    
    # Initialize structure constants
    f_abc = np.zeros((n, n, n), dtype=complex)
    
    # Compute normalization factors (Tr(F_c F_c†))
    norms = np.zeros(n)
    for c, F_c in enumerate(operators):
        norms[c] = np.real(np.trace(F_c @ F_c.conj().T))
        if norms[c] < tol:
            raise ValueError(f"Operator {c} has near-zero norm: {norms[c]:.2e}")
    
    # Compute structure constants
    for a, F_a in enumerate(operators):
        for b, F_b in enumerate(operators):
            # Compute commutator [F_a, F_b]
            commutator = F_a @ F_b - F_b @ F_a
            
            # Project onto each basis element
            for c, F_c in enumerate(operators):
                # f_abc = Tr([F_a, F_b] F_c†) / (2i Tr(F_c F_c†))
                projection = np.trace(commutator @ F_c.conj().T)
                f_abc[a, b, c] = projection / (2.0j * norms[c])
    
    # Structure constants should be real for Hermitian generators
    # Small imaginary parts are numerical error
    if np.max(np.abs(f_abc.imag)) < tol:
        f_abc = f_abc.real
    else:
        max_imag = np.max(np.abs(f_abc.imag))
        import warnings
        warnings.warn(f"Structure constants have imaginary parts up to {max_imag:.2e}. "
                     f"This may indicate non-Hermitian operators.")
    
    return f_abc


def verify_lie_algebra(operators: List[np.ndarray], 
                      f_abc: np.ndarray,
                      tol: float = 1e-8) -> ValidationReport:
    """
    Verify that operators and structure constants satisfy Lie algebra relations.
    
    Checks: [F_a, F_b] = 2i Σ_c f_abc F_c
    
    Parameters
    ----------
    operators : List[np.ndarray]
        List of operators
    f_abc : np.ndarray, shape (n, n, n)
        Structure constants
    tol : float
        Tolerance for verification
        
    Returns
    -------
    report : ValidationReport
        Validation report with all checks
    """
    n = len(operators)
    report = ValidationReport("Lie Algebra Verification")
    
    # Check each commutator
    max_error = 0.0
    for a in range(n):
        for b in range(n):
            # Compute commutator
            commutator = operators[a] @ operators[b] - operators[b] @ operators[a]
            
            # Reconstruct from structure constants
            reconstructed = sum(2.0j * f_abc[a, b, c] * operators[c] 
                              for c in range(n))
            
            # Compare
            error = np.max(np.abs(commutator - reconstructed))
            max_error = max(max_error, error)
    
    passed = max_error < tol
    report.add_check(f"Commutator relations [F_a,F_b] = 2i Σ f_abc F_c",
                    passed, max_error, tol,
                    f"Max error over all {n*(n-1)//2} commutators")
    
    return report


def verify_jacobi_identity(f_abc: np.ndarray, 
                          tol: float = 1e-8) -> ValidationReport:
    """
    Verify Jacobi identity for structure constants.
    
    Checks: Σ_d (f_abd f_dce + f_bcd f_dae + f_cad f_dbe) = 0
    for all a,b,c,e
    
    Parameters
    ----------
    f_abc : np.ndarray, shape (n, n, n)
        Structure constants
    tol : float
        Tolerance for verification
        
    Returns
    -------
    report : ValidationReport
        Validation report with Jacobi identity check
        
    Notes
    -----
    The Jacobi identity is equivalent to:
    [F_a, [F_b, F_c]] + [F_b, [F_c, F_a]] + [F_c, [F_a, F_b]] = 0
    """
    n = f_abc.shape[0]
    report = ValidationReport("Jacobi Identity Verification")
    
    max_violation = 0.0
    total_checks = 0
    
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for e in range(n):
                    # Compute Jacobi sum
                    jacobi_sum = 0.0
                    for d in range(n):
                        jacobi_sum += (f_abc[a, b, d] * f_abc[d, c, e] +
                                      f_abc[b, c, d] * f_abc[d, a, e] +
                                      f_abc[c, a, d] * f_abc[d, b, e])
                    
                    violation = abs(jacobi_sum)
                    max_violation = max(max_violation, violation)
                    total_checks += 1
    
    passed = max_violation < tol
    report.add_check("Jacobi identity", passed, max_violation, tol,
                    f"Max violation over {total_checks} checks")
    
    return report


def verify_antisymmetry(f_abc: np.ndarray, 
                       tol: float = 1e-10) -> ValidationReport:
    """
    Verify antisymmetry of structure constants.
    
    Checks: f_abc = -f_bac
    
    Parameters
    ----------
    f_abc : np.ndarray, shape (n, n, n)
        Structure constants
    tol : float
        Tolerance for verification
        
    Returns
    -------
    report : ValidationReport
        Validation report with antisymmetry check
    """
    report = ValidationReport("Antisymmetry Verification")
    
    # f_abc + f_bac should be zero
    antisymmetry_error = np.max(np.abs(f_abc + np.transpose(f_abc, (1, 0, 2))))
    
    passed = antisymmetry_error < tol
    report.add_check("Antisymmetry f_abc = -f_bac", passed, 
                    antisymmetry_error, tol)
    
    return report


# Cache for commonly used structure constants
_CACHED_STRUCTURE_CONSTANTS = {}


def get_cached_structure_constants(algebra_type: str) -> Optional[np.ndarray]:
    """
    Get cached structure constants for common algebras.
    
    Parameters
    ----------
    algebra_type : str
        Type of algebra: "su2" or "su3"
        
    Returns
    -------
    f_abc : np.ndarray or None
        Cached structure constants, or None if not in cache
    """
    return _CACHED_STRUCTURE_CONSTANTS.get(algebra_type.lower())


def cache_structure_constants(algebra_type: str, f_abc: np.ndarray):
    """
    Cache structure constants for reuse.
    
    Parameters
    ----------
    algebra_type : str
        Type of algebra
    f_abc : np.ndarray
        Structure constants to cache
    """
    _CACHED_STRUCTURE_CONSTANTS[algebra_type.lower()] = f_abc.copy()


def compute_and_cache_structure_constants(operators: List[np.ndarray],
                                         algebra_type: str,
                                         force_recompute: bool = False,
                                         tol: float = 1e-10) -> np.ndarray:
    """
    Compute structure constants with caching.
    
    Parameters
    ----------
    operators : List[np.ndarray]
        List of operators
    algebra_type : str
        Type of algebra (for caching)
    force_recompute : bool
        If True, recompute even if cached
    tol : float
        Tolerance for computation
        
    Returns
    -------
    f_abc : np.ndarray
        Structure constants
    """
    algebra_type = algebra_type.lower()
    
    # Check cache
    if not force_recompute:
        cached = get_cached_structure_constants(algebra_type)
        if cached is not None:
            return cached
    
    # Compute
    f_abc = compute_structure_constants(operators, tol=tol)
    
    # Cache
    cache_structure_constants(algebra_type, f_abc)
    
    return f_abc


def verify_all_properties(f_abc: np.ndarray,
                         operators: Optional[List[np.ndarray]] = None,
                         algebra_name: str = "unknown",
                         tol_antisymmetry: float = 1e-10,
                         tol_jacobi: float = 1e-8,
                         tol_commutator: float = 1e-8) -> ValidationReport:
    """
    Run all verification checks on structure constants.
    
    Parameters
    ----------
    f_abc : np.ndarray
        Structure constants
    operators : List[np.ndarray], optional
        Operators (if provided, verify commutator relations)
    algebra_name : str
        Name for reporting
    tol_antisymmetry : float
        Tolerance for antisymmetry check
    tol_jacobi : float
        Tolerance for Jacobi identity
    tol_commutator : float
        Tolerance for commutator verification
        
    Returns
    -------
    report : ValidationReport
        Combined validation report
    """
    report = ValidationReport(f"Structure Constants: {algebra_name}")
    
    # Antisymmetry
    anti_report = verify_antisymmetry(f_abc, tol=tol_antisymmetry)
    for check in anti_report.checks:
        report.checks.append(check)
    
    # Jacobi identity
    jacobi_report = verify_jacobi_identity(f_abc, tol=tol_jacobi)
    for check in jacobi_report.checks:
        report.checks.append(check)
    
    # Commutator relations (if operators provided)
    if operators is not None:
        comm_report = verify_lie_algebra(operators, f_abc, tol=tol_commutator)
        for check in comm_report.checks:
            report.checks.append(check)
    
    return report

