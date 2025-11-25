"""
Tolerance Framework for CIP-0004: Quantum Algorithm Test Suite

This module provides scientifically-derived tolerance categories and utility functions
for consistent numerical validation in quantum algorithm tests.

Tolerance categories are derived from rigorous error analysis of quantum operations:
- Matrix exponentiation, eigenvalue decomposition, Fisher metrics, etc.

See docs/cip0004_precision_analysis.md for mathematical justification.
"""

import numpy as np
from typing import Union, Optional


# ============================================================================
# Tolerance Categories: Scientifically Derived Bounds
# ============================================================================

class QuantumTolerances:
    """
    Tolerance categories for quantum algorithm validation.

    Categories based on mathematical precision analysis of quantum operations.
    See docs/cip0004_precision_analysis.md for derivation.
    """

    # Category A: Machine Precision Operations (≤ 1e-14)
    # Pure algebraic operations with minimal error accumulation
    A = {
        'rtol': 1e-14,
        'atol': 1e-15,  # Near machine epsilon
        'description': 'Machine precision operations (traces, algebra)'
    }

    # Category B: Quantum State Properties (≤ 1e-12)
    # Fundamental quantum constraints (unit trace, hermiticity)
    B = {
        'rtol': 1e-12,
        'atol': 1e-13,
        'description': 'Quantum state properties (unit trace, hermiticity)'
    }

    # Category C: Entanglement & Information Metrics (≤ 1e-10)
    # Information-theoretic quantities sensitive to eigenvalue ratios
    C = {
        'rtol': 1e-10,
        'atol': 1e-11,
        'description': 'Information metrics (entropy, mutual information)'
    }

    # Category D: Analytical Derivatives (≤ 1e-8)
    # Gradient and Hessian computations with error propagation
    D = {
        'rtol': 1e-8,
        'atol': 1e-9,
        'description': 'Analytical derivatives (Fisher metric, Jacobians)'
    }

    # Category E: Numerical Integration (≤ 1e-6)
    # ODE solvers and long-time trajectory computations
    E = {
        'rtol': 1e-6,
        'atol': 1e-7,
        'description': 'Numerical integration (dynamics, trajectories)'
    }

    # Category F: Physical Validation (≤ 1e-4)
    # Statistical significance for research conclusions
    F = {
        'rtol': 1e-4,
        'atol': 1e-5,
        'description': 'Physical validation (optimality claims, statistics)'
    }


# ============================================================================
# Tolerance Selection Logic
# ============================================================================

def select_tolerance_category(operation_type: str) -> dict:
    """
    Select appropriate tolerance category based on operation type.

    Parameters
    ----------
    operation_type : str
        Type of quantum operation being tested

    Returns
    -------
    category : dict
        Tolerance parameters with rtol, atol, description
    """
    category_map = {
        # Category A: Exact arithmetic
        'trace': QuantumTolerances.A,
        'hermitian': QuantumTolerances.A,
        'unitary': QuantumTolerances.A,
        'commutator': QuantumTolerances.A,
        'tensor_product': QuantumTolerances.A,

        # Category B: Quantum states
        'density_matrix': QuantumTolerances.B,
        'unit_trace': QuantumTolerances.B,
        'purity': QuantumTolerances.B,
        'state_preparation': QuantumTolerances.B,

        # Category C: Information theory
        'entropy': QuantumTolerances.C,
        'mutual_information': QuantumTolerances.C,
        'marginal_entropy': QuantumTolerances.C,
        'entanglement_measure': QuantumTolerances.C,

        # Category D: Derivatives
        'fisher_metric': QuantumTolerances.D,
        'bkm_metric': QuantumTolerances.D,
        'jacobian': QuantumTolerances.D,
        'constraint_gradient': QuantumTolerances.D,
        'constraint_hessian': QuantumTolerances.D,
        'third_cumulant': QuantumTolerances.D,

        # Category E: Integration
        'dynamics': QuantumTolerances.E,
        'trajectory': QuantumTolerances.E,
        'time_evolution': QuantumTolerances.E,
        'constraint_preservation': QuantumTolerances.E,

        # Category F: Validation
        'optimality': QuantumTolerances.F,
        'phase_transition': QuantumTolerances.F,
        'generic_decomposition': QuantumTolerances.F,
        'research_claim': QuantumTolerances.F,
    }

    return category_map.get(operation_type, QuantumTolerances.D)  # Default to D


# ============================================================================
# Assertion Utilities
# ============================================================================

def quantum_assert_close(
    actual: np.ndarray,
    expected: np.ndarray,
    operation_type: str = 'default',
    err_msg: str = '',
    **kwargs
) -> None:
    """
    Assert arrays are close using quantum-appropriate tolerances.

    Parameters
    ----------
    actual : array_like
        Computed result
    expected : array_like
        Expected result
    operation_type : str
        Type of quantum operation (determines tolerance category)
    err_msg : str
        Custom error message
    **kwargs
        Additional arguments passed to np.allclose
    """
    tol = select_tolerance_category(operation_type)

    # Override with provided tolerances if specified
    rtol = kwargs.pop('rtol', tol['rtol'])
    atol = kwargs.pop('atol', tol['atol'])

    success = np.allclose(actual, expected, rtol=rtol, atol=atol, **kwargs)

    if not success:
        # Compute actual error for better error message
        err_abs = np.abs(actual - expected)
        err_rel = err_abs / (np.abs(expected) + np.finfo(float).eps)

        max_err_abs = np.max(err_abs)
        max_err_rel = np.max(err_rel)

        default_msg = (
            f"Arrays not close for {operation_type} (tolerance category: {tol['description']})\n"
            f"Max absolute error: {max_err_abs:.2e} (atol: {atol:.0e})\n"
            f"Max relative error: {max_err_rel:.2e} (rtol: {rtol:.0e})"
        )

        if err_msg:
            full_msg = f"{err_msg}\n{default_msg}"
        else:
            full_msg = default_msg

        raise AssertionError(full_msg)


def quantum_assert_scalar_close(
    actual: Union[float, complex],
    expected: Union[float, complex],
    operation_type: str = 'default',
    err_msg: str = '',
    **kwargs
) -> None:
    """
    Assert scalars are close using quantum-appropriate tolerances.

    Parameters
    ----------
    actual : float or complex
        Computed result
    expected : float or complex
        Expected result
    operation_type : str
        Type of quantum operation
    err_msg : str
        Custom error message
    **kwargs
        Additional arguments
    """
    # Convert to arrays for consistent handling
    actual_arr = np.array([actual])
    expected_arr = np.array([expected])

    quantum_assert_close(actual_arr, expected_arr, operation_type, err_msg, **kwargs)


def quantum_assert_symmetric(
    matrix: np.ndarray,
    operation_type: str = 'fisher_metric',
    err_msg: str = 'Matrix is not symmetric'
) -> None:
    """
    Assert matrix is symmetric using quantum-appropriate tolerances.

    Parameters
    ----------
    matrix : array_like
        Matrix to check for symmetry
    operation_type : str
        Operation type for tolerance selection
    err_msg : str
        Error message if not symmetric
    """
    quantum_assert_close(matrix, matrix.T.conj(), operation_type, err_msg)


def quantum_assert_hermitian(
    matrix: np.ndarray,
    operation_type: str = 'density_matrix',
    err_msg: str = 'Matrix is not Hermitian'
) -> None:
    """
    Assert matrix is Hermitian using quantum-appropriate tolerances.

    Parameters
    ----------
    matrix : array_like
        Matrix to check for hermiticity
    operation_type : str
        Operation type for tolerance selection
    err_msg : str
        Error message if not Hermitian
    """
    quantum_assert_close(matrix, matrix.T.conj(), operation_type, err_msg)


def quantum_assert_unit_trace(
    rho: np.ndarray,
    operation_type: str = 'density_matrix',
    err_msg: str = 'Density matrix does not have unit trace'
) -> None:
    """
    Assert density matrix has unit trace.

    Parameters
    ----------
    rho : array_like
        Density matrix
    operation_type : str
        Operation type (should be density_matrix related)
    err_msg : str
        Error message if trace ≠ 1
    """
    trace = np.trace(rho)
    quantum_assert_scalar_close(trace, 1.0, operation_type, err_msg)


# ============================================================================
# Tolerance Validation Utilities
# ============================================================================

def validate_tolerance_choice(
    operation_type: str,
    rtol: float,
    atol: float,
    expected_category: Optional[str] = None
) -> dict:
    """
    Validate that chosen tolerances are appropriate for the operation type.

    Returns warning if tolerances are too loose or too strict.

    Parameters
    ----------
    operation_type : str
        Type of quantum operation
    rtol : float
        Relative tolerance used
    atol : float
        Absolute tolerance used
    expected_category : str, optional
        Expected tolerance category (for validation)

    Returns
    -------
    validation : dict
        Validation results with warnings and recommendations
    """
    recommended = select_tolerance_category(operation_type)

    validation = {
        'operation_type': operation_type,
        'recommended_rtol': recommended['rtol'],
        'recommended_atol': recommended['atol'],
        'used_rtol': rtol,
        'used_atol': atol,
        'warnings': [],
        'recommendations': []
    }

    # Check if tolerances are too loose
    if rtol > recommended['rtol'] * 10:
        validation['warnings'].append(
            f"Relative tolerance ({rtol:.0e}) is much looser than recommended "
            f"({recommended['rtol']:.0e}) for {operation_type}"
        )

    if atol > recommended['atol'] * 10:
        validation['warnings'].append(
            f"Absolute tolerance ({atol:.0e}) is much looser than recommended "
            f"({recommended['atol']:.0e}) for {operation_type}"
        )

    # Check if tolerances are too strict (might indicate numerical issues)
    if rtol < recommended['rtol'] * 0.01:
        validation['recommendations'].append(
            f"Relative tolerance ({rtol:.0e}) is much stricter than needed. "
            f"Consider using recommended {recommended['rtol']:.0e} for {operation_type}"
        )

    if atol < recommended['atol'] * 0.01:
        validation['recommendations'].append(
            f"Absolute tolerance ({atol:.0e}) is much stricter than needed. "
            f"Consider using recommended {recommended['atol']:.0e} for {operation_type}"
        )

    return validation


# ============================================================================
# Legacy Tolerance Compatibility
# ============================================================================

def get_legacy_tolerance(operation_type: str) -> tuple[float, float]:
    """
    Get legacy tolerance values for gradual migration.

    This provides backward compatibility during the transition to
    the new tolerance framework.

    Parameters
    ----------
    operation_type : str
        Type of quantum operation

    Returns
    -------
    rtol, atol : tuple of float
        Relative and absolute tolerance values
    """
    tol = select_tolerance_category(operation_type)
    return tol['rtol'], tol['atol']


# ============================================================================
# Usage Examples and Documentation
# ============================================================================

"""
Usage Examples:

# Basic assertion with automatic tolerance selection
quantum_assert_close(computed_entropy, expected_entropy, 'entropy')

# Symmetric matrix validation
quantum_assert_symmetric(fisher_metric, 'fisher_metric')

# Density matrix validation
quantum_assert_hermitian(rho, 'density_matrix')
quantum_assert_unit_trace(rho, 'density_matrix')

# Tolerance validation
validation = validate_tolerance_choice('fisher_metric', 1e-6, 1e-7)
if validation['warnings']:
    print("Tolerance warnings:", validation['warnings'])

Migration Guide:

Old code:
    np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-9)

New code:
    quantum_assert_close(a, b, 'fisher_metric')  # Automatic tolerances
    # or
    quantum_assert_close(a, b, 'fisher_metric', rtol=1e-8, atol=1e-9)  # Override
"""
