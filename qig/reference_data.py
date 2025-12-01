"""
Reference data for validation of GENERIC decomposition computations.

This module provides known-correct values for structure constants and
other quantities used in validation tests.

References
----------
SU(2) structure constants:
    Completely antisymmetric epsilon tensor ε_ijk
    [σ_i, σ_j] = 2i ε_ijk σ_k where σ are Pauli matrices
    
SU(3) structure constants:
    Standard structure constants for Gell-Mann matrices
    Values from: Gell-Mann, M. (1962). "Symmetries of Baryons and Mesons"
    Also verified against: Particle Data Group reviews
"""

import numpy as np
from typing import Dict, Any


def get_su2_structure_constants() -> np.ndarray:
    """
    Get structure constants for SU(2) algebra (Pauli matrices).
    
    The Pauli matrices satisfy: [σ_i, σ_j] = 2i ε_ijk σ_k
    where ε_ijk is the Levi-Civita symbol.
    
    Returns
    -------
    f_abc : np.ndarray, shape (3, 3, 3)
        Structure constants with normalization [F_a, F_b] = 2i Σ_c f_abc F_c
        
    Notes
    -----
    The non-zero components are:
        f_123 = +1  (cyclic permutations)
        f_231 = +1
        f_312 = +1
        f_213 = -1  (anti-cyclic permutations)
        f_132 = -1
        f_321 = -1
    """
    f = np.zeros((3, 3, 3))
    
    # Cyclic permutations: f_123 = f_231 = f_312 = +1
    f[0, 1, 2] = 1.0
    f[1, 2, 0] = 1.0
    f[2, 0, 1] = 1.0
    
    # Anti-cyclic permutations: f_213 = f_132 = f_321 = -1
    f[1, 0, 2] = -1.0
    f[0, 2, 1] = -1.0
    f[2, 1, 0] = -1.0
    
    return f


def get_su3_structure_constants() -> np.ndarray:
    """
    Get structure constants for SU(3) algebra (Gell-Mann matrices).
    
    The Gell-Mann matrices λ_a satisfy: [λ_a, λ_b] = 2i Σ_c f_abc λ_c
    
    Returns
    -------
    f_abc : np.ndarray, shape (8, 8, 8)
        Structure constants for SU(3)
        
    Notes
    -----
    The structure constants have the following properties:
    - Totally antisymmetric: f_abc = -f_bac
    - Real valued
    - Satisfy Jacobi identity
    
    Non-zero structure constants (up to permutations and sign):
    - f_123 = 1
    - f_147 = f_246 = f_257 = f_345 = 1/2
    - f_156 = f_367 = -1/2
    - f_458 = f_678 = sqrt(3)/2
    
    Reference: Gell-Mann (1962), Particle Data Group
    """
    f = np.zeros((8, 8, 8))
    
    # Note: Using 0-based indexing (subtract 1 from physical indices)
    
    # f_123 = 1
    f[0, 1, 2] = 1.0
    f[1, 2, 0] = 1.0
    f[2, 0, 1] = 1.0
    f[1, 0, 2] = -1.0
    f[0, 2, 1] = -1.0
    f[2, 1, 0] = -1.0
    
    # f_147 = 1/2
    indices = [(0, 3, 6), (3, 6, 0), (6, 0, 3)]
    for i, j, k in indices:
        f[i, j, k] = 0.5
    indices = [(3, 0, 6), (0, 6, 3), (6, 3, 0)]
    for i, j, k in indices:
        f[i, j, k] = -0.5
    
    # f_156 = -1/2
    indices = [(0, 4, 5), (4, 5, 0), (5, 0, 4)]
    for i, j, k in indices:
        f[i, j, k] = -0.5
    indices = [(4, 0, 5), (0, 5, 4), (5, 4, 0)]
    for i, j, k in indices:
        f[i, j, k] = 0.5
    
    # f_246 = 1/2
    indices = [(1, 3, 5), (3, 5, 1), (5, 1, 3)]
    for i, j, k in indices:
        f[i, j, k] = 0.5
    indices = [(3, 1, 5), (1, 5, 3), (5, 3, 1)]
    for i, j, k in indices:
        f[i, j, k] = -0.5
    
    # f_257 = 1/2
    indices = [(1, 4, 6), (4, 6, 1), (6, 1, 4)]
    for i, j, k in indices:
        f[i, j, k] = 0.5
    indices = [(4, 1, 6), (1, 6, 4), (6, 4, 1)]
    for i, j, k in indices:
        f[i, j, k] = -0.5
    
    # f_345 = 1/2
    indices = [(2, 3, 4), (3, 4, 2), (4, 2, 3)]
    for i, j, k in indices:
        f[i, j, k] = 0.5
    indices = [(3, 2, 4), (2, 4, 3), (4, 3, 2)]
    for i, j, k in indices:
        f[i, j, k] = -0.5
    
    # f_367 = -1/2
    indices = [(2, 5, 6), (5, 6, 2), (6, 2, 5)]
    for i, j, k in indices:
        f[i, j, k] = -0.5
    indices = [(5, 2, 6), (2, 6, 5), (6, 5, 2)]
    for i, j, k in indices:
        f[i, j, k] = 0.5
    
    # f_458 = sqrt(3)/2
    sqrt3_half = np.sqrt(3.0) / 2.0
    indices = [(3, 4, 7), (4, 7, 3), (7, 3, 4)]
    for i, j, k in indices:
        f[i, j, k] = sqrt3_half
    indices = [(4, 3, 7), (3, 7, 4), (7, 4, 3)]
    for i, j, k in indices:
        f[i, j, k] = -sqrt3_half
    
    # f_678 = sqrt(3)/2
    indices = [(5, 6, 7), (6, 7, 5), (7, 5, 6)]
    for i, j, k in indices:
        f[i, j, k] = sqrt3_half
    indices = [(6, 5, 7), (5, 7, 6), (7, 6, 5)]
    for i, j, k in indices:
        f[i, j, k] = -sqrt3_half
    
    return f


def verify_structure_constant_properties(f_abc: np.ndarray, 
                                        algebra_name: str = "unknown") -> Dict[str, Any]:
    """
    Verify mathematical properties of structure constants.
    
    Parameters
    ----------
    f_abc : np.ndarray, shape (n, n, n)
        Structure constants
    algebra_name : str
        Name of algebra for reporting
        
    Returns
    -------
    results : Dict[str, Any]
        Dictionary containing verification results:
        - antisymmetry_error: Max |f_abc + f_bac|
        - jacobi_error: Max Jacobi identity violation
        - is_real: Whether all entries are real
        - passed: Whether all checks passed
    """
    n = f_abc.shape[0]
    
    # Check antisymmetry: f_abc = -f_bac
    antisymmetry_error = np.max(np.abs(f_abc + np.transpose(f_abc, (1, 0, 2))))
    
    # Check Jacobi identity: Σ_d (f_abd f_dce + f_bcd f_dae + f_cad f_dbe) = 0
    jacobi_violations = []
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for e in range(n):
                    jacobi_sum = 0.0
                    for d in range(n):
                        jacobi_sum += (f_abc[a,b,d] * f_abc[d,c,e] +
                                      f_abc[b,c,d] * f_abc[d,a,e] +
                                      f_abc[c,a,d] * f_abc[d,b,e])
                    jacobi_violations.append(abs(jacobi_sum))
    
    jacobi_error = max(jacobi_violations) if jacobi_violations else 0.0
    
    # Check if real
    is_real = np.allclose(f_abc.imag, 0.0)
    
    # Overall pass/fail
    passed = (antisymmetry_error < 1e-10 and 
             jacobi_error < 1e-8 and 
             is_real)
    
    results = {
        'algebra': algebra_name,
        'antisymmetry_error': antisymmetry_error,
        'jacobi_error': jacobi_error,
        'is_real': is_real,
        'passed': passed,
        'n_generators': n
    }
    
    return results


# Generate and verify reference data on module import
_SU2_REFERENCE = get_su2_structure_constants()
_SU3_REFERENCE = get_su3_structure_constants()

_SU2_VERIFICATION = verify_structure_constant_properties(_SU2_REFERENCE, "SU(2)")
_SU3_VERIFICATION = verify_structure_constant_properties(_SU3_REFERENCE, "SU(3)")


def get_reference_structure_constants(algebra_type: str) -> np.ndarray:
    """
    Get reference structure constants for a given algebra.
    
    Parameters
    ----------
    algebra_type : str
        Type of algebra: "su2" or "su3"
        
    Returns
    -------
    f_abc : np.ndarray
        Structure constants
        
    Raises
    ------
    ValueError
        If algebra_type is not recognized
    """
    algebra_type = algebra_type.lower().replace("(", "").replace(")", "")
    
    if algebra_type == "su2":
        return _SU2_REFERENCE.copy()
    elif algebra_type == "su3":
        return _SU3_REFERENCE.copy()
    else:
        raise ValueError(f"Unknown algebra type: {algebra_type}. "
                        f"Supported types: 'su2', 'su3'")


def print_reference_verification():
    """
    Print verification results for reference structure constants.
    """
    print("\n" + "="*70)
    print("Reference Structure Constants Verification")
    print("="*70)
    
    for name, results in [("SU(2)", _SU2_VERIFICATION), 
                          ("SU(3)", _SU3_VERIFICATION)]:
        print(f"\n{name} Algebra ({results['n_generators']} generators):")
        print(f"  Antisymmetry: {results['antisymmetry_error']:.2e}")
        print(f"  Jacobi identity: {results['jacobi_error']:.2e}")
        print(f"  Real valued: {results['is_real']}")
        status = "✓ PASSED" if results['passed'] else "✗ FAILED"
        print(f"  Status: {status}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Verification when run as script
    print_reference_verification()

