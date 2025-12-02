"""
Export symbolic computation results to human-readable files.

This saves the symbolic expressions as:
1. Python code (can be imported and used directly)
2. LaTeX (for documentation/papers)

Related to CIP-0007: Analytic Forms for S and A via Lie Algebra Structure
"""

import sympy as sp
from pathlib import Path
from typing import Tuple
import os


# Output directory for exported results
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def export_symbolic_expressions(
    n_params: int = 6,
    order: int = 2,
    output_dir: Path = RESULTS_DIR
) -> dict:
    """
    Export key symbolic expressions to files.
    
    Parameters
    ----------
    n_params : int
        Number of symbolic parameters (rest padded with 0)
    order : int
        Order of Taylor expansion
    output_dir : Path
        Directory for output files
        
    Returns
    -------
    files : dict
        Dictionary of output file paths
    """
    from qig.symbolic import (
        symbolic_constraint_gradient_su9_pair,
        symbolic_lagrange_multiplier_su9_pair,
        symbolic_grad_lagrange_multiplier_su9_pair,
        symbolic_antisymmetric_part_su9_pair,
    )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create symbolic parameters
    theta_sym = sp.symbols(f'theta1:{n_params+1}', real=True)
    theta_full = tuple(list(theta_sym) + [sp.Integer(0)] * (80 - n_params))
    
    print(f"Computing symbolic expressions with {n_params} parameters...")
    
    # Compute expressions (will use cache if available)
    print("  Computing constraint gradient a...")
    a = symbolic_constraint_gradient_su9_pair(theta_full, order)
    
    print("  Computing Lagrange multiplier ν...")
    nu = symbolic_lagrange_multiplier_su9_pair(theta_full, order)
    
    print("  Computing gradient ∇ν...")
    grad_nu = symbolic_grad_lagrange_multiplier_su9_pair(theta_full, order)
    
    print("  Computing antisymmetric part A...")
    A = symbolic_antisymmetric_part_su9_pair(theta_full, order)
    
    files = {}
    
    # Export as Python module
    py_file = output_dir / f"symbolic_expressions_{n_params}params.py"
    with open(py_file, 'w') as f:
        f.write('"""\n')
        f.write(f'Symbolic expressions for su(9) pair GENERIC decomposition.\n')
        f.write(f'Generated with {n_params} symbolic parameters, order-{order} expansion.\n')
        f.write(f'\n')
        f.write(f'These expressions are the ANALYTIC forms for:\n')
        f.write(f'  - Constraint gradient a = ∇(h₁ + h₂)\n')
        f.write(f'  - Lagrange multiplier ν = (aᵀGθ)/(aᵀa)\n')
        f.write(f'  - Gradient ∇ν\n')
        f.write(f'  - Antisymmetric part A = (1/2)[a⊗(∇ν)ᵀ - (∇ν)⊗aᵀ]\n')
        f.write(f'\n')
        f.write(f'Related to CIP-0007.\n')
        f.write('"""\n\n')
        f.write('import sympy as sp\n')
        f.write('from sympy import Matrix, Rational, sqrt, log\n\n')
        
        # Symbols
        f.write(f'# Symbolic parameters\n')
        f.write(f'theta = sp.symbols("theta1:{n_params+1}", real=True)\n\n')
        
        # Constraint gradient (first n_params components)
        f.write('# Constraint gradient: a = ∇(h₁ + h₂)\n')
        f.write('# Shape: (80, 1), only first {} components shown\n'.format(n_params))
        f.write('a = Matrix([\n')
        for i in range(n_params):
            expr = sp.simplify(a[i, 0])
            f.write(f'    [{repr(expr)}],\n')
        f.write('])\n\n')
        
        # Lagrange multiplier
        f.write('# Lagrange multiplier: ν = (aᵀGθ)/(aᵀa)\n')
        f.write(f'nu = {repr(sp.simplify(nu))}\n\n')
        
        # Gradient of nu (first n_params components)
        f.write('# Gradient of Lagrange multiplier: ∇ν\n')
        f.write('# Shape: (80, 1), only first {} components shown\n'.format(n_params))
        f.write('grad_nu = Matrix([\n')
        for i in range(n_params):
            expr = sp.simplify(grad_nu[i, 0])
            f.write(f'    [{repr(expr)}],\n')
        f.write('])\n\n')
        
        # Antisymmetric part (n_params x n_params block)
        f.write('# Antisymmetric part: A = (1/2)[a⊗(∇ν)ᵀ - (∇ν)⊗aᵀ]\n')
        f.write('# Shape: (80, 80), only first {}×{} block shown\n'.format(n_params, n_params))
        f.write('A = Matrix([\n')
        for i in range(n_params):
            row = []
            for j in range(n_params):
                expr = sp.simplify(A[i, j])
                row.append(repr(expr))
            f.write(f'    [{", ".join(row)}],\n')
        f.write('])\n')
    
    files['python'] = py_file
    print(f"  ✓ Saved Python: {py_file}")
    
    # Export key results as LaTeX
    tex_file = output_dir / f"symbolic_expressions_{n_params}params.tex"
    with open(tex_file, 'w') as f:
        f.write('% Symbolic expressions for su(9) pair GENERIC decomposition\n')
        f.write(f'% Generated with {n_params} symbolic parameters, order-{order} expansion\n\n')
        
        f.write('\\section{Lagrange Multiplier}\n')
        f.write('The Lagrange multiplier $\\nu = (a^\\top G \\theta)/(a^\\top a)$:\n')
        f.write('\\begin{equation}\n')
        f.write(f'\\nu = {sp.latex(sp.simplify(nu))}\n')
        f.write('\\end{equation}\n\n')
        
        f.write('\\section{Constraint Gradient}\n')
        f.write('First few components of $a = \\nabla(h_1 + h_2)$:\n')
        f.write('\\begin{align}\n')
        for i in range(min(3, n_params)):
            expr = sp.simplify(a[i, 0])
            f.write(f'a_{{{i+1}}} &= {sp.latex(expr)} \\\\\n')
        f.write('\\end{align}\n\n')
        
        f.write('\\section{Key Result: Structural Identity Broken}\n')
        f.write('Unlike the local basis where $a = -\\theta/9$,\n')
        f.write('the su(9) pair basis has additional terms showing\n')
        f.write('the structural identity $G\\theta = -a$ is \\textbf{broken}.\n')
    
    files['latex'] = tex_file
    print(f"  ✓ Saved LaTeX: {tex_file}")
    
    return files


if __name__ == "__main__":
    files = export_symbolic_expressions(n_params=4, order=2)
    print("\nExported files:")
    for fmt, path in files.items():
        print(f"  {fmt}: {path}")

