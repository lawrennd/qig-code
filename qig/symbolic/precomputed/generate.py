"""
Generate precomputed symbolic expressions in stages.

Run stages individually:
    python -m qig.symbolic.precomputed.generate --stage 1
    python -m qig.symbolic.precomputed.generate --stage 2
    ...

Or all at once (slower):
    python -m qig.symbolic.precomputed.generate --all
"""

import sys
import pickle
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent
CACHE_DIR = OUTPUT_DIR / "_cache"
CACHE_DIR.mkdir(exist_ok=True)


def stage1_basics():
    """Stage 1: Compute C and ψ."""
    import sympy as sp
    from sympy import symbols
    from qig.symbolic.lme_exact import exact_psi_lme, exact_constraint_lme
    
    print("Stage 1: Computing C and ψ...")
    a = symbols('a', real=True)
    c = symbols('c', real=True)
    theta = {'λ3⊗I': a, 'λ1⊗λ1': c}
    
    C = exact_constraint_lme(theta)
    psi = exact_psi_lme(theta)
    
    result = {'a': a, 'c': c, 'theta': theta, 'C': C, 'psi': psi}
    with open(CACHE_DIR / "stage1.pkl", 'wb') as f:
        pickle.dump(result, f)
    print("  ✓ Saved to stage1.pkl")
    return result


def stage2_derivatives():
    """Stage 2: Compute G and a_vec."""
    import sympy as sp
    from qig.symbolic.lme_exact import (
        exact_fisher_information_lme, exact_constraint_gradient_lme
    )
    
    print("Stage 2: Computing G and a_vec...")
    with open(CACHE_DIR / "stage1.pkl", 'rb') as f:
        s1 = pickle.load(f)
    
    theta_list = [s1['a'], s1['c']]
    G = exact_fisher_information_lme(s1['theta'], theta_list)
    a_vec = exact_constraint_gradient_lme(s1['theta'], theta_list)
    
    result = {**s1, 'theta_list': theta_list, 'G': G, 'a_vec': a_vec}
    with open(CACHE_DIR / "stage2.pkl", 'wb') as f:
        pickle.dump(result, f)
    print("  ✓ Saved to stage2.pkl")
    return result


def stage3_lagrange():
    """Stage 3: Compute ν and ∇ν."""
    from qig.symbolic.lme_exact import (
        exact_lagrange_multiplier_lme, exact_grad_lagrange_multiplier_lme
    )
    
    print("Stage 3: Loading stage2...", flush=True)
    with open(CACHE_DIR / "stage2.pkl", 'rb') as f:
        s2 = pickle.load(f)
    print("  loaded", flush=True)
    
    print("Stage 3a: Computing ν...", flush=True)
    nu = exact_lagrange_multiplier_lme(s2['a_vec'], s2['G'], s2['theta_list'])
    print("  ✓ ν done", flush=True)
    
    print("Stage 3b: Computing ∇ν (may take ~10s)...", flush=True)
    grad_nu = exact_grad_lagrange_multiplier_lme(
        s2['a_vec'], s2['G'], nu, s2['theta_list'], do_simplify=False
    )
    print("  ✓ ∇ν done", flush=True)
    
    result = {**s2, 'nu': nu, 'grad_nu': grad_nu}
    with open(CACHE_DIR / "stage3.pkl", 'wb') as f:
        pickle.dump(result, f)
    print("  ✓ Saved to stage3.pkl")
    return result


def stage4_jacobian():
    """Stage 4: Compute M, S, A."""
    from qig.symbolic.lme_exact import (
        exact_constraint_hessian_lme, exact_nabla_G_theta_lme
    )
    
    print("Stage 4: Loading stage3...", flush=True)
    with open(CACHE_DIR / "stage3.pkl", 'rb') as f:
        s3 = pickle.load(f)
    print("  loaded", flush=True)
    
    print("Stage 4a: Computing ∇²C...", flush=True)
    hess_C = exact_constraint_hessian_lme(s3['theta'], s3['theta_list'], do_simplify=False)
    print("  ✓ ∇²C done", flush=True)
    
    print("Stage 4b: Computing (∇G)[θ]...", flush=True)
    nabla_G_theta = exact_nabla_G_theta_lme(s3['G'], s3['theta_list'])
    print("  ✓ (∇G)[θ] done", flush=True)
    
    print("Stage 4c: Assembling M, S, A...", flush=True)
    M = -s3['G'] - nabla_G_theta + s3['nu'] * hess_C + s3['a_vec'] * s3['grad_nu'].T
    S = (M + M.T) / 2
    A = (M - M.T) / 2
    print("  ✓ M, S, A done", flush=True)
    
    result = {**s3, 'hess_C': hess_C, 'nabla_G_theta': nabla_G_theta, 
              'M': M, 'S': S, 'A': A}
    with open(CACHE_DIR / "stage4.pkl", 'wb') as f:
        pickle.dump(result, f)
    print("  ✓ Saved to stage4.pkl")
    return result


def export_python():
    """Export final results to Python file."""
    print("Exporting to Python file...")
    with open(CACHE_DIR / "stage4.pkl", 'rb') as f:
        data = pickle.load(f)
    
    output_file = OUTPUT_DIR / "two_param_chain.py"
    with open(output_file, 'w') as f:
        f.write('"""\n')
        f.write('Precomputed symbolic expressions for 2-parameter LME chain.\n')
        f.write('Parameters: a (local λ₃⊗I), c (entangling λ₁⊗λ₁)\n')
        f.write('"""\n\n')
        f.write('import sympy as sp\n')
        f.write('from sympy import symbols, Matrix, exp, log, sqrt, cosh, sinh, Rational\n\n')
        f.write('a = symbols("a", real=True)\n')
        f.write('c = symbols("c", real=True)\n')
        f.write('theta_list = [a, c]\n\n')
        
        for name in ['C', 'psi', 'nu']:
            f.write(f'{name} = {data[name]}\n\n')
        
        for name in ['G', 'a_vec', 'grad_nu', 'hess_C', 'nabla_G_theta', 'M', 'S', 'A']:
            f.write(f'{name} = Matrix({data[name].tolist()})\n\n')
    
    print(f"  ✓ Exported to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--stage":
        stage = int(sys.argv[2])
        if stage == 1:
            stage1_basics()
        elif stage == 2:
            stage2_derivatives()
        elif stage == 3:
            stage3_lagrange()
        elif stage == 4:
            stage4_jacobian()
        elif stage == 5:
            export_python()
    else:
        # Run all stages
        stage1_basics()
        stage2_derivatives()
        stage3_lagrange()
        stage4_jacobian()
        export_python()
        print("\n✓ All stages complete!")
