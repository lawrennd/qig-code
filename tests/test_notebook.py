#!/usr/bin/env python3
"""
Test script for Jupyter Notebooks

This script executes Jupyter notebooks from the project root and validates they 
complete successfully. It can be run with different configurations via environment variables.

Usage:
    # Test default notebook (generate-paper-figures.ipynb)
    python tests/test_notebook.py
    
    # Test specific notebook
    python tests/test_notebook.py quantum_qutrit_experiments.ipynb
    
    # Test all notebooks in root directory
    python tests/test_notebook.py --all
    
    # With custom configuration
    QUTRIT_DIM=4 DYNAMICS_POINTS=10 python tests/test_notebook.py
    
    # Quick test mode
    DYNAMICS_POINTS=5 DYNAMICS_T_MAX=0.5 python tests/test_notebook.py

Environment Variables:
    QUTRIT_DIM: Qutrit dimension (default: 3)
    QUBIT_DIM: Qubit dimension (default: 2)
    N_PAIRS: Number of pairs (default: 1)
    DYNAMICS_POINTS: Number of integration points (default: 20)
    DYNAMICS_T_MAX: Maximum integration time (default: 2.0)
    RANDOM_SEED_1: Random seed for test 1 (default: 42)
    RANDOM_SEED_2: Random seed for test 2 (default: 123)
    RANDOM_SEED_3: Random seed for test 3 (default: 456)
    THETA_SCALE: Parameter scaling (default: 0.5)
    TOLERANCE: Numerical tolerance (default: 1e-6)

Note: All notebooks are located in the project root directory.
"""

import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path

def run_notebook(notebook_path, output_path=None):
    """
    Execute a Jupyter notebook using nbconvert.
    
    Parameters
    ----------
    notebook_path : str or Path
        Path to input notebook
    output_path : str or Path, optional
        Path to save executed notebook (temp file if not provided)
        
    Returns
    -------
    success : bool
        True if notebook executed successfully
    output : str
        Path to executed notebook
    """
    notebook_path = Path(notebook_path)
    
    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return False, None
    
    # Create temp file if no output specified
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = Path(temp_dir) / f"{notebook_path.stem}_executed.ipynb"
    else:
        output_path = Path(output_path)
    
    # Execute notebook
    cmd = [
        'jupyter', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        '--ExecutePreprocessor.timeout=600',
        '--output', str(output_path),
        str(notebook_path)
    ]
    
    print(f"Executing notebook: {notebook_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"\nConfiguration from environment:")
    config_vars = [
        'QUTRIT_DIM', 'QUBIT_DIM', 'N_PAIRS', 
        'DYNAMICS_POINTS', 'DYNAMICS_T_MAX',
        'RANDOM_SEED_1', 'RANDOM_SEED_2', 'RANDOM_SEED_3',
        'THETA_SCALE', 'TOLERANCE'
    ]
    for var in config_vars:
        value = os.getenv(var, 'default')
        print(f"  {var}={value}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Notebook executed successfully!")
        return True, output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Notebook execution failed!")
        print(f"\nStderr:\n{e.stderr}")
        if e.stdout:
            print(f"\nStdout:\n{e.stdout}")
        return False, None
    except FileNotFoundError:
        print("❌ jupyter not found. Install with: pip install jupyter nbconvert")
        return False, None

def check_notebook_outputs(notebook_path):
    """
    Check that notebook contains expected success markers.
    
    Parameters
    ----------
    notebook_path : str or Path
        Path to executed notebook
        
    Returns
    -------
    success : bool
        True if all expected outputs found
    """
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Look for "PASSED" in outputs
    passed_count = 0
    failed_count = 0
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'outputs' in cell:
            for output in cell['outputs']:
                if 'text' in output:
                    text = ''.join(output['text'])
                    if '✅ PASSED' in text:
                        passed_count += 1
                    if '❌ FAILED' in text:
                        failed_count += 1
    
    print(f"\nResults:")
    print(f"  ✅ Passed: {passed_count}")
    print(f"  ❌ Failed: {failed_count}")
    
    # Expect 4 experiments
    expected_passes = 4
    if passed_count >= expected_passes and failed_count == 0:
        print(f"\n✅ All {expected_passes} experiments passed!")
        return True
    else:
        print(f"\n❌ Expected {expected_passes} passes, got {passed_count}")
        return False

def main():
    """Run the notebook test suite."""
    print("="*70)
    print("NOTEBOOK INTEGRATION TEST")
    print("="*70)
    print()
    
    # Find notebook(s) - look in project root (parent of tests/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Check if specific notebook requested via command line
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            # Test all notebooks in root directory
            notebooks = sorted(project_root.glob('*.ipynb'))
            if not notebooks:
                print("❌ No notebooks found in project root")
                sys.exit(1)
            print(f"Found {len(notebooks)} notebook(s) to test:")
            for nb in notebooks:
                print(f"  - {nb.name}")
            print()
        else:
            # Test specific notebook
            notebook_path = project_root / sys.argv[1]
            notebooks = [notebook_path]
    else:
        # Default: test the main validation notebook
        notebook_path = project_root / "generate-paper-figures.ipynb"
        notebooks = [notebook_path]
    
    # Run all notebooks
    all_success = True
    for notebook_path in notebooks:
        print(f"\n{'='*70}")
        print(f"Testing: {notebook_path.name}")
        print(f"{'='*70}\n")
        
        # Execute notebook
        success, output_path = run_notebook(notebook_path)
        
        if not success:
            print(f"\n❌ TEST FAILED: {notebook_path.name} execution error")
            all_success = False
            continue
        
        # Check outputs
        if not check_notebook_outputs(output_path):
            print(f"\n❌ TEST FAILED: Not all experiments passed in {notebook_path.name}")
            all_success = False
            continue
        
        print(f"\n✅ {notebook_path.name} passed!")
    
    if all_success:
        print("\n" + "="*70)
        print("✅✅✅ ALL TESTS PASSED ✅✅✅")
        print("="*70)
        sys.exit(0)
    else:
        print("\n" + "="*70)
        print("❌ SOME TESTS FAILED")
        print("="*70)
        sys.exit(1)

if __name__ == "__main__":
    main()

