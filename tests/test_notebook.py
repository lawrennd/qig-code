#!/usr/bin/env python3
"""
Test script for Jupyter Notebooks

This script executes Jupyter notebooks and validates they complete successfully. 
It can be run with different configurations via environment variables.

Usage:
    # Test default notebook (examples/generate-origin-paper-figures.ipynb)
    python tests/test_notebook.py
    
    # Test specific notebook (relative to project root)
    python tests/test_notebook.py examples/generate-origin-paper-figures.ipynb
    
    # Test all notebooks in root directory and examples/
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

Note: Notebooks may be located in the project root directory or in examples/.
"""

import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    pytest = None
    HAS_PYTEST = False

# Check if jupyter nbconvert is available
import subprocess
try:
    subprocess.run(['jupyter', 'nbconvert', '--version'], 
                   capture_output=True, check=True)
    HAS_NBCONVERT = True
except (subprocess.CalledProcessError, FileNotFoundError):
    HAS_NBCONVERT = False

# Decorator that marks test as slow and skips if nbconvert not available
def slow_test(func):
    """Mark test as slow and skip if nbconvert not available."""
    if not HAS_PYTEST:
        return func
    
    # Apply both slow marker and skip condition
    func = pytest.mark.slow(func)
    func = pytest.mark.skipif(
        not HAS_NBCONVERT,
        reason="jupyter nbconvert not installed (install with: pip install nbconvert)"
    )(func)
    return func

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
        print("❌ jupyter nbconvert not found. Install with: pip install nbconvert")
        print("   Or: pip install jupyter[nbconvert]")
        return False, None

def check_notebook_outputs(notebook_path, require_pass_markers=False):
    """
    Check that notebook contains expected success markers.
    
    Parameters
    ----------
    notebook_path : str or Path
        Path to executed notebook
    require_pass_markers : bool, optional
        If True, require specific ✅ PASSED markers in outputs.
        If False, just check that notebook executed without errors.
        
    Returns
    -------
    success : bool
        True if validation criteria met
    """
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Check for execution errors
    error_count = 0
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'outputs' in cell:
            for output in cell['outputs']:
                if output.get('output_type') == 'error':
                    error_count += 1
                    print(f"\n❌ Error in cell: {output.get('ename', 'Unknown')}: {output.get('evalue', '')}")
    
    if error_count > 0:
        print(f"\n❌ Found {error_count} error(s) in notebook execution")
        return False
    
    if not require_pass_markers:
        print(f"\n✅ Notebook executed successfully with no errors")
        return True
    
    # Look for "PASSED" markers in outputs
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
            # Test all notebooks in root directory and examples/
            notebooks = sorted(project_root.glob('*.ipynb'))
            notebooks.extend(sorted((project_root / 'examples').glob('*.ipynb')))
            if not notebooks:
                print("❌ No notebooks found in project root or examples/")
                sys.exit(1)
            print(f"Found {len(notebooks)} notebook(s) to test:")
            for nb in notebooks:
                print(f"  - {nb.relative_to(project_root)}")
            print()
        else:
            # Test specific notebook
            notebook_path = project_root / sys.argv[1]
            notebooks = [notebook_path]
    else:
        # Default: test the main validation notebook in examples/
        notebook_path = project_root / "examples" / "generate-origin-paper-figures.ipynb"
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
        # Figure generation notebooks don't need PASSED markers
        require_markers = 'test' in notebook_path.name.lower() or 'validation' in notebook_path.name.lower()
        if not check_notebook_outputs(output_path, require_pass_markers=require_markers):
            print(f"\n❌ TEST FAILED: Validation failed for {notebook_path.name}")
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

@slow_test
def test_default_notebook():
    """Pytest-compatible test for the default notebook.
    
    This test is SKIPPED BY DEFAULT because:
    - It requires jupyter nbconvert
    - It takes several minutes to run
    - It's primarily for CI/CD validation
    
    To run this test:
      pytest tests/test_notebook.py::test_default_notebook -v
    
    Or run the standalone script:
      python tests/test_notebook.py
    """
    if not HAS_PYTEST:
        raise ImportError("pytest is required to run this test")
    
    if not HAS_NBCONVERT:
        pytest.skip("jupyter nbconvert not installed")
    
    # Find notebook - look in project root (parent of tests/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    notebook_path = project_root / "examples" / "generate-origin-paper-figures.ipynb"
    
    if not notebook_path.exists():
        pytest.skip(f"Notebook not found: {notebook_path}")
    
    # Execute notebook
    success, output_path = run_notebook(notebook_path)
    assert success, f"Notebook execution failed: {notebook_path.name}"
    
    # Check outputs (figure generation notebook, no pass markers required)
    assert check_notebook_outputs(output_path, require_pass_markers=False), \
        f"Notebook validation failed: {notebook_path.name}"

if __name__ == "__main__":
    main()

