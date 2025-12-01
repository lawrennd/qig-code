"""
Caching system for expensive symbolic computations.

Symbolic expressions for su(9) with 80 parameters are expensive to compute
(~15 min for constraint gradient). This module provides caching to:
1. Compute once
2. Save to disk (pickled SymPy expressions)
3. Load instantly on subsequent runs

Related to CIP-0007: Analytic Forms for S and A via Lie Algebra Structure
"""

import os
import pickle
import hashlib
from pathlib import Path
from typing import Any, Callable, Optional
import sympy as sp


# Cache directory
CACHE_DIR = Path(__file__).parent / "_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _compute_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """
    Compute a unique cache key for a function call.
    
    Parameters
    ----------
    func_name : str
        Name of the function
    args : tuple
        Positional arguments
    kwargs : dict
        Keyword arguments
        
    Returns
    -------
    key : str
        Unique cache key (hex string)
    """
    # Convert arguments to string representation
    args_str = str(args)
    kwargs_str = str(sorted(kwargs.items()))
    
    # Hash the function signature
    signature = f"{func_name}:{args_str}:{kwargs_str}"
    key = hashlib.sha256(signature.encode()).hexdigest()[:16]
    
    return key


def cached_symbolic(func: Callable) -> Callable:
    """
    Decorator to cache symbolic computation results.
    
    Usage
    -----
    @cached_symbolic
    def expensive_symbolic_function(theta_symbols, order=2):
        # ... expensive computation ...
        return result
    
    The result is pickled and saved to disk. On subsequent calls with
    the same arguments, the cached result is loaded instantly.
    
    Notes
    -----
    - Cache files are stored in qig/symbolic/_cache/
    - Cache key is based on function name and arguments
    - To clear cache, delete files in _cache/ directory
    - Safe for SymPy expressions (picklable)
    
    Examples
    --------
    >>> @cached_symbolic
    ... def compute_something(n):
    ...     print("Computing...")
    ...     return sp.symbols(f'x1:{n+1}')
    >>> result1 = compute_something(5)  # Prints "Computing..."
    Computing...
    >>> result2 = compute_something(5)  # Loads from cache (instant)
    >>> result1 == result2
    True
    """
    def wrapper(*args, **kwargs):
        # Compute cache key
        func_name = func.__name__
        cache_key = _compute_cache_key(func_name, args, kwargs)
        cache_file = CACHE_DIR / f"{func_name}_{cache_key}.pkl"
        
        # Try to load from cache
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                print(f"[Cache] Loaded {func_name} from cache")
                return result
            except Exception as e:
                print(f"[Cache] Failed to load cache: {e}")
                # Fall through to recompute
        
        # Compute and save
        print(f"[Cache] Computing {func_name}... (this may take a while)")
        result = func(*args, **kwargs)
        
        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            print(f"[Cache] Saved {func_name} to cache: {cache_file.name}")
        except Exception as e:
            print(f"[Cache] Failed to save cache: {e}")
        
        return result
    
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def clear_cache(func_name: Optional[str] = None):
    """
    Clear cached symbolic computations.
    
    Parameters
    ----------
    func_name : str, optional
        If provided, clear only cache for this function.
        If None, clear all cache files.
        
    Examples
    --------
    >>> clear_cache('symbolic_constraint_gradient_su9_pair')  # Clear specific
    >>> clear_cache()  # Clear all
    """
    if func_name is None:
        # Clear all
        for cache_file in CACHE_DIR.glob("*.pkl"):
            cache_file.unlink()
            print(f"Removed: {cache_file.name}")
    else:
        # Clear specific function
        for cache_file in CACHE_DIR.glob(f"{func_name}_*.pkl"):
            cache_file.unlink()
            print(f"Removed: {cache_file.name}")


def cache_info() -> dict:
    """
    Get information about cached computations.
    
    Returns
    -------
    info : dict
        Dictionary mapping function names to list of cache files
        
    Examples
    --------
    >>> info = cache_info()
    >>> for func, files in info.items():
    ...     print(f"{func}: {len(files)} cached results")
    """
    from collections import defaultdict
    
    info = defaultdict(list)
    for cache_file in CACHE_DIR.glob("*.pkl"):
        # Parse function name from filename
        func_name = cache_file.stem.rsplit('_', 1)[0]
        info[func_name].append(cache_file.name)
    
    return dict(info)

