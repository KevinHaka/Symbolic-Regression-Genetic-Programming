import time
import random
import warnings

import numpy as np

from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

def timeit(
    func: Callable, 
    *args: Any, 
    n_runs: int = 1, 
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Measure execution time of a callable.

    Behavior
    --------
    - If n_runs == 1: runs once and returns a dict with the single run time and the function results.
    - If n_runs > 1: runs multiple times and returns timing statistics (no function results).

    Parameters
    ----------
    func : callable
        Function to execute.
    *args :
        Positional arguments passed to func.
    n_runs : int, default=1
        Number of executions.
    **kwargs :
        Keyword arguments passed to func.

    Returns
    -------
    dict
        Single run:
            {
                'time': np.float64,
                'results': Any
            }

        Multiple runs:
            {
                'average_time': np.float64,
                'std_time': np.float64,
                'all_times': np.ndarray,
            }
    """

    times: np.ndarray = np.empty(n_runs, dtype=np.float64)

    # Execute n_runs times
    for i in range(n_runs):
        start = time.perf_counter()
        results = func(*args, **kwargs)
        times[i] = time.perf_counter() - start

    if n_runs == 1:
        return {
            'time': times[0],
            'results': results
        }

    return {
        'average_time': times.mean(),
        'std_time': times.std(),
        'all_times': times
    }

@contextmanager
def temporary_seed(
    seed: Optional[int] = None
):
    """
    Context manager that temporarily sets the global random seed for
    both `numpy` (legacy RNG) and Python's `random` module,
    and restores the previous state upon exit.
    """

    # If seed is None, do nothing
    if seed is None:
        yield
        return

    # Save current states
    old_np_state = np.random.get_state()
    old_random_state = random.getstate()
    
    # Set new seed
    np.random.seed(seed)
    random.seed(seed)
    
    try: yield
    finally: # Restore previous states
        np.random.set_state(old_np_state)
        random.setstate(old_random_state)

def warnings_manager(
    func: Callable,
    filters: Optional[List[dict]] = None,
    *args,
    **kwargs
) -> Any:
    """
    Execute a function with flexible warning filters.
    
    Parameters
    ----------
    func : Callable
        The function to execute
    filters : list of dict, optional
        List of filter dictionaries with keys:
        - 'action': str (required) - 'ignore', 'error', 'always', etc.
        - 'message': str (optional) - regex pattern for message
        - 'category': Warning class (optional) - warning category
        - 'module': str (optional) - regex pattern for module
    *args, **kwargs
        Arguments to pass to func
    
    Returns
    -------
    Any
        The return value of the executed function
    """

    # Set up warning filters
    with warnings.catch_warnings():
        if filters: # If filters are provided
            for filter_spec in filters: # Apply each filter
                action = filter_spec.get('action', 'ignore')
                message = filter_spec.get('message', '')
                category = filter_spec.get('category', Warning)
                module = filter_spec.get('module', '')
                
                warnings.filterwarnings(
                    action=action,
                    message=message,
                    category=category,
                    module=module,
                )
        
        return func(*args, **kwargs)