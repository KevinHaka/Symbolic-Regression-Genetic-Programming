import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

def all_values_real(
    a: Iterable
) -> np.bool_:
    """
    Return True if every element of array-like `a` is a real, finite number.
    Real means imag == 0 (for complex types). Finite means not NaN or ±Inf.
    """

    arr = np.asarray(a)
    return np.isreal(arr).all() and np.isfinite(arr).all()

def organize_results(
    task_results: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Organizes results from a list of task runs.

    This function takes a list of result dictionaries (task_results)
    and reorganizes them into three main dictionaries (for losses,
    equations, and features), grouped by dataset name and method name.

    Args:
        task_results (List[Dict[str, Any]]): A list of dictionaries, where each
            dictionary contains the results of a single run
            (e.g., 'dataset_name', 'method_name', 'losses', etc.).

    Returns:
        Tuple[Dict, Dict, Dict]: A tuple containing three dictionaries:
        1.  `losses`: A dictionary of losses, nested by dataset and method.
        2.  `equations`: A dictionary of equations, nested by dataset and method.
        3.  `features`: A dictionary of features, nested by dataset and method.
    """

    # Identify unique dataset and method names.
    dataset_names = set(task_result['dataset_name'] for task_result in task_results)
    method_names = set(task_result['method_name'] for task_result in task_results)

    # Initialize the output structures.
    losses = {}
    equations = {}
    features = {}

    # Initialize dataset-level dictionaries.
    for dataset_name in dataset_names:
        losses[dataset_name] = {}
        equations[dataset_name] = {}
        features[dataset_name] = {}

        # Initialize method-level dictionaries.
        for method_name in method_names:
            losses[dataset_name][method_name] = {
                "training_losses": [],
                "validation_losses": [],
                "test_losses": [],
            }
            equations[dataset_name][method_name] = []
            features[dataset_name][method_name] = []

    # Iterate through the task results.
    for task_result in task_results:
        # Extract key information from the current result.
        dataset = task_result['dataset_name']
        method = task_result['method_name']
        
        # Assign the values to their respective structures.
        losses_data = task_result['losses']
        losses[dataset][method]['training_losses'].append(losses_data['training_losses'])
        losses[dataset][method]['validation_losses'].append(losses_data['validation_losses'])
        losses[dataset][method]['test_losses'].append(losses_data['test_losses'])
        equations[dataset][method].append(task_result['equations'])
        features[dataset][method].append(task_result['features'])

    return losses, equations, features

def flatten_dict(
    nested: Dict
) -> Dict:
    """
    Flatten a nested dict into a single-level dict where keys are tuple paths.

    Behavior:
    - Each leaf value is stored under a tuple key representing its path from the root.
    - Top-level leaves are stored with a 1-tuple key, e.g. ('x',).

    Parameters:
    - nested: A dict that may contain other dicts as values.

    Returns:
    - Dict with tuple keys (path from root) mapped to the corresponding leaf values.
    """

    flat = {}  # Accumulator for the flattened result

    # Iterate through each key-value pair in the current dict
    for key, value in nested.items():
        if isinstance(value, dict): # If the value is a dict, recurse into it
            sub_flat = flatten_dict(value)  # Recursively flatten the child dict

            # Incorporate the flattened sub-dict into the main flat dict
            for sub_key, sub_val in sub_flat.items():
                path = (key, *sub_key) # Create the full path tuple
                flat[path] = sub_val # Store the value under the full path key
        
        # If the value is not a dict, store it with a 1-tuple key
        else: flat[(key,)] = value
    return flat

def results_to_dataframe(
    results: Dict,
    epochs: Sequence
) -> pd.DataFrame:
    """
    Convert nested results dict into a pandas DataFrame with MultiIndex columns.

    Parameters:
    - results: Nested dict with structure results[dataset][method][metric] = list of arrays
    - epochs: Sequence of epoch values corresponding to the data points

    Returns:
    - pd.DataFrame with MultiIndex columns (dataset, method, metric) and MultiIndex rows (run, epoch)
    """

    data = flatten_dict(results) # Flatten the nested dict
    n_runs = len(next(iter(data.values()))) # Number of runs
    data = {k: np.concatenate(v) for k, v in data.items()} # Concatenate arrays for each key

    # Create MultiIndex for rows: (run, epoch)
    row_index = pd.MultiIndex.from_tuples([
        (run, epoch)
        for run in range(n_runs)
        for epoch in epochs
    ], names=['run', 'epoch'])

    df = pd.DataFrame(data, index=row_index) # Create DataFrame with flattened data and row index
    df.columns.names = ['dataset', 'method', 'metric'] # Name the column levels
    return df

def train_val_test_split(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: Optional[int] = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset into training, validation, and test sets.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : np.ndarray
        Target vector.
    test_size : float, optional
        Proportion of data to use for testing (default: 0.2).
    val_size : float, optional
        Proportion of data to use for validation (default: 0.2).
    random_state : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple
        Tuple containing (X_train, X_val, X_test, y_train, y_val, y_test).
    """

    # Initialize random number generator
    rng = np.random.default_rng(random_state)
    
    # Adjust validation size
    adjusted_val_size = val_size / (1 - test_size) 

    # Split the data into train/val/test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=rng.integers(0, 2**32)
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=adjusted_val_size, 
        random_state=rng.integers(0, 2**32)
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
