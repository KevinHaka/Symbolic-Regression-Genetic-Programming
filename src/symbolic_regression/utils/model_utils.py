import os
import dill

import numpy as np
import pandas as pd
import sympy as sp

from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pysr import PySRRegressor
from pysr.export_numpy import CallableEquation
from sklearn.model_selection import train_test_split

from ..methods.base import BaseMethod
from .data_utils import all_values_real

def best_equation(
    model: PySRRegressor, 
    X: pd.DataFrame,
    y: np.ndarray,
    loss_function: Callable[[np.ndarray, np.ndarray], float],
    X_check: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    Select the best equation from a fitted PySRRegressor.

    Workflow
    --------
    - For each candidate equation in `model.equations_`:
      - Evaluate `lambda_format(X)`. If any prediction is non-real or non-finite,
        discard the equation.
      - If `X_check` is provided, also require real and finite predictions on
        `X_check`.
      - Compute `loss = loss_function(y, y_pred)` and store it in the `'loss'`
        column.
    - Keep only the subset of equations that forms a strictly decreasing loss
      frontier with respect to their original order (each kept row has a lower
      loss than any previously kept row).
    - For each consecutive pair on this frontier (i > 0), compute the score:
          score_i = (log(loss_{i-1}) - log(loss_i)) / (complexity_i - complexity_{i-1})
    - Return the row (equation) with the maximum score.

    Parameters
    ----------
    model : PySRRegressor
        Trained symbolic regression model.
    X : pandas.DataFrame
        Feature matrix used to evaluate candidate equations.
    y : np.ndarray
        True target values aligned with `X`.
    loss_function : Callable[[np.ndarray, np.ndarray], float]
        Function that computes a scalar loss between true and predicted values.
    X_check : pandas.DataFrame, optional
        Additional dataset on which candidate equations must also produce real,
        finite predictions.

    Returns
    -------
    pandas.Series
        The selected equation row from `model.equations_` (with computed `'loss'`
        and `'score'` fields for the kept subset).
    """

    # Ensure that model.equations_ is a pandas DataFrame
    assert isinstance(model.equations_, pd.DataFrame), "model.equations_ is not a DataFrame. The model may not be fitted yet."
    
    # Create a copy to avoid modifying the original
    df_copy = model.equations_.copy()
    mask = [False]*len(df_copy) # Mask to track valid equations

    # Evaluate each equation
    for idx in range(len(df_copy)):
        eq = df_copy.iloc[idx]
        y_pred = eq.lambda_format(X)

        # Check if all predicted values are real
        if all_values_real(y_pred):
            # If X_check is provided, ensure predictions are real on that set too
            if X_check is not None:
                y_check_pred = eq.lambda_format(X_check)

                if not all_values_real(y_check_pred): continue

            # If all values are real, compute the loss
            mask[idx] = True
            df_copy.at[idx, 'loss'] = loss_function(y, y_pred)

    # Filter out equations that did not produce real values
    df_filtered = df_copy.loc[mask].copy().reset_index(drop=True)
    mask = [True]+[False]*(len(df_filtered)-1) # Mask to track equations with decreasing loss
    losses = df_filtered['loss'] # Extract losses
    min_loss = losses.iloc[0] # Initialize minimum loss

    # Select equations with strictly decreasing loss
    for idx in range(1, len(df_filtered)):
        current_loss = losses[idx]

        # If current loss is less than the minimum loss, keep it
        if current_loss < min_loss:
            mask[idx] = True
            min_loss = current_loss

    # Final filtering
    df_final = df_filtered.loc[mask].copy().reset_index(drop=True)

    # Compute score based on loss improvement and complexity increase
    for idx in range(1, len(df_final)):
        prev_loss = df_final.at[idx-1, 'loss']
        curr_loss = df_final.at[idx, 'loss']

        prev_complexity = df_final.at[idx-1, 'complexity']
        curr_complexity = df_final.at[idx, 'complexity']

        df_final.at[idx, 'score'] = (np.log(prev_loss) - np.log(curr_loss)) / (curr_complexity - prev_complexity) # type: ignore

    best_row_idx = df_final['score'].idxmax() # Find index of the best equation
    best_row = df_final.loc[best_row_idx] # Get the best equation row
    
    # Ensure that best_row is a pandas Series.
    assert isinstance(best_row, pd.Series), "Best row is not a Series."
    return best_row 

def fit_and_evaluate_best_equation(
    train_val_test_set: Tuple[
        pd.DataFrame, # X_train
        pd.DataFrame, # X_val
        pd.DataFrame, # X_test
        np.ndarray,   # y_train
        np.ndarray,   # y_val
        np.ndarray    # y_test
    ],
    loss_function: Callable,
    record_interval: Optional[int],
    resplit_interval: Optional[int],
    pysr_params: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.Series]]:
    """
    Train a PySRRegressor iteratively (warm-start) and record losses using the
    "best" discovered equation at configurable recording intervals.

    Parameters
    ----------
    train_val_test_set : tuple
        Tuple containing (X_train, X_val, X_test, y_train, y_val, y_test).
    loss_function : Callable
        Function to compute the loss between true and predicted values.
    record_interval : Optional[int]
        Number of iterations between recoridings. 
    resplit_interval : Optional[int]
        Number of iterations between each re-splitting of the training and 
        validation sets. The data is re-shuffled and split every 
        `resplit_interval` iterations to reduce overfitting.
    pysr_params : dict, optional
        Parameters to pass to PySRRegressor (default: None).

    Returns
    -------
    training_losses : np.ndarray
        Array of training losses at each recording interval using the best equation.
    validation_losses : np.ndarray
        Array of validation losses at each recording interval using the best equation.
    test_losses : np.ndarray
        Array of test losses at each recording interval using the best equation.
    best_eqs : list[pd.Series]
        List of the best equations selected at each recording interval.
    """

    # Set default parameters if not provided
    if pysr_params is None: pysr_params = {}

    # Get default niterations from PySRRegressor if not provided
    niterations = pysr_params.get("niterations", signature(PySRRegressor).parameters['niterations'].default)

    assert isinstance(niterations, int) and niterations > 0, "niterations must be a positive integer."

    # Set record and resplit intervals, defaulting to niterations if not provided
    record_interval = record_interval if record_interval else niterations
    resplit_interval = resplit_interval if resplit_interval else niterations

    # Calculate total number of records based on niterations and record_interval
    n_records = niterations // record_interval

    # Handle the case where no features are selected (empty DataFrame)
    if train_val_test_set[0].empty:
        # Get complexity_of_constants from pysr_params or default
        complexity_of_constants = pysr_params.get(
            "complexity_of_constants", 
            signature(PySRRegressor).parameters['complexity_of_constants'].default
        )

        return _create_mean_baseline(
            y_train=train_val_test_set[3],
            y_val=train_val_test_set[4],
            y_test=train_val_test_set[5],
            loss_function=loss_function,
            complexity_of_constants=complexity_of_constants,
            n_records=n_records,
        )

    # Random state for reproducibility
    random_state = pysr_params.get("random_state", None)
    rng = np.random.default_rng(random_state)

    # Determine record and resplit points
    record_points = list(range(record_interval, niterations + 1, record_interval))
    resplit_points = list(range(resplit_interval, niterations, resplit_interval))

    # Build event schedule for recording and resplitting
    events: Dict[int, set] = {}
    for it in record_points: events.setdefault(it, set()).add("record")
    for it in resplit_points: events.setdefault(it, set()).add("resplit")

    # Initialize arrays to store losses at each interval
    training_losses = np.empty(n_records, dtype=float)
    validation_losses = np.empty(n_records, dtype=float)
    test_losses = np.empty(n_records, dtype=float)
    best_eqs = [] # To store the best equations at each recording interval

    # Unpack the train/val/test splits
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_set

    # Initialize the PySRRegressor with warm_start=True to allow iterative fitting
    model = PySRRegressor(warm_start=True, **pysr_params)

    prev_iter = 0 # To track the previous iteration count
    record_idx = 0 # To track the index in the losses arrays

    # Process events in chronological order
    for it in sorted(events.keys()):
        step = it - prev_iter # Number of iterations to run since last event

        model.set_params(niterations=step) # Update niterations for this step
        model.fit(X_train, y_train) # Fit the model for the current step
        
        prev_iter = it # Update previous iteration count
        actions = events[it] # Actions to perform at this iteration

        # If recording is scheduled, evaluate and store losses
        if "record" in actions:
            # Concatenate the training and test sets
            if X_test.empty: X_check = X_train
            else: X_check = pd.concat([X_train, X_test], ignore_index=True)

            # Select the best equation based on validation loss
            best_eq = best_equation(model, X_val, y_val, loss_function, X_check)
            best_eqs.append(best_eq)
            lambda_expr = best_eq.lambda_format

            # Compute losses on training, validation and test sets
            training_losses[record_idx] = loss_function(y_train, lambda_expr(X_train))
            validation_losses[record_idx] = loss_function(y_val, lambda_expr(X_val))
            test_losses[record_idx] = loss_function(y_test, lambda_expr(X_test)) if len(X_test) > 0 else np.nan # Handle case with empty test set
            
            record_idx += 1 # Increment record index

        # If resplitting is scheduled, create new train/val splits
        if "resplit" in actions:
            # Concatenate the training and validation sets
            X_train_val = pd.concat([X_train, X_val], ignore_index=True)
            y_train_val = np.concatenate([y_train, y_val], axis=0)

            # Compute adjusted train size and split the data into new train/val sets
            adjusted_train_size = len(y_train)/(len(y_train_val))
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, 
                train_size=adjusted_train_size, 
                random_state=rng.integers(0, 2**32)
            )

    return training_losses, validation_losses, test_losses, best_eqs

def process_task(
    method_class: Callable[..., BaseMethod],
    method_params: Dict[str, Any],
    dataset_name: str, 
    method_name: str, 
    run: int, 
    train_val_test_set: Tuple, 
    output_dir: str,
    return_results: bool = False,
    random_state: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Executes a single run of a symbolic regression method on a given dataset.

    This function serves as a wrapper to call the `run` method of a specific
    symbolic regression approach (`BaseMethod`) and formats the output into a
    dictionary for later aggregation and analysis.

    Parameters
    ----------
    method_class : BaseMethod
        The class of the symbolic regression method to be instantiated.
    method_params : Dict[str, Any]
        Parameters to be passed to the method class constructor.
    dataset_name : str
        The name of the dataset being processed.
    method_name : str
        The name of the symbolic regression method being used.
    run : int
        The current run number for identification.
    train_val_test_set : Tuple
        A tuple containing the training, validation, and test data splits.
        Expected format: (X_train, X_val, X_test, y_train, y_val, y_test).
    output_dir : str
        The directory where the output files will be saved.
    return_results : bool
        Flag that determines whether to return the results dictionary.
    random_state : int or None
        Random state for reproducibility.

    Returns
    -------
    Dict[str, Any] or None
        If `return_results` is True, returns a dictionary containing:
        - 'dataset_name': str
        - 'method_name': str
        - 'run': int
        - 'losses': Dict[str, np.ndarray]
        - 'equations': List[pd.Series]
        - 'features': List[Any]
        Otherwise, results are only saved to disk as pickle files and None is returned.
    """

    # Set random state for reproducibility
    rng = np.random.default_rng(random_state)

    # Instantiate the method class with the provided parameters
    method = method_class(**method_params)

    # Set the random state in the method's PySR parameters
    pysr_rs = rng.integers(0, 2**32) if method.pysr_params.get("deterministic") else None
    method.pysr_params["random_state"] = pysr_rs

    # Run the method on the provided dataset splits
    temp_losses, temp_best_eqs, temp_features = method.run(train_val_test_set, int(rng.integers(0, 2**32)))

    # Organize the results into a dictionary
    results = {
        'dataset_name': dataset_name,
        'method_name': method_name,
        'run': run,
        'losses': {
            'training_losses': temp_losses[0],
            'validation_losses': temp_losses[1],
            'test_losses': temp_losses[2],
        },
        'equations': temp_best_eqs,
        'features': temp_features
    }

    # create a data directory if it doesn't exist
    final_dir = os.path.join(output_dir, dataset_name, method_name)
    os.makedirs(final_dir, exist_ok=True)

    # Save the results to a file
    filename = f"run_{run}.pkl"
    with open(os.path.join(final_dir, filename), "wb") as f: dill.dump(results, f)

    # Return results if specified
    if return_results: return results

def _create_mean_baseline(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    loss_function: Callable[[np.ndarray, np.ndarray], np.float64],
    complexity_of_constants: Optional[Union[int, float]],
    n_records: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.Series]]:
    """
    Create a baseline model that predicts the mean of y_train for all samples.
    
    This function is used when no features are selected or available, returning
    a constant model that predicts the training set mean. The losses are 
    replicated across all recording intervals.
    
    Parameters
    ----------
    y_train : np.ndarray
        Training target values.
    y_val : np.ndarray
        Validation target values.
    y_test : np.ndarray
        Test target values.
    loss_function : Callable
        Function to compute the loss between true and predicted values.
    complexity_of_constants : int or float or None
        Complexity value assigned to constant.
    n_records : int
        Number of recording intervals for which to replicate losses.
    
    Returns
    -------
    training_losses : np.ndarray
        Training losses replicated n_records times.
    validation_losses : np.ndarray
        Validation losses replicated n_records times.
    test_losses : np.ndarray
        Test losses replicated n_records times.
    best_eqs : List[pd.Series]
        Single-element list containing the baseline equation metadata.
    """
    
    # Compute mean once and reuse
    mean_value = y_train.mean()
    y_train_pred = np.full_like(y_train, mean_value)
    y_val_pred = np.full_like(y_val, mean_value)
    y_test_pred = np.full_like(y_test, mean_value)
    
    # Compute losses for all splits
    training_losses = np.ones(n_records) * loss_function(y_train, y_train_pred)
    validation_losses = np.ones(n_records) * loss_function(y_val, y_val_pred)
    test_losses = np.ones(n_records) * loss_function(y_test, y_test_pred)
    
    # Create baseline equation metadata
    best_eqs = [pd.Series({
        "complexity": 1 if complexity_of_constants is None else complexity_of_constants,
        "loss": loss_function(y_train, y_train_pred),
        "equation": str(mean_value),
        "sympy_format": sp.Float(mean_value),
        "lambda_format": CallableEquation(mean_value, [])
    }) for _ in range(n_records)]
    
    return training_losses, validation_losses, test_losses, best_eqs