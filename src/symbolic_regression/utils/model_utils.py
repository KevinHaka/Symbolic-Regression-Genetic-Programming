import os
import dill

import numpy as np
import pandas as pd

from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Tuple

from pysr import PySRRegressor
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
    record_interval: Optional[int] = 1,
    resplit_interval: Optional[int] = None,
    pysr_params: Dict[str, Any] = {}
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
    record_interval : int, optional
        Number of epochs (iterations) between each recording of training, validation, and test losses.
        For example, if record_interval=2, losses are recorded every 2 epochs. Default is 1.
    resplit_interval : int, optional
        Number of epochs (iterations) between each resplitting of the training and validation sets.
        For example, if resplit_interval=3, the training and validation sets are resplit every 3 epochs.
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

    # Total iterations from params
    niterations = pysr_params.get("niterations", signature(PySRRegressor).parameters['niterations'].default)

    # Random state for reproducibility
    random_state = pysr_params.get("random_state", None)
    rng = np.random.default_rng(random_state)
    
    # Ensure niterations is an integer
    assert isinstance(niterations, int), "niterations must be an integer."

    # Set default intervals if not provided
    if record_interval is None: record_interval = niterations
    if resplit_interval is None: resplit_interval = niterations

    # Build event schedule
    record_points = list(range(record_interval, niterations + 1, record_interval))
    resplit_points = list(range(resplit_interval, niterations, resplit_interval))
    events: Dict[int, set] = {}
    for it in record_points: events.setdefault(it, set()).add("record")
    for it in resplit_points: events.setdefault(it, set()).add("resplit")

    # Arrays sized by number of record events
    n_records = len(record_points)

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
            X_check = pd.concat([X_train, X_test], ignore_index=True)

            # Select the best equation based on validation loss
            best_eq = best_equation(model, X_val, y_val, loss_function, X_check)
            best_eqs.append(best_eq)
            lambda_expr = best_eq.lambda_format

            # Compute losses on training, validation and test sets
            training_losses[record_idx] = loss_function(y_train, lambda_expr(X_train))
            validation_losses[record_idx] = loss_function(y_val, lambda_expr(X_val))
            test_losses[record_idx] = loss_function(y_test, lambda_expr(X_test))
            
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
    dataset_name: str, 
    method_name: str, 
    run: int, 
    train_val_test_set: Tuple, 
    method: BaseMethod, 
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
    dataset_name : str
        The name of the dataset being processed.
    method_name : str
        The name of the symbolic regression method being used.
    run : int
        The current run number for identification.
    train_val_test_set : Tuple
        A tuple containing the training, validation, and test data splits.
        Expected format: (X_train, X_val, X_test, y_train, y_val, y_test).
    method : BaseMethod
        An instance of a class that inherits from `BaseMethod` and implements
        the `run` method.
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