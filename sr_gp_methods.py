# Libraries and modules
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from typing import Callable, Optional
from inspect import signature
from tqdm import tqdm

from pysr import PySRRegressor
from pysr_utils import best_equation

from fs_techniques import cmi_feature_selection, shap_feature_selection
from pysr_utils import train_val_test_split, fit_and_evaluate_best_equation, cumulative_lambda

def gp(
    X: pd.DataFrame,
    y: np.ndarray,
    loss_function: Callable,
    n_runs: int = 100,
    record_interval: int = 1,
    test_size: float = 0.2,
    val_size: float = 0.25,
    pysr_params: Optional[dict] = None
) -> tuple[dict, list, list, tuple[list, list, list, list, list, list]]:
    """
    Perform symbolic regression using PySR over multiple runs.

    For each run, this function:
      - Splits the data into train/validation/test sets.
      - Fits a symbolic regressor (PySR) to the training data.
      - Records training, validation, and test losses at each recording interval.
      - Stores the best equation found in each run.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : array-like
        Target vector.
    loss_function : Callable
        Function to compute the loss between true and predicted values.
    n_runs : int, optional
        Number of runs for model training and evaluation (default: 100).
    record_interval : int, optional
        Number of generations (iterations) between each recording of training, validation, and test losses.
        For example, if record_interval=2, losses are recorded every 2 generations. Default is 1.
    test_size : float, optional
        Proportion of data to use for testing (default: 0.2).
    val_size : float, optional
        Proportion of remaining data to use for validation (default: 0.25).
    pysr_params : dict, optional
        Parameters to pass to PySRRegressor (default: None).

    Returns
    -------
    results : dict
        Dictionary containing arrays of training, validation, and test losses for each run.
        Format: results['GP']['training_loss'][run, :], results['GP']['validation_loss'][run, :], 
        results['GP']['test_loss'][run, :].
    features : list of str
        List of feature names (column names) used in the regression.
    best_eqs : list
        List containing the best equation object from each run.
    train_val_test_sets_list : tuple of lists
        Tuple of six lists: (X_train_list, X_val_list, X_test_list, y_train_list, y_val_list, y_test_list),
        where each list contains the corresponding split for each run.

    Notes
    -----
    The function performs multiple independent runs with different train/val/test splits.
    Losses are recorded at each interval as determined by record_interval.
    """

    if pysr_params is None: pysr_params = {}
    method_name = "GP"

    # Determine the total number of iterations and how many times to record losses
    niterations = pysr_params.get("niterations", signature(PySRRegressor).parameters['niterations'].default)
    n_records = niterations // record_interval
    
    # Initialize variables
    results = { method_name: {
        "training_loss": np.empty((n_runs, n_records)),
        "validation_loss": np.empty((n_runs, n_records)),
        "test_loss": np.empty((n_runs, n_records))
    } }
    features = X.columns.to_list()
    best_eqs = []

    # Create six lists, one for each split
    X_train_list, X_val_list, X_test_list = [], [], []
    y_train_list, y_val_list, y_test_list = [], [], []

    print(f"{method_name+' ':-<20}")

    def _run_once():
        # Split data into train/val/test sets
        train_val_test_sets = train_val_test_split(X, y, test_size, val_size)

        # Fit and evaluate the model
        evaluation = fit_and_evaluate_best_equation(
            train_val_test_sets, loss_function, record_interval, pysr_params
        )

        return train_val_test_sets, evaluation

    # Use Parallel processing to run the function in parallel
    outputs = Parallel(n_jobs=-1)(
        delayed(_run_once)() for _ in tqdm(
            range(n_runs), 
            desc=f"{method_name} "
        )
    )

    # Unpack the outputs into the respective lists
    for run, (train_val_test_sets, evaluation) in enumerate(outputs):
        # Store each split separately
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_sets
        X_train_list.append(X_train)
        X_val_list.append(X_val)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_val_list.append(y_val)
        y_test_list.append(y_test)

        # Store results
        training_losses, validation_losses, test_losses, best_eq = evaluation
        results[method_name]["training_loss"][run] = training_losses
        results[method_name]["validation_loss"][run] = validation_losses
        results[method_name]["test_loss"][run] = test_losses
        best_eqs.append(best_eq)

    # Return as a tuple of lists for clarity
    train_val_test_sets_list = (X_train_list, X_val_list, X_test_list, y_train_list, y_val_list, y_test_list)
    return results, features, best_eqs, train_val_test_sets_list

def gpshap(
    X: pd.DataFrame,
    y: np.ndarray,
    loss_function: Callable,
    n_runs: int = 100,
    record_interval: int = 1,
    top_features_ratio: Optional[float] = None,
    gp_best_equations: Optional[list] = None,
    train_val_test_sets_list: Optional[tuple] = None,
    test_size: float = 0.2,
    val_size: float = 0.25,
    pysr_params: Optional[dict] = None
) -> tuple[dict, list, list]:

    if pysr_params is None: pysr_params = {}
    method_name = "GPSHAP"

    # Determine the total number of iterations and how many times to record losses
    niterations = pysr_params.get("niterations", signature(PySRRegressor).parameters['niterations'].default)
    n_records = niterations // record_interval
    
    # Initialize variables
    results = { method_name: {
        'training_loss': np.zeros((n_runs, n_records)),
        'validation_loss': np.zeros((n_runs, n_records)),
        'test_loss': np.zeros((n_runs, n_records))
    } }
    best_eqs = []
    
    # Check if gp_best_equations is provided and get its length
    n_equations = len(gp_best_equations) if gp_best_equations is not None else 0
    
    # Select top features using SHAP values
    selected_features, _ = shap_feature_selection(
        X, 
        y, 
        loss_function, 
        n_runs, 
        record_interval,
        test_size, 
        val_size,
        top_features_ratio,
        gp_best_equations,
        train_val_test_sets_list,
        pysr_params
    )

    print(f"{method_name+' ':-<20}")

    def _run_once():
        # Split data into train/val/test sets
        train_val_test_sets = train_val_test_split(X, y, test_size, val_size)

        # Fit and evaluate the model
        evaluation = fit_and_evaluate_best_equation(
            train_val_test_sets, loss_function, record_interval, pysr_params
        )

        return train_val_test_sets, evaluation
    
    # Use Parallel processing to run the function in parallel
    outputs = Parallel(n_jobs=-1)(
        delayed(_run_once)() for _ in tqdm(
            range(n_runs), 
            desc=f"{method_name} "
        )
    )

    # Unpack the outputs into the respective lists
    for run, (_, evaluation) in enumerate(outputs):
        # Store results
        training_losses, validation_losses, test_losses, best_eq = evaluation
        results[method_name]["training_loss"][run] = training_losses
        results[method_name]["validation_loss"][run] = validation_losses
        results[method_name]["test_loss"][run] = test_losses
        best_eqs.append(best_eq)

    return results, selected_features, best_eqs

def gpcmi(
    X: pd.DataFrame, 
    y: np.ndarray,
    loss_function: Callable,
    n_runs: int = 100,
    record_interval: int = 1,
    test_size: float = 0.2,
    val_size: float = 0.25,
    fs_params: Optional[dict] = {},
    pysr_params: Optional[dict] = None
) -> tuple[dict, list, list]:
    """
    Perform feature selection using conditional mutual information (CMI),
    then fit symbolic regression models on the selected features.

    For each run, this function:
      - Selects features using the CMI method.
      - Trains a PySR model using only the selected features.
      - Evaluates and records training and test losses.
      - Stores the best equation for each run.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : np.ndarray
        Target vector.
    loss_function : Callable
        Function to compute the loss between true and predicted values.
    n_runs : int, optional
        Number of runs for model training and evaluation (default: 100).
    record_interval : int, optional
        Number of generations (iterations) between each recording of training, validation, and test losses.
        For example, if record_interval=2, losses are recorded every 2 generations. Default is 1.
    test_size : float, optional
        Proportion of data to use for testing (default: 0.2).
    val_size : float, optional
        Proportion of remaining data to use for validation (default: 0.25).
    fs_params : dict, optional
        Parameters to pass to the MI feature selection function (default: None).
    pysr_params : dict, optional
        Parameters to pass to PySRRegressor (default: None).

    Returns
    -------
    results : dict
        Dictionary containing training and test losses for each run.
        Format: results[method][loss_type][run]
    selected_features : list of str
        List of selected feature names, ordered by importance.
    best_eqs : list
        List containing the best equation object from each run.
    """

    if pysr_params is None: pysr_params = {}
    if fs_params is None: fs_params = {}
    method_name = "gpCMI"

    # Determine the total number of iterations and how many times to record losses
    niterations = pysr_params.get("niterations", signature(PySRRegressor).parameters['niterations'].default)
    n_records = niterations // record_interval

    # Initialize variables
    results = { method_name: {
        "training_loss": np.zeros((n_runs, n_records)),
        "validation_loss": np.zeros((n_runs, n_records)),
        "test_loss": np.zeros((n_runs, n_records))
    } }
    selected_features_list = []
    best_eqs = []

    print(f"{method_name+" ":-<20}")

    for run in range(n_runs):
        # Progress indicator
        print(f". {run+1}\n" if ((run % 10 == 9) or (run + 1 == n_runs)) else ".", end="")

        # Split data into train/val/test sets
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, test_size, val_size)

        # Select features using CMI feature selection
        selected_features, _ = cmi_feature_selection(X_train, y_train, **fs_params)

        split_selected_features = (
            X_train[selected_features],
            X_val[selected_features],
            X_test[selected_features],
            y_train,
            y_val,
            y_test
        )

         # Fit and evaluate the model using only the selected features
        training_losses, validation_losses, test_losses, best_eq = fit_and_evaluate_best_equation(
            split_selected_features, loss_function, record_interval, pysr_params
        )

        # Store results for this run
        results[method_name]["training_loss"][run] = training_losses
        results[method_name]["validation_loss"][run] = validation_losses
        results[method_name]["test_loss"][run] = test_losses
        selected_features_list.append(selected_features)
        best_eqs.append(best_eq)

    return results, selected_features_list, best_eqs

def new_method(
    X: pd.DataFrame,
    y: np.ndarray,
    loss_function: Callable,
    n_submodels: int = 2,
    n_runs: int = 100,
    record_interval: int = 1,
    test_size: float = 0.2,
    val_size: float = 0.25,
    fs_params: Optional[dict] = None,
    pysr_params: Optional[dict] = None
) -> tuple[dict, list, list]:
    """
    Perform iterative feature selection and symbolic regression with residual fitting.

    For each run, this function:
      - Splits the data into train/val/test sets.
      - Iteratively selects features using CMI and fits a symbolic regressor to the residuals.
      - Stores the best equations and selected features for each part.
      - Aggregates training and test losses.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : np.ndarray
        Target vector.
    loss_function : Callable
        Function to compute the loss between true and predicted values.
    n_submodels : int, optional
        Number of submodels (parts) to fit in each run (default: 2).
    n_runs : int, optional
        Number of runs for model training and evaluation (default: 100).
    record_interval : int, optional
        Number of generations (iterations) between each recording of training, validation, and test losses.
        For example, if record_interval=2, losses are recorded every 2 generations. Default is 1.
    test_size : float, optional
        Proportion of data to use for testing (default: 0.2).
    val_size : float, optional
        Proportion of remaining data to use for validation (default: 0.25).
    fs_params : dict, optional
        Parameters to pass to the MI feature selection function (default: None).
    pysr_params : dict, optional
        Parameters to pass to PySRRegressor (default: None).

    Returns
    -------
    results : dict
        Dictionary containing training and test losses for each run.
        Format: results[method][loss_type][run]
    selected_features_list : list
        List of lists, each containing the selected feature names for each part in a run.
    best_eqs_list : list
        List of lists, each containing the best equation objects for each part in a run.
    """

    if pysr_params is None: pysr_params = {}
    if fs_params is None: fs_params = {}
    method_name = "gpNEW"

    # Determine the total number of iterations and how many times to record losses
    niterations = pysr_params.get("niterations", signature(PySRRegressor).parameters['niterations'].default)
    n_records = niterations // record_interval

    results = { method_name: {
        "training_loss": np.zeros((n_runs, n_records)),
        "validation_loss": np.zeros((n_runs, n_records)),
        "test_loss": np.zeros((n_runs, n_records))
    } }
    selected_features_list = []
    best_eqs_list = []

    print(f"{method_name+' ':-<20}")

    for run in range(n_runs):
        # Progress indicator
        print(f". {run+1}\n" if ((run % 10 == 9) or (run + 1 == n_runs)) else ".", end="")

        # Initialize variables for this run
        lambda_models = []
        run_best_eqs = []
        run_selected_features = []

        # Split data into train/val/test sets
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, test_size, val_size)
        
        y_train_residual = y_train.copy()
        y_val_residual   = y_val.copy()

        for sub_i in range(n_submodels):
            # Initialize the PySRRegressor with warm_start=True to allow iterative fitting
            model = PySRRegressor(warm_start=True, **pysr_params)
            model.set_params(niterations=record_interval) # Adjust the number of generations per interval

            low = int(n_records * sub_i / n_submodels)
            high = int(n_records * (sub_i + 1) / n_submodels)

            # Select features using CMI feature selection
            selected_features, _ = cmi_feature_selection(X_train, y_train_residual, **fs_params)

            X_train_curr = X_train[selected_features]
            X_val_curr = X_val[selected_features]
            X_test_curr = X_test[selected_features]

            # Compute cumulative predictions from current submodels
            train_cumulative_pred = cumulative_lambda(X_train, lambda_models)
            val_cumulative_pred = cumulative_lambda(X_val, lambda_models)
            test_cumulative_pred = cumulative_lambda(X_test, lambda_models)

            # Iteratively fit the model and record losses at each interval
            for interval_idx in range(low, high):
                # Fit the model for the current interval
                model.fit(X_train_curr, y_train_residual)

                # Cooncatenate the training and test sets
                X_check = pd.concat([X_train_curr, X_test_curr], ignore_index=True)

                # Select the best equation based on validation loss
                best_eq = best_equation(model, X_val_curr, y_val_residual, loss_function, X_check)

                assert best_eq is not None, "No valid equation found during fitting."
                lambda_expr = best_eq.lambda_format 

                # Compute losses on training, validation and test sets
                results[method_name]["training_loss"][run, interval_idx] = loss_function(
                    y_train, 
                    train_cumulative_pred + lambda_expr(X_train_curr)
                )
                results[method_name]["validation_loss"][run, interval_idx] = loss_function(
                    y_val, 
                    val_cumulative_pred + lambda_expr(X_val_curr)
                )
                results[method_name]["test_loss"][run, interval_idx] = loss_function(
                    y_test, 
                    test_cumulative_pred + lambda_expr(X_test_curr)
                )

            # Store the lambda model and selected features
            lambda_models.append((lambda_expr, selected_features))

            # Update residuals for next part
            y_train_residual -= lambda_expr(X_train_curr)
            y_val_residual -= lambda_expr(X_val_curr)

            # Store the best equation and selected features for this part
            run_best_eqs.append(best_eq)
            run_selected_features.append(selected_features)
        
        # Store results for this run
        selected_features_list.append(run_selected_features)
        best_eqs_list.append(run_best_eqs)

    return results, selected_features_list, best_eqs_list