import datetime
import os
import pickle
import random
import time

from typing import Iterable
from contextlib  import contextmanager

import pandas as pd
import numpy as np
import sympy as sp

from sklearn.model_selection import train_test_split
from pysr import PySRRegressor

from ..methods.base import BaseMethod

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from typing import Optional, Callable, Dict, List, Tuple, Any
from inspect import signature
import smtplib
from email.message import EmailMessage

def mse_loss(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> np.float64:
    """
    Computes the Mean Squared Error (MSE) between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns
    -------
    np.float64
        The MSE value.
    """
    return ((y_true - y_pred) ** 2).mean()

def rmse_loss(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> np.float64:
    """
    Computes the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns
    -------
    np.float64
        The RMSE value.
    """
    return np.sqrt(mse_loss(y_true, y_pred))

def nrmse_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = 'range',
    iqr_quantiles: Tuple[float, float] = (25.0, 75.0)
) -> np.float64:
    """
    Computes the Normalized Root Mean Squared Error (NRMSE) between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    method : str
        Normalization method to use:
            - 'mean': Normalizes by the mean of y_true.
            - 'range': Normalizes by the range (max - min) of y_true.
            - 'std': Normalizes by the standard deviation of y_true.
            - 'iqr': Normalizes by the interquantile range (IQR) defined by iqr_quantiles.
    iqr_quantiles : tuple(float, float), optional
        Percentiles for the IQR when method == 'iqr'. Defaults to (25.0, 75.0).

    Returns
    -------
    np.float64
        The NRMSE value.
    """
    rmse = rmse_loss(y_true, y_pred)
    method = method.lower()

    match method:
        case 'mean':
            denominator = y_true.mean()

        case 'range':
            denominator = y_true.max() - y_true.min()

        case 'std':
            denominator = y_true.std()

        case 'iqr':
            q_low, q_high = float(iqr_quantiles[0]), float(iqr_quantiles[1])

            if not (0.0 <= q_low < q_high <= 100.0):
                raise ValueError("iqr_quantiles must satisfy 0 <= low < high <= 100")
            
            denominator = np.percentile(y_true, q_high) - np.percentile(y_true, q_low)

        case _:
            raise ValueError(f"Unknown normalization method: {method}")

    return rmse / denominator

def all_values_real(a: Iterable) -> np.bool_:
    """
    Return True if every element of array-like `a` is a real, finite number.
    Real means imag == 0 (for complex types). Finite means not NaN or ±Inf.
    """
    arr = np.asarray(a)

    return np.isreal(arr).all() and np.isfinite(arr).all()

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
    mask2 = [True]+[False]*(len(df_filtered)-1) # Mask to track equations with decreasing loss
    losses = df_filtered['loss'] # Extract losses
    min_loss = losses.iloc[0] # Initialize minimum loss

    # Select equations with strictly decreasing loss
    for idx in range(1, len(df_filtered)):
        current_loss = losses[idx]

        # If current loss is less than the minimum loss, keep it
        if current_loss < min_loss:
            mask2[idx] = True
            min_loss = current_loss

    # Final filtering
    df_final = df_filtered.loc[mask2].copy().reset_index(drop=True)

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

def results_to_dataframe(
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    epochs: np.ndarray
) -> pd.DataFrame:
    """
    Converts a nested dictionary of results into a pandas DataFrame with a MultiIndex for columns.

    Parameters
    ----------
    results : dict
        Dictionary with structure {dataset: {method: {metric: values}}}, where values are np.ndarray of measurements.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with MultiIndex columns (dataset, method, metric) and rows corresponding to runs.
    """
    
    data = {
        (dataset_name, method, metric): [loss for losses in run_losses for loss in losses]
        for dataset_name, models in results.items()
        for method, metrics in models.items()
        for metric, run_losses  in metrics.items()
    }

    temp_results = results.copy()
    while isinstance(temp_results, dict):
        key = next(iter(temp_results))
        temp_results = temp_results[key]
    n_runs = len(temp_results)

    row_index = pd.MultiIndex.from_tuples([
        (run, epoch)
        for run in range(n_runs)
        for epoch in epochs
    ], names=['run', 'epoch'])

    df = pd.DataFrame(data, index=row_index)
    df.columns.names = ['dataset', 'method', 'metric'] # Name the column levels

    return df

def plot_results(
    dataframe: pd.DataFrame,
    nrows: int = 1,
    ncols: Optional[int] = None,
    group_level: str = "dataset",
    value_level: str = "metric",
    value_key: str = "training_losses",
    plotting_function: Callable = sns.boxenplot
) -> tuple[Figure, np.ndarray]:
    """
    Plots boxenplots for a given pandas DataFrame with MultiIndex columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame with MultiIndex columns. Each level corresponds to a grouping (e.g., dataset, method, metric).
    nrows : int, optional
        Number of rows of subplots (default is 1).
    ncols : int or None, optional
        Number of columns of subplots. If None, it is set to the number of unique groups (default is None).
    group_level : str, optional
        The column MultiIndex level to use for grouping subplots (default is "dataset").
     value_level : str, optional
        The column MultiIndex level whose value (specified by `value_key`) will be plotted.
    value_key : str, optional
        The value in `value_level` to plot (default is "test_loss").
    plotting_function : callable
        A function that takes a DataFrame and an Axes object to create the plot (default is sns.boxenplot).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    axes : np.ndarray
        Array of matplotlib Axes objects.
    """

    # Get unique group names from the specified MultiIndex level
    group_names = dataframe.columns.get_level_values(group_level).unique()
    n = len(group_names)

    # Set number of columns if not provided
    if ncols is None: ncols = n

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, sharex=True)

    # Ensure axes is always iterable
    if nrows * ncols == 1: axes = np.array([axes])
    if nrows > 1: axes = axes.flatten()

    # Plot for each group
    for ax, group_name in zip(axes, group_names):
        # Select columns for the current group in group_level
        df = dataframe.xs(key=group_name, level=group_level, axis=1)

        # Select columns for the specified value_key in value_level
        df = df.xs(key=value_key, level=value_level, axis=1) # type: ignore

        df.columns.name = None # Remove column name for clarity
        plotting_function(data=df, ax=ax) # type: ignore
        ax.set_title(group_name)

    return fig, axes

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
    
    adjusted_val_size = val_size / (1 - test_size)  # Adjust validation size

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
        Number of generations (iterations) between each recording of training, validation, and test losses.
        For example, if record_interval=2, losses are recorded every 2 generations. Default is 1.
    resplit_interval : int, optional
        Number of generations (iterations) between each resplitting of the training and validation sets.
        For example, if resplit_interval=3, the training and validation sets are resplit every 3 generations.
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
        if step <= 0: print(f"Warning: Non-positive step size {step} at iteration {it}. Skipping.")

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

            # Compute adjusted validation size and split the data into new train/val sets
            adjusted_val_size = len(y_val)/(len(y_train_val))
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, 
                test_size=adjusted_val_size, 
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
    _: Any = None,
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
        The current run number (e.g., for repeated experiments).
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
    _ : Any
        Placeholder parameter for creating proper Dask task dependencies.

    Returns
    -------
    Dict[str, Any] or None
        If dependency is not None, returns the results dictionary.
        Otherwise, results are only saved to disk as pickle files and None is returned.
    """

    # pickle the current parameters to be able to recover them later if needed
    # method.pysr_params['random_state'] = run

    # if method_name == "RFGPCMI":
    #     method.method_params['random_state'] = run  # type: ignore

    # parameters = {
    #     'dataset_name': dataset_name,
    #     'method_name': method_name,
    #     'run': run,
    #     'train_val_test_set': train_val_test_set,
    #     'method': method,
    #     'output_dir': output_dir,
    #     'return_results': return_results
    # }
    # with open("parameters.pkl", "wb") as f:
    #     pickle.dump(parameters, f)

    # Run the symbolic regression method and get the results
    start = time.time()
    with open("logfile.log", "a", encoding="utf-8") as lf:
        lf.write(f"Starting: {dataset_name} - {method_name} - Run {run}\n")

    temp_losses, temp_best_eqs, temp_features = method.run(train_val_test_set)
    
    with open("logfile.log", "a", encoding="utf-8") as lf:
        lf.write(f"Completed: {dataset_name} - {method_name} - Run {run} in {time.time() - start:.2f} seconds\n")

    # Organize the results into a dictionary
    results = {
        'dataset_name': dataset_name,
        'method_name': method_name,
        'run': run,
        'losses': temp_losses,
        'equations': temp_best_eqs,
        'features': temp_features
    }

    # Save the results to a file
    filename = f"results_{dataset_name}_{method_name}_{run}.pkl"
    with open(os.path.join(output_dir, filename), "wb") as f:
        pickle.dump(results, f)

    if return_results:
        return results
    else:
        return None

def send_email(
    subject: str, 
    body_message: str, 
    sender_email: str, 
    receiver_email: str,
    app_password: str, 
    smtp_server: str, 
    smtp_port: int, 
):
    if receiver_email is None:
        receiver_email = sender_email

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.set_content(body_message)

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            # Connect to the SMTP server and send the email
            server.login(sender_email, app_password)
            server.send_message(msg)
            print("The notification email has been sent successfully!")
            
    except Exception as e:
        print(f"Failed to send email: {e}")

def gather_splits(split_results, index):
    """Gathers a specific part of the train-val-test split results (e.g., index 0 for X_train)."""
    return tuple(split[index] for split in split_results)

def extract_equations(results, index):
    """Extracts the equations from the results."""
    return [result['equations'][index] for result in results]

def inv(x):
    return 1 / x

def sqrt_sympy(x, evaluate=True):
    return sp.sqrt(x, evaluate=evaluate)

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
                'time': float,
                'results': Any
            }

        Multiple runs:
            {
                'average_time': float,
                'std_time': float,
                'all_times': list[float],
            }
    """

    times: list[float] = []

    # Execute n_runs times
    for _ in range(n_runs):
        start = time.perf_counter()
        results = func(*args, **kwargs)
        times.append(time.perf_counter() - start)

    if n_runs == 1:
        return {
            'time': times[0],
            'results': results
        }

    return {
        'average_time': np.mean(times),
        'std_time': np.std(times),
        'all_times': times,
    }

def load_task_results(
    directory: str
) -> list:
    """Load all .pkl task result files from a directory."""

    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            filepath = os.path.join(directory, filename)

            with open(filepath, "rb") as f:
                results.append(pickle.load(f))

    return results

def collect_results(
    task_results: list, 
    datasets: Dict, 
    methods: Dict
) -> Tuple[Dict, Dict, Dict]:
    results = {}
    equations = {}
    features = {}

    for dataset_name in datasets.keys():
        results[dataset_name] = {}
        equations[dataset_name] = {}
        features[dataset_name] = {}

        for method_name in methods.keys():
            results[dataset_name][method_name] = {
                "training_losses": [],
                "validation_losses": [],
                "test_losses": [],
            }
            equations[dataset_name][method_name] = []
            features[dataset_name][method_name] = []

    for result in task_results:
        dataset_name = result['dataset_name']
        method_name = result['method_name']
        losses = result['losses']

        results[dataset_name][method_name]["training_losses"].append(losses[0])
        results[dataset_name][method_name]["validation_losses"].append(losses[1])   
        results[dataset_name][method_name]["test_losses"].append(losses[2])

        equations[dataset_name][method_name].append(result['equations'])
        features[dataset_name][method_name].append(result['features'])

    return results, equations, features

def save_results(
    df: pd.DataFrame, 
    equations: Dict, 
    features: Dict, 
    prefix: str = "data"
) -> str:
    """Save results, equations, and features to a pickle file with a timestamped filename."""

    data = {
        'df': df,
        'equations': equations,
        'features': features
    }

    timestamp = datetime.datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
    filename = f"{prefix}_{timestamp}.pickle"

    with open(filename, "wb") as f:
        pickle.dump(data, f)

    return filename

def permutation_test(
    test_statistic: Callable,
    data: pd.Series,
    observed_statistic: Optional[float] = None,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    alternative: str = 'two-sided',
    decision_by: str = 'p_value',
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Perform a permutation test for a univariate statistic.
    
    The p-value is computed using the Phipson & Smyth (2010, 
    Permutation P-values should never be zero) adjustment.

    Parameters
    ----------
    test_statistic : Callable
        Function computing the test statistic given the data.
    data : np.ndarray
        Sample used to build the null distribution.
    observed_statistic : float, optional
        Precomputed observed statistic; if None it is evaluated from ``data``.
    n_permutations : int, default=1000
        Number of random permutations used to approximate the null distribution.
    alpha : float, default=0.05
        Significance level for confidence interval or p-value decision.
    alternative : {'two-sided', 'greater', 'less'}, default='two-sided'
        Alternative hypothesis controlling tail calculations.
    decision_by : {'p_value', 'interval'}, default='p_value'
        Whether to reject the null using the p-value or the confidence interval.
    random_state : int, optional
        Seed for the permutation generator.

    Returns
    -------
    Dict[str, Any]
        Dictionary with the following keys:
        - 'observed_statistic': The observed statistic.
        - 'null_distribution': The null distribution of the statistic.
        - 'p_value': The computed p-value.
        - 'reject_null': Boolean indicating whether to reject the null hypothesis.
        - 'confidence_interval': The (lower, upper) bounds of the confidence interval.
    """
    
    # Set seed for reproducibility
    rng = np.random.default_rng(random_state)

    # Calculate the observed statistic if not provided
    if observed_statistic is None: observed_statistic = test_statistic(data)
    assert isinstance(observed_statistic, float), "The observed statistic must be a numeric value."

    # Generate the null distribution and sort it
    null_distribution = np.array([
        test_statistic(rng.permutation(data)) for _ in range(n_permutations)
    ])
    null_distribution.sort()

    # Calculate p-value and confidence intervals based on the alternative hypothesis
    match alternative:
        case 'two-sided':
            count_greater = np.sum(null_distribution >= observed_statistic)
            count_less = np.sum(null_distribution <= observed_statistic)
            p_value = 2 * min(
                (count_greater + 1) / (n_permutations + 1),
                (count_less + 1) / (n_permutations + 1)
            )
            p_value = min(1.0, p_value)   # clip to 1
            lower_bound = np.quantile(null_distribution, alpha / 2)
            upper_bound = np.quantile(null_distribution, 1 - alpha / 2)

        case 'greater':
            count = np.sum(null_distribution >= observed_statistic)
            p_value = (count + 1) / (n_permutations + 1)
            lower_bound = None
            upper_bound = np.quantile(null_distribution, 1 - alpha)

        case 'less':
            count = np.sum(null_distribution <= observed_statistic)
            p_value = (count + 1) / (n_permutations + 1)
            lower_bound = np.quantile(null_distribution, alpha)
            upper_bound = None

        case _:
            raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'.")
        
    # Decide whether to reject the null hypothesis
    match decision_by:
        case 'p_value': reject_null = p_value <= alpha
        case 'interval':
            match alternative:
                case 'two-sided':
                    assert lower_bound is not None and upper_bound is not None
                    reject_null = not (lower_bound < observed_statistic < upper_bound)
                
                case 'greater':
                    assert upper_bound is not None
                    reject_null = observed_statistic >= upper_bound
                
                case 'less':
                    assert lower_bound is not None
                    reject_null = observed_statistic <= lower_bound
        
        case _: raise ValueError("decision_by must be 'p_value' or 'interval'.")

    return {
        "observed_statistic": observed_statistic,
        "null_distribution": null_distribution,
        "p_value": p_value,
        "reject_null": reject_null,
        "confidence_interval": (lower_bound, upper_bound)
    }

@contextmanager
def temporary_seed(seed: Optional[int] = None):
    """
    Context manager that temporarily sets the global random seed for
    both `numpy` (legacy RNG) and Python's `random` module,
    and restores the previous state upon exit.
    """

    if seed is not None:
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
    
    # If seed is None, do nothing
    else: yield