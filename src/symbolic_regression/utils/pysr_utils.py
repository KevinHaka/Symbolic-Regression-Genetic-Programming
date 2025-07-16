# Libraries and modules
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from pysr import PySRRegressor

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from typing import Optional, Callable, Dict, List, Tuple, Any
from inspect import signature

def rmse_loss(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
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
    float
        The RMSE value.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def nrmse_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Computes the Normalized Root Mean Squared Error (NRMSE) between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns
    -------
    float
        The NRMSE value.
    """
    return rmse_loss(y_true, y_pred) / (np.max(y_true) - np.min(y_true))

def best_equation(
    model: PySRRegressor, 
    X: pd.DataFrame,
    y: pd.Series,
    loss_function: Callable[[np.ndarray, np.ndarray], float],
    X_check: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    Selects the best equation from a PySRRegressor model based on 
    the lowest value of the provided loss function on the given data.

    Parameters
    ----------
    model : PySRRegressor
        Trained symbolic regression model.
    X : DataFrame
        Feature matrix to evaluate equations.
    y : array-like
        True target values.
    loss_function : callable
        Function to compute the loss between true and predicted values.
    X_check: DataFrame, optional
        Additional dataset to check if the equation produces real values.

    Returns
    -------
    best_eq : pandas.Series or None
        The equation (row from model.equations_) with the lowest loss.
    """

    # Assert that model.equations_ is a DataFrame
    assert isinstance(model.equations_, pd.DataFrame), "model.equations_ is not a DataFrame. The model may not be fitted yet."

    n_equations = len(model.equations_)
    temp_loss = np.inf
    best_eq = None 

    # Iterate over all equations and find the one with the lowest loss
    for idx in range(n_equations):
        y_pred = model.predict(X, idx)
        loss = loss_function(y, y_pred)

        # Update best equation if current one has lower loss
        if loss < temp_loss:

            # Ensure the equation produces real values
            y_check_pred = model.predict(X_check, idx)
            if np.iscomplexobj(y_check_pred): continue

            temp_loss = loss
            best_eq = model.equations_.iloc[idx]

    return best_eq

def results_to_dataframe(
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]]
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
    
    # Create a DataFrame with MultiIndex columns from the nested dictionary
    df = pd.DataFrame({
        (dataset_name, method, metric): values
        for dataset_name, models in results.items()
        for method, metrics in models.items()
        for metric, values in metrics.items()
    })

    df.columns.names = ['dataset', 'method', 'metric'] # Name the column levels
    df.index.name = "run" # Name the index (rows)
    return df

def plot_results(
    dataframe: pd.DataFrame,
    nrows: int = 1,
    ncols: Optional[int] = None,
    group_level: str = "dataset",
    value_level: str = "metric",
    value_key: str = "test_loss"
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
        sns.boxenplot(df, ax=ax) # type: ignore
        ax.set_title(group_name)

    return fig, axes

def train_val_test_split(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.25
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
     f data to use for testing (default: 0.2).
    val_size : float, optional
        Proportion of remaining data to use for validation (default: 0.25).

    Returns
    -------
    tuple
        Tuple containing (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    
    # Split the data into train/val/test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size)

    return X_train, X_val, X_test, y_train, y_val, y_test

def cumulative_lambda(
    X: pd.DataFrame,
    lambda_models: list[tuple[Callable, list[str]]]
) -> np.ndarray:
    """
    Compute the cumulative prediction of an ensemble of symbolic regression models.

    This function takes a list of lambda models (each consisting of a callable function and
    the list of feature names it uses) and applies each model to the corresponding columns
    of X. The predictions from all models are summed to produce the cumulative prediction
    for each sample in X.

    Parameters
    ----------
    X : pandas.DataFrame
        Input feature matrix for which to compute the cumulative prediction.
    lambda_models : list of (Callable, list of str)
        List of tuples, where each tuple contains:
            - A callable (e.g., a sympy lambda function) that takes as input the selected features.
            - A list of feature names (columns of X) that the callable expects as input.

    Returns
    -------
    y_pred : np.ndarray
        Array of shape (n_samples,) containing the sum of predictions from all lambda models
        for each sample in X.
    """

    y_pred = np.zeros(X.shape[0])  # Initialize predictions to zero

    for func, features in lambda_models:
        # Add the predictions of each lambda model for its selected features
        y_pred += func(X[features])

    return y_pred

def fit_and_evaluate_best_equation(
    train_val_test_set: Tuple[
        pd.DataFrame, # X_train
        pd.DataFrame, # X_val
        pd.DataFrame, # X_test
        pd.Series,   # y_train
        pd.Series,   # y_val
        pd.Series    # y_test
    ],
    loss_function: Callable,
    record_interval: int = 1,
    pysr_params: Dict[str, Any] = {}
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.Series]]:
    """
    Fit a symbolic regression model using PySR and evaluate its performance over multiple intervals.

    This function trains a PySRRegressor on the provided training set, selects the best equation
    based on validation loss at each recording interval, and computes the training, validation,
    and test losses using the best equation found at each step.

    Parameters
    ----------
    train_val_test_set : tuple
        Tuple containing (X_train, X_val, X_test, y_train, y_val, y_test).
    loss_function : Callable
        Function to compute the loss between true and predicted values.
    record_interval : int, optional
        Number of generations (iterations) between each recording of training, validation, and test losses.
        For example, if record_interval=2, losses are recorded every 2 generations. Default is 1.
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

    Notes
    -----
    The model is trained iteratively with warm_start=True, and at each interval the best equation
    is selected based on validation loss.
    """

    if pysr_params is None: pysr_params = {}
    best_eqs = []

    # Determine the total number of iterations and how many times to record losses
    niterations = pysr_params.get("niterations", signature(PySRRegressor).parameters['niterations'].default)
    n_records = niterations // record_interval

     # Initialize arrays to store losses at each interval
    training_losses = np.empty(n_records)
    validation_losses = np.empty(n_records)
    test_losses = np.empty(n_records)

    # Unpack the train/val/test sets
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_set

    # Initialize the PySRRegressor with warm_start=True to allow iterative fitting
    model = PySRRegressor(warm_start=True, **pysr_params)
    model.set_params(niterations=record_interval) # Adjust the number of generations per interval

     # Iteratively fit the model and record losses at each interval
    for interval_idx in range(n_records):
        # Fit the model for the current interval
        model.fit(X_train, y_train)

        # Concatenate the training and test sets
        X_check = pd.concat([X_train, X_test], ignore_index=True)

        # Select the best equation based on validation loss
        best_eq = best_equation(model, X_val, y_val, loss_function, X_check)
        best_eqs.append(best_eq)
        lambda_expr = best_eq.lambda_format 

        # Compute losses on training, validation and test sets
        training_losses[interval_idx] = loss_function(y_train, lambda_expr(X_train))
        validation_losses[interval_idx] = loss_function(y_val, lambda_expr(X_val))
        test_losses[interval_idx] = loss_function(y_test, lambda_expr(X_test))

    return training_losses, validation_losses, test_losses, best_eqs