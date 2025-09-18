import numpy as np
import pandas as pd

from sympy import lambdify
from shap import SamplingExplainer, utils

from typing import Callable, Optional, List, Tuple

from symbolic_regression.methods.gp import GP
from symbolic_regression.utils.pysr_utils import train_val_test_split

def _lambda_func_shap(
    X: np.ndarray, 
    lambda_func: Callable
) -> np.ndarray:
    return lambda_func(*[X[:, i] for i in range(X.shape[1])])

def _get_shap_values(
    X_train: pd.DataFrame,
    gp_equation: pd.Series
) -> Tuple[List[str], np.ndarray]:
    """
    Compute SHAP values for features based on GP equations.

    Args:
        X_trains: Tuple of training feature sets.
        gp_equations: List of best GP equations.

    Returns:
        Tuple containing selected feature names and their SHAP values.
    """

    # Convert GP equation to sympy format and extract variables
    sympy_expr = gp_equation.sympy_format
    expr_variables = sorted(sympy_expr.free_symbols, key=lambda s: str(s))

    # Skip equations with no variables
    if len(expr_variables) >= 1:
        # Create numpy-compatible lambda function from sympy expression
        lambda_func = lambdify(expr_variables, sympy_expr, modules="numpy")
        str_variables = [str(var) for var in expr_variables]

        # Generate a random seed to ensure non-deterministic sampling
        random_seed = np.random.randint(np.iinfo(np.int32).max)
        
        # Sample data for SHAP analysis
        X_background = utils.sample(X_train[str_variables], 100, random_state=random_seed)
        X_foreground = utils.sample(X_train[str_variables], 1000, random_state=random_seed)

        # Create SHAP explainer for the equation function
        explainer = SamplingExplainer(
            lambda X: _lambda_func_shap(X, lambda_func),
            X_background
        )

        # Compute SHAP values for each feature in the equation
        shap_values = explainer.shap_values(X_foreground, silent=True)
        feature_shap_values = np.mean(shap_values, axis=0)

    else:
        str_variables = []
        feature_shap_values = np.array([])

    return str_variables, feature_shap_values

def select_features(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.25,
    n_runs: int = 30,
    n_top_features: Optional[int] = None,
    **gp_params
) -> Tuple[List[str], List[float]]:

    gp = GP(**gp_params)
    X_trains = []
    gp_equations = []

    if (n_top_features is None):
        n_features = len(X.columns)
        n_top_features = max(1, round(np.log2(n_features)))

    else:
        n_top_features = n_top_features

    for _ in range(n_runs):
        # Split the data into training, validation, and test sets
        train_val_test_set = train_val_test_split(
            X, y, test_size=test_size, val_size=val_size
        )

        # Train a GP model and get the best equations
        _, temp_best_eqs, _ = gp.run(
            train_val_test_set
        )

        X_trains.append(train_val_test_set[0])
        gp_equations.append(temp_best_eqs[-1])

    selected_features, mean_shap_values_selected_features = select_features_from_pretrained_models(
        X_trains=tuple(X_trains),
        gp_equations=gp_equations,
        n_top_features=n_top_features
    )
    return selected_features, mean_shap_values_selected_features

def select_features_from_pretrained_models(
    X_trains: Tuple[pd.DataFrame],
    gp_equations: List[pd.Series],
    n_top_features: Optional[int] = None
) -> Tuple[List[str], List[float]]:
    """
    Select top features based on SHAP values from GP equations.

    Args:
        X_trains: List or tuple of training DataFrame objects.
        gp_equations: List of best GP equations.
        n_top_features: Number of top features to select. If None, defaults to log2(n_features).
    
    Returns:
        Tuple containing selected feature names and their SHAP values.
    """

    if (n_top_features is None):
        n_features = len(X_trains[0].columns)
        n_top_features = max(1, round(np.log2(n_features)))

    else:
        n_top_features = n_top_features

    mean_shap_values = {}
    n_equations = len(gp_equations)

    for gp_equation, X_train in zip(gp_equations, X_trains):
        str_variables, feature_shap_values = _get_shap_values(X_train, gp_equation)

        # Aggregate SHAP values across equations (accumulate absolute values)
        for feature_shap_value, var_name in zip(feature_shap_values, str_variables):
            mean_shap_values[var_name] = mean_shap_values.get(var_name, 0) + abs(feature_shap_value)

    # Normalize by number of equations after aggregation
    for var_name in mean_shap_values:
        mean_shap_values[var_name] /= n_equations

    # Select top features by mean SHAP value
    selected_features = sorted(list(mean_shap_values.keys()), key=lambda k: mean_shap_values[k], reverse=True)[:n_top_features]
    mean_shap_values_selected_features = [mean_shap_values[feature] for feature in selected_features]

    return selected_features, mean_shap_values_selected_features