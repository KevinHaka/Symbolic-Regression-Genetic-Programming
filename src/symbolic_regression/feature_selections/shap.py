import numpy as np
import pandas as pd

from sympy import lambdify
from shap import SamplingExplainer

from typing import Any, Dict, List, Optional, Tuple

from symbolic_regression.methods.gp import GP
from symbolic_regression.utils.pysr_utils import train_val_test_split

def get_shap_values(
    X_train: pd.DataFrame,
    gp_equation: pd.Series,
    random_state: Optional[int] = None
) -> Tuple[List[str], np.ndarray]:
    """
    Compute SHAP values for features based on GP equations.

    Args:
        X_train: Training DataFrame.
        gp_equation: GP equation from pysr.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple containing selected feature names and their SHAP values.
    """

    # Convert GP equation to sympy format and extract variables
    sympy_expr = gp_equation.sympy_format
    expr_variables = sorted(sympy_expr.free_symbols, key=lambda s: str(s))

    # Skip equations with no variables
    if len(expr_variables) >= 1:
        rng = np.random.default_rng(random_state) # Set random seed for reproducibility

        # Create numpy-compatible lambda function from sympy expression
        lambda_func = lambdify(expr_variables, sympy_expr, modules="numpy")
        str_variables = [str(var) for var in expr_variables]
        
        # Create SHAP explainer for the equation function
        explainer = SamplingExplainer(
            lambda X: lambda_func(*X.T),
            X_train[str_variables],
            seed=int(rng.integers(0, 2**32))
        )

        # Compute SHAP values for each feature in the equation
        shap_values = explainer.shap_values(X_train[str_variables], silent=True)
        feature_shap_values = np.mean(np.abs(shap_values), axis=0)

    else:
        str_variables = []
        feature_shap_values = np.array([])

    return str_variables, feature_shap_values

def select_features(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    n_runs: int = 30,
    n_top_features: Optional[int] = None,
    random_state: Optional[int] = None,
    gp_params: Optional[Dict[str, Any]] = None
) -> Tuple[List[str], List[float]]:
    """
    Select top features based on SHAP values from GP models.

    Args:
        X: Input DataFrame.
        y: Target array.
        test_size: Proportion of data to use as test set.
        val_size: Proportion of data to use as validation set.
        n_runs: Number of GP runs to perform.
        n_top_features: Number of top features to select. If None, defaults to log2(n_features).
        random_state: Random seed for reproducibility.
        gp_params: Parameters for the GP model.
    
    Returns:
        Tuple containing selected feature names and their SHAP values.
    """
    
    # Set default GP parameters if none provided
    if gp_params is None: gp_params = {}

    rng = np.random.default_rng(random_state) # Set random seed for reproducibility
    gp = GP(**gp_params) # Initialize GP with provided parameters
    X_trains = [] # To hold training data for each run
    gp_equations = [] # To hold best GP equations

    # Determine number of top features if not provided
    if (n_top_features is None):
        n_features = len(X.columns)
        n_top_features = max(1, round(np.log2(n_features)))

    # Perform multiple runs to gather GP equations and training data
    for _ in range(n_runs):

        # Split the data into training, validation, and test sets
        train_val_test_set = train_val_test_split(
            X, y, 
            test_size=test_size, 
            val_size=val_size, 
            random_state=rng.integers(0, 2**32)
        )

        # Train a GP model and get the best equations
        temp_best_eqs = gp.run(train_val_test_set)[1]

        # Store training data and best equation from this run
        X_trains.append(train_val_test_set[0])
        gp_equations.append(temp_best_eqs[-1])

    # Select top features based on SHAP values from the gathered GP equations
    selected_features, mean_shap_values_selected_features = select_features_from_pretrained_models(
        X_trains=tuple(X_trains),
        gp_equations=gp_equations,
        n_top_features=n_top_features,
        random_state=int(rng.integers(0, 2**32))
    )
    return selected_features, mean_shap_values_selected_features

def select_features_from_pretrained_models(
    X_trains: Tuple[pd.DataFrame],
    gp_equations: List[pd.Series],
    n_top_features: Optional[int] = None,
    random_state: Optional[int] = None
) -> Tuple[List[str], List[float]]:
    """
    Select top features based on SHAP values from GP equations.

    Args:
        X_trains: List or tuple of training DataFrame objects.
        gp_equations: List of best GP equations.
        n_top_features: Number of top features to select. If None, defaults to log2(n_features).
        random_state: Random seed for reproducibility.
    
    Returns:
        Tuple containing selected feature names and their SHAP values.
    """

    rng = np.random.default_rng(random_state) # Set random seed for reproducibility
    feature_names = X_trains[0].columns.tolist() # List of all feature names
    mean_shap_values = {feature: 0.0 for feature in feature_names} # Initialize dictionary to hold mean SHAP values
    n_equations = len(gp_equations) # Total number of GP equations
    
    # Determine number of top features if not provided
    if (n_top_features is None):
        n_features = len(feature_names)
        n_top_features = max(1, round(np.log2(n_features)))

    # Compute SHAP values for each GP equation and aggregate
    for gp_equation, X_train in zip(gp_equations, X_trains):

        # Get SHAP values for the current equation
        str_variables, feature_shap_values = get_shap_values(X_train, gp_equation, rng.integers(0, 2**32))

        # Aggregate SHAP values across equations
        for feature_shap_value, var_name in zip(feature_shap_values, str_variables):
            mean_shap_values[var_name] += feature_shap_value

    # Normalize by number of equations after aggregation
    for feature in feature_names: 
        mean_shap_values[feature] /= n_equations

    # Select top features by mean SHAP value
    selected_features = sorted(list(mean_shap_values.keys()), key=lambda k: mean_shap_values[k], reverse=True)[:n_top_features]
    mean_shap_values_selected_features = [mean_shap_values[feature] for feature in selected_features]

    return selected_features, mean_shap_values_selected_features