# Libraries and modules
import numpy as np
import pandas as pd

from sympy import lambdify
from shap import SamplingExplainer, utils
from typing import Callable, Optional

from sklearn.preprocessing import StandardScaler
from npeet import entropy_estimators as ee

def _lambda_func_shap(
    X: np.ndarray, 
    lambda_func: Callable
) -> np.ndarray:
    """
    Helper function to apply a sympy lambdified function to a 2D numpy array X,
    unpacking columns as separate arguments.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (n_samples, n_features).
    lambda_func : callable
        Function returned by sympy.lambdify, expecting each feature as a separate argument.

    Returns
    -------
    np.ndarray
        The result of applying lambda_func to X.
    """
        
    # Unpack each column of X as a separate argument to lambda_func
    return lambda_func(*[X[:, i] for i in range(X.shape[1])])

def shap_feature_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    loss_function: Callable,
    n_runs: int = 100,
    record_interval: int = 1,
    test_size: float = 0.2,
    val_size: float = 0.25,
    top_features_ratio: Optional[float] = None,
    gp_best_equations: Optional[list] = None,
    train_val_test_sets_list: Optional[tuple] = None,
    pysr_params: Optional[dict] = None
) -> tuple[list[str], list[float], tuple]:
    """
    Perform feature selection using SHAP (SHapley Additive exPlanations) values.
    
    This function uses genetic programming to discover equations and then applies
    SHAP analysis to identify the most important features across all discovered
    equations. Features are ranked by their mean absolute SHAP values.
    
    The method works by:
    1. Running genetic programming to discover multiple equations (if not provided)
    2. For each equation, computing SHAP values to measure feature importance
    3. Aggregating SHAP values across all equations
    4. Selecting top features based on mean SHAP importance
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with shape (n_samples, n_features).
    y : np.ndarray
        Target vector with shape (n_samples,).
    loss_function : callable
        Loss function to evaluate equation quality during GP.
    n_runs : int, default=100
        Number of GP runs to perform (if equations not provided).
    record_interval : int, default=1
        Interval for recording GP progress.
    test_size : float, default=0.2
        Proportion of data for testing split.
    val_size : float, default=0.25
        Proportion of training data for validation split.
    top_features_ratio : float, optional
        Ratio of features to select (e.g., 0.1 for top 10%).
        If None, uses log2(n_features) as default.
    gp_best_equations : list, optional
        Pre-computed list of best equations from GP runs.
        If None, GP will be run to generate equations.
    train_val_test_sets_list : tuple, optional
        Pre-computed data splits. If None, splits will be generated.
    pysr_params : dict, optional
        Parameters for PySR if used in GP.
        
    Returns
    -------
    selected_features : list of str
        Names of selected features ranked by SHAP importance.
    mean_shap_values_selected_features : list of float
        Mean SHAP values for the selected features.
    train_val_test_sets_list : tuple
        Data splits used for analysis (for consistency with other methods).
    """
    
    # If no precomputed equations or train_val_test_sets_list, run gp() to get them
    if (gp_best_equations is None) or (train_val_test_sets_list is None):
        from sr_gp_methods import gp  # Import here to avoid circular import
        
        _, _, gp_best_equations, train_val_test_sets_list = gp(
            X, 
            y, 
            loss_function, 
            n_runs, 
            record_interval,
            test_size, 
            val_size, 
            pysr_params
        )
    
    # Initialize SHAP value aggregation
    mean_shap_values = {}
    n_equations = len(gp_best_equations)
    
    # Determine number of top features to select
    if isinstance(top_features_ratio, float):
        n_top_features = max(1, round(top_features_ratio * X.shape[1]))
    else:
        # Default: select log2(n_features) features if ratio not specified
        n_top_features = max(1, round(np.log2(X.shape[1])))

    # Extract training data for SHAP analysis
    X_train_list = train_val_test_sets_list[0]

    # Process each equation to compute SHAP values
    for gp_best_equation, X_train in zip(gp_best_equations, X_train_list):
        # Convert GP equation to sympy format and extract variables
        sympy_expr = gp_best_equation.sympy_format.simplify()
        expr_variables = sorted(sympy_expr.free_symbols, key=lambda s: str(s))

        # Skip equations with no variables
        if len(expr_variables) >= 1:
            # Create numpy-compatible lambda function from sympy expression
            lambda_func = lambdify(expr_variables, sympy_expr, modules="numpy")
            str_variables = [str(var) for var in expr_variables]
            
            # Sample data for SHAP analysis
            # Background data provides baseline for SHAP explanations
            X_background = utils.sample(X_train[str_variables], 100, random_state=None)
            # Foreground data is what we want to explain
            X_foreground = utils.sample(X_train[str_variables], 1000, random_state=None)

            # Create SHAP explainer for the equation function
            explainer = SamplingExplainer(
                lambda X: _lambda_func_shap(X, lambda_func),
                X_background
            )

            # Compute SHAP values for each feature in the equation
            shap_values = explainer.shap_values(X_foreground, silent=True)
            # Use absolute values and average across samples to get feature importance
            feature_shap_values = np.abs(shap_values).mean(axis=0)

            # Aggregate SHAP values across equations (normalize by number of equations)
            for feature_shap_value, var_name in zip(feature_shap_values, str_variables):
                mean_shap_values[var_name] = mean_shap_values.get(var_name, 0) + feature_shap_value/n_equations

    # Select top features by mean SHAP value
    selected_features = sorted(mean_shap_values, key=mean_shap_values.get, reverse=True)[:n_top_features]
    mean_shap_values_selected_features = [mean_shap_values[feature] for feature in selected_features]

    return selected_features, mean_shap_values_selected_features, train_val_test_sets_list

def cmi_feature_selection2(
    X: pd.DataFrame,
    y: np.ndarray,
    k: int = 3,
    top_features_ratio: Optional[float] = None,
    min_relative_mi_gain: float = 0.05
) -> tuple[list[str], list[float]]:
    """
    Perform feature selection using Conditional Mutual Information (CMI).
    
    This function implements an iterative feature selection algorithm based on
    conditional mutual information. It greedily selects features that have the
    highest mutual information with the target variable, conditioned on the
    already selected features.
    
    The algorithm works as follows:
    1. Start with no selected features
    2. For each remaining feature, compute CMI with target given selected features
    3. Select the feature with highest CMI
    4. Repeat until stopping criterion is met
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with shape (n_samples, n_features).
    y : np.ndarray
        Target vector with shape (n_samples,).
    k : int, default=3
        Number of nearest neighbors for MI estimation using k-NN entropy estimator.
        Higher values provide more stable estimates but are computationally more expensive.
    top_features_ratio : float, optional
        Ratio of features to select (e.g., 0.1 for top 10% of features).
        If None, uses adaptive stopping criterion based on MI gain.
    min_relative_mi_gain : float, default=0.05
        Minimum relative MI gain for adaptive stopping. Only used when
        top_features_ratio is None. Selection stops when the MI gain of
        the next feature is less than this fraction of cumulative MI.
        
    Returns
    -------
    selected_features : list of str
        Names of selected features in order of selection (highest CMI first).
    cmi_values_selected_features : list of float
        CMI values for the selected features in order of selection.
    """
    
    # Initialize algorithm state
    scaler = StandardScaler()
    remaining_features = list(X.columns)
    cumulative_mi = 0  # Track total MI for stopping criterion
    cmi_values_selected_features = []
    selected_features = []

    # Determine stopping criterion
    if isinstance(top_features_ratio, float):
        # Fixed number of features to select
        n_top_features = max(1, round(top_features_ratio * X.shape[1]))
    else:
        # Adaptive stopping based on MI gain
        n_top_features = None

    # Standardize data for stable MI estimation
    # MI estimation via k-NN requires normalized features
    X_scaled = scaler.fit_transform(X.copy())
    y_scaled = scaler.fit_transform(y.copy().reshape(-1, 1))
    X_scaled = pd.DataFrame(X_scaled, columns=remaining_features)

    # Greedy feature selection loop
    while remaining_features and ((n_top_features is None) or (len(selected_features) < n_top_features)):
        cmi_per_feature = {}

        # Compute CMI for each remaining feature
        for feature in remaining_features:
            # Compute I(feature; target | selected_features)
            # If no features selected yet, this is just I(feature; target)
            cmi_per_feature[feature] = ee.mi(
                X_scaled[feature], 
                y_scaled, 
                X_scaled[selected_features] if selected_features else None,
                k=k
            )

        # Select feature with highest CMI
        max_cmi_feature = max(cmi_per_feature, key=cmi_per_feature.get) 
        max_cmi_value = cmi_per_feature[max_cmi_feature]
        cumulative_mi += max_cmi_value
        
        # Check adaptive stopping criterion
        if (not isinstance(top_features_ratio, float)) and (max_cmi_value < min_relative_mi_gain * cumulative_mi):
            # Stop if MI gain is too small relative to cumulative MI
            break
        
        # Add selected feature to results and remove from candidates
        selected_features.append(max_cmi_feature)
        cmi_values_selected_features.append(cmi_per_feature[max_cmi_feature])
        remaining_features.remove(max_cmi_feature)
    
    return selected_features, cmi_values_selected_features

def cmi_feature_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    k: int = 3,
    top_features_ratio: Optional[float] = None,
    alpha: float = 0.01,
) -> tuple[list[str], list[float]]:
    
    # Initialize algorithm state
    scaler = StandardScaler()
    remaining_features = list(X.columns)
    cmi_values_selected_features = []
    selected_features = []

    # Determine stopping criterion
    if isinstance(top_features_ratio, float):
        # Fixed number of features to select
        n_top_features = max(1, round(top_features_ratio * X.shape[1]))
    else:
        # Adaptive stopping based on MI gain
        n_top_features = None

    # Standardize data for stable MI estimation
    # MI estimation via k-NN requires normalized features
    X_scaled = scaler.fit_transform(X.copy())
    y_scaled = scaler.fit_transform(y.copy().reshape(-1, 1))
    X_scaled = pd.DataFrame(X_scaled, columns=remaining_features)

    # Greedy feature selection loop
    while remaining_features and ((n_top_features is None) or (len(selected_features) < n_top_features)):
        cmi_per_feature = {}

        # Compute CMI for each remaining feature
        for feature in remaining_features:
            # Compute I(feature; target | selected_features)
            # If no features selected yet, this is just I(feature; target)
            cmi_per_feature[feature] = ee.mi(
                X_scaled[feature], 
                y_scaled, 
                X_scaled[selected_features] if selected_features else None,
                k=k
            )

        # Select feature with highest CMI
        max_cmi_feature = max(cmi_per_feature, key=cmi_per_feature.get) 
        max_cmi_value = cmi_per_feature[max_cmi_feature]

        _, ci = ee.shuffle_test(
            ee.mi,
            X_scaled[feature], 
            y_scaled,
            X_scaled[selected_features].to_numpy().tolist() if selected_features else None,
            k=k,
            ns=1000,
            alpha=alpha
        )

        # Check if CMI is statistically significant
        if ci[0] < max_cmi_value < ci[1]: break
        
        # Add selected feature to results and remove from candidates
        selected_features.append(max_cmi_feature)
        cmi_values_selected_features.append(cmi_per_feature[max_cmi_feature])
        remaining_features.remove(max_cmi_feature)
    
    return selected_features, cmi_values_selected_features