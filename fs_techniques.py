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
    n_top_features: int,
    X_train_list: Optional[tuple],
    gp_best_equations: Optional[list]
) -> tuple[list[str], list[float], tuple]:
    
    # Initialize SHAP value aggregation
    mean_shap_values = {}
    n_equations = len(gp_best_equations)

    # Process each equation to compute SHAP values
    for gp_best_equation, X_train in zip(gp_best_equations, X_train_list):
        # Convert GP equation to sympy format and extract variables
        sympy_expr = gp_best_equation.sympy_format
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
            feature_shap_values = np.mean(shap_values, axis=0)
            # feature_shap_values = np.abs(shap_values).mean(axis=0)

            # Aggregate SHAP values across equations (normalize by number of equations)
            for feature_shap_value, var_name in zip(feature_shap_values, str_variables):
                mean_shap_values[var_name] = mean_shap_values.get(var_name, 0) + feature_shap_value/n_equations

    # Select top features by mean SHAP value
    selected_features = sorted(mean_shap_values, key=mean_shap_values.get, reverse=True)[:n_top_features]
    mean_shap_values_selected_features = [mean_shap_values[feature] for feature in selected_features]

    return selected_features, mean_shap_values_selected_features

def cmi_feature_selection2(
    X: pd.DataFrame,
    y: np.ndarray,
    k: int = 3,
    top_features_ratio: Optional[float] = None,
    min_relative_mi_gain: float = 0.05
) -> tuple[list[str], list[float]]:
    
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
    alpha: float = 0.01,
) -> tuple[list[str], list[float]]:
    
    # Initialize algorithm state
    scaler = StandardScaler()
    remaining_features = list(X.columns)
    cmi_values_selected_features = []
    selected_features = []

    # Standardize data for stable MI estimation
    # MI estimation via k-NN requires normalized features
    X_scaled = scaler.fit_transform(X.copy())
    y_scaled = scaler.fit_transform(y.copy().reshape(-1, 1))
    X_scaled = pd.DataFrame(X_scaled, columns=remaining_features)

    # Greedy feature selection loop
    while remaining_features:
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
            ns=100,
            alpha=alpha
        )

        # Check if CMI is statistically significant
        if ci[0] < max_cmi_value < ci[1]: break
        
        # Add selected feature to results and remove from candidates
        selected_features.append(max_cmi_feature)
        cmi_values_selected_features.append(cmi_per_feature[max_cmi_feature])
        remaining_features.remove(max_cmi_feature)
    
    return selected_features, cmi_values_selected_features