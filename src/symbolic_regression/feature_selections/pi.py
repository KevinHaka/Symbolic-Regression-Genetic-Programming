import numpy as np
import pandas as pd

from typing import Any, Dict, List, Optional, Sequence, Tuple

from symbolic_regression.methods.gp import GP
from symbolic_regression.utils.pysr_utils import train_val_test_split, nrmse_loss

def select_features(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    n_runs: int = 30,
    random_state: Optional[int] = None,
    gp_params: Optional[Dict[str, Any]] = None
) -> Tuple[List[str], List[float]]:
    """
    Select features using permutation importance method with GP (Genetic Programming).
    
    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (np.ndarray): Target variable.
        test_size (float): Proportion of data for testing.
        val_size (float): Proportion of data for validation.
        n_runs (int): Number of GP runs.
        random_state (Optional[int]): Random seed for reproducibility.
        gp_params (Optional[Dict[str, Any]]): Parameters for GP.
        
    Returns:
        Tuple[List[str], List[float]]: Selected feature names and their importances.
    """

    # Set default GP parameters if none provided
    if gp_params is None: gp_params = {} 
    
    rng = np.random.default_rng(random_state) # Set random seed for reproducibility
    gp = GP(**gp_params) # Initialize GP with provided parameters
    err_org = np.empty(n_runs) # To hold original errors
    gp_equations = [] # To hold best GP equations
    test_sets = [] # To hold test sets
    
    # Perform multiple runs to gather GP equations and test sets
    for n_run in range(n_runs):
        # Split the data into training, validation, and test sets
        train_val_test_set = train_val_test_split(
            X, y, 
            test_size=test_size, 
            val_size=val_size, 
            random_state=rng.integers(0, 2**32)
        )

        # Train a GP model and get the losses and best equations
        losses, temp_best_eqs, _ = gp.run(train_val_test_set)

        # Store the original error, best equation and test set from this run
        err_org[n_run] = losses[-1][-1]
        gp_equations.append(temp_best_eqs[-1])
        test_sets.append((train_val_test_set[2], train_val_test_set[5]))

    # Select features using permutation importance from the pretrained GP models
    selected_features, scaled_importances = select_features_from_pretrained_models(
        test_sets=test_sets,
        err_org=err_org,
        gp_equations=gp_equations,
        random_state=random_state
    )
    
    return selected_features, scaled_importances

def select_features_from_pretrained_models(
    test_sets: Sequence[Tuple[pd.DataFrame, np.ndarray]],
    err_org: np.ndarray,
    gp_equations: Sequence[pd.Series],
    random_state: Optional[int] = None
) -> Tuple[List[str], List[float]]:
    """ 
    Select features using permutation importance from pretrained GP (Genetic Programming) models.

    Args:
        test_sets (Sequence[Tuple[pd.DataFrame, np.ndarray]]): Sequence of test sets (X_test, y_test).
        err_org (np.ndarray): Original errors from the GP models on the train sets.
        gp_equations (Sequence[pd.Series]): Sequence of GP equations.
        random_state (Optional[int]): Random seed for reproducibility.
    
    Returns:
        Tuple[List[str], List[float]]: Selected feature names and their importances.
    """

    rng = np.random.default_rng(random_state) # Random number generator for reproducibility
    length = len(gp_equations) # Number of runs/models
    # Initialize raw feature importances
    raw_FI = {feature_name: np.zeros(length) for feature_name in test_sets[0][0].columns.tolist()}

    # Compute permutation importance for each GP model
    for n_run, (equation, (X_test, y_test)) in enumerate(zip(gp_equations, test_sets)):
        sympy_expr = equation.sympy_format # Get sympy expression
        expr_variables = sorted(sympy_expr.free_symbols, key=lambda s: str(s)) # Extract variables from the expression
        str_variables = [str(var) for var in expr_variables] # Convert variables to string format

        # Permute each variable and compute the raw feature importance
        for var in str_variables:
            X_test_pmt = X_test.copy() # Create a copy of the test set
            X_test_pmt[var] = rng.permutation(X_test_pmt[var]) # Permute the selected feature
            y_pred = equation.lambda_format(X_test_pmt) # Predict using the permuted test set
            err_pmt = nrmse_loss(y_test, y_pred) # Compute error with permuted feature
            raw_FI[var][n_run] = err_pmt - err_org[n_run] # Store the raw feature importance

    mean_raw_FI = {var: arr.mean() for var, arr in raw_FI.items()} # Compute mean raw feature importances
    
    selected_features = sorted(list(mean_raw_FI.keys()), key=lambda k: mean_raw_FI[k], reverse=True) # Sort features by importance
    selected_features = [var for var in selected_features if mean_raw_FI[var] > 0] # Keep only features with positive importance
    scaled_importances = [mean_raw_FI[var] for var in selected_features] # Get corresponding importances

    return selected_features, scaled_importances