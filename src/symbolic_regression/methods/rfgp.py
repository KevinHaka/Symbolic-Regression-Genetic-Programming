import numpy as np
import pandas as pd

import copy
from typing import Any, Callable, Dict, Optional, Tuple, List, Type
from inspect import signature

from pysr import PySRRegressor

from .base import BaseMethod
from .gpcmi import GPCMI

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

    # Initialize predictions to zero
    y_pred = np.zeros(X.shape[0])  

    # Iterate over each lambda model and its selected features
    for func, features in lambda_models:

        # Add the predictions of each lambda model for its selected features
        y_pred += func(X[features])

    return y_pred

class RFGP(BaseMethod):
    def __init__(
        self,
        n_submodels: int = 2,
        method_class: Type[BaseMethod] = GPCMI,
        method_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the Residual Fitting Genetic Programming (RFGP) method.

        Args:
            n_submodels (int): Number of submodels to fit sequentially.
            method_class (Type[BaseMethod]): The method class to use for each submodel (e.g., GPCMI) 
            must be a filter method.
            method_params (Optional[Dict[str, Any]]): Parameters for the sub-method.
        """

        # Set default method parameters if none provided
        if method_params is None: method_params = {}

        # Extract pysr_params if provided, else set to empty dict
        pysr_params = method_params.get('pysr_params', {})

        # Get the signature of the method class to extract default parameters
        method_sig = signature(method_class)
        loss_function = method_params.get('loss_function', method_sig.parameters['loss_function'].default)
        record_interval = method_params.get('record_interval', method_sig.parameters['record_interval'].default)
        resplit_interval = method_params.get('resplit_interval', method_sig.parameters['resplit_interval'].default)

        super().__init__(loss_function, record_interval, resplit_interval, pysr_params)
        self.n_submodels = n_submodels
        self.method_class = method_class
        self.method_params = copy.deepcopy(method_params)

    def run(
        self,
        train_val_test_set: Tuple[
            pd.DataFrame,     # X_train
            pd.DataFrame,     # X_val
            pd.DataFrame,     # X_test
            np.ndarray,       # y_train
            np.ndarray,       # y_val
            np.ndarray        # y_test
        ],
        random_state: Optional[int] = None
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray],  # training, validation, test losses
        List[List[pd.Series]],                      # best-equation objects for each submodel
        List[List[str]],                            # list of feature names for each submodel
    ]:
        """ Execute one full train/validation/test run using RFGP.

        Args:
            train_val_test_set: Tuple containing training, validation, and test sets.
            random_state: Random seed for reproducibility.

        Returns:
            Tuple containing:
                - Tuple of arrays with training, validation, and test losses at each record interval.
                - List of lists containing best-equation objects for each submodel.
                - List of lists containing feature names used by each submodel.
        """

        # Set random seed for reproducibility
        rng = np.random.default_rng(random_state) 

        # Unpack the sets
        X_train_orig, X_val_orig, X_test_orig, y_train_orig, y_val_orig, y_test_orig = train_val_test_set
        
        # Initialize residuals as original targets
        y_train_residual = y_train_orig.copy()
        y_val_residual = y_val_orig.copy()
        y_test_residual = y_test_orig.copy()

        # Get niterations from pysr_params or default from PySRRegressor
        niterations = self.pysr_params.get("niterations", signature(PySRRegressor).parameters['niterations'].default)

        # Determine record interval, defaulting to niterations if not provided
        record_interval = self.record_interval if self.record_interval else niterations
        n_records = niterations // record_interval 

        # Distribute records among submodels, including remainders
        base_records_per_submodel = n_records // self.n_submodels
        remainder = n_records % self.n_submodels
        self.records_per_submodel = [
            base_records_per_submodel + 1 if (i < remainder) else base_records_per_submodel
            for i in range(self.n_submodels)
        ]

        # Initialize results
        training_losses = np.zeros(n_records)
        validation_losses = np.zeros(n_records)
        test_losses = np.zeros(n_records)
        
        # To hold results from all submodels
        all_best_eqs = []
        all_selected_features = []
        lambda_models_per_interval = []
        prev_best_lambda_models = []

        # Prepare the method
        method = self.method_class(**self.method_params)

        # Track the starting index for each submodel's records
        current_interval_start = 0

        # Fit each submodel sequentially
        for sub_i in range(self.n_submodels):

            # Number of records to fit for this submodel
            n_records_for_this_submodel = self.records_per_submodel[sub_i]

            # Update the number of iterations for the submodel
            method.pysr_params['niterations'] = n_records_for_this_submodel * self.record_interval

            # Prepare data for the current sub-model
            current_train_val_test_set = (
                X_train_orig, X_val_orig, X_test_orig,
                y_train_residual, y_val_residual, y_test_residual
            )

            # Run the sub-method
            _, sub_best_eqs, sub_features = method.run(current_train_val_test_set, rng.integers(0, 2**32))

            # Store results
            all_best_eqs.append(sub_best_eqs)
            all_selected_features.append(sub_features)

            # Determine the start and end indices for this submodel's records
            start_interval = current_interval_start
            end_interval = start_interval + n_records_for_this_submodel

            # Update lambda models and losses for each interval
            for j, interval_idx in enumerate(range(start_interval, end_interval)): 

                # Create new model component for this interval
                new_model_component = (sub_best_eqs[j].lambda_format, sub_features)

                # Update the list of lambda models
                lambda_models_per_interval.append(prev_best_lambda_models.copy())
                lambda_models_per_interval[interval_idx].append(new_model_component)

                # Calculate cumulative predictions and losses
                y_train_pred = cumulative_lambda(X_train_orig, lambda_models_per_interval[interval_idx])
                y_val_pred = cumulative_lambda(X_val_orig, lambda_models_per_interval[interval_idx])
                y_test_pred = cumulative_lambda(X_test_orig, lambda_models_per_interval[interval_idx])

                # Compute and store losses
                training_losses[interval_idx] = self.loss_function(y_train_orig, y_train_pred)
                validation_losses[interval_idx] = self.loss_function(y_val_orig, y_val_pred)
                test_losses[interval_idx] = self.loss_function(y_test_orig, y_test_pred)

            # Update previous best lambda models for the next submodel
            prev_best_lambda_models = lambda_models_per_interval[end_interval - 1]

            # Update residuals for the next submodel using the last equation from the current one
            last_eq_lambda = sub_best_eqs[-1].lambda_format

            # Get the subset of features used in the last equation
            X_train_sub = X_train_orig[sub_features]
            X_val_sub = X_val_orig[sub_features]
            X_test_sub = X_test_orig[sub_features]

            # Update residuals for the next submodel
            y_train_residual -= last_eq_lambda(X_train_sub)
            y_val_residual -= last_eq_lambda(X_val_sub)
            y_test_residual -= last_eq_lambda(X_test_sub)
            
            # Move to the next interval start
            current_interval_start = end_interval

        return (training_losses, validation_losses, test_losses), all_best_eqs, all_selected_features