import numpy as np
import pandas as pd

from typing import Any, Dict, Optional, Tuple, List, Type
from inspect import signature

from .base import BaseMethod
from ..utils.pysr_utils import cumulative_lambda
from .gpcmi import GPCMI

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
            method_class (Type[BaseMethod]): The class of the method to be used for each submodel (e.g., GP, GPCMI).
            method_params (Optional[Dict[str, Any]]): Parameters for the sub-method.
        """

        if method_params is None:
            method_params = {}

        pysr_params = method_params.get('pysr_params', {})

        # Get the signature of the method class to extract default parameters
        method_sig = signature(method_class)
        loss_function = method_params.get('loss_function', method_sig.parameters['loss_function'].default)
        record_interval = method_params.get('record_interval', method_sig.parameters['record_interval'].default)

        super().__init__(loss_function, record_interval, pysr_params)
        self.n_submodels = n_submodels
        self.method_class = method_class
        self.method_params = method_params

        # Distribute records among submodels, including remainders
        base_records_per_submodel = self.n_records // self.n_submodels
        remainder = self.n_records % self.n_submodels
        self.records_per_submodel = [
            base_records_per_submodel + 1 if (i < remainder) else base_records_per_submodel
            for i in range(self.n_submodels)
        ]

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
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray],  # training, validation, test losses
        List[List[pd.Series]],                      # best-equation objects for each submodel
        List[List[str]],                            # list of feature names for each submodel
    ]:
        """ Execute one full train/validation/test run using RFGP."""

        X_train_orig, X_val_orig, X_test_orig, y_train_orig, y_val_orig, y_test_orig = train_val_test_set
        
        y_train_residual = y_train_orig.copy()
        y_val_residual = y_val_orig.copy()
        y_test_residual = y_test_orig.copy()

        # Initialize results
        training_losses = np.zeros(self.n_records)
        validation_losses = np.zeros(self.n_records)
        test_losses = np.zeros(self.n_records)
        
        all_best_eqs = []
        all_selected_features = []
        lambda_models_per_interval = []
        prev_best_lambda_models = []

        # Prepare the method
        method = self.method_class(**self.method_params)

        current_interval_start = 0

        for sub_i in range(self.n_submodels):
            n_records_for_this_submodel = self.records_per_submodel[sub_i]
            if n_records_for_this_submodel == 0: continue

            # Update the number of iterations for the submodel
            method.pysr_params['niterations'] = n_records_for_this_submodel * self.record_interval

            # Prepare data for the current sub-model
            current_train_val_test_set = (
                X_train_orig, X_val_orig, X_test_orig,
                y_train_residual, y_val_residual, y_test_residual
            )

            # Run the sub-method
            _, sub_best_eqs, sub_features = method.run(current_train_val_test_set)

            all_best_eqs.append(sub_best_eqs)
            all_selected_features.append(sub_features)

            # Update residuals and calculate cumulative losses
            start_interval = current_interval_start
            end_interval = start_interval + n_records_for_this_submodel

            for j, interval_idx in enumerate(range(start_interval, end_interval)): 
                new_model_component = (sub_best_eqs[j].lambda_format, sub_features)
                lambda_models_per_interval.append(prev_best_lambda_models.copy())
                lambda_models_per_interval[interval_idx].append(new_model_component)

                # Calculate cumulative predictions and losses
                y_train_pred = cumulative_lambda(X_train_orig, lambda_models_per_interval[interval_idx])
                y_val_pred = cumulative_lambda(X_val_orig, lambda_models_per_interval[interval_idx])
                y_test_pred = cumulative_lambda(X_test_orig, lambda_models_per_interval[interval_idx])

                training_losses[interval_idx] = self.loss_function(y_train_orig, y_train_pred)
                validation_losses[interval_idx] = self.loss_function(y_val_orig, y_val_pred)
                test_losses[interval_idx] = self.loss_function(y_test_orig, y_test_pred)

            prev_best_lambda_models = lambda_models_per_interval[end_interval - 1]

            # Update residuals for the next submodel using the last equation from the current one
            last_eq_lambda = sub_best_eqs[-1].lambda_format

            X_train_sub = X_train_orig[sub_features]
            X_val_sub = X_val_orig[sub_features]
            X_test_sub = X_test_orig[sub_features]

            y_train_residual -= last_eq_lambda(X_train_sub)
            y_val_residual -= last_eq_lambda(X_val_sub)
            y_test_residual -= last_eq_lambda(X_test_sub)
            
            current_interval_start = end_interval

        return (training_losses, validation_losses, test_losses), all_best_eqs, all_selected_features