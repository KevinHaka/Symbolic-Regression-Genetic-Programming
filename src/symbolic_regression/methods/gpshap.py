import numpy as np
import pandas as pd

from typing import Any, Callable, List, Tuple, Dict

from .base import BaseMethod
from ..utils.pysr_utils import fit_and_evaluate_best_equation, nrmse_loss

class GPSHAP(BaseMethod):
    def __init__(
        self,
        loss_function: Callable[[np.ndarray, np.ndarray], float] = nrmse_loss,
        record_interval: int = 1,
        pysr_params: Dict[str, Any] = {},
    ):
        """
        Initialize the GPSHAP method.

        This method first runs GP, then uses SHAP to select the most important
        features from the resulting equations, and finally runs GP again on the
        subset of selected features.

        Parameters
        ----------
        loss_function : Callable
            Function to compute the loss between true and predicted values.
        record_interval : int
            Interval at which to record statistics.
        pysr_params : Dict
            Parameters for PySRRegressor.
        n_top_features : int, optional
            Number of top features to select using SHAP. If None, it defaults to
            log2(number of features).
        """

        super().__init__(loss_function, record_interval, pysr_params)

    def run(
        self,
        train_val_test_set: Tuple[
            pd.DataFrame,     # X_train
            pd.DataFrame,     # X_val
            pd.DataFrame,     # X_test
            np.ndarray,    # y_train
            np.ndarray,    # y_val
            np.ndarray     # y_test
        ],
        selected_features: List[str],
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray],  # training, validation, test losses
        List[pd.Series],                            # best-equation objects
        List[str],                                  # feature names
    ]:
        """ Execute one full train/validation/test run using GP and SHAP."""

        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_set

        # Create a new data split with only the selected features
        train_val_test_set_filtered = (
            X_train[selected_features],
            X_val[selected_features],
            X_test[selected_features],
            y_train,
            y_val,
            y_test
        )

        # Run GP on the filtered dataset
        training_losses, validation_losses, test_losses, best_eqs = fit_and_evaluate_best_equation(
            train_val_test_set_filtered,
            self.loss_function,
            self.record_interval,
            self.pysr_params
        )

        return (training_losses, validation_losses, test_losses), best_eqs, selected_features