import numpy as np
import pandas as pd

from multiprocessing import Manager
from typing import Callable, List, Optional, Tuple

from .base import BaseMethod
from ..utils.pysr_utils import fit_and_evaluate_best_equation, nrmse_loss
from ..feature_selections.shap import select_features as shap_sf
from ..feature_selections.shap import select_features_from_pretrained_models as shap_pretrained_sf

class GPSHAP(BaseMethod):
    def __init__(
        self,
        loss_function: Callable[[np.ndarray, np.ndarray], float] = nrmse_loss,
        record_interval: int = 1,
        **pysr_params
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
        """

        super().__init__(loss_function, record_interval, **pysr_params)
        self._feature_cache = Manager().dict()

    def clear_cache(self):
        """Clear the feature cache."""
        self._feature_cache.clear()

    @staticmethod
    def _get_dataset_key(X: pd.DataFrame) -> int:
        return hash(tuple(X.columns))
    
    def precompute_features(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray,
        **shap_params
    ) -> List[str]:
        """
        Precompute and cache the selected features using SHAP.
        
        Args:
            X (pd.DataFrame): Feature DataFrame.
            y (np.ndarray): Target variable.
            shap_params (dict): Additional parameters for SHAP feature selection.
        
        Returns:
            List[str]: List of selected feature names.
        """

        dataset_key = self._get_dataset_key(X)

        if dataset_key not in self._feature_cache:
            selected_features, _ = shap_sf(X, y, **shap_params)
            self._feature_cache[dataset_key] = selected_features

        return selected_features

    def precompute_features_from_pretrained_models(
        self,
        X_trains: Tuple[pd.DataFrame],
        gp_equations: List[pd.Series],
        n_top_features: Optional[int] = None
    ) -> List[str]:
        """
        Precompute and cache the selected features from pretrained GP models.

        Args:
            X_trains (Tuple[pd.DataFrame]): Tuple of training DataFrames.
            gp_equations (List[pd.Series]): List of GP equations.
            n_top_features (Optional[int]): Number of top features to select.

        Returns:
            List[str]: List of selected feature names.
        """

        dataset_key = self._get_dataset_key(X_trains[0])

        if dataset_key not in self._feature_cache:
            selected_features, _ = shap_pretrained_sf(X_trains, gp_equations, n_top_features)
            self._feature_cache[dataset_key] = selected_features

        return selected_features
        
    def run(
        self,
        train_val_test_set: Tuple[
            pd.DataFrame,     # X_train
            pd.DataFrame,     # X_val
            pd.DataFrame,     # X_test
            np.ndarray,    # y_train
            np.ndarray,    # y_val
            np.ndarray     # y_test
        ]
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray],  # training, validation, test losses
        List[pd.Series],                            # best-equation objects
        List[str],                                  # feature names
    ]:
        """ Execute one full train/validation/test run using GP and SHAP."""

        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_set

        # Get a unique key for the dataset to cache features
        dataset_key = self._get_dataset_key(X_train)

        if dataset_key not in self._feature_cache:
            X = pd.concat([X_train, X_val, X_test], ignore_index=True)
            y = np.concatenate([y_train, y_val, y_test])
            selected_features = self.precompute_features(X, y, **self.pysr_params)
            self._feature_cache[dataset_key] = selected_features

        else:
            selected_features = self._feature_cache[dataset_key]

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