import numpy as np
import pandas as pd

from multiprocessing import Manager
from typing import Any, Callable, Dict, List, Optional, Tuple
from inspect import signature

from .gp import GP
from .base import BaseMethod
from ..utils.model_utils import fit_and_evaluate_best_equation
from ..feature_selections.shap import select_features as shap_sf
from ..feature_selections.shap import select_features_from_pretrained_models as shap_pretrained_sf

class GPSHAP(BaseMethod):
    def __init__(
        self,
        test_size: float = signature(shap_sf).parameters['test_size'].default,
        val_size: float = signature(shap_sf).parameters['val_size'].default,
        n_runs: int = signature(shap_sf).parameters['n_runs'].default,
        n_top_features: int = signature(shap_sf).parameters['n_top_features'].default,
        loss_function: Callable[[np.ndarray, np.ndarray], np.float64] = signature(GP).parameters['loss_function'].default,
        record_interval: Optional[int] = 1,
        resplit_interval: Optional[int] = None,
        pysr_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the GPSHAP method.

        This method first runs GP, then uses SHAP to select the most important
        features from the resulting equations, and finally runs GP again on the
        subset of selected features.

        Parameters
        ----------
        test_size : float
            Proportion of the dataset to include in the test split.
        val_size : float
            Proportion of the dataset to include in the validation split.
        n_runs : int
            Number of runs to perform.
        n_top_features : int
            Number of top features to select.
        loss_function : Callable
            Function to compute the loss between true and predicted values.
        record_interval : int
            Interval at which to record statistics.
        resplit_interval : int
            Interval at which to resplit the training and validation sets.
        pysr_params : Dict
            Parameters for PySRRegressor.
        """

        # Set default PySR parameters if none provided
        if pysr_params is None: pysr_params = {}

        super().__init__(loss_function, record_interval, resplit_interval, pysr_params)
        self._feature_cache = Manager().dict() # Cache for storing selected features
        self.test_size = test_size
        self.val_size = val_size
        self.n_runs = n_runs
        self.n_top_features = n_top_features

    @staticmethod
    def _get_dataset_key(
        X: pd.DataFrame
    ) -> Tuple[str, ...]:
        """ Generate a unique key for the dataset based on its feature names. """

        return tuple(sorted(X.columns))
    
    def clear_feature_cache(
        self
    ) -> None:
        """ Clear the feature cache. """
        self._feature_cache.clear()

    def precompute_features(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray,
        random_state: Optional[int] = None
    ) -> List[str]:
        """
        Precompute and cache the selected features using SHAP.
        
        Args:
            X (pd.DataFrame): Feature DataFrame.
            y (np.ndarray): Target variable.
            random_state (Optional[int]): Random seed for reproducibility.
        
        Returns:
            List[str]: List of selected feature names.
        """

        # Get a unique key for the dataset to cache features
        dataset_key = self._get_dataset_key(X)

        # If features not cached, compute and store them
        if dataset_key not in self._feature_cache:
            gp_params = {
                "loss_function": self.loss_function,
                "record_interval": self.record_interval,
                "resplit_interval": self.resplit_interval,
                "pysr_params": self.pysr_params
            }

            # Compute selected features using SHAP
            selected_features = shap_sf(
                X, y,
                test_size=self.test_size,
                val_size=self.val_size,
                n_runs=self.n_runs,
                n_top_features=self.n_top_features,
                random_state=random_state,
                gp_params=gp_params
            )[0]

            # Cache the selected features
            self._feature_cache[dataset_key] = selected_features

        else:
            selected_features = self._feature_cache[dataset_key]

        return selected_features

    def precompute_features_from_pretrained_models(
        self,
        X_trains: Tuple[pd.DataFrame],
        gp_equations: List[pd.Series],
        n_top_features: Optional[int] = None,
        random_state: Optional[int] = None
    ) -> List[str]:
        """
        Precompute and cache the selected features from pretrained GP models.

        Args:
            X_trains (Tuple[pd.DataFrame]): Tuple of training DataFrames.
            gp_equations (List[pd.Series]): List of GP equations.
            n_top_features (Optional[int]): Number of top features to select.
            random_state (Optional[int]): Random seed for reproducibility.

        Returns:
            List[str]: List of selected feature names.
        """

        # Get a unique key for the dataset to cache features
        dataset_key = self._get_dataset_key(X_trains[0])

        # If features not cached, compute and store them
        if dataset_key not in self._feature_cache:
            # Compute selected features using SHAP from pretrained models
            selected_features, _ = shap_pretrained_sf(
                X_trains,
                gp_equations,
                n_top_features=n_top_features,
                random_state=random_state
            )

            # Cache the selected features
            self._feature_cache[dataset_key] = selected_features

        else:
            selected_features = self._feature_cache[dataset_key]

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
        ],
        random_state: Optional[int] = None
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray],  # training, validation, test losses
        List[pd.Series],                            # best-equation objects
        List[str],                                  # feature names
    ]:
        """ 
        Execute one full train/validation/test run using GP and SHAP.

        Parameters
        ----------
            train_val_test_set: Tuple containing training, validation, and test sets.
            random_state: Random seed for reproducibility. It is not used in this method

        Returns
        -------
        Tuple containing:
            - training, validation, and test losses (each as np.ndarray),
            - list of best-equation objects (pd.Series),
            - list of feature names (List[str]).
        """

        del random_state  # Unused variable

        # Unpack the sets
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_set

        # Get a unique key for the dataset to cache features
        dataset_key = self._get_dataset_key(X_train)

        # If features not cached, retrieve them
        if dataset_key in self._feature_cache.keys():
            selected_features = self._feature_cache[dataset_key]

        else:
            raise ValueError(
                "Features have not been precomputed.\n"+
                "Please run `precompute_features` or\n" +
                "`precompute_features_from_pretrained_models` before calling `run`."
            )
        
        # If no features were selected, choose one feature at random
        if not selected_features:
            selected_features = np.random.choice(X_train.columns, 1).tolist()

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
            self.resplit_interval,
            self.pysr_params
        )

        return (training_losses, validation_losses, test_losses), best_eqs, selected_features