import numpy as np
import pandas as pd

from typing import Any, Callable, Dict, List, Optional, Tuple
from inspect import signature

from .gp import GP
from .base import BaseMethod
from ..utils.model_utils import fit_and_evaluate_best_equation
from ..feature_selections.gpshap import select_features as gpshap_sf
from ..feature_selections.gpshap import select_features_from_pretrained_models as gpshap_pretrained_sf

class GPSHAP(BaseMethod):
    def __init__(
        self,
        n_runs: int = signature(gpshap_sf).parameters['n_runs'].default,
        n_top_features: int = signature(gpshap_sf).parameters['n_top_features'].default,
        loss_function: Callable[[np.ndarray, np.ndarray], np.float64] = signature(GP).parameters['loss_function'].default,
        record_interval: Optional[int] = 1,
        resplit_interval: Optional[int] = None,
        pysr_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the GPSHAP method.

        This method first runs GP, then uses Shapley values to select the most important
        features from the resulting equations, and finally runs GP again on the
        subset of selected features.

        Parameters
        ----------
        n_runs : int
            Number of GP runs to perform.
        n_top_features : Optional[int]
            Number of top features to select. If None, defaults to log2(n_features).
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
        self.n_runs = n_runs
        self.n_top_features = n_top_features
        
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
        Run the GPSHAP method on the provided dataset.

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

        # Unpack the sets
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_set

        # Combine training and validation sets for feature selection
        X_train_val = pd.concat([X_train, X_val], axis=0)
        y_train_val = np.concatenate([y_train, y_val], axis=0)

        # Compute selected features using GPSHAP
        selected_features = gpshap_sf(
            X_train_val, y_train_val,
            val_size=len(y_val) / (len(y_train) + len(y_val) + len(y_test)),
            n_runs=self.n_runs,
            n_top_features=self.n_top_features,
            random_state=random_state,
            pysr_params=self.pysr_params
        )[0]

        # Create a new data split with only the selected features
        train_val_test_set_filtered = (
            X_train[selected_features],
            X_val[selected_features],
            X_test[selected_features],
            y_train,
            y_val,
            y_test
        )

        # Train and evaluate the best equation on the filtered dataset
        training_losses, validation_losses, test_losses, best_eqs = fit_and_evaluate_best_equation(
            train_val_test_set_filtered,
            self.loss_function,
            self.events,
            self.pysr_params
        )

        return (training_losses, validation_losses, test_losses), best_eqs, selected_features