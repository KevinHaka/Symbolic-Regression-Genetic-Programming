import numpy as np
import pandas as pd

from typing import Any, Callable, Dict, List, Optional, Tuple
from inspect import signature

from .gp import GP
from .base import BaseMethod
from ..utils.model_utils import fit_and_evaluate_best_equation
from ..feature_selections.gppi import select_features as gppi_sf

class GPPI(BaseMethod):
    def __init__(
        self,
        val_size: float = signature(gppi_sf).parameters['val_size'].default,
        sub_test_size: float = signature(gppi_sf).parameters['sub_test_size'].default,
        n_runs: int = signature(gppi_sf).parameters['n_runs'].default,
        loss_function: Callable[[np.ndarray, np.ndarray], np.float64] = signature(GP).parameters['loss_function'].default,
        record_interval: Optional[int] = 1,
        resplit_interval: Optional[int] = None,
        pysr_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the GPPPI method.
        
        This method first runs GP, then uses Permutation Importance to select the most important
        features from the resulting equations, and finally runs GP again on the subset of selected features.
        
        Parameters
        ----------
        val_size : float
            Proportion of data to use for validation (only for feature selection).
        sub_test_size : float
            Proportion of data to use for sub-testing (only for feature selection).
        n_runs : int
            Number of GP runs.
        loss_function : Callable[[np.ndarray, np.ndarray], np.float64]
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
        self.val_size = val_size
        self.sub_test_size = sub_test_size
      
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
        Run the GPPI (Genetic Programming with Permutation Importance) method on the provided dataset.

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

        # Compute selected features using GPPI
        selected_features = gppi_sf(
            X_train_val, y_train_val,
            val_size=self.val_size,
            sub_test_size=self.sub_test_size,
            n_runs=self.n_runs,
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

        # Fit and evaluate the best equation using PySR on the filtered dataset
        training_losses, validation_losses, test_losses, best_eqs = fit_and_evaluate_best_equation(
            train_val_test_set_filtered,
            self.loss_function,
            self.events,
            self.pysr_params
        )

        return (training_losses, validation_losses, test_losses), best_eqs, selected_features