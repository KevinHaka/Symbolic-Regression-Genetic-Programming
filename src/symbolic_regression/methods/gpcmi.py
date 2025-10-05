import numpy as np
import pandas as pd

from typing import Any, Callable, Dict, Optional, Tuple, List
from inspect import signature

from .base import BaseMethod
from ..utils.pysr_utils import fit_and_evaluate_best_equation, nrmse_loss
from ..feature_selections.cmi import select_features as cmi_sf

class GPCMI(BaseMethod):
    def __init__(
        self,
        n_permutations: int = signature(cmi_sf).parameters['n_permutations'].default,
        alpha: float = signature(cmi_sf).parameters['alpha'].default,
        k_nearest_neighbors: int = signature(cmi_sf).parameters['k_nearest_neighbors'].default,
        loss_function: Callable[[np.ndarray, np.ndarray], np.float64] = nrmse_loss,
        record_interval: Optional[int] = 1,
        resplit_interval: Optional[int] = None,
        pysr_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the GPCMI method.

        Args:
            n_permutations (int): Number of permutations for the permutation test.
            alpha (float): Significance level for the permutation test.
            k_nearest_neighbors (int): Number of neighbors for k-NN entropy estimation.
            loss_function (Callable): Function to calculate the loss.
            record_interval (int): Interval at which to record statistics.
            resplit_interval (int): Interval at which to resplit the training and validation sets.
            pysr_params (Dict[str, Any]): Parameters for PySR.
        """

        if pysr_params is None: pysr_params = {}

        super().__init__(loss_function, record_interval, resplit_interval, pysr_params)
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.k_nearest_neighbors = k_nearest_neighbors

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
        Run the GPCMI method on the provided dataset.

        Args:
            train_val_test_set (Tuple): A tuple containing the training, validation, and test sets.
            random_state (Optional[int]): Random seed for reproducibility.

        Returns:
            Tuple containing:
            - training, validation, and test losses (np.ndarray),
            - list of best-equation objects (List[pd.Series]),
            - list of feature names used (List[str]).
        """

        # Unpack train/val/test set
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_set

        # Select features using CMI-based selection
        selected_features, _ = cmi_sf(
            X=X_train, 
            y=y_train,
            n_permutations=self.n_permutations,
            alpha=self.alpha,
            k_nearest_neighbors=self.k_nearest_neighbors,
            random_state=random_state
        )

        # If no features were selected, choose one feature at random
        if not selected_features:
            selected_features = np.random.choice(X_train.columns, 1).tolist()

        # Filter datasets to only include selected features
        train_val_test_set_filtered = (
            X_train[selected_features],
            X_val[selected_features],
            X_test[selected_features],
            y_train,
            y_val,
            y_test,
        )

        # Get feature names
        features = train_val_test_set_filtered[0].columns.tolist()

        # Fit and evaluate the best equation using the filtered data
        training_losses, validation_losses, test_losses, best_eqs = fit_and_evaluate_best_equation(
            train_val_test_set_filtered,
            self.loss_function,
            self.record_interval,
            self.resplit_interval,
            self.pysr_params
        )

        return (training_losses, validation_losses, test_losses), best_eqs, features