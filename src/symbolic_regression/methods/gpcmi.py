import numpy as np
import pandas as pd

from typing import Callable, Optional, Tuple, List
from inspect import signature

from .base import BaseMethod
from ..utils.pysr_utils import fit_and_evaluate_best_equation, nrmse_loss
from ..feature_selections.cmi import select_features as cmi_sf

class GPCMI(BaseMethod):
    def __init__(
        self,
        ns: int = signature(cmi_sf).parameters['ns'].default,
        ci: float = signature(cmi_sf).parameters['ci'].default,
        k: int = signature(cmi_sf).parameters['k'].default,
        loss_function: Callable[[np.ndarray, np.ndarray], float] = nrmse_loss,
        record_interval: int = 1,
        **pysr_params
    ) -> None:
        """
        Initialize the GPCMI method.

        Args:
            ns (int): Number of samples for CMI estimation.
            ci (float): Confidence interval for CMI significance testing.
            k (int): Number of neighbors for k-NN.
            loss_function (Callable): Function to calculate the loss.
            record_interval (int): Interval at which to record statistics.
            pysr_params (Dict[str, Any]): Parameters for PySR.
        """

        super().__init__(loss_function, record_interval, **pysr_params)
        self.ns = ns
        self.ci = ci
        self.k = k

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
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray],  # training, validation, test losses
        List[pd.Series],                            # best-equation objects
        List[str],                                  # feature names
    ]:
        """ Execute one full train/validation/test run using GP and CMI."""

        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_set

        selected_features, _ = cmi_sf(
            X_train, 
            y_train,
            ns=self.ns,
            ci=self.ci,
            k=self.k
        )

        if not selected_features:
            selected_features = np.random.choice(X_train.columns, 1).tolist()

        # Create a new data split with only the selected features
        train_val_test_set_filtered = (
            X_train[selected_features],
            X_val[selected_features],
            X_test[selected_features],
            y_train,
            y_val,
            y_test,
        )

        features = train_val_test_set_filtered[0].columns.tolist()

        training_losses, validation_losses, test_losses, best_eqs = fit_and_evaluate_best_equation(
            train_val_test_set_filtered,
            self.loss_function,
            self.record_interval,
            self.pysr_params
        )

        return (training_losses, validation_losses, test_losses), best_eqs, features