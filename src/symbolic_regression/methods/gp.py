# from symbolic_regression.methods.base import BaseMethod
from base import BaseMethod
from ..utils.pysr_utils import fit_and_evaluate_best_equation, nrmse_loss

import numpy as np
from pandas import DataFrame, Series

from pysr import PySRRegressor
from typing import Any, Callable, Tuple, Dict



class GP(BaseMethod):
    def __init__(
        self,
        loss_function: Callable[[np.ndarray, np.ndarray], float] = nrmse_loss,
        record_interval: int = 1,
        pysr_params: Dict[str, Any] = {}
    ):
        """
        Initialize the GP method with loss function, record interval, and PySR parameters.

        Parameters
        ----------
        loss_function : Callable
            Function to compute the loss between true and predicted values.
        record_interval : int
            Interval at which to record statistics.
        pysr_params : Dict
            Parameters for PySRRegressor.
        """
        super().__init__(loss_function, record_interval, pysr_params)

        # Initialize PySRRegressor with provided parameters
        self.pysr_regressor = PySRRegressor(**self.pysr_params)

    def run(
        self,
        train_val_test_set: Tuple[
            DataFrame,  # X_train
            DataFrame,  # X_val
            DataFrame,  # X_test
            Series,     # y_train
            Series,     # y_val
            Series      # y_test
        ],
    ) -> Tuple[
            Dict[str, list],  # loss histories
            list[str],        # feature names
            list[Series]      # best-equation objects
        ]:
        """
        Execute one full train/validation/test run using GP.
        """

        features = train_val_test_set[0].columns.tolist()

        training_losses, validation_losses, test_losses, best_eqs = fit_and_evaluate_best_equation(
            train_val_test_set,
            self.loss_function,
            self.record_interval,
            self.pysr_params
        )

        results = {
            "training_loss": training_losses,
            "validation_loss": validation_losses,
            "test_loss": test_losses
        }

        return results, features, best_eqs