import pandas as pd
import numpy as np

from typing import Any, Callable, List, Tuple, Dict

from .base import BaseMethod
from ..utils.pysr_utils import fit_and_evaluate_best_equation, nrmse_loss

class GP(BaseMethod):
    def __init__(
        self,
        loss_function: Callable[[np.ndarray, np.ndarray], float] = nrmse_loss,
        record_interval: int = 1,
        pysr_params: Dict[str, Any] = {}
    ) -> None:
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

    def run(
        self,
        train_val_test_set: Tuple[
            pd.DataFrame,  # X_train
            pd.DataFrame,  # X_val
            pd.DataFrame,  # X_test
            np.ndarray,    # y_train
            np.ndarray,    # y_val
            np.ndarray     # y_test
        ],
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray],  # training, validation, test losses
        List[pd.Series],                            # best-equation objects
        List[str],                                  # feature names
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

        return (training_losses, validation_losses, test_losses), best_eqs, features