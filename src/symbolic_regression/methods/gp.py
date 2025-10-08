import pandas as pd
import numpy as np

from typing import Callable, List, Optional, Tuple, Dict, Any

from .base import BaseMethod
from ..utils.pysr_utils import fit_and_evaluate_best_equation, nrmse_loss

class GP(BaseMethod):
    def __init__(
        self,
        loss_function: Callable[[np.ndarray, np.ndarray], np.float64] = nrmse_loss,
        record_interval: Optional[int] = 1,
        resplit_interval: Optional[int] = None,
        pysr_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Genetic Programming (GP) method using PySR.

        Parameters
        ----------
        loss_function : Callable
            Function to compute the loss between true and predicted values.
        record_interval : Optional[int]
            Interval at which to record statistics.
        resplit_interval : Optional[int]
            Interval at which to resplit the training and validation sets.
        pysr_params : Dict
            Parameters for PySRRegressor.
        """

        if pysr_params is None: pysr_params = {}

        super().__init__(loss_function, record_interval, resplit_interval, pysr_params)

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
        random_state: Optional[int] = None
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray],  # training, validation, test losses
        List[pd.Series],                            # best-equation objects
        List[str],                                  # feature names
    ]:
        """
        Execute one full train/validation/test run using GP.

        Parameters
        ----------
        train_val_test_set : Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]
            The training, validation, and test sets.
        random_state : Optional[int], optional
            Random seed, but it is not used since PySR has its own random state handling.
            It is included for API consistency.
        
        Returns
        -------
        Tuple containing:
            - training, validation, and test losses (each as np.ndarray),
            - list of best-equation objects (pd.Series),
            - list of feature names (List[str]).
        """

        # Extract feature names
        features = train_val_test_set[0].columns.tolist() 
        
        # Fit and evaluate the best equation using PySR
        training_losses, validation_losses, test_losses, best_eqs = fit_and_evaluate_best_equation(
            train_val_test_set,
            self.loss_function,
            self.record_interval,
            self.resplit_interval,
            self.pysr_params
        )

        return (training_losses, validation_losses, test_losses), best_eqs, features