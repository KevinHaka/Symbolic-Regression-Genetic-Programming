import numpy as np
import pandas as pd

from inspect import signature
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, List, Dict, Any

from pysr import PySRRegressor

class BaseMethod(ABC):
    """Abstract interface for a symbolic-regression method."""

    def __init__(
        self,
        loss_function: Callable[[np.ndarray, np.ndarray], np.float64],
        record_interval: Optional[int],
        resplit_interval: Optional[int],
        pysr_params: Dict[str, Any],
    ) -> None:
        """
        Shared init logic for all methods.

        Args:
            loss_function: function (y_true, y_pred) -> float
            record_interval: record stats every N iterations
            resplit_interval: resplit train/val every N iterations
            pysr_params: parameters for PySRRegressor
        """

        # Get default niterations from PySRRegressor if not provided
        niterations = pysr_params.get("niterations", signature(PySRRegressor).parameters['niterations'].default)

        # Store parameters
        self.loss_function = loss_function
        self.record_interval = record_interval if record_interval else niterations
        self.resplit_interval = resplit_interval if resplit_interval else niterations
        self.pysr_params = pysr_params

        # Calculate total number of records based on niterations and record_interval
        self.n_records = niterations // self.record_interval
        
        # Determine record and resplit points
        self.record_points = list(range(self.record_interval, niterations + 1, self.record_interval))
        self.resplit_points = list(range(self.resplit_interval, niterations, self.resplit_interval))

        # Build event schedule for recording and resplitting
        self.events: Dict[int, set] = {}
        for it in self.record_points: self.events.setdefault(it, set()).add("record")
        for it in self.resplit_points: self.events.setdefault(it, set()).add("resplit")

    @abstractmethod
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
        Run the symbolic regression method.

        Args:
            train_val_test_set: tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
            random_state: random seed for reproducibility.
        """

        raise NotImplementedError