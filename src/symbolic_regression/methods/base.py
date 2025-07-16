from typing import Callable, Tuple, Dict, List, Any
from abc import ABC, abstractmethod
from inspect import signature

import numpy as np
from pandas import DataFrame, Series


class BaseMethod(ABC):
    """Abstract interface for a symbolic-regression method."""

    def __init__(
        self,
        loss_function: Callable[[np.ndarray, np.ndarray], float],
        record_interval: int,
        pysr_params: Dict[str, Any],
    ) -> None:
        """
        Shared init logic for all methods.

        Args:
            loss_function: function (y_true, y_pred) -> float
            record_interval: record stats every N iterations
            pysr_params: parameters for PySRRegressor
        """

        self.loss_function = loss_function
        self.record_interval = record_interval
        self.pysr_params = pysr_params

        self.niterations = self.pysr_params.get("niterations", signature(PySRRegressor).parameters['niterations'].default)
        self.n_records = self.niterations // self.record_interval

    @abstractmethod
    def run(
        self,
        data_split: Tuple[
            DataFrame,  # X_train
            DataFrame,  # X_val
            DataFrame,  # X_test
            Series,     # y_train
            Series,     # y_val
            Series      # y_test
        ],
    ) -> Tuple[
        Dict[str, List[float]],  # loss histories
        List[str],               # feature names
        List[Series]             # best-equation objects
    ]:
        """
        Execute one full train/validation/test run.

        Returns:
            A tuple of:
            - dict with keys 'training_loss','validation_loss','test_loss'
            - list of feature names
            - list of best-equations captured at each record interval
        """
        raise NotImplementedError