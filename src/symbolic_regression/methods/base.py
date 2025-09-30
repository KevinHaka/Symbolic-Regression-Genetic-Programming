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

        self.loss_function = loss_function
        self.record_interval = record_interval
        self.resplit_interval = resplit_interval
        self.pysr_params = pysr_params

        niterations = self.pysr_params.get("niterations", signature(PySRRegressor).parameters['niterations'].default)
        self.n_records = niterations // self.record_interval

    @abstractmethod
    def run(self, *args, **kwargs) -> Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray],  # training, validation, test losses
        List[pd.Series],                            # best-equation objects
        List[str],                                  # feature names
    ]:
        """
        Run the symbolic regression method.
        """

        raise NotImplementedError