import pandas as pd
import numpy as np

from joblib import Parallel, delayed
from pysr_utils import nrmse_loss, train_val_test_split, fit_and_evaluate_best_equation

from typing import Callable, Optional
# from inspect import signature

# from pysr import PySRRegressor

class BaseSR:
    def __init__(
        self,
        loss_function: Callable = nrmse_loss,
        record_interval: int = 1,
        test_size: float = 0.2,
        val_size: float = 0.25,
        pysr_params: Optional[dict] = None,
    ):
        self.loss_function = loss_function
        self.record_interval = record_interval
        self.test_size = test_size
        self.val_size = val_size
        self.pysr_params = pysr_params or {}
        
        # if "niterations" not in self.pysr_params:
        #     self.pysr_params["niterations"] = signature(PySRRegressor).parameters["niterations"].default

    def _run_once(self, X, y, train_val_test_sets=None):
        if train_val_test_sets is None:
            train_val_test_sets = train_val_test_split(
                X, y, self.test_size, self.val_size
            )
        return fit_and_evaluate_best_equation(
            train_val_test_sets,
            self.loss_function,
            self.record_interval,
            self.pysr_params,
        )

    def run(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        n_runs: int = 100
    ):
        n_records = self.pysr_params['niterations'] // self.record_interval
        results = {
            "training_loss": np.empty((n_runs, n_records)),
            "validation_loss": np.empty((n_runs, n_records)),
            "test_loss": np.empty((n_runs, n_records)),
        }
        outputs = Parallel(n_jobs=-1)(
            delayed(self._run_once)(X, y) for _ in range(n_runs)
        )
        for i, (tr, va, te, _) in enumerate(outputs):
            results["training_loss"][i] = tr
            results["validation_loss"][i] = va
            results["test_loss"][i] = te
        return results, [output[3] for output in outputs]
    
class GP(BaseSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method_name = "GP"

class GPSHAP(BaseSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method_name = "GPSHAP"

    def feature_selection(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        n_runs: int = 100, 
        n_features: int = 10
    ):
        train_val_test_sets = train_val_test_split(X, y, self.test_size, self.val_size)
        return shap_feature_selection(
            train_val_test_sets,
            self.run_once,
            n_runs,
            n_features,
            self.loss_function,
            self.pysr_params
        )