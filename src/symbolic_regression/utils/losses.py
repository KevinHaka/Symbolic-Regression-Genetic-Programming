import numpy as np
from typing import Tuple

def mse_loss(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> np.float64:
    """
    Computes the Mean Squared Error (MSE) between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns
    -------
    np.float64
        The MSE value.
    """
    return ((y_true - y_pred) ** 2).mean()

def rmse_loss(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> np.float64:
    """
    Computes the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns
    -------
    np.float64
        The RMSE value.
    """
    return np.sqrt(mse_loss(y_true, y_pred))

def nrmse_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = 'range',
    iqr_quantiles: Tuple[float, float] = (25.0, 75.0)
) -> np.float64:
    """
    Computes the Normalized Root Mean Squared Error (NRMSE) between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    method : str
        Normalization method to use:
            - 'mean': Normalizes by the mean of y_true.
            - 'range': Normalizes by the range (max - min) of y_true.
            - 'std': Normalizes by the standard deviation of y_true.
            - 'iqr': Normalizes by the interquantile range (IQR) defined by iqr_quantiles.
    iqr_quantiles : tuple(float, float), optional
        Percentiles for the IQR when method == 'iqr'. Defaults to (25.0, 75.0).

    Returns
    -------
    np.float64
        The NRMSE value.
    """
    rmse = rmse_loss(y_true, y_pred)
    method = method.lower()

    match method:
        case 'mean':
            denominator = y_true.mean()

        case 'range':
            denominator = y_true.max() - y_true.min()

        case 'std':
            denominator = y_true.std()

        case 'iqr':
            q_low, q_high = float(iqr_quantiles[0]), float(iqr_quantiles[1])

            if not (0.0 <= q_low < q_high <= 100.0):
                raise ValueError("iqr_quantiles must satisfy 0 <= low < high <= 100")
            
            denominator = np.percentile(y_true, q_high) - np.percentile(y_true, q_low)

        case _:
            raise ValueError(f"Unknown normalization method: {method}")

    return rmse / denominator