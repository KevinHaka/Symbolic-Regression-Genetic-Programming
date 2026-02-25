import numpy as np
import pandas as pd

from typing import Any, Dict, List, Optional, Tuple

from sklearn.preprocessing import StandardScaler
from npeet import entropy_estimators as ee
from joblib import Parallel, delayed

from ..utils.stats import permutation_test
from ..utils.system_utils import temporary_seed

def compute_cmi(
    x: pd.Series, 
    y: np.ndarray, 
    z: Optional[pd.DataFrame], 
    k: int = 3,
    base: Optional[int] = 2,
    alpha: Optional[float] = 0,
    random_state: Optional[int] = None
) -> np.float64:
    """
    Compute the conditional mutual information I(x; y | z) using k-NN estimators from the npeet package.
    """
    
    cmi_kwargs = {
        'y': y,
        'z': z,
        'k': k,
        'base': base,
        'alpha': alpha
    }

    # Set a temporary random seed for reproducibility of the k-NN estimation
    with temporary_seed(random_state):
        return ee.mi(x=x, **cmi_kwargs)

def select_features(
    X: pd.DataFrame,
    y: np.ndarray,
    n_permutations: int = 1000,
    alpha: float = 0.01,
    k_nearest_neighbors: int = 3,
    random_state: Optional[int] = None
) -> Tuple[List[str], List[float]]:
    """
    Conditional Mutual Information with Nearest Neighbors (CMINN) feature selection algorithm.

    This module implements a greedy feature selection algorithm based on
    conditional mutual information estimated with k-NN entropy estimators
    (from the npeet package). At each step, it selects the remaining feature
    that maximizes I(X_i; y | S), where S is the set of already selected
    features, and uses a permutation (shuffle) test to decide when to stop.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with shape (n_samples, n_features).
    y : np.ndarray
        Target vector with shape (n_samples,).
    n_permutations : int, optional
        Number of permutations to use in the shuffle test for significance (default: 1000).
    alpha : float
        Significance level for the permutation one-sided test (default: 0.01).
    k_nearest_neighbors : int, optional
        Number of neighbors used by the k-NN entropy estimators (default: 4).
    random_state : Optional[int], optional
        Random seed for reproducibility (default: None).

    Returns
    -------
    Tuple[List[str], List[float]]
        A tuple containing:
        - selected_features: ordered list of selected feature names (strings),
        - cmi_values_selected_features: corresponding estimated CMI values.
    """

    # Initialize random number generator
    rng = np.random.default_rng(random_state)

    # Initialize variables
    remaining_features = list(X.columns)
    cmi_values_selected_features = []
    selected_features = []

    # Standardize data for stable MI estimation
    # MI estimation via k-NN requires normalized features
    X_scaled = StandardScaler().fit_transform(X)
    y_scaled = StandardScaler().fit_transform(y.reshape(-1, 1))
    X_scaled = pd.DataFrame(X_scaled, columns=remaining_features)

    # Precompute CMI estimation parameters that don't change during selection
    cmi_kwargs: Dict[str, Any] = {
        "y": y_scaled,
        'k': k_nearest_neighbors
    }

    # Greedy feature selection loop
    while remaining_features:
        # Update the conditioning set for CMI estimation
        cmi_kwargs['z'] = X_scaled[selected_features] if selected_features else None

        # Compute CMI for each remaining feature IN PARALLEL
        results = Parallel(n_jobs=-1)(
            delayed(compute_cmi)(X_scaled[feature], **cmi_kwargs, random_state=int(rng.integers(0, 2**32))) 
            for feature in remaining_features
        )

        # Find the best feature from parallel results
        np_results = np.array(results)
        best_idx = np.argmax(np_results)
        best_feature = remaining_features[best_idx]
        best_value = np_results[best_idx]

        # Perform a permutation test to see if max CMI is significant
        # This tests the null hypothesis that I(feature; target | selected_features) = 0
        reject_null = permutation_test(
            test_statistic = lambda data, random_state: compute_cmi(data, **cmi_kwargs, random_state=random_state),
            data = X_scaled[best_feature],
            observed_statistic = best_value,
            n_permutations = n_permutations,
            alpha = alpha,
            alternative = 'greater',
            decision_by = 'p_value',
            random_state = rng.integers(0, 2**32)
        )['reject_null']

        # Check if cmi is significant; if not, stop selection
        if reject_null:
            # Add selected feature to results and remove from candidates
            selected_features.append(best_feature)
            cmi_values_selected_features.append(best_value)
            remaining_features.remove(best_feature)

        else: break
    return selected_features, cmi_values_selected_features