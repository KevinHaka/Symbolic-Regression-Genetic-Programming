import numpy as np
import pandas as pd

from typing import Any, Dict, List, Optional, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from npeet import entropy_estimators as ee

from ..utils.stats import permutation_test
from ..utils.system_utils import temporary_seed


def select_features(
    X: pd.DataFrame,
    y: np.ndarray,
    n_permutations: int = 1000,
    alpha: float = 0.01,
    k_nearest_neighbors: int = 5,
    random_state: Optional[int] = None
) -> Tuple[List[str], List[float]]:
    """
    Conditional Mutual Information (CMI) feature selection.

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
        Number of neighbors used by the k-NN entropy estimators (default: 5).
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

    # Set random seed for reproducibility
    with temporary_seed(int(rng.integers(0, 2**32))):
        # Initialize variables
        remaining_features = list(X.columns)
        cmi_values_selected_features = []
        selected_features = []
        cmi_kwargs: Dict[str, Any] = {'k': k_nearest_neighbors}

        # Standardize data for stable MI estimation
        # MI estimation via k-NN requires normalized features
        X_scaled = StandardScaler().fit_transform(X)
        y_scaled = StandardScaler().fit_transform(y.reshape(-1, 1))
        X_scaled = pd.DataFrame(X_scaled, columns=remaining_features)

        # HACK: Sampling for large datasets to speed up CMI estimation
        # Check if dataset exceeds threshold for sampling
        sample_threshold = 200
        should_sample = X_scaled.shape[0] > sample_threshold

        # Greedy feature selection loop
        while remaining_features:
            # Track the feature with the highest CMI
            best_feature = ""
            best_value = -np.inf

            # Sample data if necessary for efficiency
            if should_sample:
                X_sample, _, y_sample, _ = train_test_split(
                    X_scaled, y_scaled, test_size=sample_threshold, random_state=rng.integers(0, 2**32)
                )

            else:
                X_sample = X_scaled
                y_sample = y_scaled

            # Update CMI calculation parameters
            cmi_kwargs.update({
                'y': y_sample,
                'z': X_sample[selected_features] if selected_features else None
            })

            # Compute CMI for each remaining feature
            for feature in remaining_features:
                # Compute I(feature; target | selected_features)
                # If no features selected yet, this is just I(feature; target)
                current_cmi = ee.mi(
                    x=X_sample[feature], 
                    **cmi_kwargs
                )

                # Update max CMI if current is greater
                if current_cmi > best_value:
                    best_value = current_cmi
                    best_feature = feature

            # Perform a permutation test to see if max CMI is significant
            # This tests the null hypothesis that I(feature; target | selected_features) = 0
            reject_null = permutation_test(
                test_statistic = lambda data: ee.mi(x=data, **cmi_kwargs),
                data = X_sample[best_feature],
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