import numpy as np
import pandas as pd

from ..utils.pysr_utils import permutation_test, temporary_seed

from sklearn.preprocessing import StandardScaler
from npeet import entropy_estimators as ee
from typing import List, Optional, Tuple

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

    # Create a main RNG and global seed for reproducibility
    if random_state is None:
        global_seed = None
        main_rng = None
        perm_seed = None

    else:
        main_rng = np.random.default_rng(random_state)
        global_seed = int(main_rng.integers(0, 2**32))

    # Set random seed for reproducibility
    with temporary_seed(global_seed):

        # Initialize variables
        remaining_features = list(X.columns)
        cmi_values_selected_features = []
        selected_features = []

        # Standardize data for stable MI estimation
        # MI estimation via k-NN requires normalized features
        X_scaled = StandardScaler().fit_transform(X)
        y_scaled = StandardScaler().fit_transform(y.reshape(-1, 1))
        X_scaled = pd.DataFrame(X_scaled, columns=remaining_features)

        # Initial CMI calculation parameters
        cmi_kwargs = {'y': y_scaled, 'k': k_nearest_neighbors}

        # Greedy feature selection loop
        while remaining_features:

            # Track the feature with the highest CMI
            best_feature = ""
            best_value = -np.inf

            # Compute CMI for each remaining feature
            for feature in remaining_features:
                
                # Compute I(feature; target | selected_features)
                # If no features selected yet, this is just I(feature; target)
                current_cmi = ee.mi(
                    x=X_scaled[feature], 
                    y=y_scaled, 
                    z=X_scaled[selected_features] if selected_features else None,
                    k=k_nearest_neighbors
                )
                
                # Update max CMI if current is greater
                if current_cmi > best_value:
                    best_value = current_cmi
                    best_feature = feature

            # Update CMI calculation parameters
            cmi_kwargs['z'] = X_scaled[selected_features] if selected_features else None

            # Produce a new random seed for the permutation test
            if main_rng is not None: perm_seed = int(main_rng.integers(0, 2**32))

            # Perform a permutation test to see if max CMI is significant
            # This tests the null hypothesis that I(feature; target | selected_features) = 0
            res = permutation_test(
                test_statistic = lambda data: ee.mi(x=data, **cmi_kwargs),
                data = X_scaled[best_feature],
                observed_statistic = best_value,
                n_permutations = n_permutations,
                alpha = alpha,
                alternative = 'greater',
                decision_by = 'p_value',
                random_state = perm_seed
            )

            # Check if cmi is significant; if not, stop selection
            if res['reject_null']:

                # Add selected feature to results and remove from candidates
                selected_features.append(best_feature)
                cmi_values_selected_features.append(best_value)
                remaining_features.remove(best_feature)

            else: break
        
        return selected_features, cmi_values_selected_features