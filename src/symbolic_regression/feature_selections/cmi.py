import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from npeet import entropy_estimators as ee
from typing import List, Tuple

def select_features(
    X: pd.DataFrame,
    y: np.ndarray,
    ns: int = 1000,
    ci: float = 0.99,
    k: int = 5
) -> Tuple[List[str], List[float]]:
    
    # Initialize algorithm state
    scaler = StandardScaler()
    remaining_features = list(X.columns)
    cmi_values_selected_features = []
    selected_features = []

    # Standardize data for stable MI estimation
    # MI estimation via k-NN requires normalized features
    X_scaled = scaler.fit_transform(X.copy())
    y_scaled = scaler.fit_transform(y.copy().reshape(-1, 1))
    X_scaled = pd.DataFrame(X_scaled, columns=remaining_features)

    # Greedy feature selection loop
    while remaining_features:
        cmi_per_feature = {}

        # Compute CMI for each remaining feature
        for feature in remaining_features:
            # Compute I(feature; target | selected_features)
            # If no features selected yet, this is just I(feature; target)
            cmi_per_feature[feature] = ee.mi(
                X_scaled[feature], 
                y_scaled, 
                X_scaled[selected_features] if selected_features else None,
                k=k
            )

        # Select feature with highest CMI
        max_cmi_feature = max(cmi_per_feature.keys(), key=lambda k: cmi_per_feature[k])
        max_cmi_value = cmi_per_feature[max_cmi_feature]

        _, (lower_bound, upper_bound) = ee.shuffle_test(
            measure=ee.mi,
            x=X_scaled[feature],
            y=y_scaled,
            z=X_scaled[selected_features].to_numpy().tolist() if selected_features else None,  # type: ignore
            ns=ns,
            ci=ci,
            k=k
        )

        # Check if CMI is statistically significant
        if lower_bound < max_cmi_value < upper_bound: break

        # Add selected feature to results and remove from candidates
        selected_features.append(max_cmi_feature)
        cmi_values_selected_features.append(cmi_per_feature[max_cmi_feature])
        remaining_features.remove(max_cmi_feature)
    
    return selected_features, cmi_values_selected_features