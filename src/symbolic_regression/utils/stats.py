import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from typing import Any, Callable, Dict, Optional

def permutation_test(
    test_statistic: Callable,
    data: pd.Series,
    observed_statistic: Optional[float] = None,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    alternative: str = 'two-sided',
    decision_by: str = 'p_value',
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Perform a permutation test for a univariate statistic.
    
    The p-value is computed using the Phipson & Smyth (2010, 
    Permutation P-values should never be zero) adjustment.

    Parameters
    ----------
    test_statistic : Callable
        Function computing the test statistic given the data and a random state for reproducibility.
    data : np.ndarray
        Sample used to build the null distribution.
    observed_statistic : float, optional
        Precomputed observed statistic; if None it is evaluated from ``data``.
    n_permutations : int, default=1000
        Number of random permutations used to approximate the null distribution.
    alpha : float, default=0.05
        Significance level for confidence interval or p-value decision.
    alternative : {'two-sided', 'greater', 'less'}, default='two-sided'
        Alternative hypothesis controlling tail calculations.
    decision_by : {'p_value', 'interval'}, default='p_value'
        Whether to reject the null using the p-value or the confidence interval.
    random_state : int, optional
        Seed for the permutation generator.

    Returns
    -------
    Dict[str, Any]
        Dictionary with the following keys:
        - 'observed_statistic': The observed statistic.
        - 'null_distribution': The null distribution of the statistic.
        - 'p_value': The computed p-value.
        - 'reject_null': Boolean indicating whether to reject the null hypothesis.
        - 'confidence_interval': The (lower, upper) bounds of the confidence interval.
    """
    
    # Set seed for reproducibility
    rng = np.random.default_rng(random_state)

    # Calculate the observed statistic if not provided
    if observed_statistic is None: observed_statistic = test_statistic(data, random_state=int(rng.integers(0, 2**32)))
    assert isinstance(observed_statistic, float), "The observed statistic must be a numeric value."

    # Generate the null distribution and sort it
    null_distribution = np.array(
        Parallel(n_jobs=-1)(
            delayed(test_statistic)(rng.permutation(data), int(rng.integers(0, 2**32))) 
            for _ in range(n_permutations)
        )
    )
    null_distribution.sort()

    # Calculate p-value and confidence intervals based on the alternative hypothesis
    match alternative:
        case 'two-sided':
            count_greater = np.sum(null_distribution >= observed_statistic)
            count_less = np.sum(null_distribution <= observed_statistic)
            p_value = 2 * min(
                (count_greater + 1) / (n_permutations + 1),
                (count_less + 1) / (n_permutations + 1)
            )
            p_value = min(1.0, p_value) # clip to 1
            lower_bound = np.quantile(null_distribution, alpha / 2)
            upper_bound = np.quantile(null_distribution, 1 - alpha / 2)

        case 'greater':
            count = np.sum(null_distribution >= observed_statistic)
            p_value = (count + 1) / (n_permutations + 1)
            lower_bound = None
            upper_bound = np.quantile(null_distribution, 1 - alpha)

        case 'less':
            count = np.sum(null_distribution <= observed_statistic)
            p_value = (count + 1) / (n_permutations + 1)
            lower_bound = np.quantile(null_distribution, alpha)
            upper_bound = None

        case _: raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'.")
        
    # Decide whether to reject the null hypothesis
    match decision_by:
        case 'p_value': reject_null = p_value <= alpha
        case 'interval':
            match alternative:
                case 'two-sided':
                    assert lower_bound is not None and upper_bound is not None
                    reject_null = not (lower_bound < observed_statistic < upper_bound)
                
                case 'greater':
                    assert upper_bound is not None
                    reject_null = observed_statistic >= upper_bound
                
                case 'less':
                    assert lower_bound is not None
                    reject_null = observed_statistic <= lower_bound
        
        case _: raise ValueError("decision_by must be 'p_value' or 'interval'.")

    return {
        "observed_statistic": observed_statistic,
        "null_distribution": null_distribution,
        "p_value": p_value,
        "reject_null": reject_null,
        "confidence_interval": (lower_bound, upper_bound)
    }