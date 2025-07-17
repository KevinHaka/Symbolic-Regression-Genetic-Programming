from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

class BaseFeatureSelector(ABC):
    """
    Abstract base class for feature selection methods.
    Provides a common interface for feature selection implementations.
    """

    def __init__(
            self,
            n_top_features: Optional[int] = None
    ) -> None:
        """
        Initialize the feature selection class.

        Parameters
        ----------
        n_top_features : int, optional
            Number of top features to select.
        """

        self.n_top_features = n_top_features

    
    @abstractmethod
    def select_features(self, *args, **kwargs) -> Tuple[List[str], List[float]]:
        """
        Select the most important features.
        This method should be implemented by subclasses to perform feature selection.
        
        Returns
        -------
        Tuple[List[str], List[float]]
            - List of selected feature names
            - List of importance scores for the selected features
        """
        raise NotImplementedError("Subclasses must implement this method.")