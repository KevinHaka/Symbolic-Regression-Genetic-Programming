# Libraries and modules
import pandas as pd
import numpy as np

from typing import Callable, Optional, Sequence, Union, Tuple, Dict, List

from pmlb import fetch_data
from ucimlrepo import fetch_ucirepo, dotdict

# List all available datasets
def list_available_datasets() -> List[str]:
    return sorted(_DATASET_REGISTRY.keys())

# Custom dataset loading functions
def load_datasets(
    dataset_names: Sequence[Union[str, Tuple[str, str]]]
) -> Dict[str, Dict[str, Union[pd.DataFrame, np.ndarray]]]:
    """
    Load multiple datasets by name with optional renaming.

    Each element in `dataset_names` can be:
      - a string: the dataset's registered name (used as the key)
      - a (name, alias) tuple: original name plus a custom dictionary key

    Parameters
    ----------
    dataset_names : sequence of (str | (str, str))
        Names or (name, alias) tuples referring to registered loaders.

    Returns
    -------
    datasets : dict
        Dictionary with keys as dataset names (or renames) 
        and values as dicts {'X': features, 'y': target}.
    """

    datasets = {}
    unknown_datasets = []
    available_datasets = list_available_datasets()

    # Process each requested dataset
    for item in dataset_names:
        # Support both string names and (name, rename) tuples
        if isinstance(item, str):
            name, rename = item, item
        else:
            name, rename = item

        # Check if the dataset is available
        if name in available_datasets:
            # Load the dataset using its registered function
            X, y = _DATASET_REGISTRY[name]()
            datasets[rename] = {'X': X, 'y': y}

        else:
            # Collect unknown dataset names for reporting
            unknown_datasets.append(name)

    # Report any unknown datasets
    if unknown_datasets:
        print("Unknown datasets:")

        for name in unknown_datasets:
            print(f" - {name}")
        
        print("\nFor a list of available datasets, use list_available_datasets().")

    return datasets

# Functions to load specific datasets from PMLB
def _4544_geographical_original_of_music(
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load the 4544 Geographical Original of Music dataset from PMLB.
    This dataset contains audio/music features extracted from various music tracks,
    along with their geographical origin labels.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix containing audio/music features.
    y : np.ndarray
        Target values representing geographical regions.
    """
    
    # Fetch the dataset from PMLB
    dataset = fetch_data("4544_GeographicalOriginalofMusic")
    assert isinstance(dataset, pd.DataFrame), "Expected pandas DataFrame"

    # Separate features and target
    X = dataset.drop(columns="target")
    y = dataset['target'].to_numpy()
    
    return X, y

def _505_tecator(
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load the 505 Tecator dataset from PMLB.
    This dataset contains near-infrared spectroscopy data for food analysis.
    
    Returns
    -------
    X : pandas.DataFrame
        Feature matrix containing spectroscopy measurements.
    y : np.ndarray
        Target values representing percent fat content in food samples.
    """
    
    # Fetch the dataset from PMLB
    dataset = fetch_data("505_tecator")
    assert isinstance(dataset, pd.DataFrame), "Expected pandas DataFrame"
    
    # Separate features and target
    X = dataset.drop(columns="target")
    y = dataset['target'].to_numpy()
    
    return X, y

# Functions to load specific datasets from UCI ML repository
def _communities_and_crime(
) -> Tuple[pd.DataFrame, np.ndarray]:
    """ 
    Load the Communities and Crime dataset from UCI ML repository.
    
    This dataset contains socio-economic data for predicting violent crime rates
    in communities across the United States. The function performs data cleaning
    including removal of non-predictive features and handling missing values.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix containing socio-economic and demographic features.
    y : np.ndarray
        Target values representing violent crimes per population (ViolentCrimesPerPop).
    """

    # Define non-predictive features to remove
    removing_features = ['state', 'county', 'community', 'communityname', 'fold']

    # Fetch dataset from UCI ML repository
    dataset = fetch_ucirepo(name="Communities and Crime")

    # Validate dataset structure
    assert isinstance(dataset.data, dotdict), "Expected dataset.data to be a dotdict"
    assert isinstance(dataset.data.original, pd.DataFrame), "Expected dataset.data.original to be a DataFrame"

    # Remove unwanted columns
    dataset.data.original = dataset.data.original.drop(removing_features, axis=1)
    
    # Replace "?" with np.nan for proper missing value handling
    dataset.data.original.replace("?", np.nan, inplace=True)
    
    # Drop rows with any missing values
    dataset.data.original.dropna(inplace=True)
    dataset.data.original.reset_index(drop=True, inplace=True)

    # Separate features and target
    X = dataset.data.original.drop(columns="ViolentCrimesPerPop")
    y = dataset.data.original["ViolentCrimesPerPop"].to_numpy()

    # Convert all features to numeric types
    X = X.apply(pd.to_numeric, errors='raise')
    
    return X, y

def _communities_and_crime_unnormalized(
    target: str = 'violentPerPop'
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load the Communities and Crime Unnormalized dataset from UCI ML repository.
    
    This dataset contains the unnormalized version of the Communities and Crime data,
    allowing for selection of different target variables. The function performs
    data cleaning and allows flexible target selection.

    Parameters
    ----------
    target : str, default='violentPerPop'
        Name of the target variable to predict. Must be one of the available
        target columns in the dataset.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix containing socio-economic and demographic features.
    y : np.ndarray
        Target values for the specified target variable.
    """
    
    # Fetch dataset from UCI ML repository
    dataset = fetch_ucirepo(name="Communities and Crime Unnormalized")

    # Validate dataset structure
    assert isinstance(dataset.data, dotdict), "Expected dataset.data to be a dotdict"
    assert isinstance(dataset.data.targets, pd.DataFrame), "Expected dataset.data.targets to be a DataFrame"
    assert isinstance(dataset.data.original, pd.DataFrame), "Expected dataset.data.original to be a DataFrame"

    # Define features to remove: all other targets (except chosen one) + identifier columns
    removing_features = dataset.data.targets.columns.to_list() + [
        'communityname', 'countyCode', 'communityCode', 'State', 'fold'
    ]
    removing_features.remove(target)  # Keep the target we want to predict

    # Remove unwanted columns
    dataset.data.original = dataset.data.original.drop(removing_features, axis=1)

    # Clean missing values
    dataset.data.original.dropna(inplace=True)
    dataset.data.original.reset_index(drop=True, inplace=True)
    
    # Fix column names: replace '-' with '_' for valid Python identifiers
    dataset.data.original.columns = dataset.data.original.columns.str.replace('-', '_')

    # Separate features and target
    X = dataset.data.original.drop(columns=target)
    y = dataset.data.original[target].to_numpy()
    
    return X, y

# Functions to generate synthetic datasets
def _generate_F1_dataset(
    n_samples: int = 100,
    n_noise_features: int = 50,
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate the F1 synthetic dataset based on Newton's law of gravitation.
    
    This function creates a synthetic dataset with 3 informative features and 'n' noise
    features to test feature selection methods. The target follows Newton's gravitational
    law formula.

    Formula: F1 = -g * (X1 * X2) / X3**2
    where g = 6.67408e-11 (gravitational constant)

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    n_noise_features : int, default=50
        Number of noise features to include.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix with shape (n_samples, n_noise_features+3). Contains X1, X2, X3 (informative)
        and 'n_noise_features' noise features (noise_1, noise_2, ..., noise_n).
    y : np.ndarray
        Target values with shape (n_samples,). Gravitational force values.
    """
    # Set random seed for reproducibility
    rng = np.random.default_rng(random_state)
    
    # Generate true input features
    X1 = rng.uniform(0, 1, n_samples)  # Mass 1
    X2 = rng.uniform(0, 1, n_samples)  # Mass 2
    X3 = rng.uniform(1, 2, n_samples)  # Distance (> 0 to avoid division by zero)

    # Compute target using Newton's law of gravitation
    g = 6.67408e-11  # Gravitational constant
    y = -g * (X1 * X2) / (X3**2)

    # Generate noise features (irrelevant for prediction)
    noise_features = rng.uniform(0, 1, (n_samples, n_noise_features))

    # Create column names
    variable_names = ['X1', 'X2', 'X3'] + [f'noise_{i+1}' for i in range(n_noise_features)]
    variable_names[-1] += '_F1'  # Unique name for last noise feature

    # Combine informative and noise features
    variable_values = np.column_stack((X1, X2, X3, noise_features))

    # Build DataFrame with informative and noise features
    X = pd.DataFrame(variable_values, columns=variable_names)

    return X, y

def _generate_F2_dataset(
    n_samples: int = 11000,
    n_noise_features: int = 50,
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate the F2 synthetic dataset as described in Wang et al., 2024.
    
    This function creates a synthetic dataset based on the F2 benchmark from
    "Improving Generalization of Genetic Programming for High-Dimensional 
    Symbolic Regression with Shapley Value Based Feature Selection" paper.

    Formula: F2 = 30 * X1 * X3 / ((X1 - 10) * X2**2)

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix with shape (n_samples, n_noise_features+3). Contains X1, X2, X3 (informative)
        and 'n_noise_features' noise features (noise_1, noise_2, ..., noise_n).
    y : np.ndarray
        Target values with shape (n_samples,). Rational function values following F2 formula.
    """

    # Set random seed for reproducibility
    rng = np.random.default_rng(random_state)
    
    # Generate true input features with specified ranges
    X1 = rng.uniform(-1, 1, n_samples)  # Symmetric around zero
    X3 = rng.uniform(-1, 1, n_samples)  # Symmetric around zero
    X2 = rng.uniform(1, 2, n_samples)   # Positive values for stable denominator
    
    # Compute target using the F2 rational function
    y = 30 * X1 * X3 / ((X1 - 10) * (X2 ** 2))
    
    # Generate noise features (irrelevant for prediction)
    noise_features = rng.uniform(0, 1, size=(n_samples, n_noise_features))

    # Create column names
    variable_names = ['X1', 'X2', 'X3'] + [f'noise_{i+1}' for i in range(n_noise_features)]
    variable_names[-1] += '_F1'  # Unique name for last noise feature

    # Combine informative and noise features
    variable_values = np.column_stack((X1, X2, X3, noise_features))
    
    # Build DataFrame with informative and noise features
    X = pd.DataFrame(variable_values, columns=variable_names)
    
    return X, y

# Registry of available datasets and their loading functions
_DATASET_REGISTRY: Dict[str, Callable] = {
    # Synthetic
    "F1": _generate_F1_dataset,
    "F2": _generate_F2_dataset,

    # PMLB
    "4544_GeographicalOriginalofMusic": _4544_geographical_original_of_music,
    "505_tecator": _505_tecator,

    # UCI
    "Communities and Crime": _communities_and_crime,
    "Communities and Crime Unnormalized": _communities_and_crime_unnormalized,
}