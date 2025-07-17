# Libraries and modules
import pandas as pd
import numpy as np

from typing import Sequence, Union, Tuple, Dict

from pmlb import fetch_data
from ucimlrepo import fetch_ucirepo 

# Custom dataset loading functions
def load_datasets(
    dataset_names: Sequence[Union[str, Tuple[str, str]]]
) -> Dict[str, dict]:
    """
    Load multiple datasets by name, with optional renaming.

    Accepts a list of dataset names or (name, rename) tuples.
    Returns a dictionary where each key is a dataset name (or its rename)
    and each value is a dict with 'X' (features) and 'y' (target).

    Parameters
    ----------
    dataset_names : list of str or (str, str)
        List of dataset names or (name, rename) tuples. Each item can be:
        - str: Dataset name (used as both identifier and key)
        - tuple: (original_name, renamed_key) for custom naming

    Returns
    -------
    datasets : dict
        Dictionary with keys as dataset names (or renames) 
        and values as dicts {'X': features, 'y': target}.
    """

    datasets = {}
    
    # Define mapping of dataset names to generation functions
    custom_functions = {
        'F1': _generate_F1_dataset,
        'F2': _generate_F2_dataset,
    }
    
    # Define PMLB (Penn Machine Learning Benchmarks) dataset names and functions
    pmlb_dataset_names = {
        "4544_GeographicalOriginalofMusic": _4544_geographical_original_of_music,
        "505_tecator": _505_tecator,
    }
    
    # Define UCI ML Repository datasets with their loading functions
    uci_dataset_names = {
        "Communities and Crime": communities_and_crime,
        "Communities and Crime Unnormalized": communities_and_crime_unnormalized,
    }

    # Process each requested dataset
    for item in dataset_names:
        # Support both string names and (name, rename) tuples
        if isinstance(item, str):
            name, rename = item, item
        else:
            name, rename = item

        # Check if the dataset is a custom synthetic dataset
        if name in custom_functions:
            # Generate synthetic dataset using custom function
            X, y = custom_functions[name]()
            datasets[rename] = {'X': X, 'y': y}

        # Check if the dataset is available in PMLB
        elif name in pmlb_dataset_names:
            # Load dataset from PMLB using specific function
            X, y = pmlb_dataset_names[name]()
            datasets[rename] = {'X': X, 'y': y}
        
        # Check if the dataset is available in UCI ML repository
        elif name in uci_dataset_names:
            # Load dataset from UCI ML repository using specific function
            X, y = uci_dataset_names[name]()
            datasets[rename] = {'X': X, 'y': y}

        else:
            # Dataset not found in any source
            print(f'Dataset "{name}" not found in available sources.')

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
    assert type(dataset) is pd.DataFrame, "Expected dataset to be a pandas DataFrame"

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
    assert type(dataset) is pd.DataFrame, "Expected dataset to be a pandas DataFrame"
    
    # Separate features and target
    X = dataset.drop(columns="target")
    y = dataset['target'].to_numpy()
    
    return X, y

# Functions to load specific datasets from UCI ML repository
def communities_and_crime(
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

def communities_and_crime_unnormalized(
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
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate the F1 synthetic dataset based on Newton's law of gravitation.
    
    This function creates a synthetic dataset with 3 informative features and 50 noise
    features to test feature selection methods. The target follows Newton's gravitational
    law formula.

    Formula: F1 = -g * (X1 * X2) / X3**2
    where g = 6.67408e-11 (gravitational constant)

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix with shape (100, 53). Contains X1, X2, X3 (informative)
        and 50 noise features (noise_1, noise_2, ..., noise_50).
    y : np.ndarray
        Target values with shape (100,). Gravitational force values.
    """

    n_noise_features = 50
    n_samples = 100
    
    # Generate true input features
    X1 = np.random.uniform(0, 1, n_samples)  # Mass 1
    X2 = np.random.uniform(0, 1, n_samples)  # Mass 2
    X3 = np.random.uniform(1, 2, n_samples)  # Distance (> 0 to avoid division by zero)
    
    # Compute target using Newton's law of gravitation
    g = 6.67408e-11  # Gravitational constant
    y = -g * (X1 * X2) / (X3**2)

    # Generate noise features (irrelevant for prediction)
    noise_features = np.random.uniform(0, 1, (n_samples, n_noise_features))
    
    # Build DataFrame with informative and noise features
    X = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        **{f'noise_{i+1}': noise_features[:,i] for i in range(n_noise_features)}
    })

    return X, y

def _generate_F2_dataset(
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
        Feature matrix with shape (11000, 53). Contains X1, X2, X3 (informative)
        and 50 noise features (noise_1, noise_2, ..., noise_50).
    y : np.ndarray
        Target values with shape (11000,). Rational function values following F2 formula.
    """

    n_noise_features = 50
    n_samples = 11000  # Larger sample size for better statistical power
    
    # Generate true input features with specified ranges
    X1 = np.random.uniform(-1, 1, n_samples)  # Symmetric around zero
    X3 = np.random.uniform(-1, 1, n_samples)  # Symmetric around zero
    X2 = np.random.uniform(1, 2, n_samples)   # Positive values for stable denominator
    
    # Compute target using the F2 rational function
    y = 30 * X1 * X3 / ((X1 - 10) * (X2 ** 2))
    
    # Generate noise features (irrelevant for prediction)
    noise = np.random.uniform(0, 1, size=(n_samples, n_noise_features))
    
    # Build DataFrame with informative and noise features
    X = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        **{f'noise_{i+1}': noise[:, i] for i in range(n_noise_features)}
    })
    
    return X, y
