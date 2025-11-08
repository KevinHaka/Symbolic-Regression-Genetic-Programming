# Libraries and modules
import pandas as pd
import numpy as np

from pmlb import fetch_data
from ucimlrepo import fetch_ucirepo, dotdict
from typing import Callable, Optional, Sequence, Union, Tuple, Dict, List
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3

# List all available datasets
def list_available_datasets() -> List[str]:
    """
    List all available datasets in the registry.
    """
    
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
    This dataset contains 117 audio/music features extracted from 1059 traditional
    music tracks originating from 33 countries/regions.
    The target variable is the geographic latitude (a continuous value in degrees)
    of the track's origin, making this a regression problem.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix containing 117 audio/music features.
    y : np.ndarray
        Target values representing the latitude of the geographical origin.
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
    This dataset contains near-infrared spectroscopy data for 240 meat samples.
    It includes 124 features: 100 raw absorbance measurements (wavelengths 850-1050 nm)
    and 24 additional derived features.
    The target variable is the percent fat content (% by weight) in each sample,
    making this a regression problem commonly used in chemometrics and symbolic regression.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix of shape (240, 124) containing all available spectroscopic features.
    y : np.ndarray
        Target values representing the fat content (%) of the meat samples.
    """
    
    # Fetch the dataset from PMLB
    dataset = fetch_data("505_tecator")
    assert isinstance(dataset, pd.DataFrame), "Expected pandas DataFrame"
    
    # Separate features and target
    X = dataset.drop(columns="target")
    y = dataset['target'].to_numpy()
    
    return X, y

def _542_pollution(
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    This dataset contains 15 environmental features related to air pollution monitoring.
    It consists of 60 samples collected from real-world measurements.
    The target variable is a continuous index representing overall pollution level.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix of shape (60, 15) containing environmental measurement features.
    y : np.ndarray
        Target values representing the pollution index of each sample.
    """
    
    # Fetch the dataset from PMLB
    dataset = fetch_data("542_pollution")
    assert isinstance(dataset, pd.DataFrame), "Expected pandas DataFrame"
    
    # Separate features and target
    X = dataset.drop(columns="target")
    y = dataset['target'].to_numpy()
    
    return X, y

# Functions to load specific datasets from UCI ML repository
def _communities_and_crime(
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    This dataset contains socio-economic, demographic, and law enforcement
    features for 1994 communities in the United States, with the goal of
    predicting violent crime rates per capita (ViolentCrimesPerPop).

    Preprocessing steps:
    - Removes non-predictive identifiers: 'state', 'county', 'community',
      'communityname', and 'fold'.
    - Replaces missing values (encoded as '?') with NaN and drops all rows
      with any missing data (reduces dataset to 319 samples).
    - Converts all features to numeric types.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix of shape (319, 122) containing socio-economic and
        demographic predictors.
    y : np.ndarray
        Target values representing violent crimes per population (continuous).
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
    The dataset includes 125 socio-economic and law enforcement features,
    18 crime-related target variables, and non-predictive identifiers.
    After preprocessing (removing identifiers, categorical features, other targets,
    and rows with missing values), the resulting dataset contains **319 samples**
    and **124 numeric features**.

    Parameters
    ----------
    target : str, default='violentPerPop'
        Target variable to predict. Must be one of:
        ['murders', 'murdPerPop', 'rapes', 'rapesPerPop', 'robberies',
         'robbbPerPop', 'assaults', 'assaultPerPop', 'burglaries',
         'burglPerPop', 'larcenies', 'larcPerPop', 'autoTheft',
         'autoTheftPerPop', 'arsons', 'arsonsPerPop', 'violentPerPop',
         'nonViolPerPop']

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix of shape (319, 124).
    y : np.ndarray
        Target values (continuous) of shape (319,).
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

def _superconductivity(
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    This dataset contains 81 numeric features derived from the chemical composition
    of 21263 superconductor materials, with the goal of predicting the critical
    temperature (in Kelvin) at which the material becomes superconducting.

    The features include statistical aggregations (mean, entropy, range, etc.)
    of atomic properties (e.g., atomic mass, density, electron affinity) across
    the elements in each compound.

    The dataset has no missing values and is widely used in symbolic regression
    and materials informatics.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix of shape (21263, 81).
    y : np.ndarray
        Target values representing critical temperature (in Kelvin), shape (21263,).
    """

    # Fetch dataset from UCI ML repository
    dataset = fetch_ucirepo(name='Superconductivty Data')

    # Validate dataset structure
    assert dataset.data is not None, "Dataset loading failed, data is None"

    # Extract features and target
    X = dataset.data.features
    y = dataset.data.targets['critical_temp'].to_numpy()

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

def _generate_friedman1_dataset(
    n_samples: int = 100,
    n_features: int = 10,
    noise: float = 0.,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate the Friedman #1 synthetic regression dataset.

    The Friedman #1 function was introduced by Jerome H. Friedman in 1991 to evaluate the 
    Multivariate Adaptive Regression Splines (MARS) method. It defines a target 
    variable y as a nonlinear combination of the first five features, while any additional 
    features (beyond the first five) are pure noise and unrelated to the output.

    The ground-truth relationship is:

        y = 10 * sin(π * X1 * X2) + 20 * (X3 - 0.5)^2 + 10 * X4 + 5 * X5 + ε,

    where X1-X5 are independent and uniformly distributed on [0, 1], and ε is Gaussian noise 
    with zero mean and standard deviation equal to the `noise` parameter.

    Arguments
    ---------
    n_samples : int, default=100
        Number of samples to generate.
    n_features : int, default=10
        Number of features to generate. Must be at least 5.
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to the output.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix of shape (n_samples, n_features) containing both informative
        and noise features.
    y : np.ndarray
        Target values of shape (n_samples,) generated according to the Friedman #1 function
    """

    # Generate dataset using sklearn's built-in function
    X, y = make_friedman1(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)

    # Create column names
    variable_names = ['X1', 'X2', 'X3', 'X4', 'X5'] + [f'noise_{i+1}' for i in range(n_features - 5)]
    variable_names[-1] += '_friedman1'  # Unique name for last noise feature

    # Build DataFrame with informative and noise features
    X = pd.DataFrame(X, columns=variable_names)
    
    return X, y

def _generate_friedman2_dataset(
    n_samples: int = 100,
    noise: float = 0.,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate the Friedman #2 synthetic regression dataset.

    The Friedman #2 function was introduced by Jerome H. Friedman in 1991 to evaluate the
    Multivariate Adaptive Regression Splines (MARS) method. It defines a target
    variable y as a nonlinear combination of four features with different uniform
    distributions. There are no additional noise features; the problem is fixed to 4 inputs.

    The ground-truth relationship is:

        y = sqrt( X1^2 + (X2 * X3 - 1 / (X2 * X4))^2 ) + ε,

    where the input features are independently sampled from:
        X1 ~ U(0, 100)
        X2 ~ U(40π, 560π)
        X3 ~ U(0, 1)
        X4 ~ U(1, 11)

    and ε is Gaussian noise with zero mean and standard deviation equal to the `noise` parameter.
    
    Arguments
    ---------
    n_samples : int, default=100
        Number of samples to generate.
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to the output.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix of shape (n_samples, 4) with columns ['X1', 'X2', 'X3', 'X4_friedman2'].
    y : np.ndarray
        Target values of shape (n_samples,) generated according to the Friedman #2 function.
    """

    # Generate dataset using sklearn's built-in function
    X, y = make_friedman2(n_samples=n_samples, noise=noise, random_state=random_state)

    # Create column names
    variable_names = ['X1', 'X2', 'X3', 'X4']
    variable_names[-1] += '_friedman2'  # Unique name for last noise feature

    # Build DataFrame with informative and noise features
    X = pd.DataFrame(X, columns=variable_names)
    
    return X, y

def _generate_friedman3_dataset(
    n_samples: int = 100,
    noise: float = 0.,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate the Friedman #3 synthetic regression dataset.

    The Friedman #3 function was introduced by Jerome H. Friedman in 1991 to evaluate the
    Multivariate Adaptive Regression Splines (MARS) method. It defines a target
    variable y using an arctangent transformation of a nonlinear combination of four features,
    each drawn from a different uniform distribution. There are no additional noise features;
    the problem is fixed to 4 inputs.

    The ground-truth relationship is:

        y = arctan( (X2 * X3 - 1 / (X2 * X4)) / X1 ) + ε,

    where the input features are independently sampled from:
        X1 ~ U(0, 100)
        X2 ~ U(40π, 560π)
        X3 ~ U(0, 1)
        X4 ~ U(1, 11)

    and ε is Gaussian noise with zero mean and standard deviation equal to the `noise` parameter.

    Arguments
    ---------
    n_samples : int, default=100
        Number of samples to generate.
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to the output.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix of shape (n_samples, 4) with columns ['X1', 'X2', 'X3', 'X4_friedman3'].
    y : np.ndarray
        Target values of shape (n_samples,) generated according to the Friedman #3 function.
    """

    # Generate dataset using sklearn's built-in function
    X, y = make_friedman3(n_samples=n_samples, noise=noise, random_state=random_state)

    # Create column names
    variable_names = ['X1', 'X2', 'X3', 'X4']
    variable_names[-1] += '_friedman3'  # Unique name for last feature

    # Build DataFrame
    X = pd.DataFrame(X, columns=variable_names)
    
    return X, y

# Registry of available datasets and their loading functions
_DATASET_REGISTRY: Dict[str, Callable] = {
    # Synthetic
    "F1": _generate_F1_dataset,
    "F2": _generate_F2_dataset,
    "Friedman1": _generate_friedman1_dataset,
    "Friedman2": _generate_friedman2_dataset,
    "Friedman3": _generate_friedman3_dataset,

    # PMLB
    "4544_GeographicalOriginalofMusic": _4544_geographical_original_of_music,
    "505_tecator": _505_tecator,
    "542_pollution": _542_pollution,

    # UCI
    "Communities and Crime": _communities_and_crime,
    "Communities and Crime Unnormalized": _communities_and_crime_unnormalized,
    "Superconductivty Data": _superconductivity,
}