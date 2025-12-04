import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Callable, Optional
from matplotlib.figure import Figure

def plot_results(
    dataframe: pd.DataFrame,
    nrows: int = 1,
    ncols: Optional[int] = None,
    group_level: str = "dataset",
    value_level: str = "metric",
    value_key: str = "test_losses",
    plotting_function: Callable = sns.boxenplot
) -> tuple[Figure, np.ndarray]:
    """
    Create subplots for each group in a MultiIndex DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame with MultiIndex columns.
    nrows : int, optional
        Number of rows of subplots (default is 1).
    ncols : int or None, optional
        Number of columns of subplots. If None, it is set to ceil(n / nrows) where n is the number of unique groups.
    group_level : str, optional
        The column MultiIndex level to use for grouping subplots (default is "dataset").
    value_level : str, optional
        The column MultiIndex level to use for selecting values (default is "metric").
    value_key : str, optional
        The specific key in value_level to plot (default is "test_losses").
    plotting_function : callable
        A function that takes a DataFrame and an Axes object to create the plot (default is sns.boxenplot).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    axes : np.ndarray
        Array of matplotlib Axes objects.
    """

    # Get unique group names from the specified MultiIndex level
    group_names = dataframe.columns.get_level_values(group_level).unique()
    n = len(group_names)

    # Determine number of columns if not provided
    if ncols is None: ncols = int(np.ceil(n / nrows))

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, sharex=True)

    # Ensure axes is always iterable
    if nrows * ncols == 1: axes = np.array([axes])
    if nrows > 1: axes = axes.flatten()

    # Plot for each group
    for idx, (ax, group_name) in enumerate(zip(axes, group_names)):
        # Select columns for the current group in group_level
        df = dataframe.xs(key=group_name, level=group_level, axis=1)

        # Select columns for the specified value_key in value_level
        df = df.xs(key=value_key, level=value_level, axis=1) # type: ignore

        df.columns.name = None # Remove column name for clarity
        plotting_function(data=df, ax=ax)
        ax.set_title(group_name)

        # Set y-label only for the first column
        if idx % ncols == 0: ax.set_ylabel("Loss")

    fig.suptitle(value_key) # Set overall title
    return fig, axes
