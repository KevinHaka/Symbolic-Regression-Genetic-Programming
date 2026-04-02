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
    subplot_kwargs: Optional[dict] = None,
    group_level: str = "dataset",
    value_level: Optional[str] = "metric",
    value_key: Optional[str] = "test_losses",
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
    subplot_kwargs : dict, optional
        Additional keyword arguments to pass to plt.subplots() for subplot creation.
    group_level : str, optional
        The column MultiIndex level to use for grouping subplots (default is "dataset").
    value_level : str or None, optional
        The column MultiIndex level to use for selecting values (default is "metric").
    value_key : str or None, optional
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
    fig, axes = plt.subplots(nrows, ncols, **(subplot_kwargs or {}))

    # Ensure axes is always iterable
    if nrows * ncols == 1: axes = np.array([axes])
    else: axes = axes.flatten()

    # Plot for each group
    for ax, group_name in zip(axes, group_names):
        # Select columns for the current group in group_level
        df = dataframe.xs(key=group_name, level=group_level, axis=1)

        if (value_key is not None) and (value_level is not None):
            # Select columns for the specified value_key in value_level
            df = df.xs(key=value_key, level=value_level, axis=1) # type: ignore

        df.columns.name = None # Remove column name for clarity
        plotting_function(data=df, ax=ax)
        ax.set_title(group_name)

    # Turn off any unused subplots
    for ax in axes[len(group_names):]: ax.axis('off')

    return fig, axes