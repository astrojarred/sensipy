"""Detectability analysis and visualization module.

This module provides tools for analyzing and visualizing source detectability data
from lookup tables. It supports flexible column naming, custom filtering, and
modern plotting features for creating detectability heatmaps.

Example:
    >>> from sensipy.detectability import LookupData
    >>> data = LookupData("lookup_table.parquet")
    >>> data.set_filters(("irf_site", "==", "north"))
    >>> ax = data.plot(title="North Site Detectability")
"""
from pathlib import Path
from typing import Callable, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm

from sensipy.logging import logger

log = logger(__name__)


def convert_time(seconds: float) -> str:
    """Convert a time in seconds to a human-readable string representation.

    Args:
        seconds: Time in seconds.

    Returns:
        String representation (e.g., "30s", "2m", "1h", "1d").
    """
    if seconds < 60:
        return f"{int(seconds):.0f}s"
    elif seconds < 3600:
        return f"{int(seconds / 60):.0f}m"
    elif seconds < 86400:
        return f"{int(seconds / 3600):.0f}h"
    else:
        return f"{int(seconds / 86400):.0f}d"


class LookupData:
    """A class for reading, filtering, and visualizing detectability data from lookup tables.

    This class provides a flexible interface for working with detectability data stored
    in Parquet or CSV format. It supports custom column names, flexible filtering,
    and customizable visualization options.

    Attributes:
        df (pd.DataFrame): The current (filtered) data frame.
        observation_times (np.ndarray): The observation times used for calculations.
        results (pd.DataFrame): The calculated detectability results.

    Example:
        >>> data = LookupData("data.parquet", delay_column="obs_delay")
        >>> data.set_filters(("irf_site", "==", "north"))
        >>> data.plot(title="North Site Analysis")
    """

    def __init__(
        self,
        input_file: str | Path,
        delay_column: str = "obs_delay",
        obs_time_column: str = "obs_time",
    ):
        """Initialize LookupData from a file.

        Args:
            input_file: Path to the input file (Parquet or CSV).
            delay_column: Name of the column containing delay times. Defaults to "obs_delay".
            obs_time_column: Name of the column containing observation times. Defaults to "obs_time".

        Raises:
            ValueError: If file type is not supported or required columns are missing.
        """
        # Store the absolute path to the input file.
        self._input_file = Path(input_file).absolute()

        # Determine the file type from the file extension.
        self._file_type = self._input_file.suffix

        # Load the data from the file.
        if self._file_type == ".parquet":
            self._data = pd.read_parquet(self._input_file)
        elif self._file_type == ".csv":
            self._data = pd.read_csv(self._input_file)
        else:
            raise ValueError("File type not supported, please use .parquet or .csv")

        # Store column names
        self._delay_column = delay_column
        self._obs_time_column = obs_time_column

        # Validate required columns exist
        if delay_column not in self._data.columns:
            raise ValueError(
                f"Column '{delay_column}' (delay_column) not found in data. "
                f"Available columns: {list(self._data.columns)}"
            )
        if obs_time_column not in self._data.columns:
            raise ValueError(
                f"Column '{obs_time_column}' (obs_time_column) not found in data. "
                f"Available columns: {list(self._data.columns)}"
            )

        # Set the initial data to the full data set.
        self._current_data = self._data.copy()

        self._obs_times = self._default_obs_times

        self._results = pd.DataFrame(
            columns=["delay", "obs_time", "n_seen", "total", "percent_seen"]
        )

    @property
    def df(self) -> pd.DataFrame:
        """Access the current (filtered) data frame.

        Returns:
            The current (filtered) data frame.
        """
        return self._current_data

    @property
    def observation_times(self) -> np.ndarray:
        """Access the observation times.

        Returns:
            Array of observation times used for calculations.
        """
        return self._obs_times

    @property
    def results(self) -> pd.DataFrame:
        """Access the calculated detectability results.

        Returns:
            DataFrame containing detectability results with columns:
            delay, obs_time, n_seen, total, percent_seen
        """
        if len(self._results) == 0:
            self._calculate_results()
        return self._results

    def __len__(self) -> int:
        """Get the length of the current data frame.

        Returns:
            Number of rows in the current data frame.
        """
        return len(self._current_data)

    def __repr__(self) -> str:
        """Get a string representation of the LookupData object.

        Returns:
            String representation showing the input file path.
        """
        return f"LookupData({self._input_file})"

    @property
    def _default_obs_times(self) -> np.ndarray:
        """Default observation times for calculations."""
        return np.logspace(1, np.log10(1 * 3600 + 0.1), 50, dtype=int)

    def _calculate_results(self) -> None:
        """Calculate detectability percentages and store in results DataFrame.

        This method computes the fraction of sources detected at various delay times
        and observation times, storing the results in the _results attribute.
        """
        data = self._current_data
        delay_col = self._delay_column
        obs_time_col = self._obs_time_column

        # Filter out rows with observation time <= 0 (non-detections)
        valid_data = data[data[obs_time_col] > 0]

        # Group by delay and observation time
        groups = valid_data.groupby([delay_col, obs_time_col])
        seen = groups[obs_time_col].count()

        # Calculate total number of sources for each delay
        total = data.groupby(delay_col)[obs_time_col].count()

        # Create a DataFrame with unique pairs of delay and observation time
        unique_delays = data[delay_col].unique().tolist()
        pairs = (
            pd.MultiIndex.from_product(
                [unique_delays, self._obs_times.tolist()],
                names=[delay_col, obs_time_col],
            )
            .to_frame()
            .reset_index(drop=True)
        )

        # Rename columns to standard names for consistency
        pairs = pairs.rename(columns={delay_col: "delay", obs_time_col: "obs_time"})

        # Calculate n_seen and total for each pair
        pairs[["n_seen", "total"]] = pairs.apply(
            lambda row: (
                seen.where(
                    (seen.index.get_level_values(delay_col) == row.delay)
                    & (seen.index.get_level_values(obs_time_col) <= row.obs_time)
                )
                .dropna()
                .sum(),
                total.get(row.delay, 0),
            ),
            result_type="expand",
            axis=1,
        )

        # Set n_seen and total to integer values
        pairs["n_seen"] = pairs["n_seen"].astype(int)
        pairs["total"] = pairs["total"].astype(int)
        pairs["percent_seen"] = pairs["n_seen"] / pairs["total"]

        self._results = pairs

    def set_filters(self, *args) -> None:
        """Set filters on the current data frame.

        Filters are applied sequentially, and each filter must match for a row
        to be included. Filters reset the current data to the full dataset before
        applying new filters.

        Args:
            *args: Filter tuples, each of the form (column, operator, value).
                Operators: ==, =, <, >, <=, >=, in, not in, notin
                Values can be single values or lists (for 'in' operations).

        Raises:
            TypeError: If a filter is not a tuple.
            ValueError: If operator is invalid or column doesn't exist.

        Example:
            >>> data.set_filters(
            ...     ("irf_site", "==", "north"),
            ...     ("irf_zenith", "<", 40)
            ... )
        """
        self._current_data = self._data.copy()
        self._results = pd.DataFrame(
            columns=["delay", "obs_time", "n_seen", "total", "percent_seen"]
        )

        for a in args:
            if not isinstance(a, tuple):
                raise TypeError("Filters must be passed as tuples")
            if len(a) != 3:
                raise ValueError("Each filter must be a tuple of (column, operator, value)")

            column, op, value = a

            # Validate column exists
            if column not in self._current_data.columns:
                raise ValueError(
                    f"Column '{column}' not found in data. "
                    f"Available columns: {list(self._current_data.columns)}"
                )

            if op not in ["==", "=", "<", ">", "<=", ">=", "in", "not in", "notin"]:
                raise ValueError(
                    "Filter operation must be one of ==, =, <, >, <=, >=, in, not in, notin"
                )

            if op == "in":
                self._current_data = self._current_data[
                    self._current_data[column].isin(value)
                ]
            elif op == "not in" or op == "notin":
                self._current_data = self._current_data[
                    ~self._current_data[column].isin(value)
                ]
            elif op == "==" or op == "=":
                self._current_data = self._current_data[
                    self._current_data[column] == value
                ]
            elif op == "<":
                self._current_data = self._current_data[
                    self._current_data[column] < value
                ]
            elif op == ">":
                self._current_data = self._current_data[
                    self._current_data[column] > value
                ]
            elif op == "<=":
                self._current_data = self._current_data[
                    self._current_data[column] <= value
                ]
            elif op == ">=":
                self._current_data = self._current_data[
                    self._current_data[column] >= value
                ]

    def set_observation_times(self, obs_times: np.ndarray | list[int]) -> None:
        """Set the observation times for calculations.

        Args:
            obs_times: Array or list of observation times in seconds.
        """
        self._obs_times = np.array(obs_times)
        # Reset results to force recalculation
        self._results = pd.DataFrame(
            columns=["delay", "obs_time", "n_seen", "total", "percent_seen"]
        )

    def reset(self) -> None:
        """Reset the current data frame to the full data set and clear results."""
        self._current_data = self._data.copy()
        self._results = pd.DataFrame(
            columns=["delay", "obs_time", "n_seen", "total", "percent_seen"]
        )

    def plot(
        self,
        ax: plt.Axes | None = None,
        output_file: str | Path | None = None,
        annotate: bool = False,
        x_tick_labels: np.ndarray | list[str] | None = None,
        y_tick_labels: np.ndarray | list[str] | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
        color_scheme: str = "mako",
        color_scale: str | None = None,
        as_percent: bool = False,
        title: str | None = None,
        title_callback: Callable[[pd.DataFrame, pd.DataFrame], str] | None = None,
        subtitle: str | None = None,
        n_labels: int = 10,
        square: bool = True,
        return_ax: bool = True,
        tick_fontsize: int = 12,
        label_fontsize: int = 16,
        max_exposure: float | None = None,
    ) -> plt.Axes | None:
        """Generate a heatmap of detectability data.

        Creates a heatmap showing the fraction of sources detected at various
        delay times and observation times.

        Args:
            ax: Matplotlib axes to plot on. If None, creates a new figure.
            output_file: Path to save the plot. If None, plot is not saved.
            annotate: Whether to annotate cells with values.
            x_tick_labels: Custom labels for x-axis ticks. If None, auto-generated.
            y_tick_labels: Custom labels for y-axis ticks. If None, auto-generated.
            min_value: Minimum value for color scale.
            max_value: Maximum value for color scale.
            color_scheme: Colormap name (default: "mako").
            color_scale: Color scale type ("log" for logarithmic, None for linear).
            as_percent: If True, display values as percentages (0-100).
            title: Custom title string. If None and title_callback is None, no title.
            title_callback: Callable that receives (data, results) DataFrames and returns title string.
            subtitle: Deprecated, use title instead.
            n_labels: Number of tick labels to display on axes.
            square: If True, make cells square-shaped.
            return_ax: If True, return axes object; if False, show plot and return None.
            tick_fontsize: Font size for tick labels.
            label_fontsize: Font size for axis labels.
            max_exposure: Maximum exposure time in hours. If provided, overrides default obs_times.

        Returns:
            Matplotlib axes object if return_ax=True, otherwise None.
        """
        # Handle max_exposure parameter
        if max_exposure is not None:
            obs_times = [
                round(i) for i in np.logspace(1, np.log10(max_exposure * 3600), 50)
            ]
            self.set_observation_times(obs_times)

        # Get the results dataframe
        df = self.results.copy()
        df.rename(columns={"obs_time": "exposure time"}, inplace=True)

        # Set the plot style using seaborn
        sns.set_theme()

        # Convert the results to percentages, if requested
        if as_percent:
            df["percent_seen"] = df["percent_seen"] * 100

        # Pivot the data for heatmap
        pivot = df.pivot(
            index="exposure time", columns="delay", values="percent_seen"
        ).astype(float)

        # Create a new figure and axis if needed
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 9))

        # Set the colorbar options
        cbar_label = "Percentage of sources detected" if as_percent else "Fraction of sources detected"
        cbar_kws = {"label": cbar_label, "orientation": "vertical"}

        # Set the color scale if logarithmic scale is selected
        if color_scale == "log":
            norm = LogNorm(vmin=min_value, vmax=max_value)
        else:
            norm = None

        # Set the x-axis tick labels
        if x_tick_labels is None:
            x_delays = np.sort(self._results.delay.unique())
            step = max(1, int(len(x_delays) / n_labels))
            label_delays = x_delays[::step]
            x_tick_pos = np.arange(len(x_delays))[::step]
            if x_delays[-1] != label_delays[-1]:
                label_delays = np.append(label_delays, x_delays[-1])
                x_tick_pos = np.append(x_tick_pos, len(x_delays) - 1)
            x_tick_labels = [convert_time(x) for x in label_delays]
        else:
            x_delays = np.sort(self._results.delay.unique())
            step = max(1, int(len(x_delays) / n_labels))
            x_tick_pos = np.arange(len(x_delays))[::step]
            if len(x_delays) - 1 not in x_tick_pos:
                x_tick_pos = np.append(x_tick_pos, len(x_delays) - 1)

        # Set the y-axis tick labels
        if y_tick_labels is None:
            step = max(1, int(len(self.observation_times) / n_labels))
            label_obs_times = self.observation_times[::step]
            y_tick_pos = np.arange(len(self.observation_times))[::step]
            if self.observation_times[-1] != label_obs_times[-1]:
                label_obs_times = np.append(label_obs_times, self.observation_times[-1])
                y_tick_pos = np.append(y_tick_pos, len(self.observation_times) - 1)
            y_tick_labels = [convert_time(x) for x in label_obs_times]
        else:
            step = max(1, int(len(self.observation_times) / n_labels))
            y_tick_pos = np.arange(len(self.observation_times))[::step]
            if len(self.observation_times) - 1 not in y_tick_pos:
                y_tick_pos = np.append(y_tick_pos, len(self.observation_times) - 1)

        # Create the heatmap
        heatmap = sns.heatmap(
            pivot,
            annot=True if annotate else None,
            fmt=".0f" if annotate else ".2g",
            linewidths=0.5 if annotate else 0,
            ax=ax,
            cmap=color_scheme,
            vmin=min_value,
            vmax=max_value,
            xticklabels=cast(Sequence[str], x_tick_labels) if x_tick_labels is not None else x_tick_labels,
            yticklabels=cast(Sequence[str], y_tick_labels) if y_tick_labels is not None else y_tick_labels,
            cbar_kws=cbar_kws,
            norm=norm,
            square=not square,
        )

        # Invert the y-axis
        heatmap.invert_yaxis()

        # Generate title
        plot_title = None
        if title_callback is not None:
            plot_title = title_callback(self._current_data, self._results)
        elif title is not None:
            plot_title = title
        elif subtitle is not None:
            # Legacy support for subtitle parameter
            n_total = self._results.groupby("delay").total.first().iloc[0] if len(self._results) > 0 else 0
            plot_title = f"Source Detectability: {subtitle} (n={n_total})"

        if plot_title:
            ax.set_title(plot_title, fontsize=label_fontsize)

        # Set axis labels
        ax.set_xlabel("$t_{L}$", fontsize=label_fontsize)
        ax.set_ylabel(r"$t_{\mathrm{exp}}$", fontsize=label_fontsize)
        ax.set_xticks(x_tick_pos, x_tick_labels, rotation=45, fontsize=tick_fontsize)
        ax.set_yticks(y_tick_pos, y_tick_labels, fontsize=tick_fontsize)

        # Set tick parameters
        ax.tick_params(
            axis="both",
            length=5,
            color="black",
            direction="out",
            bottom=True,
            left=True,
        )

        # Save figure if requested
        fig = heatmap.get_figure()
        log.debug(f"ax: {ax}, output_file: {output_file}, filetype: {self._file_type}")
        if output_file and fig:
            log.debug(f"saving plot to {output_file}")
            fig.savefig(output_file, bbox_inches="tight", pad_inches=0)

        if not return_ax:
            plt.show()
            return None

        return ax


def create_heatmap_grid(
    lookup_data_list: list[LookupData],
    grid_size: tuple[int, int],
    max_exposure: float | int | None = None,
    max_value: float | None = None,
    title: str | None = None,
    subtitles: list[str] | None = None,
    n_labels: int = 10,
    tick_fontsize: int = 12,
    label_fontsize: int = 16,
    square: bool = False,
    cmap: str = "mako",
    **plot_kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    """Create a grid of detectability heatmaps.

    Args:
        lookup_data_list: List of LookupData instances to plot.
        grid_size: Tuple (rows, cols) specifying grid dimensions.
        max_exposure: Maximum exposure time in hours. If None, uses default obs_times.
        max_value: Maximum value for color scale. If None, auto-determined.
        title: Overall title for the grid.
        subtitles: List of titles for each subplot. If None, uses default titles.
        n_labels: Number of tick labels on axes.
        tick_fontsize: Font size for tick labels.
        label_fontsize: Font size for axis labels.
        square: Whether to make cells square-shaped.
        cmap: Colormap name.
        **plot_kwargs: Additional keyword arguments passed to plot() method.

    Returns:
        Tuple of (figure, axes array).
    """
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    # Flatten axes into a list if more than one subplot exists
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    num_heatmaps = min(len(lookup_data_list), len(axes))
    for i in range(num_heatmaps):
        # Use subtitle from list if provided
        subtitle = subtitles[i] if subtitles and i < len(subtitles) else None

        lookup_data_list[i].plot(
            ax=axes[i],
            title=subtitle,
            n_labels=n_labels,
            tick_fontsize=tick_fontsize,
            label_fontsize=label_fontsize,
            square=square,
            color_scheme=cmap,
            max_value=max_value,
            max_exposure=max_exposure,
            **plot_kwargs,
        )

    # Turn off any unused subplots
    for j in range(num_heatmaps, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=20)

    fig.tight_layout(pad=2)
    return fig, axes
