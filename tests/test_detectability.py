from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sensipy.detectability import LookupData, convert_time, create_heatmap_grid
from sensipy.data.create_mock_lookup import create_mock_lookup_table


@pytest.fixture
def mock_lookup_table(tmp_path):
    """Make a lookup table from scratch for the testing module"""
    return create_mock_lookup_table(
        event_ids=[1, 2, 3],
        sites=["north"],
        zeniths=[20],
        ebl_models=["franceschini"],  # Use single EBL model for simpler test data
        delays=[10, 100, 1000],
        output_dir=tmp_path,
        output_filename="test_lookup_table.parquet",
        use_random_metadata=False,  # Use deterministic metadata for reproducibility
    )


@pytest.fixture
def lookup_table_with_custom_columns(tmp_path):
    """Make a lookup table from scratch but with custom column names."""
    df = pd.DataFrame(
        {
            "delay_col": [10, 100, 1000, 10, 100, 1000],
            "time_col": [50, 200, 500, 60, 250, 600],
            "event_id": [1, 1, 1, 2, 2, 2],
        }
    )
    path = tmp_path / "custom_columns.parquet"
    df.to_parquet(path)
    return path


def test_lookupdata_init_with_parquet(mock_lookup_table):
    """Test LookupData initialization with Parquet file."""
    data = LookupData(mock_lookup_table)
    assert data._input_file == Path(mock_lookup_table).absolute()
    assert data._file_type == ".parquet"
    assert len(data.df) > 0


def test_lookupdata_init_with_csv(tmp_path):
    """Test LookupData initialization with CSV file."""
    df = pd.DataFrame(
        {
            "obs_delay": [10, 100, 1000],
            "obs_time": [50, 200, 500],
        }
    )
    csv_file = tmp_path / "test.csv"
    df.to_csv(csv_file, index=False)

    data = LookupData(csv_file)
    assert data._file_type == ".csv"
    assert len(data.df) == 3


def test_lookupdata_init_invalid_file_type(tmp_path):
    """Test LookupData initialization raises error for unsupported file type."""
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("test")

    with pytest.raises(ValueError, match="File type not supported"):
        LookupData(str(invalid_file))


def test_lookupdata_init_missing_column(tmp_path):
    """Test LookupData initialization raises error when missing a required column."""
    df = pd.DataFrame(
        {
            "obs_delay": [10, 100],
            "other_col": [1, 2],
        }
    )
    parquet_file = tmp_path / "test.parquet"
    df.to_parquet(parquet_file)

    with pytest.raises(ValueError, match="obs_time_column"):
        LookupData(parquet_file, obs_time_column="obs_time")


def test_lookupdata_custom_column_names(lookup_table_with_custom_columns):
    """Test LookupData with custom column names."""
    data = LookupData(
        lookup_table_with_custom_columns,
        delay_column="delay_col",
        obs_time_column="time_col",
    )
    assert len(data.df) == 6
    assert "delay_col" in data.df.columns
    assert "time_col" in data.df.columns


def test_lookupdata_df_property(mock_lookup_table):
    """Ensure that LookupData.df exists and works as intended."""
    data = LookupData(mock_lookup_table)
    df = data.df
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "obs_delay" in df.columns
    assert "obs_time" in df.columns


def test_lookupdata_observation_times_property(mock_lookup_table):
    """Test LookupData observation_times property."""
    data = LookupData(mock_lookup_table)
    obs_times = data.observation_times
    assert isinstance(obs_times, np.ndarray)
    assert len(obs_times) > 0


def test_lookupdata_set_observation_times(mock_lookup_table):
    """Test setting your own observation times."""
    data = LookupData(mock_lookup_table)
    new_times = [10, 100, 1000, 10000]
    data.set_observation_times(new_times)
    assert np.array_equal(data.observation_times, new_times)
    # Results should be reset
    assert len(data._results) == 0


def test_lookupdata_reset(mock_lookup_table):
    """Test reset method on your filters."""
    data = LookupData(mock_lookup_table)
    initial_len = len(data.df)

    # Apply a filter
    data.set_filters(("event_id", "==", 1))
    assert len(data.df) < initial_len

    # Reset
    data.reset()
    assert len(data.df) == initial_len
    assert len(data._results) == 0


# Test that the various filter operations work / don't work when expected
def test_lookupdata_set_filters_equals(mock_lookup_table):
    """Test set_filters with == operation."""
    data = LookupData(mock_lookup_table)
    initial_len = len(data.df)
    data.set_filters(("event_id", "==", 1))
    assert len(data.df) < initial_len
    assert all(data.df["event_id"] == 1)


def test_lookupdata_set_filters_less_than(mock_lookup_table):
    """Test set_filters with < operation."""
    data = LookupData(mock_lookup_table)
    data.set_filters(("obs_delay", "<", 100))
    assert all(data.df["obs_delay"] < 100)


def test_lookupdata_set_filters_greater_than(mock_lookup_table):
    """Test set_filters with > operation."""
    data = LookupData(mock_lookup_table)
    data.set_filters(("obs_delay", ">", 100))
    assert all(data.df["obs_delay"] > 100)


def test_lookupdata_set_filters_in(mock_lookup_table):
    """Test set_filters with 'in' operation."""
    data = LookupData(mock_lookup_table)
    data.set_filters(("event_id", "in", [1, 2]))
    assert all(data.df["event_id"].isin([1, 2]))


def test_lookupdata_set_filters_not_in(mock_lookup_table):
    """Test set_filters with 'not in' operation."""
    data = LookupData(mock_lookup_table)
    data.set_filters(("event_id", "not in", [1]))
    assert all(data.df["event_id"] != 1)


def test_lookupdata_set_filters_multiple(mock_lookup_table):
    """Test set_filters with multiple filters."""
    data = LookupData(mock_lookup_table)
    data.set_filters(
        ("event_id", "==", 1),
        ("obs_delay", "<", 1000),
    )
    assert all(data.df["event_id"] == 1)
    assert all(data.df["obs_delay"] < 1000)


def test_lookupdata_set_filters_invalid_tuple(mock_lookup_table):
    """Test set_filters raises error for non-tuple filter."""
    data = LookupData(mock_lookup_table)
    with pytest.raises(TypeError, match="Filters must be passed as tuples"):
        data.set_filters("not a tuple")


def test_lookupdata_set_filters_invalid_tuple_length(mock_lookup_table):
    """Test set_filters raises error for invalid tuple length."""
    data = LookupData(mock_lookup_table)
    with pytest.raises(ValueError, match="tuple of \\(column, operator, value\\)"):
        data.set_filters(("col1", "col2"))


def test_lookupdata_set_filters_invalid_operation(mock_lookup_table):
    """Test set_filters raises error for invalid operation."""
    data = LookupData(mock_lookup_table)
    with pytest.raises(ValueError, match="Filter operation must be one of"):
        data.set_filters(("obs_delay", "invalid_op", 1000))


def test_lookupdata_set_filters_missing_column(mock_lookup_table):
    """Test set_filters raises error for missing column."""
    data = LookupData(mock_lookup_table)
    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        data.set_filters(("nonexistent", "==", 1))


def test_lookupdata_len(mock_lookup_table):
    """Test LookupData __len__ method."""
    data = LookupData(mock_lookup_table)
    length = len(data)
    assert isinstance(length, int)
    assert length >= 0


def test_lookupdata_repr(mock_lookup_table):
    """Test LookupData __repr__ method."""
    data = LookupData(mock_lookup_table)
    repr_str = repr(data)
    assert "LookupData" in repr_str
    assert str(mock_lookup_table) in repr_str


def test_lookupdata_results_property(mock_lookup_table):
    """Test LookupData results property."""
    data = LookupData(mock_lookup_table)
    data.set_observation_times([10, 100, 1000])
    results = data.results
    assert isinstance(results, pd.DataFrame)
    assert "delay" in results.columns
    assert "obs_time" in results.columns
    assert "n_seen" in results.columns
    assert "total" in results.columns
    assert "percent_seen" in results.columns
    assert len(results) > 0


def test_lookupdata_results_calculation(mock_lookup_table):
    """Test that results calculation works correctly."""
    data = LookupData(mock_lookup_table)
    data.set_observation_times([10, 100, 1000])
    results = data.results

    # Check that percent_seen is between 0 and 1
    assert all(results["percent_seen"] >= 0)
    assert all(results["percent_seen"] <= 1)

    # Check that n_seen <= total
    assert all(results["n_seen"] <= results["total"])


def test_convert_time():
    """Test convert_time function."""
    assert convert_time(30) == "30s"
    assert convert_time(90) == "1m"  # 90 seconds = 1.5 minutes, rounds to 1m
    assert convert_time(3600) == "1h"
    assert convert_time(86400) == "1d"


def test_lookupdata_plot_basic(mock_lookup_table, tmp_path):
    """Test LookupData plot method (basic test)."""
    data = LookupData(mock_lookup_table)
    data.set_observation_times([10, 100, 1000])

    output_file = tmp_path / "test_plot.png"
    ax = data.plot(output_file=str(output_file), return_ax=True)
    assert ax is not None
    assert output_file.exists()


def test_lookupdata_plot_with_title(mock_lookup_table, tmp_path):
    """Test LookupData plot method with custom title."""
    data = LookupData(mock_lookup_table)
    data.set_observation_times([10, 100, 1000])

    ax = data.plot(title="Custom Title", return_ax=True)
    assert ax is not None
    assert ax.get_title() == "Custom Title"


def test_lookupdata_plot_with_title_callback(mock_lookup_table, tmp_path):
    """Test LookupData plot method with title callback."""
    data = LookupData(mock_lookup_table)
    data.set_observation_times([10, 100, 1000])

    def title_callback(df, results):
        return f"Test Title: {len(df)} rows"

    ax = data.plot(title_callback=title_callback, return_ax=True)
    assert ax is not None
    assert "Test Title" in ax.get_title()


def test_lookupdata_plot_with_annotations(mock_lookup_table, tmp_path):
    """Test LookupData plot method with annotations."""
    data = LookupData(mock_lookup_table)
    data.set_observation_times([10, 100, 1000])

    output_file = tmp_path / "test_plot_annotated.png"
    ax = data.plot(
        output_file=str(output_file),
        annotate=True,
        return_ax=True,
    )
    assert ax is not None


def test_lookupdata_plot_as_percent(mock_lookup_table, tmp_path):
    """Test LookupData plot method with as_percent=True."""
    data = LookupData(mock_lookup_table)
    data.set_observation_times([10, 100, 1000])

    ax = data.plot(as_percent=True, return_ax=True)
    assert ax is not None


def test_lookupdata_plot_max_exposure(mock_lookup_table):
    """Test LookupData plot method with max_exposure parameter."""
    data = LookupData(mock_lookup_table)

    # Set max_exposure to 2 hours
    ax = data.plot(max_exposure=2, return_ax=True)
    assert ax is not None
    # Check that observation times were updated
    assert max(data.observation_times) <= 2 * 3600


def test_lookupdata_plot_custom_colors(mock_lookup_table):
    """Test LookupData plot method with custom color scheme."""
    data = LookupData(mock_lookup_table)
    data.set_observation_times([10, 100, 1000])

    ax = data.plot(color_scheme="viridis", return_ax=True)
    assert ax is not None


def test_lookupdata_plot_log_scale(mock_lookup_table):
    """Test LookupData plot method with logarithmic color scale."""
    data = LookupData(mock_lookup_table)
    data.set_observation_times([10, 100, 1000])

    ax = data.plot(color_scale="log", return_ax=True)
    assert ax is not None


def test_lookupdata_plot_custom_axes(mock_lookup_table):
    """Test LookupData plot method with custom axes."""
    import matplotlib.pyplot as plt

    data = LookupData(mock_lookup_table)
    data.set_observation_times([10, 100, 1000])

    fig, ax = plt.subplots()
    result_ax = data.plot(ax=ax, return_ax=True)
    assert result_ax is ax
    plt.close(fig)


def test_create_heatmap_grid(mock_lookup_table):
    """Test create_heatmap_grid function."""
    data1 = LookupData(mock_lookup_table)
    data2 = LookupData(mock_lookup_table)
    data2.set_filters(("event_id", "==", 2))

    fig, axes = create_heatmap_grid(
        [data1, data2],
        grid_size=(1, 2),
        max_exposure=1,
    )

    assert fig is not None
    assert len(axes) == 2
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_create_heatmap_grid_with_titles(mock_lookup_table):
    """Test create_heatmap_grid with custom titles."""
    data1 = LookupData(mock_lookup_table)
    data2 = LookupData(mock_lookup_table)

    fig, axes = create_heatmap_grid(
        [data1, data2],
        grid_size=(1, 2),
        max_exposure=1,
        title="Overall Title",
        subtitles=["Plot 1", "Plot 2"],
    )

    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_lookupdata_with_custom_columns_results(lookup_table_with_custom_columns):
    """Test that results calculation works with custom column names."""
    data = LookupData(
        lookup_table_with_custom_columns,
        delay_column="delay_col",
        obs_time_column="time_col",
    )
    data.set_observation_times([10, 100, 1000])
    results = data.results

    assert len(results) > 0
    assert "delay" in results.columns
    assert "obs_time" in results.columns


def test_lookupdata_plot_custom_fontsizes(mock_lookup_table):
    """Test LookupData plot with custom font sizes."""
    data = LookupData(mock_lookup_table)
    data.set_observation_times([10, 100, 1000])

    ax = data.plot(
        tick_fontsize=10,
        label_fontsize=14,
        return_ax=True,
    )
    assert ax is not None


def test_lookupdata_empty_data_after_filtering(mock_lookup_table):
    """Test LookupData behavior when filters result in empty data."""
    data = LookupData(mock_lookup_table)
    # Filter to non-existent event
    data.set_filters(("event_id", "==", 999))

    # Should still work, just with empty results
    assert len(data.df) == 0
    data.set_observation_times([10, 100, 1000])
    results = data.results
    # Results should be empty or have zero totals
    assert len(results) == 0 or all(results["total"] == 0)
