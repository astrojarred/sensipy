"""Script to create a mock lookup dataframe for documentation examples.

This module provides functionality to generate mock lookup tables for testing
and documentation. It can create deterministic tables (for documentation) or
flexible tables with customizable parameters (for testing).
"""
import numpy as np
import pandas as pd
from pathlib import Path

from sensipy.util import get_data_path


def create_mock_lookup_table(
    n_events: int | None = None,
    event_ids: list[int] | None = None,
    sites: list[str] | None = None,
    zeniths: list[int] | None = None,
    ebl_models: list[str] | None = None,
    delays: list[int] | None = None,
    output_dir: Path | None = None,
    output_filename: str = "mock_lookup_table.parquet",
    delay_column: str = "obs_delay",
    obs_time_column: str = "obs_time",
    include_metadata: bool = True,
    event_metadata: dict[int, dict[str, float]] | None = None,
    use_random_metadata: bool = False,
    seed: int | None = None,
) -> Path:
    """Generate a source-agnostic mock lookup table for detectability analysis.

    Creates a comprehensive lookup table with multiple events, various observation
    configurations, and delay times. This is useful for testing and demonstrating
    the detectability analysis tools.

    Args:
        n_events: Number of events to generate if event_ids is None and event_metadata is None.
        event_ids: List of event IDs. If None, uses keys from event_metadata or generates sequential IDs.
        sites: List of site names (e.g., ["north", "south"]). Defaults to ["north", "south"].
        zeniths: List of zenith angles in degrees. Defaults to [20, 40, 60].
        ebl_models: List of EBL model names. Defaults to ["franceschini", "dominguez11"].
        delays: List of delay times in seconds. Defaults to [10, 30, 100, 300, 1000, 3000, 10000].
        output_dir: Output directory. If None, uses package data directory.
        output_filename: Name of the output file. Defaults to "mock_lookup_table.parquet".
        delay_column: Name of the delay column. Defaults to "obs_delay".
        obs_time_column: Name of the observation time column. Defaults to "obs_time".
        include_metadata: If True, includes metadata columns (site, zenith, etc.).
        event_metadata: Dictionary mapping event_id to metadata dict with keys: long, lat, dist.
            If None and use_random_metadata=False, uses default deterministic metadata.
        use_random_metadata: If True, generates random metadata for events. Requires seed for reproducibility.
        seed: Random seed for reproducibility when use_random_metadata=True.

    Returns:
        Path to the created parquet file.

    Example:
        >>> from sensipy.data.create_mock_lookup import create_mock_lookup_table
        >>> path = create_mock_lookup_table(n_events=5, sites=["north"])
        >>> print(f"Created lookup table: {path}")
    """
    if seed is not None:
        np.random.seed(seed)

    # Determine output directory
    if output_dir is None:
        output_dir = get_data_path("mock_data")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Set defaults
    if sites is None:
        sites = ["north", "south"]
    if zeniths is None:
        zeniths = [20, 40, 60]
    if ebl_models is None:
        ebl_models = ["franceschini"]
    if delays is None:
        delays = [10, 30, 70, 100, 300, 700, 1000, 3000, 7000, 10000]

    # Determine event_ids and metadata
    if event_metadata is None:
        if use_random_metadata:
            # Generate random metadata
            if event_ids is None:
                if n_events is None:
                    n_events = 10
                event_ids = list(range(1, n_events + 1))
            event_metadata = {}
            for event_id in event_ids:
                event_metadata[event_id] = {
                    "long": np.random.uniform(-np.pi, np.pi),
                    "lat": np.random.uniform(-np.pi / 2, np.pi / 2),
                    "dist": np.random.uniform(30000, 850000),  # kpc
                }
        else:
            # Use default deterministic metadata (for documentation compatibility)
            default_metadata = {
                1: {"long": 0.5, "lat": 0.3, "dist": 50000},
                2: {"long": -0.3, "lat": 0.7, "dist": 60000},
                3: {"long": 1.2, "lat": -0.4, "dist": 70000},
                4: {"long": 0.0, "lat": 0.0, "dist": 80000},
                5: {"long": -1.0, "lat": 0.5, "dist": 90000},
                6: {"long": -1.0, "lat": 0.5, "dist": 100000},
                7: {"long": -2.0, "lat": 0.2, "dist": 150000},
                8: {"long": 2.0, "lat": -0.2, "dist": 200000},
                9: {"long": -1.0, "lat": 0.5, "dist": 250000},
                10: {"long": 1.0, "lat": -0.5, "dist": 300000},
                11: {"long": -0.3, "lat": 0.7, "dist": 350000},
                12: {"long": 1.3, "lat": -0.5, "dist": 400000},
                13: {"long": -1.1, "lat": 0.2, "dist": 450000},
                14: {"long": 0.1, "lat": -0.1, "dist": 500000},
                15: {"long": 2.0, "lat": 0.3, "dist": 550000},
            }
            if event_ids is None:
                event_ids = list(default_metadata.keys())
            event_metadata = {eid: default_metadata.get(eid, default_metadata[1]) for eid in event_ids}
    else:
        # Use provided metadata
        if event_ids is None:
            event_ids = list(event_metadata.keys())

    rows = []

    for event_id in event_ids:
        metadata = event_metadata[event_id]

        for site in sites:
            for zenith in zeniths:
                for ebl_model in ebl_models:
                    for delay in delays:
                        # Calculate observation time based on delay and configuration
                        # Typical behavior: longer delays require longer observation times
                        base_time = (delay-9.5) ** 0.3

                        # Site effect: south is slightly more sensitive
                        site_factor = 0.9 if site == "south" else 1.0

                        # Zenith effect: lower zenith = better sensitivity
                        zenith_factor = 3.0 * (1.0 + (zenith - 20) / 100.0)

                        # EBL effect: franceschini is more absorptive
                        ebl_factor = 1.2 if ebl_model == "franceschini" else 1.0

                        # Event-specific factors
                        dist_factor = (metadata["dist"] / 1000) ** 1.1 # Inverse distance scaling

                        # Calculate observation time
                        obs_time = (
                            base_time
                            * site_factor
                            * zenith_factor
                            * ebl_factor
                            * dist_factor
                        ) / 10

                        # Add deterministic variation based on parameters
                        variation = 1.0 + (event_id % 10) / 100.0
                        obs_time *= variation

                        # Ensure minimum observation time
                        obs_time = max(obs_time, 1.0)

                        # Some events/configurations are not detectable at certain delays
                        # Make some obs_times negative to indicate non-detection
                        if delay > 10000 and zenith > 40:
                            obs_time = -1  # Not detectable
                        elif delay < 30 and obs_time > 3600:
                            obs_time = -1  # Too faint

                        row = {
                            delay_column: delay,
                            obs_time_column: obs_time,
                        }

                        if include_metadata:
                            row.update(
                                {
                                    "event_id": event_id,
                                    "irf_site": site,
                                    "irf_zenith": zenith,
                                    "irf_ebl": True,
                                    "irf_ebl_model": ebl_model,
                                    "irf_config": "alpha",
                                    "irf_duration": 1800,
                                    "long": metadata["long"],
                                    "lat": metadata["lat"],
                                    "dist": metadata["dist"],
                                }
                            )

                        rows.append(row)

    df = pd.DataFrame(rows)

    # Save to parquet
    output_path = output_dir / output_filename
    df.to_parquet(output_path, index=False)

    return output_path


if __name__ == "__main__":
    """Generate mock lookup table when run as a script.
    
    This creates the default mock_lookup_table.parquet file used in
    documentation examples and tests.
    """
    # Use default parameters for documentation compatibility
    output_path = create_mock_lookup_table()
    
    print(f"Created mock lookup table with {len(pd.read_parquet(output_path))} rows")
    print(f"Saved to: {output_path}")
    df = pd.read_parquet(output_path)
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nEvent IDs: {sorted(df['event_id'].unique())}")
    print(f"Sites: {sorted(df['irf_site'].unique())}")
    print(f"Zeniths: {sorted(df['irf_zenith'].unique())}")
    print(f"EBL models: {sorted(df['irf_ebl_model'].unique())}")
    print(f"Delays: {sorted(df['obs_delay'].unique())}")
