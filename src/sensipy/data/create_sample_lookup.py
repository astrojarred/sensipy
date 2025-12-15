"""Script to create a sample lookup dataframe for documentation examples.

This creates a dummy sample lookup table that works with the examples
in the followups documentation.
"""
import numpy as np
import pandas as pd
from pathlib import Path

# Define the parameter grid
event_ids = [1, 2, 3, 4, 5]
sites = ["north", "south"]
zeniths = [20, 40, 60]
ebl_models = ["franceschini", "dominguez11"]
delays = [10, 30, 100, 300, 1000, 3000, 10000]  # seconds

# Create dummy source metadata for each event
event_metadata = {
    1: {
        "long": 0.5,  # radians
        "lat": 0.3,
        "dist": 50000,  # kpc
    },
    2: {
        "long": -0.3,
        "lat": 0.7,
        "dist": 80000,
    },
    3: {
        "long": 1.2,
        "lat": -0.4,
        "dist": 30000,
    },
    4: {
        "long": 0.0,
        "lat": 0.0,
        "dist": 60000,
    },
    5: {
        "long": -1.0,
        "lat": 0.5,
        "dist": 100000,
    },
}

rows = []

for event_id in event_ids:
    metadata = event_metadata[event_id]
    
    for site in sites:
        for zenith in zeniths:
            for ebl_model in ebl_models:
                # Create observation times that vary with delay
                # Typical behavior: longer delays require longer observation times
                # Use a power-law relationship with some noise
                for delay in delays:
                    # Add variations based on site, zenith, and ebl
                    base_time = delay ** 0.5
                    
                    # Site effect: south is slightly more sensitive
                    site_factor = 0.9 if site == "south" else 1.0
                    
                    # Zenith effect: lower zenith = better sensitivity
                    zenith_factor = 1.0 / (1.0 + (zenith - 20) / 100.0)
                    
                    # EBL effect: franceschini is more absorptive
                    ebl_factor = 1.2 if ebl_model == "franceschini" else 1.0
                    
                    # Event-specific factors
                    dist_factor = 60000 / metadata["dist"]  # Inverse distance scaling
                    
                    # Calculate observation time
                    obs_time = (
                        base_time
                        * site_factor
                        * zenith_factor
                        * ebl_factor
                        * dist_factor
                    )
                    
                    # Add deterministic variation based on parameters
                    # (removed randomness for reproducibility)
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
                    
                    rows.append({
                        "event_id": event_id,
                        "obs_delay": delay,
                        "obs_time": obs_time,
                        "irf_site": site,
                        "irf_zenith": zenith,
                        "irf_ebl": True,
                        "irf_ebl_model": ebl_model,  # Also include for other_info
                        "irf_config": "alpha",
                        "irf_duration": 1800,
                        "long": metadata["long"],
                        "lat": metadata["lat"],
                        "dist": metadata["dist"],
                    })

df = pd.DataFrame(rows)

# Save to parquet
output_path = Path(__file__).parent / "mock_data" / "sample_lookup_table.parquet"
df.to_parquet(output_path, index=False)
print(f"Created sample lookup table with {len(df)} rows")
print(f"Saved to: {output_path}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nEvent IDs: {sorted(df['event_id'].unique())}")
print(f"Sites: {sorted(df['irf_site'].unique())}")
print(f"Zeniths: {sorted(df['irf_zenith'].unique())}")
print(f"EBL models: {sorted(df['irf_ebl'].unique())}")
print(f"Delays: {sorted(df['obs_delay'].unique())}")
