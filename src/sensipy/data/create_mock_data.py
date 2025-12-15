"""Script to generate mock GRB spectral data files for testing.

This module provides functionality to generate mock FITS and CSV files
with time-resolved spectral data, including metadata in the new flexible format.
"""

import numpy as np
import pandas as pd
from astropy.io import fits
import astropy.units as u
from pathlib import Path

from sensipy.util import get_data_path


def power_law(energy: u.Quantity, index: float, amplitude: u.Quantity) -> u.Quantity:
    """Generate a power-law spectrum.
    
    Args:
        energy: Energy array (astropy Quantity)
        index: Spectral index (power-law index)
        amplitude: Normalization amplitude (astropy Quantity)
    
    Returns:
        Differential flux array (astropy Quantity)
    """
    return amplitude * (energy / (1 * u.TeV)) ** -index


def create_mock_data(
    grb_id: int = 42,
    output_dir: Path | None = None,
    start_time: float = 1.0,
    end_time: float = 1e5,
    time_steps: int = 101,
    start_energy: float = 0.001,
    end_energy: float = 10.0,
    energy_steps: int = 101,
    spectral_index: float = 2.0,
    max_amplitude: float = 1e-17,
    min_amplitude: float = 1e-21,
) -> tuple[Path, Path, Path]:
    """Generate mock GRB spectral data files (FITS and CSV formats).
    
    This function creates mock time-resolved spectral data for a GRB event,
    generating both FITS and CSV files along with a metadata CSV file.
    
    Args:
        grb_id: GRB event ID (default: 42)
        output_dir: Output directory for files. If None, uses package data directory.
        start_time: Start time in seconds (default: 1.0)
        end_time: End time in seconds (default: 1e5)
        time_steps: Number of time steps (default: 101)
        start_energy: Start energy in TeV (default: 0.001)
        end_energy: End energy in TeV (default: 10.0)
        energy_steps: Number of energy steps (default: 101)
        spectral_index: Power-law spectral index (default: 2.0)
        max_amplitude: Maximum amplitude in cm-2 s-1 TeV-1 (default: 1e-17)
        min_amplitude: Minimum amplitude in cm-2 s-1 TeV-1 (default: 1e-21)
    
    Returns:
        Tuple of (fits_path, csv_path, metadata_csv_path) Path objects
    """
    # Determine output directory
    if output_dir is None:
        output_dir = get_data_path("mock_data")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate time grid
    time = np.logspace(np.log10(start_time), np.log10(end_time), time_steps)
    time = time * u.s
    
    # Generate energy grid
    energy = np.logspace(np.log10(start_energy), np.log10(end_energy), energy_steps)
    energy = energy * u.TeV
    
    # Generate amplitude array (decreasing with time)
    amplitudes = np.logspace(np.log10(max_amplitude), np.log10(min_amplitude), time_steps)
    amplitudes = amplitudes * u.Unit("cm-2 s-1 TeV-1")
    
    # Generate lightcurves (power-law spectra at each time)
    lightcurves_list: list[u.Quantity] = []
    for amplitude in amplitudes:
        lightcurves_list.append(power_law(energy, spectral_index, amplitude))
    
    # Convert to GeV for FITS format
    energy_gev = energy.to(u.GeV).value
    time_s = time.to(u.s).value
    
    # Create energy bins (FITS format)
    initial_energy = energy_gev[:-1]
    final_energy = energy_gev[1:]
    energy_dtype = np.dtype([('Initial Energy', '>f4'), ('Final Energy', '>f4')])
    energy_rec = np.rec.fromarrays([initial_energy, final_energy], dtype=energy_dtype)
    
    # Create time bins (FITS format)
    initial_time = time_s[:-1]
    final_time = time_s[1:]
    time_dtype = np.dtype([('Initial Time', '>f4'), ('Final Time', '>f4')])
    time_rec = np.rec.fromarrays([initial_time, final_time], dtype=time_dtype)
    
    # Prepare spectra data (convert TeV-1 to GeV-1: multiply by 1000)
    # Convert list of Quantity arrays to numpy array
    lightcurves_array = np.array([lc.value for lc in lightcurves_list])
    lightcurves_gev = lightcurves_array * 1000
    
    # Get bin centers
    time_bin_centers = (time_s[:-1] + time_s[1:]) / 2
    energy_bin_centers_gev = (energy_gev[:-1] + energy_gev[1:]) / 2
    
    # Use flux at bin edges (remove last time and energy bin)
    spectra_data: np.ndarray = lightcurves_gev[:-1, :-1]
    
    # Create FITS_rec for spectra
    spectra_cols = []
    for i in range(len(energy_bin_centers_gev)):
        col_name = f'col{i}'
        spectra_cols.append((col_name, '>f8'))
    
    spectra_dtype = np.dtype(spectra_cols)
    spectra_rec = np.rec.array([tuple(row) for row in spectra_data], dtype=spectra_dtype)
    
    # Create FITS file structure
    # Create PrimaryHDU first (which automatically adds SIMPLE=True)
    primary_hdu = fits.PrimaryHDU()
    
    # Add metadata to header in new flexible format
    primary_hdu.header["EVENT_ID"] = (grb_id, "event_id")
    primary_hdu.header["LONG"] = (0.0, "longitude [rad]")
    primary_hdu.header["LAT"] = (1.0, "latitude [rad]")
    primary_hdu.header["EISO"] = (2e50, "eiso [erg]")
    primary_hdu.header["DISTANCE"] = (100000.0, "distance [kpc]")
    primary_hdu.header["AUTHOR"] = ("", "Copernicus")  # Empty value - should be ignored
    primary_hdu.header["PROJECT"] = ("", "sensipy")  # Empty value - should be ignored
    primary_hdu.header["HISTORY"] = ("Sample time-resolved spectrum",)
    energy_hdu = fits.BinTableHDU(data=energy_rec, name='ENERGIES')
    time_hdu = fits.BinTableHDU(data=time_rec, name='TIMES')
    spectra_hdu = fits.BinTableHDU(data=spectra_rec, name='SPECTRA')
    
    hdul = fits.HDUList([primary_hdu, energy_hdu, time_hdu, spectra_hdu])
    
    # Save FITS file
    fits_filename = output_dir / f"GRB_{grb_id}_mock.fits"
    hdul.writeto(fits_filename, overwrite=True)
    
    # Create CSV file
    # Flatten arrays for CSV format
    time_flat = np.repeat(time_bin_centers, len(energy_bin_centers_gev))
    energy_flat = np.tile(energy_bin_centers_gev, len(time_bin_centers))
    flux_flat = spectra_data.flatten()
    
    csv_data = pd.DataFrame({
        'time [s]': time_flat,
        'energy [GeV]': energy_flat,
        'flux [cm-2 s-1 GeV-1]': flux_flat
    })
    
    csv_filename = output_dir / f"GRB_{grb_id}_mock.csv"
    csv_data.to_csv(csv_filename, index=False)
    
    # Create metadata CSV file
    metadata_data = pd.DataFrame({
        'parameter': ['event_id', 'longitude', 'latitude', 'distance'],
        'value': [float(grb_id), 0.0, 1.0, 100000.0],
        'units': ['', 'rad', 'rad', 'kpc']
    })
    
    metadata_filename = output_dir / f"GRB_{grb_id}_mock_metadata.csv"
    metadata_data.to_csv(metadata_filename, index=False)
    
    return fits_filename, csv_filename, metadata_filename


if __name__ == "__main__":
    """Generate mock data files when run as a script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate mock GRB spectral data files")
    parser.add_argument(
        "--grb-id",
        type=int,
        default=42,
        help="GRB event ID (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: package data directory)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    fits_path, csv_path, metadata_path = create_mock_data(
        grb_id=args.grb_id,
        output_dir=output_dir
    )
    
    print(f"Created FITS file: {fits_path}")
    print(f"Created CSV file: {csv_path}")
    print(f"Created metadata CSV file: {metadata_path}")
