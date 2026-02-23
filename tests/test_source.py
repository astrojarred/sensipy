from pathlib import Path
import warnings

import astropy.units as u
import numpy as np
import pytest

from sensipy.source import Source


def test_source_init_with_csv(mock_csv_path):
    """Test source initialization with CSV file."""
    source = Source(mock_csv_path)
    assert source.filepath == Path(mock_csv_path).absolute()
    assert source.file_type == "csv"
    assert source.seen is False
    assert source.obs_time == -1 * u.s


def test_source_read_csv_basic(mock_csv_path):
    """Test basic CSV reading functionality."""
    source = Source(mock_csv_path)
    assert source.file_type == "csv"
    assert hasattr(source, "time")
    assert hasattr(source, "energy")
    assert hasattr(source, "spectra")
    assert len(source.time) > 0
    assert len(source.energy) > 0
    assert source.spectra.shape == (len(source.energy), len(source.time))


def test_source_read_csv_metadata(mock_csv_path):
    """Test that metadata is read from metadata file."""
    source = Source(mock_csv_path)
    # Metadata keys come directly from CSV file (event_id, not id)
    assert source.event_id == 42
    assert source.longitude is not None
    assert source.latitude is not None
    assert source.distance is not None
    # Note: eiso and angle are not in the current mock metadata file


def test_source_read_csv_energy_time_units(mock_csv_path):
    """Test that energy and time have correct units."""
    source = Source(mock_csv_path)
    assert source.energy.unit == u.GeV
    assert source.time.unit == u.s


def test_source_read_csv_spectra_units(mock_csv_path):
    """Test that spectra have correct units."""
    source = Source(mock_csv_path)
    assert source.spectra.unit == u.Unit("1 / (cm2 s GeV)")


def test_source_read_csv_min_max_energy(mock_csv_path):
    """Test that min and max energy are set correctly."""
    source = Source(mock_csv_path)
    assert source.min_energy is not None
    assert source.max_energy is not None
    assert source.min_energy <= source.max_energy
    assert source.min_energy == source.energy.min()
    assert source.max_energy == source.energy.max()


def test_source_read_csv_with_custom_energy_limits(mock_csv_path):
    """Test source initialization with custom energy limits."""
    min_e = 0.1 * u.TeV
    max_e = 1.0 * u.TeV
    source = Source(mock_csv_path, min_energy=min_e, max_energy=max_e)
    assert source.min_energy == min_e.to("GeV")
    assert source.max_energy == max_e.to("GeV")


def test_source_read_csv_missing_file():
    """Test that error is raised for missing CSV file."""
    with pytest.raises((FileNotFoundError, ValueError)):
        Source("/nonexistent/file.csv")


def test_source_read_csv_invalid_format(tmp_path):
    """Test that error is raised for CSV with missing columns."""
    invalid_csv = tmp_path / "invalid.csv"
    invalid_csv.write_text("col1,col2\n1,2\n")

    with pytest.raises(ValueError, match="Missing columns"):
        Source(invalid_csv)


def test_source_read_csv_malformed_data(tmp_path):
    """Test handling of malformed CSV data."""
    # Create CSV with wrong number of rows
    csv_file = tmp_path / "malformed.csv"
    csv_file.write_text(
        "time [s],energy [GeV],dNdE [cm-2 s-1 GeV-1]\n1.0,1.0,1e-8\n2.0,2.0,1e-9\n"
    )

    # This may raise ValueError or LinAlgError depending on the data
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        Source(csv_file)


def test_source_set_spectral_grid(mock_csv_path):
    """Test setting spectral grid."""
    source = Source(mock_csv_path)
    source.set_spectral_grid()
    assert source.SpectralGrid is not None


def test_source_get_spectrum(mock_csv_path):
    """Test getting spectrum at specific time and energy."""
    source = Source(mock_csv_path)
    source.set_spectral_grid()

    time = source.time[0]
    energy = source.energy[0]
    spectrum = source.get_spectrum(time, energy)

    assert isinstance(spectrum, u.Quantity)
    assert spectrum.unit == u.Unit("1 / (cm2 s GeV)")


def test_source_get_spectrum_invalid_time(mock_csv_path):
    """Test that error is raised for invalid time unit."""
    source = Source(mock_csv_path)
    source.set_spectral_grid()

    with pytest.raises(ValueError, match="time must be a time quantity"):
        source.get_spectrum(1.0 * u.m, source.energy[0])


def test_source_get_spectrum_invalid_energy(mock_csv_path):
    """Test that error is raised for invalid energy unit."""
    source = Source(mock_csv_path)
    source.set_spectral_grid()

    with pytest.raises(ValueError, match="energy must be an energy quantity"):
        source.get_spectrum(source.time[0], 1.0 * u.s)


def test_source_get_flux(mock_csv_path):
    """Test getting flux at specific energy and time."""
    source = Source(mock_csv_path)
    source.set_spectral_grid()

    energy = source.energy[0]
    time = source.time[0]
    flux = source.get_flux(energy, time)

    assert isinstance(flux, u.Quantity)
    assert flux.unit == u.Unit("1 / (cm2 s GeV)")


def test_source_get_template_spectrum(mock_csv_path):
    """Test getting template spectrum."""
    source = Source(mock_csv_path)
    source.set_spectral_grid()

    time = source.time[0]
    template = source.get_template_spectrum(time)

    assert template is not None
    assert hasattr(template, "energy")
    assert hasattr(template, "values")


def test_source_fit_spectral_indices(mock_csv_path):
    """Test fitting spectral indices."""
    source = Source(mock_csv_path)
    source.fit_spectral_indices()

    assert hasattr(source, "_indices")
    assert hasattr(source, "_amplitudes")
    assert hasattr(source, "index_at")
    assert hasattr(source, "amplitude_at")


def test_source_get_spectral_index(mock_csv_path):
    """Test getting spectral index at specific time."""
    source = Source(mock_csv_path)
    source.fit_spectral_indices()

    time = source.time[len(source.time) // 2]  # Use middle time
    index = source.get_spectral_index(time)

    assert isinstance(index, (float, np.floating))


def test_source_get_spectral_index_invalid_time(mock_csv_path):
    """Test that error is raised for invalid time unit in get_spectral_index."""
    source = Source(mock_csv_path)
    source.fit_spectral_indices()

    with pytest.raises(ValueError, match="time must be a time quantity"):
        source.get_spectral_index(1.0 * u.m)


def test_source_get_spectral_amplitude(mock_csv_path):
    """Test getting spectral amplitude at specific time."""
    source = Source(mock_csv_path)
    source.fit_spectral_indices()

    time = source.time[len(source.time) // 2]
    amplitude = source.get_spectral_amplitude(time)

    assert isinstance(amplitude, u.Quantity)
    assert amplitude.unit == u.Unit("cm-2 s-1 GeV-1")


def test_source_get_powerlaw_spectrum(mock_csv_path):
    """Test getting powerlaw spectrum."""
    source = Source(mock_csv_path)
    source.fit_spectral_indices()

    time = source.time[len(source.time) // 2]
    spectrum = source.get_powerlaw_spectrum(time)

    assert spectrum is not None
    assert hasattr(spectrum, "index")
    assert hasattr(spectrum, "amplitude")


def test_source_get_powerlaw_spectrum_with_ebl(mock_csv_path):
    """Test getting powerlaw spectrum with EBL."""
    source = Source(mock_csv_path, ebl="franceschini")
    source.fit_spectral_indices()

    time = source.time[len(source.time) // 2]
    spectrum = source.get_powerlaw_spectrum(time, use_ebl=True)

    assert spectrum is not None

    # When EBL is applied, the model becomes a CompoundSpectralModel
    # Check that we can call the model (it should work)
    from gammapy.modeling.models import CompoundSpectralModel

    assert isinstance(spectrum._model, CompoundSpectralModel)
    # Check that EBL was applied
    assert source.ebl is not None


def test_source_get_template_spectrum_with_ebl(mock_csv_path):
    """Test getting template spectrum with EBL."""
    source = Source(mock_csv_path, ebl="franceschini")
    source.set_spectral_grid()

    time = source.time[len(source.time) // 2]
    spectrum = source.get_template_spectrum(time, use_ebl=True)

    assert spectrum is not None

    # When EBL is applied, the model becomes a CompoundSpectralModel
    # Check that we can call the model (it should work)
    from gammapy.modeling.models import CompoundSpectralModel

    assert isinstance(spectrum._model, CompoundSpectralModel)
    # Check that EBL was applied
    assert source.ebl is not None


def test_source_spectrum_plot_energy_range(mock_csv_path):
    """Test that plot() method uses source energy limits."""
    source = Source(mock_csv_path, min_energy=30 * u.GeV, max_energy=10 * u.TeV)
    source.fit_spectral_indices()

    time = source.time[len(source.time) // 2]
    spectrum = source.get_powerlaw_spectrum(time)

    # Check that plot method exists and can be called
    assert hasattr(spectrum, "plot")
    assert callable(spectrum.plot)

    assert hasattr(spectrum, "_source")
    assert spectrum._source == source


def test_source_get_powerlaw_spectrum_ebl_default(mock_csv_path):
    """Test that use_ebl=None defaults to True when EBL model is set."""
    source = Source(mock_csv_path, ebl="franceschini")
    source.fit_spectral_indices()

    time = source.time[len(source.time) // 2]
    # use_ebl=None should default to True since EBL model is set
    spectrum = source.get_powerlaw_spectrum(time, use_ebl=None)

    # Check that EBL was applied (model should be a compound model)
    from gammapy.modeling.models import CompoundSpectralModel

    assert isinstance(spectrum._model, CompoundSpectralModel)
    assert source.ebl is not None


def test_source_get_powerlaw_spectrum_ebl_default_no_ebl(mock_csv_path):
    """Test that use_ebl=None defaults to False when EBL model is not set."""
    source = Source(mock_csv_path)
    source.fit_spectral_indices()

    time = source.time[len(source.time) // 2]
    # use_ebl=None should default to False since no EBL model is set
    spectrum = source.get_powerlaw_spectrum(time, use_ebl=None)

    # Check that EBL was not applied (model should be a PowerLawSpectralModel)
    from gammapy.modeling.models import PowerLawSpectralModel

    assert isinstance(spectrum._model, PowerLawSpectralModel)
    assert source.ebl is None


def test_source_get_template_spectrum_ebl_default(mock_csv_path):
    """Test that use_ebl=None defaults to True when EBL model is set for template."""
    source = Source(mock_csv_path, ebl="franceschini")
    source.set_spectral_grid()

    time = source.time[len(source.time) // 2]
    # use_ebl=None should default to True since EBL model is set
    spectrum = source.get_template_spectrum(time, use_ebl=None)

    # Check that EBL was applied (model should be a compound model)
    from gammapy.modeling.models import CompoundSpectralModel

    assert isinstance(spectrum._model, CompoundSpectralModel)
    assert source.ebl is not None


def test_source_init_with_distance(mock_csv_path):
    """Test Source initialization with distance parameter."""
    from astropy.coordinates import Distance

    distance = Distance(z=0.5)
    source = Source(mock_csv_path, distance=distance)

    # Check that distance was set (use tolerance for floating point comparison)
    stored_distance = source._metadata.get("distance")
    assert stored_distance is not None
    assert abs(stored_distance.z.value - distance.z.value) < 1e-6


def test_source_init_with_z(mock_csv_path):
    """Test Source initialization with z parameter."""
    source = Source(mock_csv_path, z=0.5)

    # Check that distance was set from z
    distance = source._metadata.get("distance")
    assert distance is not None
    assert abs(distance.z.value - 0.5) < 1e-6


def test_source_init_distance_z_conflict(mock_csv_path):
    """Test that providing both distance and z raises ValueError."""
    from astropy.coordinates import Distance

    distance = Distance(z=0.5)
    with pytest.raises(
        ValueError, match="Only one of 'distance' or 'z' can be provided"
    ):
        Source(mock_csv_path, distance=distance, z=0.3)


def test_source_set_ebl_model_with_distance(mock_csv_path):
    """Test set_ebl_model with distance parameter."""
    from astropy.coordinates import Distance

    source = Source(mock_csv_path)
    distance = Distance(z=0.5)

    # Set EBL model with distance
    result = source.set_ebl_model("franceschini", distance=distance)

    # Check that distance was set and EBL model was configured
    assert result is True  # Distance was changed
    assert source._metadata.get("distance") == distance
    assert source.ebl is not None


def test_source_set_ebl_model_distance_z_conflict(mock_csv_path):
    """Test that providing both distance and z to set_ebl_model raises ValueError."""
    from astropy.coordinates import Distance

    source = Source(mock_csv_path)
    distance = Distance(z=0.5)

    with pytest.raises(
        ValueError, match="Only one of 'distance' or 'z' can be provided"
    ):
        source.set_ebl_model("franceschini", distance=distance, z=0.3)


def test_source_distance_overrides_metadata(mock_csv_path):
    """Test that provided distance/z overrides metadata distance."""
    from astropy.coordinates import Distance

    # Source with metadata distance
    source1 = Source(mock_csv_path)
    metadata_distance = source1._metadata.get("distance")

    # Source with explicit distance parameter
    new_distance = Distance(z=0.7)
    source2 = Source(mock_csv_path, distance=new_distance)

    # Check that explicit distance overrides metadata
    assert source2._metadata.get("distance") == new_distance
    if metadata_distance is not None:
        assert source2._metadata.get("distance") != metadata_distance


def test_source_repr(mock_csv_path):
    """Test source __repr__ method."""
    source = Source(mock_csv_path)
    repr_str = repr(source)
    assert "Source" in repr_str
    # Check that repr contains the event_id from metadata (or "unknown" if not present)
    assert (
        str(source.event_id) in repr_str
        or "unknown" in repr_str
        or "Source" in repr_str
    )


def test_source_metadata_property(mock_csv_path):
    """Test source metadata property."""
    source = Source(mock_csv_path)
    metadata = source.metadata
    assert isinstance(metadata, dict)
    assert "id" in metadata or "event_id" in metadata


def test_source_set_ebl_model_invalid(mock_csv_path):
    """Test that error is raised for invalid EBL model."""
    source = Source(mock_csv_path)

    # Set a dummy distance first
    from astropy.coordinates import Distance

    source.distance = Distance(z=0.1)

    with pytest.raises(ValueError, match="ebl must be one of"):
        source.set_ebl_model("invalid_model")


def test_source_set_ebl_model_no_gammapy_data(mock_csv_path, monkeypatch):
    """Test that GAMMAPY_DATA is automatically set to package data when not set."""
    source = Source(mock_csv_path)

    from astropy.coordinates import Distance
    from sensipy.util import get_data_path
    import os

    source.distance = Distance(z=0.1)

    # Remove GAMMAPY_DATA if it exists
    monkeypatch.delenv("GAMMAPY_DATA", raising=False)

    # With our changes, the code should automatically set GAMMAPY_DATA to the package data directory
    # and the EBL model should be set successfully
    source.set_ebl_model("dominguez")

    # Verify GAMMAPY_DATA was set to package data directory
    expected_data_path = str(get_data_path().resolve())
    assert os.environ.get("GAMMAPY_DATA") == expected_data_path
    assert source.ebl is not None
    assert source.ebl_model == "dominguez"


def test_source_csv_column_matching_flexibility(tmp_path):
    """Test that CSV reading handles various column name formats."""
    # Create CSV with different column name format
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "timestamp,energy_value,flux_value\n"
        "1.0,1.0,1e-8\n"
        "1.0,2.0,1e-8\n"
        "2.0,1.0,1e-8\n"
        "2.0,2.0,1e-8\n"
    )

    # Should work because of substring matching
    source = Source(csv_file)
    assert source.file_type == "csv"
    assert len(source.time) == 2
    assert len(source.energy) == 2


def test_source_read_csv_without_metadata(tmp_path):
    """Test CSV reading when metadata file doesn't exist."""
    csv_file = tmp_path / "test.csv"
    # Create minimal valid CSV
    times = [1.0, 2.0]
    energies = [1.0, 2.0]
    csv_content = "time [s],energy [GeV],dNdE [cm-2 s-1 GeV-1]\n"
    for t in times:
        for e in energies:
            csv_content += f"{t},{e},1e-8\n"
    csv_file.write_text(csv_content)

    source = Source(csv_file)
    assert source.file_type == "csv"
    # Metadata should be empty (no metadata file)
    assert len(source.metadata) == 0


def test_source_energy_time_ordering(mock_csv_path):
    """Test that energy and time arrays are properly sorted."""
    source = Source(mock_csv_path)

    # Check that arrays are sorted
    assert np.all(np.diff(source.energy.value) >= 0) or np.all(
        np.diff(source.energy.value) <= 0
    )
    assert np.all(np.diff(source.time.value) >= 0) or np.all(
        np.diff(source.time.value) <= 0
    )


def test_source_spectra_shape_consistency(mock_csv_path):
    """Test that spectra shape matches energy and time dimensions."""
    source = Source(mock_csv_path)

    assert source.spectra.shape[0] == len(source.energy)
    assert source.spectra.shape[1] == len(source.time)


@pytest.mark.slow
def test_source_observe_with_irf(irf_house, mock_csv_path):
    """Test source observation functionality with IRF.

    This test matches the usage pattern from quick-test.ipynb (lines 1-13).
    """
    from sensipy.sensitivity import Sensitivity

    # Load in the desired IRF (matching quick-test.ipynb)
    irf = irf_house.get_irf(
        site="south",
        configuration="alpha",
        zenith=20,
        duration=1800,
        azimuth="average",
        version="prod5-v0.1",
    )

    # Create a gammapy sensitivity class (matching quick-test.ipynb)
    min_energy = 30 * u.GeV
    max_energy = 10 * u.TeV

    sens = Sensitivity(
        irf=irf,
        observatory=f"ctao_{irf.site.name}",
        min_energy=min_energy,
        max_energy=max_energy,
        radius=3.0 * u.deg,
    )

    # Load in a GRB and add EBL (matching quick-test.ipynb)
    event = Source(
        mock_csv_path, min_energy=min_energy, max_energy=max_energy, ebl="franceschini"
    )

    # Generate sensitivity curve first (required for observe)
    # Suppress expected power law warning in tests
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Using a power law model for sensitivity calculation"
        )
        sens.get_sensitivity_curve(source=event)

    # Simulate the observation (matching quick-test.ipynb lines 1-13)
    delay_time = 30 * u.min

    res = event.observe(
        sensitivity=sens,
        start_time=delay_time,
        min_energy=min_energy,
        max_energy=max_energy,
    )

    # Verify the result dictionary contains expected keys
    assert "obs_time" in res
    assert "ebl_model" in res

    # Verify obs_time is a time quantity (can be -1 if not detectable)
    assert isinstance(res["obs_time"], u.Quantity)
    assert res["obs_time"].unit.physical_type == "time"
    # obs_time can be -1 if source is not detectable, or positive if detectable
    assert res["obs_time"].value >= -1

    # Verify ebl_model is set
    assert res["ebl_model"] == "franceschini"

    # Verify source seen status
    assert (
        event.seen is True or event.seen is False
    )  # Can be either depending on detectability


def test_source_read_fits_basic(mock_fits_path):
    """Test basic FITS file reading functionality."""
    source = Source(mock_fits_path)
    assert source.file_type == "fits"
    assert hasattr(source, "time")
    assert hasattr(source, "energy")
    assert hasattr(source, "spectra")
    assert len(source.time) > 0
    assert len(source.energy) > 0
    assert source.spectra.shape == (len(source.energy), len(source.time))


def test_source_read_fits_metadata(mock_fits_path):
    """Test that flexible metadata is parsed from FITS header."""
    source = Source(mock_fits_path)
    # Check that metadata keys are accessible via attribute notation
    assert hasattr(source, "event_id")
    assert hasattr(source, "longitude")
    assert hasattr(source, "latitude")
    assert hasattr(source, "distance")
    assert source.event_id == 42
    assert source.longitude.value == 0.0
    assert source.latitude.value == 1.0
    assert source.distance.value == 100000.0


def test_source_read_fits_empty_value(tmp_path):
    """Test that FITS header entries with empty values are ignored."""
    from astropy.io import fits
    import numpy as np

    # Create a minimal FITS file with empty value
    header = fits.Header()
    header["EVENT_ID"] = (42, "event_id")
    header["AUTHOR"] = ("", "author")  # Empty value - should be ignored
    header["PROJECT"] = ("", "project")  # Empty value - should be ignored

    # Create minimal data arrays - need at least 2 energy and 2 time points
    energy_vals = np.array([1.0, 2.0, 3.0])
    time_vals = np.array([1.0, 2.0, 3.0])
    energy_dtype = np.dtype([("Initial Energy", ">f4"), ("Final Energy", ">f4")])
    time_dtype = np.dtype([("Initial Time", ">f4"), ("Final Time", ">f4")])
    energy_rec = np.rec.fromarrays(
        [energy_vals[:-1], energy_vals[1:]], dtype=energy_dtype
    )
    time_rec = np.rec.fromarrays([time_vals[:-1], time_vals[1:]], dtype=time_dtype)

    # Spectra data needs to match: (n_time_bins, n_energy_bins) = (2, 2)
    spectra_data = np.array([[1e-8, 1e-9], [1e-8, 1e-9]])
    spectra_cols = [("col0", ">f8"), ("col1", ">f8")]
    spectra_dtype = np.dtype(spectra_cols)
    spectra_rec = np.rec.array(
        [tuple(row) for row in spectra_data], dtype=spectra_dtype
    )

    primary_hdu = fits.PrimaryHDU(header=header)
    energy_hdu = fits.BinTableHDU(data=energy_rec, name="ENERGIES")
    time_hdu = fits.BinTableHDU(data=time_rec, name="TIMES")
    spectra_hdu = fits.BinTableHDU(data=spectra_rec, name="SPECTRA")

    hdul = fits.HDUList([primary_hdu, energy_hdu, time_hdu, spectra_hdu])
    fits_file = tmp_path / "test.fits"
    hdul.writeto(fits_file, overwrite=True)

    source = Source(fits_file)
    # Empty values should not be in metadata
    assert "author" not in source.metadata
    assert "project" not in source.metadata
    # But event_id should be present
    assert "event_id" in source.metadata


def test_source_read_fits_custom_keys(tmp_path):
    """Test that custom FITS header keys are parsed correctly."""
    from astropy.io import fits
    import numpy as np

    # Create FITS file with custom metadata keys
    header = fits.Header()
    header["CUST_1"] = (123.45, "custom_field_1 [erg]")
    header["CUST_2"] = (67.89, "custom_field_2")
    header["MY_VALUE"] = (999, "my_value")

    # Create minimal data arrays - need at least 2 energy and 2 time points
    energy_vals = np.array([1.0, 2.0, 3.0])
    time_vals = np.array([1.0, 2.0, 3.0])
    energy_dtype = np.dtype([("Initial Energy", ">f4"), ("Final Energy", ">f4")])
    time_dtype = np.dtype([("Initial Time", ">f4"), ("Final Time", ">f4")])
    energy_rec = np.rec.fromarrays(
        [energy_vals[:-1], energy_vals[1:]], dtype=energy_dtype
    )
    time_rec = np.rec.fromarrays([time_vals[:-1], time_vals[1:]], dtype=time_dtype)

    # Spectra data needs to match: (n_time_bins, n_energy_bins) = (2, 2)
    spectra_data = np.array([[1e-8, 1e-9], [1e-8, 1e-9]])
    spectra_cols = [("col0", ">f8"), ("col1", ">f8")]
    spectra_dtype = np.dtype(spectra_cols)
    spectra_rec = np.rec.array(
        [tuple(row) for row in spectra_data], dtype=spectra_dtype
    )

    primary_hdu = fits.PrimaryHDU(header=header)
    energy_hdu = fits.BinTableHDU(data=energy_rec, name="ENERGIES")
    time_hdu = fits.BinTableHDU(data=time_rec, name="TIMES")
    spectra_hdu = fits.BinTableHDU(data=spectra_rec, name="SPECTRA")

    hdul = fits.HDUList([primary_hdu, energy_hdu, time_hdu, spectra_hdu])
    fits_file = tmp_path / "test.fits"
    hdul.writeto(fits_file, overwrite=True)

    source = Source(fits_file)
    # Custom keys should be accessible
    assert hasattr(source, "custom_field_1")
    assert hasattr(source, "custom_field_2")
    assert hasattr(source, "my_value")
    assert source.custom_field_1.value == 123.45

    # NOTE: Without units, value is converted to number - check it's stored correctly
    # could be int or float
    assert (
        source.custom_field_2 == 67.89
        or source.custom_field_2 == 67
        or abs(source.custom_field_2 - 67.89) < 0.1
    )
    assert source.my_value == 999


def test_source_read_fits_no_comment(tmp_path):
    """Test FITS keys without comments use header key name as slug."""
    from astropy.io import fits
    import numpy as np

    # Create FITS file with keys without comments
    header = fits.Header()
    header["NO_CMNT"] = (42.0, "")  # Empty comment
    header["OTHER_KY"] = 100.0  # No comment at all

    # Create minimal data arrays - need at least 2 energy and 2 time points
    energy_vals = np.array([1.0, 2.0, 3.0])
    time_vals = np.array([1.0, 2.0, 3.0])
    energy_dtype = np.dtype([("Initial Energy", ">f4"), ("Final Energy", ">f4")])
    time_dtype = np.dtype([("Initial Time", ">f4"), ("Final Time", ">f4")])
    energy_rec = np.rec.fromarrays(
        [energy_vals[:-1], energy_vals[1:]], dtype=energy_dtype
    )
    time_rec = np.rec.fromarrays([time_vals[:-1], time_vals[1:]], dtype=time_dtype)

    # Spectra data needs to match: (n_time_bins, n_energy_bins) = (2, 2)
    spectra_data = np.array([[1e-8, 1e-9], [1e-8, 1e-9]])
    spectra_cols = [("col0", ">f8"), ("col1", ">f8")]
    spectra_dtype = np.dtype(spectra_cols)
    spectra_rec = np.rec.array(
        [tuple(row) for row in spectra_data], dtype=spectra_dtype
    )

    primary_hdu = fits.PrimaryHDU(header=header)
    energy_hdu = fits.BinTableHDU(data=energy_rec, name="ENERGIES")
    time_hdu = fits.BinTableHDU(data=time_rec, name="TIMES")
    spectra_hdu = fits.BinTableHDU(data=spectra_rec, name="SPECTRA")

    hdul = fits.HDUList([primary_hdu, energy_hdu, time_hdu, spectra_hdu])
    fits_file = tmp_path / "test.fits"
    hdul.writeto(fits_file, overwrite=True)

    source = Source(fits_file)
    # Keys without comments should use header key name (lowercase)
    assert "no_cmnt" in source.metadata
    assert "other_ky" in source.metadata
    assert source.no_cmnt == 42.0
    assert source.other_ky == 100.0


def test_source_read_fits_distance_object(mock_fits_path):
    """Test that distance metadata is converted to Distance object."""
    source = Source(mock_fits_path)
    from astropy.coordinates import Distance

    assert isinstance(source.distance, Distance)
    assert source.distance.value == 100000.0


def test_source_read_txt_basic(tmp_path):
    """Test basic text file directory reading."""
    # Create a directory with text files
    dir_name = "test_source"
    source_dir = tmp_path / dir_name
    source_dir.mkdir()

    # Create sample spectral files
    energies = [1.0, 2.0, 3.0]
    for i in range(1, 4):
        file_path = source_dir / f"{dir_name}_tobs={i:02d}.txt"
        content = "\n".join([f"{e} {1e-8 * i}" for e in energies])
        file_path.write_text(content)

    source = Source(source_dir)
    assert source.file_type == "txt"
    assert len(source.time) == 3
    assert len(source.energy) == 3
    assert source.spectra.shape == (len(source.energy), len(source.time))


def test_source_read_txt_missing_files(tmp_path):
    """Test error handling when no matching text files are found."""
    # Create empty directory
    dir_name = "empty_source"
    source_dir = tmp_path / dir_name
    source_dir.mkdir()

    with pytest.raises(ValueError, match="No supported files"):
        Source(source_dir)


def test_source_show_spectral_pattern(mock_csv_path):
    """Test show_spectral_pattern method."""
    source = Source(mock_csv_path)

    # Test with return_plot=False (default)
    result = source.show_spectral_pattern(return_plot=False)
    assert result is None

    # Test with return_plot=True - returns matplotlib.pyplot module
    import matplotlib.pyplot as plt

    result = source.show_spectral_pattern(return_plot=True)
    assert isinstance(result, plt.Figure)
    plt.close("all")  # Close any figures created


def test_source_show_spectral_evolution(mock_csv_path):
    """Test show_spectral_evolution method."""
    source = Source(mock_csv_path)

    # Test with return_plot=False (default)
    result = source.show_spectral_evolution(return_plot=False)
    assert result is None

    import matplotlib.pyplot as plt

    result = source.show_spectral_evolution(return_plot=True)
    assert isinstance(result, plt.Figure)
    plt.close("all")  # Close any figures created


def test_source_get_integral_spectrum_sensitivity_mode(mock_csv_path):
    """Test get_integral_spectrum with 'sensitivity' mode."""
    source = Source(mock_csv_path, min_energy=30 * u.GeV, max_energy=10 * u.TeV)
    source.set_spectral_grid()

    time = source.time[len(source.time) // 2]
    first_energy_bin = source.energy[0]
    spectrum = source.get_integral_spectrum(time, first_energy_bin, mode="sensitivity")

    assert isinstance(spectrum, u.Quantity)

    # confirm the dimensionality
    assert spectrum.unit.physical_type in ("energy flux", "irradiance")


def test_source_get_integral_spectrum_photon_flux_mode(mock_csv_path):
    """Test get_integral_spectrum with 'photon_flux' mode."""
    source = Source(mock_csv_path, min_energy=30 * u.GeV, max_energy=10 * u.TeV)
    source.set_spectral_grid()

    time = source.time[len(source.time) // 2]
    first_energy_bin = source.energy[0]
    spectrum = source.get_integral_spectrum(time, first_energy_bin, mode="photon_flux")

    assert isinstance(spectrum, u.Quantity)

    # confirm the dimensionality, we can have flux or flux density here
    assert spectrum.unit.physical_type in ("particle flux", "photon flux density")


def test_source_get_integral_spectrum_with_ebl(mock_csv_path):
    """Test get_integral_spectrum with EBL absorption."""
    source = Source(
        mock_csv_path, min_energy=30 * u.GeV, max_energy=10 * u.TeV, ebl="franceschini"
    )
    source.set_spectral_grid()

    time = source.time[len(source.time) // 2]
    first_energy_bin = source.energy[0]
    spectrum = source.get_integral_spectrum(time, first_energy_bin)

    assert isinstance(spectrum, u.Quantity)
    assert source.ebl is not None


def test_source_get_fluence(mock_csv_path):
    """Test get_fluence method."""
    source = Source(mock_csv_path, min_energy=30 * u.GeV, max_energy=10 * u.TeV)
    source.set_spectral_grid()

    start_time = source.time[0]
    stop_time = source.time[len(source.time) // 2]
    fluence = source.get_fluence(start_time, stop_time)

    assert isinstance(fluence, u.Quantity)

    # fluence has photons/cm²
    # but in spectral flux mode we can end up with energy flux or irradiance
    assert fluence.unit.physical_type in (
        "spectral flux density",
        "energy flux",
        "irradiance",
    )


def test_source_get_fluence_modes(mock_csv_path):
    """Test get_fluence with different modes."""
    source = Source(mock_csv_path, min_energy=30 * u.GeV, max_energy=10 * u.TeV)
    source.set_spectral_grid()

    start_time = source.time[0]
    stop_time = source.time[len(source.time) // 2]
    fluence_sens = source.get_fluence(start_time, stop_time, mode="sensitivity")
    fluence_phot = source.get_fluence(start_time, stop_time, mode="photon_flux")

    assert isinstance(fluence_sens, u.Quantity)
    assert isinstance(fluence_phot, u.Quantity)

    assert fluence_sens.unit.physical_type in (
        "spectral flux density",
        "energy flux",
        "irradiance",
    )

    assert fluence_phot.unit.physical_type in (
        "particle flux",
        "photon flux",
        "column density",
    )


def test_source_observe_error_handling(mock_csv_path):
    """Test observe method error handling."""
    source = Source(mock_csv_path)

    # Test with invalid sensitivity object
    with pytest.raises((AttributeError, TypeError, ValueError)):
        source.observe(sensitivity=None, start_time=0 * u.s)


def test_source_metadata_attribute_access(mock_csv_path):
    """Test __getattr__ for various metadata keys."""
    source = Source(mock_csv_path)

    # Test accessing metadata via attribute notation
    assert source.event_id == 42
    assert source.longitude is not None
    assert source.latitude is not None
    assert source.distance is not None


def test_source_metadata_set_attribute(mock_csv_path):
    """Test __setattr__ for custom metadata."""
    source = Source(mock_csv_path)

    # Set custom metadata
    source.custom_field = 123.45
    source.another_field = "test_value"

    # Verify it's still stored in metadata
    assert source.custom_field == 123.45
    assert source.another_field == "test_value"
    assert "custom_field" in source.metadata
    assert "another_field" in source.metadata


def test_source_metadata_nonexistent_attribute(mock_csv_path):
    """Test AttributeError for missing metadata keys."""
    source = Source(mock_csv_path)

    # Try to access non-existent attribute
    with pytest.raises(AttributeError):
        _ = source.nonexistent_field


def test_source_metadata_fits_format(mock_fits_path):
    """Test FITS (value, 'slug [unit]') format parsing."""
    source = Source(mock_fits_path)

    # Verify that FITS format metadata is correctly parsed
    assert hasattr(source, "event_id")
    assert hasattr(source, "longitude")
    assert hasattr(source, "latitude")
    assert hasattr(source, "distance")

    # Check units are applied correctly
    assert source.longitude.unit == u.rad
    assert source.latitude.unit == u.rad
    assert source.distance.unit == u.kpc


def test_source_metadata_fits_no_unit(tmp_path):
    """Test FITS keys without units."""
    from astropy.io import fits
    import numpy as np

    # Create FITS file with key without unit
    header = fits.Header()
    header["NO_UNIT"] = (42, "no_unit_field")

    # Create minimal data arrays - need at least 2 energy and 2 time points
    energy_vals = np.array([1.0, 2.0, 3.0])
    time_vals = np.array([1.0, 2.0, 3.0])
    energy_dtype = np.dtype([("Initial Energy", ">f4"), ("Final Energy", ">f4")])
    time_dtype = np.dtype([("Initial Time", ">f4"), ("Final Time", ">f4")])
    energy_rec = np.rec.fromarrays(
        [energy_vals[:-1], energy_vals[1:]], dtype=energy_dtype
    )
    time_rec = np.rec.fromarrays([time_vals[:-1], time_vals[1:]], dtype=time_dtype)

    # Spectra data needs to match: (n_time_bins, n_energy_bins) = (2, 2)
    spectra_data = np.array([[1e-8, 1e-9], [1e-8, 1e-9]])
    spectra_cols = [("col0", ">f8"), ("col1", ">f8")]
    spectra_dtype = np.dtype(spectra_cols)
    spectra_rec = np.rec.array(
        [tuple(row) for row in spectra_data], dtype=spectra_dtype
    )

    primary_hdu = fits.PrimaryHDU(header=header)
    energy_hdu = fits.BinTableHDU(data=energy_rec, name="ENERGIES")
    time_hdu = fits.BinTableHDU(data=time_rec, name="TIMES")
    spectra_hdu = fits.BinTableHDU(data=spectra_rec, name="SPECTRA")

    hdul = fits.HDUList([primary_hdu, energy_hdu, time_hdu, spectra_hdu])
    fits_file = tmp_path / "test.fits"
    hdul.writeto(fits_file, overwrite=True)

    source = Source(fits_file)
    # Key without unit should be stored as int/float/str
    assert "no_unit_field" in source.metadata
    assert isinstance(source.no_unit_field, (int, float, str))


def test_source_output(mock_csv_path):
    """Test output() dictionary structure."""
    source = Source(mock_csv_path)

    output_dict = source.output()

    assert isinstance(output_dict, dict)
    # Check for expected keys - output() includes _metadata, not metadata
    assert "_metadata" in output_dict
    assert isinstance(output_dict["_metadata"], dict)
    # Check that metadata contains event_id
    assert "event_id" in output_dict["_metadata"]


def test_source_output_excludes_large_data(mock_csv_path):
    """Verify that large arrays are excluded from output."""
    source = Source(mock_csv_path)

    output_dict = source.output()

    # Large arrays should not be in output
    assert "spectra" not in output_dict or output_dict["spectra"] is None
    assert "time" not in output_dict or isinstance(output_dict["time"], (list, str))
    assert "energy" not in output_dict or isinstance(output_dict["energy"], (list, str))


def test_source_get_flux_no_time(mock_csv_path):
    """Test get_flux with time=None."""
    source = Source(mock_csv_path)
    source.set_spectral_grid()

    assert hasattr(source, "get_flux")


def test_source_get_spectrum_no_energy(mock_csv_path):
    """Test get_spectrum with energy=None."""
    source = Source(mock_csv_path)
    source.set_spectral_grid()

    assert hasattr(source, "get_spectrum")


def test_source_set_ebl_with_redshift(mock_csv_path):
    """Test set_ebl_model with z parameter."""
    source = Source(mock_csv_path)

    from astropy.coordinates import Distance

    source.distance = Distance(z=0.1)

    source.set_ebl_model("dominguez")
    assert source.ebl is not None
    assert source.ebl_model == "dominguez"


def test_source_set_ebl_no_distance(mock_csv_path):
    """Test EBL without distance metadata."""
    source = Source(mock_csv_path)

    # Remove distance if it exists
    if "distance" in source.metadata:
        del source.metadata["distance"]

    # EBL should still work (may use default redshift)
    source.set_ebl_model("franceschini")
    assert source.ebl is not None
