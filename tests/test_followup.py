import astropy.units as u
import numpy as np
import pandas as pd
import pytest

from sensipy import followup


def test_get_row_found(mock_lookup_df):
    """Test get_row when row is found."""
    row = followup.get_row(
        lookup_df=mock_lookup_df,
        event_id=1,
        irf_site="north",
        irf_zenith=20,
        irf_ebl_model="franceschini",
    )

    assert row["event_id"] == 1
    assert row["irf_site"] == "north"
    assert row["irf_zenith"] == 20


def test_get_row_not_found(mock_lookup_df):
    """Test get_row when no row is found."""
    with pytest.raises(ValueError, match="No matching row found"):
        followup.get_row(
            lookup_df=mock_lookup_df,
            event_id=999,
            irf_site="north",
            irf_zenith=20,
            irf_ebl_model="franceschini",
        )


def test_get_row_multiple_matches(mock_lookup_df):
    """Test get_row when multiple rows match, it should take the first row"""
    # Add duplicate row
    duplicate_row = mock_lookup_df.iloc[0].copy()
    duplicate_df = pd.concat(
        [mock_lookup_df, duplicate_row.to_frame().T], ignore_index=True
    )

    # Should not raise error, just use first match
    row = followup.get_row(
        lookup_df=duplicate_df,
        event_id=1,
        irf_site="north",
        irf_zenith=20,
        irf_ebl_model="franceschini",
    )
    assert row is not None


def test_get_row_empty_filters(mock_lookup_df):
    """Ensure get_row raises error when no filters are provided."""
    with pytest.raises(ValueError, match="At least one filter must be provided"):
        followup.get_row(
            lookup_df=mock_lookup_df,
        )


def test_get_row_invalid_column(mock_lookup_df):
    """Ensure get_row raises error when invalid column is specified."""
    with pytest.raises(ValueError, match="Column.*does not exist"):
        followup.get_row(
            lookup_df=mock_lookup_df,
            nonexistent_column=1,
        )


def test_extrapolate_obs_time_valid(mock_lookup_df):
    """Test extrapolate_obs_time works with valid delay."""
    result = followup.extrapolate_obs_time(
        delay=1000 * u.s,
        lookup_df=mock_lookup_df,
        filters={"event_id": 1, "irf_site": "north", "irf_zenith": 20},
    )

    assert "obs_time" in result
    assert result["obs_time"] > 0
    assert "error_message" in result


def test_extrapolate_obs_time_below_minimum(mock_lookup_df):
    """Test extrapolate_obs_time with delay below minimum."""
    with pytest.raises(ValueError, match="Minimum delay"):
        followup.extrapolate_obs_time(
            delay=5 * u.s,  # Below minimum of 10
            lookup_df=mock_lookup_df,
            filters={"event_id": 1, "irf_site": "north", "irf_zenith": 20},
        )


def test_extrapolate_obs_time_above_maximum(mock_lookup_df, capsys):
    """Test extrapolate_obs_time with delay above maximum (should warn)."""
    result = followup.extrapolate_obs_time(
        delay=200000 * u.s,  # Above maximum of 100000
        lookup_df=mock_lookup_df,
        filters={"event_id": 1, "irf_site": "north", "irf_zenith": 20},
    )

    # Should still return a result (extrapolated)
    assert "obs_time" in result
    # Check that warning was printed
    captured = capsys.readouterr()
    assert "Warning" in captured.out or "warning" in captured.out.lower()


def test_extrapolate_obs_time_other_info(mock_lookup_df):
    """Test the function works with other_info parameter."""
    result = followup.extrapolate_obs_time(
        delay=1000 * u.s,
        lookup_df=mock_lookup_df,
        filters={"event_id": 1, "irf_site": "north", "irf_zenith": 20},
        other_info=["long", "lat", "dist"],
    )

    assert "long" in result
    assert "lat" in result
    assert "dist" in result


def test_extrapolate_obs_time_no_matching_data(mock_lookup_df):
    """Test extrapolate_obs_time when no data matches filters."""
    result = followup.extrapolate_obs_time(
        delay=1000 * u.s,
        lookup_df=mock_lookup_df,
        filters={"event_id": 999, "irf_site": "north", "irf_zenith": 20},
    )

    assert result["obs_time"] == -1
    assert "error_message" in result
    assert "No matching data" in result["error_message"]


def test_get_sensitivity_from_sens_df(sample_sensitivity_df):
    """Test getting sensitivity from the mock lookup_df."""
    sens = followup.get_sensitivity(
        lookup_df=sample_sensitivity_df,
        event_id=1,
        irf_site="north",
        irf_zenith=20,
        irf_ebl_model="franceschini",
    )

    assert sens is not None
    assert sens.observatory == "ctao_north"
    assert len(sens.sensitivity_curve) > 0
    assert len(sens.photon_flux_curve) > 0


def test_get_sensitivity_from_curves():
    """Test getting sensitivity with a given sensitivity_curve and photon_flux_curve."""
    sensitivity_curve = np.logspace(-10, -12, 10) * u.Unit("erg cm-2 s-1")
    photon_flux_curve = np.logspace(-9, -11, 10) * u.Unit("cm-2 s-1")

    # test as quantities
    sens = followup.get_sensitivity(
        sensitivity_curve=sensitivity_curve,
        photon_flux_curve=photon_flux_curve,
        observatory="ctao_north",
    )

    assert sens is not None
    assert sens.observatory == "ctao_north"

    # test as numpy arrays
    sens = followup.get_sensitivity(
        sensitivity_curve=sensitivity_curve.value,
        photon_flux_curve=photon_flux_curve.value,
        observatory="ctao_north",
    )

    assert sens is not None
    assert sens.observatory == "ctao_north"

    # test as a list
    sens = followup.get_sensitivity(
        sensitivity_curve=sensitivity_curve.value.tolist(),
        photon_flux_curve=photon_flux_curve.value.tolist(),
        observatory="ctao_north",
    )

    assert sens is not None
    assert sens.observatory == "ctao_north"


def test_get_sensitivity_conflicting_inputs(mock_lookup_df):
    """Test that you cannot provide both a lookup_df and sensitivity_curve/photon_flux_curve."""
    sensitivity_curve = np.logspace(-10, -12, 10) * u.Unit("erg cm-2 s-1")

    with pytest.raises(ValueError, match="If lookup_df is provided"):
        followup.get_sensitivity(
            lookup_df=mock_lookup_df,
            sensitivity_curve=sensitivity_curve.value.tolist(),
            event_id=1,
            irf_site="north",
        )


def test_get_sensitivity_missing_inputs():
    """Test that you cannot provide neither lookup_df nor sensitivity_curve/photon_flux_curve."""
    with pytest.raises(ValueError, match="Must provide either lookup_df"):
        followup.get_sensitivity(
            observatory="ctao_north",
        )


def test_get_sensitivity_observatory_from_irf_site(sample_sensitivity_df):
    """Test that observatory is properly assumed from irf_site."""
    sens = followup.get_sensitivity(
        lookup_df=sample_sensitivity_df,
        event_id=1,
        irf_site="north",  # observatory assumed from irf_site
        irf_zenith=20,
        irf_ebl_model="franceschini",
    )

    assert sens is not None
    assert sens.observatory == "ctao_north"


def test_get_sensitivity_observatory_direct(sample_sensitivity_df):
    """Test that you can provide the observatory directly."""
    sens = followup.get_sensitivity(
        lookup_df=sample_sensitivity_df,
        event_id=1,
        observatory="ctao_south",  # observatory provided directly
        irf_zenith=20,
        irf_ebl_model="franceschini",
    )

    assert sens is not None
    assert sens.observatory == "ctao_south"


def test_get_sensitivity_observatory_required_for_curves():
    """Ensure that you must provide the observatory when using curves directly."""
    sensitivity_curve = np.logspace(-10, -12, 10) * u.Unit("erg cm-2 s-1")
    photon_flux_curve = np.logspace(-9, -11, 10) * u.Unit("cm-2 s-1")

    with pytest.raises(ValueError, match="observatory parameter is required"):
        followup.get_sensitivity(
            sensitivity_curve=sensitivity_curve.value.tolist(),
            photon_flux_curve=photon_flux_curve.value.tolist(),
        )


def test_get_exposure_with_extrapolation_df(mock_lookup_df):
    """Test using the lookup_df to extrapolate the observation time."""
    result = followup.get_exposure(
        delay=1000 * u.s,
        lookup_df=mock_lookup_df,
        event_id=1,
        irf_site="north",
        irf_zenith=20,
    )

    assert "obs_time" in result
    assert "start_time" in result
    assert "seen" in result
    assert "id" in result
    assert result["id"] == 1


def test_get_exposure_without_extrapolation_df():
    """Test that you must provide a source_filepath when lookup_df is not provided."""
    with pytest.raises(ValueError, match="Must provide source_filepath"):
        followup.get_exposure(
            delay=1000 * u.s,
            # No lookup_df provided - should require source_filepath
            event_id=1,
            irf_site="north",
            irf_zenith=20,
        )


def test_get_exposure_invalid_delay_unit(mock_lookup_df):
    """Test that you cannot provide a delay with an invalid unit."""
    with pytest.raises(ValueError, match="delay must be a time quantity"):
        followup.get_exposure(
            delay=1000 * u.m,  # Wrong unit
            lookup_df=mock_lookup_df,
            event_id=1,
            irf_site="north",
            irf_zenith=20,
        )


def test_get_exposure_invalid_energy_units(mock_lookup_df):
    """Test get_exposure raises error for invalid energy units."""
    with pytest.raises(ValueError, match="min_energy must be an energy quantity"):
        followup.get_exposure(
            delay=1000 * u.s,
            lookup_df=mock_lookup_df,
            event_id=1,
            irf_site="north",
            irf_zenith=20,
            min_energy=1.0 * u.s,  # Wrong unit
        )


def test_get_exposure_invalid_radius_unit(mock_lookup_df):
    """Test get_exposure raises error for invalid radius unit."""
    with pytest.raises(ValueError, match="radius must be an angle quantity"):
        followup.get_exposure(
            delay=1000 * u.s,
            lookup_df=mock_lookup_df,
            event_id=1,
            irf_site="north",
            irf_zenith=20,
            radius=3.0 * u.s,  # Wrong unit
        )


def test_get_exposure_custom_parameters(mock_lookup_df):
    """Test using custom parameters."""
    result = followup.get_exposure(
        delay=1000 * u.s,
        lookup_df=mock_lookup_df,
        event_id=1,
        irf_site="north",
        irf_zenith=20,
        radius=5.0 * u.deg,
        min_energy=0.1 * u.TeV,
        max_energy=5.0 * u.TeV,
        target_precision=10 * u.s,
        max_time=6 * u.h,
    )

    assert result is not None
    assert "min_energy" in result
    assert "max_energy" in result


def test_get_exposure_no_matching_data(mock_lookup_df):
    """Test that you get -1 obs_time when no data matches filters."""
    result = followup.get_exposure(
        delay=1000 * u.s,
        lookup_df=mock_lookup_df,
        event_id=999,  # Non-existent event
        irf_site="north",
        irf_zenith=20,
    )

    assert result["obs_time"] == -1
    assert not result["seen"]


def test_get_exposure_without_irf_site(mock_lookup_df):
    """Test that you can provide the observatory directly without irf_site."""
    result = followup.get_exposure(
        delay=1000 * u.s,
        lookup_df=mock_lookup_df,
        event_id=1,
        observatory="ctao_north",  # observatory provided directly
        irf_zenith=20,
    )

    assert "obs_time" in result


def test_extrapolate_obs_time_custom_column_names(mock_lookup_df):
    """Test using custom column names for delay and observation time."""
    # Rename columns
    df = mock_lookup_df.rename(
        columns={"obs_delay": "delay_col", "obs_time": "time_col"}
    )

    result = followup.extrapolate_obs_time(
        delay=12 * u.s,
        lookup_df=df,
        filters={"event_id": 1, "irf_site": "north", "irf_zenith": 20},
        delay_column="delay_col",
        obs_time_column="time_col",
    )

    assert "obs_time" in result
    assert result["obs_time"] > 0


def test_extrapolate_obs_time_missing_column(mock_lookup_df):
    """Test that you get an error when required columns don't exist."""
    result = followup.extrapolate_obs_time(
        delay=1000 * u.s,
        lookup_df=mock_lookup_df,
        filters={"event_id": 1, "irf_site": "north", "irf_zenith": 20},
        delay_column="nonexistent_column",
        obs_time_column="obs_time",
    )

    assert result["obs_time"] == -1
    assert "error_message" in result
    assert "nonexistent_column" in result["error_message"]


def test_get_exposure_custom_column_names(mock_lookup_df):
    """Test using custom column names for delay and observation time."""
    # Rename columns
    df = mock_lookup_df.rename(
        columns={"obs_delay": "delay_col", "obs_time": "time_col"}
    )

    result = followup.get_exposure(
        delay=1000 * u.s,
        lookup_df=df,
        event_id=1,
        irf_site="north",
        irf_zenith=20,
        delay_column="delay_col",
        obs_time_column="time_col",
    )

    assert "obs_time" in result
    assert result["obs_time"] > 0
