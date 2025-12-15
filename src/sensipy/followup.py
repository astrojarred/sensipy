import warnings
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from astropy import units as u
from numpy import log10
from scipy.interpolate import interp1d

from . import sensitivity, source


def get_row(
    lookup_df: pd.DataFrame,
    **filters: str | float | int | bool,
):
    """Retrieve a row from the lookup dataframe matching the specified criteria.

    Searches the lookup dataframe for a row that matches all of the provided
    filter criteria. If multiple rows match, returns the first one. Raises an error if
    no matching row is found.

    Args:
        lookup_df: DataFrame containing lookup data.
        **filters: Column-value pairs to filter the dataframe. Keyword arguments should
            be column names, values should be the values to match. This allows filtering
            on any column in the dataframe.

    Returns:
        pandas.Series: The first matching row from the dataframe.

    Raises:
        ValueError: If no row matches all the specified criteria, if no filters are
            provided, or if a specified column does not exist in the dataframe.

    Example:
        >>> row = get_row(
        ...     lookup_df=df,
        ...     event_id=42,
        ...     irf_site="north",
        ...     irf_zenith=20,
        ...     irf_ebl=False,
        ... )
    """
    if not filters:
        raise ValueError("At least one filter must be provided.")
    
    # Build the filter condition dynamically
    mask = pd.Series([True] * len(lookup_df), index=lookup_df.index)
    for column, value in filters.items():
        if column not in lookup_df.columns:
            raise ValueError(
                f"Column '{column}' specified in filters does not exist in the dataframe. "
                f"Available columns: {list(lookup_df.columns)}"
            )
        mask = mask & (lookup_df[column] == value)
    
    rows = lookup_df[mask]

    if len(rows) < 1:
        raise ValueError("No matching row found with these values.")
    if len(rows) > 1:
        # print(
        #     f"Warning: multiple ({len(rows)}) sensitivities found with these values. Will use first row."
        # )
        pass

    return rows.iloc[0]


def extrapolate_obs_time(
    delay: u.Quantity,
    lookup_df: pd.DataFrame,
    filters: dict[str, str | float | int] = {},
    other_info: list[str] = [],
    delay_column: str = "obs_delay",
    obs_time_column: str = "obs_time",
):
    """Estimate the required observation time for a given delay using interpolation.

    Uses logarithmic interpolation to estimate the observation time needed to detect
    an event at a specific delay time. The function looks up pre-computed observation
    times from the lookup dataframe and interpolates between them. If the delay
    exceeds the maximum value in the dataframe, a warning is issued and the value is
    extrapolated beyond the data range.

    Args:
        delay: Time delay from the event trigger, as an astropy Quantity with time units.
        lookup_df: DataFrame containing pre-computed observation times at various
            delays. Must contain columns for observation delay, observation time,
            and any filter columns.
        filters: Dictionary of column-value pairs to filter the dataframe.
            Keys should be column names, values should be the values to match.
            All filters must match for a row to be included.
        other_info: List of column names to include in the returned dictionary.
            These are extracted from the first matching row.
        delay_column: Name of the column containing observation delays. Defaults to "obs_delay".
        obs_time_column: Name of the column containing observation times. Defaults to "obs_time".

    Returns:
        dict: Dictionary containing:
            - "obs_time": Estimated observation time in seconds, or -1 if the event
                is not detectable or extrapolation fails.
            - "error_message": Empty string if successful, otherwise an error description.
            - Additional keys from other_info if provided.

    Raises:
        ValueError: If the requested delay is below the minimum delay available in
            the dataframe for the matching rows, or if the required columns don't exist.
    """
    res: dict[str, Any] = {}
    delay = delay.to("s").value
    event_info = lookup_df.copy()

    # Validate that required columns exist
    if delay_column not in event_info.columns:
        res["error_message"] = (
            f"Column '{delay_column}' (delay_column) does not exist in the dataframe. "
            f"Available columns: {list(event_info.columns)}"
        )
        res["obs_time"] = -1
        return res
    
    if obs_time_column not in event_info.columns:
        res["error_message"] = (
            f"Column '{obs_time_column}' (obs_time_column) does not exist in the dataframe. "
            f"Available columns: {list(event_info.columns)}"
        )
        res["obs_time"] = -1
        return res

    if filters:
        for key, value in filters.items():
            if key not in event_info.columns:
                res["error_message"] = (
                    f"Column '{key}' specified in filters does not exist in the dataframe. "
                    f"Available columns: {list(event_info.columns)}"
                )
                res["obs_time"] = -1
                return res
            event_info = event_info[event_info[key] == value]

    if other_info:
        if len(event_info) == 0:
            res["error_message"] = (
                f"No matching data found with filters {filters}"
            )
            res["obs_time"] = -1
            return res
        for key in other_info:
            if key not in event_info.columns:
                res["error_message"] = (
                    f"Column '{key}' specified in other_info does not exist in the dataframe."
                )
                res["obs_time"] = -1
                return res
            res[key] = event_info.iloc[0][key]

    if len(event_info) == 0:
        res["error_message"] = (
            f"No matching data found with filters {filters}"
        )
        res["obs_time"] = -1
        return res

    event_dict = event_info.set_index(delay_column)[obs_time_column].to_dict()

    if not event_dict:
        res["error_message"] = (
            f"No matching data found with filters {filters}"
        )
        res["obs_time"] = -1
        return res

    if delay < min(event_dict.keys()):
        res["error_message"] = (
            f"Minimum delay is {min(event_dict.keys())} seconds for this simulation"
        )
        res["obs_time"] = -1
        raise ValueError(
            f"Minimum delay is {min(event_dict.keys())} seconds for this simulation [{delay}s requested]"
        )
    elif delay > max(event_dict.keys()):
        print(
            f"Warning: delay is greater than maximum delay of {max(event_dict.keys())}s for this simulation [{delay}s requested], value will be extrapolated."
        )

    # remove negative values
    pos_event_dict = {k: v for k, v in event_dict.items() if v > 0}

    if not pos_event_dict:
        res["error_message"] = (
            f"Event is never detectable under the observation conditions {filters}"
        )
        res["obs_time"] = -1
        return res

    pairs = sorted((log10(k), log10(v)) for k, v in pos_event_dict.items())
    xs, ys = zip(*pairs)  # safe since not empty

    # perform log interpolation
    interp = interp1d(xs, ys, kind="linear", bounds_error=True)

    try:
        res["obs_time"] = 10 ** interp(log10(delay))
        res["error_message"] = ""
    except ValueError:
        res["obs_time"] = -1
        res["error_message"] = "Extrapolation failed for this simulation"

    # Add filters to result dictionary
    res.update(filters)

    return res


def get_sensitivity(
    lookup_df: pd.DataFrame | None = None,
    sensitivity_curve: list[float] | None = None,
    photon_flux_curve: list[float] | None = None,
    observatory: str | None = None,
    radius: u.Quantity = 3.0 * u.deg,
    min_energy: u.Quantity = 0.02 * u.TeV,
    max_energy: u.Quantity = 10 * u.TeV,
    **filters: str | float | int | bool,
):
    """Create a Sensitivity object for a given event and observation configuration.

    Constructs a Sensitivity instance either by looking up pre-computed sensitivity
    curves from a lookup dataframe or by using directly provided sensitivity and photon flux
    curves. The sensitivity object is configured for the specified observatory,
    energy range, and observation region.

    Args:
        lookup_df: Optional DataFrame containing pre-computed sensitivity data. If provided,
            sensitivity_curve and photon_flux_curve must be None.
        sensitivity_curve: Optional list of sensitivity values in erg cm⁻² s⁻¹. Must be
            provided along with photon_flux_curve if lookup_df is None.
        photon_flux_curve: Optional list of photon flux values in cm⁻² s⁻¹. Must be
            provided along with sensitivity_curve if lookup_df is None.
        observatory: Observatory name (e.g., "ctao_north", "ctao_south", "hess", "magic").
            Must be one of the valid observatory locations from gammapy.data.observatory_locations.
            Required if lookup_df is None. If lookup_df is provided and observatory is None,
            will attempt to construct from "irf_site" in filters (e.g., "north" -> "ctao_north").
        radius: Angular radius of the observation region. Defaults to 3.0 degrees.
        min_energy: Minimum energy for the sensitivity calculation. Defaults to 0.02 TeV.
        max_energy: Maximum energy for the sensitivity calculation. Defaults to 10 TeV.
        **filters: Column-value pairs to filter the dataframe when lookup_df is provided.
            Keyword arguments should be column names, values should be the values to match.
            This allows filtering on any column in the dataframe.

    Returns:
        Sensitivity: A configured Sensitivity object ready for use in exposure calculations.

    Raises:
        ValueError: If both lookup_df and curves are provided, if neither lookup_df nor
            both curves are provided, if sensitivity_curve is not a list or Quantity,
            or if observatory cannot be determined when needed.

    Example:
        >>> # Using lookup_df with filters
        >>> sens = get_sensitivity(
        ...     lookup_df=df,
        ...     event_id=42,
        ...     irf_zenith=20,
        ...     irf_ebl=False,
        ...     observatory="ctao_north",
        ... )
        >>> # Using curves directly
        >>> sens = get_sensitivity(
        ...     sensitivity_curve=[1e-10, 1e-11],
        ...     photon_flux_curve=[1e-9, 1e-10],
        ...     observatory="ctao_south",
        ... )
    """
    if lookup_df is not None:
        if sensitivity_curve is not None or photon_flux_curve is not None:
            raise ValueError(
                "If lookup_df is provided, sensitivity_curve and photon_flux_curve must both be None."
            )
    else:
        if sensitivity_curve is None or photon_flux_curve is None:
            raise ValueError(
                "Must provide either lookup_df or both sensitivity_curve and photon_flux_curve"
            )

    # Determine observatory name
    if lookup_df is not None:
        # Get row from dataframe using filters
        row = get_row(lookup_df=lookup_df, **filters)
        
        sensitivity_curve = row["sensitivity_curve"]
        photon_flux_curve = row["photon_flux_curve"]
        
        # If observatory not provided, try to construct from irf_site if available
        if observatory is None:
            if "irf_site" in filters:
                site = str(filters["irf_site"])
                observatory = f"ctao_{site}"
            else:
                # If no observatory and no irf_site, we can't determine it
                # But we'll allow this - observatory might not be strictly necessary
                # for some use cases. Let's default to None and let Sensitivity handle it.
                observatory = "_"
    else:
        # Using curves directly - observatory must be provided
        if observatory is None:
            raise ValueError(
                "observatory parameter is required when providing sensitivity_curve and photon_flux_curve directly."
            )

    if isinstance(sensitivity_curve, (list, u.Quantity)):
        n_sensitivity_points = len(sensitivity_curve)
    else:
        raise ValueError(
            f"sensitivity_curve must be a list or u.Quantity, got {type(sensitivity_curve)}"
        )

    sens = sensitivity.Sensitivity(
        observatory=observatory,
        radius=radius,
        min_energy=min_energy,
        max_energy=max_energy,
        n_sensitivity_points=n_sensitivity_points,
        sensitivity_curve=sensitivity_curve * u.Unit("erg cm-2 s-1"),
        photon_flux_curve=photon_flux_curve * u.Unit("cm-2 s-1"),
    )

    return sens


def get_exposure(
    delay: u.Quantity,
    source_filepath: Path | str | None = None,
    lookup_df: pd.DataFrame | Path | str | None = None,
    sensitivity_curve: list | None = None,
    photon_flux_curve: list | None = None,
    redshift: float | None = None,
    radius: u.Quantity = 3.0 * u.deg,
    min_energy: u.Quantity = 0.02 * u.TeV,
    max_energy: u.Quantity = 10 * u.TeV,
    target_precision: u.Quantity = 1 * u.s,
    max_time: u.Quantity = 12 * u.h,
    sensitivity_mode: Literal["sensitivity", "photon_flux"] = "sensitivity",
    n_time_steps: int = 10,
    other_info: list[str] = [],
    delay_column: str = "obs_delay",
    obs_time_column: str = "obs_time",
    **filters: str | float | int | bool,
):
    """Calculate exposure information for observing an event with a spectrum evolving in time.

    Determines the required observation time and other exposure parameters for detecting
    an event at a given delay. This function supports two modes of operation:

    1. **Lookup mode**: If lookup_df is provided, uses pre-computed
       observation times from simulations to quickly estimate the exposure time via
       interpolation. This is faster but requires pre-existing simulation data.

    2. **Direct calculation mode**: If source_filepath is provided, loads the source
       spectrum and performs a full observation calculation using the Source.observe()
       method. This is more accurate but computationally intensive.

    Args:
        delay: Time delay from the event trigger, as an astropy Quantity with time units.
        source_filepath: Path to the source file. Required if lookup_df is None.
        lookup_df: Optional DataFrame or path to parquet file containing
            pre-computed observation times or sensitivity data. If provided, uses lookup mode.
        sensitivity_curve: Optional list of sensitivity values. Must be provided with
            photon_flux_curve if lookup_df is None.
        photon_flux_curve: Optional list of photon flux values. Must be provided with
            sensitivity_curve if lookup_df is None.
        redshift: Optional redshift value. If provided, overrides the redshift from
            the source file for EBL calculations.
        radius: Angular radius of the observation region. Defaults to 3.0 degrees.
        min_energy: Minimum energy for the calculation. Defaults to 0.02 TeV.
        max_energy: Maximum energy for the calculation. Defaults to 10 TeV.
        target_precision: Precision for rounding observation times. Defaults to 1 second.
        max_time: Maximum allowed observation time. Defaults to 12 hours.
        sensitivity_mode: Whether to use "sensitivity" or "photon_flux" for detection
            calculations. Defaults to "sensitivity".
        n_time_steps: Number of time steps for the observation calculation. Defaults to 10.
        other_info: List of column names to include in the returned dictionary when using
            lookup mode. These are extracted from the lookup dataframe. Defaults to
            ["long", "lat", "eiso", "dist", "theta_view", "irf_ebl_model"].
        delay_column: Name of the column containing observation delays in lookup_df.
            Defaults to "obs_delay".
        obs_time_column: Name of the column containing observation times in lookup_df.
            Defaults to "obs_time".
        **filters: Column-value pairs to filter the dataframes. Keyword arguments should
            be column names, values should be the values to match. This allows filtering
            on any column in the dataframe. Common filters include:
            - event_id: Event identifier (if present in lookup table)
            - irf_site: Observatory site name (e.g., "north" or "south")
            - irf_zenith: Zenith angle in degrees
            - irf_ebl: Boolean indicating if EBL is used (True/False)
            - irf_ebl_model: EBL model name (e.g., "franceschini", "dominguez11")
            - irf_config: IRF configuration name (default: "alpha")
            - irf_duration: Observation duration in seconds (default: 1800)

    Returns:
        dict: Dictionary containing exposure information. In lookup mode, includes:
            - "obs_time": Observation time in seconds (or -1 if not detectable)
            - "start_time": Start time of observation
            - "end_time": End time of observation (or -1 if not detectable)
            - "seen": Boolean indicating if the event is detectable
            - "id": Event identifier
            - "long", "lat": Source coordinates in radians
            - "eiso": Isotropic equivalent energy in erg
            - "dist": Distance in kpc
            - "angle": Viewing angle in degrees
            - "ebl_model": EBL model name used
            - "min_energy", "max_energy": Energy range
            - "error_message": Error description if applicable

        In direct calculation mode, returns the result from Source.observe().

    Raises:
        ValueError: If unit types are incorrect, if lookup_df is None and
            source_filepath is not provided, or if delay is below the minimum in the
            lookup dataframe.
    """
    if delay.unit.physical_type != "time":
        raise ValueError(f"delay must be a time quantity, got {delay}")
    if min_energy.unit.physical_type != "energy":
        raise ValueError(f"min_energy must be an energy quantity, got {min_energy}")
    if max_energy.unit.physical_type != "energy":
        raise ValueError(f"max_energy must be an energy quantity, got {max_energy}")
    if radius.unit.physical_type != "angle":
        raise ValueError(f"radius must be an angle quantity, got {radius}")
    if target_precision.unit.physical_type != "time":
        raise ValueError(
            f"target_precision must be a time quantity, got {target_precision}"
        )
    if max_time.unit.physical_type != "time":
        raise ValueError(f"max_time must be a time quantity, got {max_time}")

    delay = delay.to("s")
    min_energy = min_energy.to("TeV")
    max_energy = max_energy.to("TeV")
    radius = radius.to("deg")
    target_precision = target_precision.to("s")
    max_time = max_time.to("s")

    # Extract event_id from filters if present (for return value)
    event_id = filters.get("event_id", None)

    # Determine observatory for get_sensitivity
    # Try to extract from filters, or construct from irf_site if available
    observatory = filters.get("observatory", None)
    if observatory is None and "irf_site" in filters:
        site = str(filters["irf_site"])
        observatory = f"ctao_{site}"

    if lookup_df is not None:
        if isinstance(lookup_df, (Path, str)):
            lookup_df = pd.read_parquet(lookup_df)

        # Build filters dict for extrapolate_obs_time (exclude observatory from filters)
        # observatory is not a column in the dataframe
        lookup_filters = {
            k: v
            for k, v in filters.items()
            if k != "observatory"
        }

        obs_info = extrapolate_obs_time(
            delay=delay,
            lookup_df=lookup_df,
            filters=lookup_filters,
            other_info=other_info,
            delay_column=delay_column,
            obs_time_column=obs_time_column,
        )

        obs_time = obs_info["obs_time"]
        if obs_time > 0:
            if obs_time > max_time.value:
                obs_info["error_message"] = (
                    f"Exposure time of {int(obs_time)} s exceeds maximum time"
                )
                obs_time = -1
            else:
                obs_time = round(obs_time / target_precision.value) * target_precision

        other_info_dict = {
            "min_energy": min_energy,
            "max_energy": max_energy,
            "seen": True if obs_time > 0 else False,
            "obs_time": obs_time if obs_time > 0 else -1,
            "start_time": delay,
            "end_time": delay + obs_time if obs_time > 0 else -1,
        }
        # Add event_id to result if it was in filters
        if event_id is not None:
            other_info_dict["id"] = event_id

        # Add filters to result dictionary
        result = {**obs_info, **other_info_dict, **filters}
        return result

    else:
        if not source_filepath:
            raise ValueError(
                "Must provide source_filepath if lookup_df is not provided"
            )

    # Extract ebl from filters if present, otherwise default to None
    ebl = filters.get("irf_ebl", None)
    
    if ebl is not None and not isinstance(ebl, str):
        raise ValueError(f"ebl must be a string or None, got {ebl} of type {type(ebl)}")
    
    if not isinstance(observatory, str):
        raise ValueError(f"observatory must be a string, got {observatory} of type {type(observatory)}")

    sens = get_sensitivity(
        lookup_df=lookup_df,
        sensitivity_curve=sensitivity_curve,
        photon_flux_curve=photon_flux_curve,
        observatory=observatory,
        radius=radius,
        min_energy=min_energy,
        max_energy=max_energy,
        **filters,
    )

    grb = source.Source(source_filepath, min_energy, max_energy, ebl=ebl)

    if redshift is not None:
        grb.set_ebl_model(ebl, z=redshift)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN slice encountered")
        result = grb.observe(
            sens,
            start_time=delay,
            min_energy=min_energy,
            max_energy=max_energy,
            target_precision=target_precision,
            max_time=max_time,
            sensitivity_mode=sensitivity_mode,
            n_time_steps=n_time_steps,
        )

    # Add filters to result dictionary
    result.update(filters)

    return result
