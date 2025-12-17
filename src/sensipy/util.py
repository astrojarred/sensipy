import logging
import warnings
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path


@contextmanager
def suppress_warnings_and_logs(logging_ok: bool = True):
    """A helper function to suppress warnings and logs.

    Args:
        logging_ok: Whether to suppress logging. Defaults to True.

    Yields:
        None
    """

    if logging_ok:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    else:
        logging.disable(logging.WARNING)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
        logging.disable(logging.NOTSET)


def get_data_path(subpath: str = "") -> Path:
    """Get the path to the package data directory or a specific subpath within it.
    
    This function provides access to data files included with the sensipy package.
    The data directory contains EBL models and mock data files that are installed
    with the package.
    
    Works correctly when installed in virtual environments (e.g., via uv, pip).
    
    Args:
        subpath: Optional subpath within the data directory (e.g., "ebl" or "mock_data/GRB_42_mock.csv").
                 If empty, returns the data directory root.
    
    Returns:
        Path object pointing to the requested data location.
        Returns a filesystem Path that works for normal package installations.
    
    Examples:
        >>> # Get the data directory root
        >>> data_dir = get_data_path()
        >>> # Get the EBL directory
        >>> ebl_dir = get_data_path("ebl")
        >>> # Get a specific file
        >>> mock_file = get_data_path("mock_data/GRB_42_mock.csv")
    """
    # Primary method: Use __file__ attribute (most reliable for installed packages)
    # This works perfectly for normal installations in venvs, site-packages, etc.
    # When installed via uv/pip, sensipy.__file__ points to __init__.py in the package
    try:
        import sensipy
        if hasattr(sensipy, '__file__') and sensipy.__file__:
            package_dir = Path(sensipy.__file__).parent
            data_path = package_dir / "data"
            if subpath:
                data_path = data_path / subpath
            # Verify the data directory exists (or at least the package directory exists)
            if data_path.exists() or (not subpath and (package_dir / "data").exists()):
                return data_path
    except (ImportError, AttributeError, TypeError):
        pass
    
    # Fallback: Use importlib.resources.files() and try to convert to Path
    # This works for most normal installations (not zipped)
    try:
        package_data = files("sensipy") / "data"
        if subpath:
            package_data = package_data / subpath
        
        # Try to convert Traversable to Path
        # For normal filesystem installations, this often works
        try:
            # Some Traversable implementations expose the path directly
            if hasattr(package_data, 'path'):
                result = Path(package_data.path)
                if subpath:
                    # If we got the data dir, append subpath
                    result = result / subpath
                if result.exists() or not subpath:
                    return result
            
            # Try string conversion (works for filesystem-based Traversables)
            path_str = str(package_data)
            # Check if it looks like a real filesystem path
            if path_str.startswith('/') or (len(path_str) > 1 and path_str[1] == ':'):
                result = Path(path_str)
                if result.exists() or not subpath:
                    return result
        except (AttributeError, TypeError, ValueError):
            pass
    except Exception:
        pass
    
    # Last resort: Search sys.path for the package
    # This handles edge cases where the package might be in an unusual location
    import sys
    for search_path in sys.path:
        if not search_path:
            continue
        try:
            potential_path = Path(search_path) / "sensipy" / "data"
            if subpath:
                potential_path = potential_path / subpath
            if potential_path.exists():
                return potential_path
        except Exception:
            continue
    
    # If we get here, we couldn't find the data directory
    raise RuntimeError(
        "Could not locate sensipy package data directory. "
        "This may indicate the package is not properly installed or data files are missing. "
        f"Looking for: sensipy/data/{subpath if subpath else ''}"
    )
