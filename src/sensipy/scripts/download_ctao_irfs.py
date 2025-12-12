"""
Download CTA Instrument Response Functions (IRFs) from Zenodo.

This module provides functionality to download official CTA IRF releases
from Zenodo. It can be used as a library or via the command line.

Usage:
    # As a CLI command (after installing sensipy)
    sensipy-download-irfs --output-dir ./IRFs

    # Or programmatically
    from sensipy.scripts.download_ctao_irfs import download_ctao_irfs
    download_ctao_irfs(output_dir="./IRFs")
"""

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm

from ..logging import logger

log = logger(__name__)


# IRF download sources from Zenodo
IRF_SOURCES = {
    "prod5-v0.1": {
        "url": "https://zenodo.org/records/5499840/files/cta-prod5-zenodo-fitsonly-v0.1.zip",
        "filename": "cta-prod5-zenodo-fitsonly-v0.1.zip",
        "description": "CTA prod5 v0.1 IRFs (Alpha configuration)",
    },
    "prod3b-v2": {
        "url": "https://zenodo.org/records/5163273/files/CTA-Performance-IRFs-prod3b-v2-v1.0.0.zip",
        "filename": "CTA-Prod3b-v2-Zenodo-FITS.zip",
        "description": "CTA prod3b v2 IRFs",
    },
}


class DownloadProgressBar(tqdm):
    """Progress bar for file downloads."""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        """
        Update progress bar.

        Args:
            b: Number of blocks transferred so far
            bsize: Size of each block (in bytes)
            tsize: Total size (in bytes), if known
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(
    url: str, output_path: Path, description: str = "Downloading"
) -> None:
    """
    Download a file with progress bar.

    Args:
        url: URL to download from
        output_path: Path to save the file
        description: Description for the progress bar
    """
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=description
    ) as pbar:
        urlretrieve(url, output_path, reporthook=pbar.update_to)


def extract_tar_files(directory: Path) -> None:
    """
    Recursively extract all .tar.gz and .tar files in a directory.

    Args:
        directory: Directory to search for tar files
    """
    tar_files = list(directory.rglob("*.tar.gz")) + list(directory.rglob("*.tar"))

    for tar_path in tar_files:
        log.info(f"  📦 Extracting {tar_path.name}...")
        try:
            with tarfile.open(tar_path, "r:*") as tar:
                tar.extractall(path=tar_path.parent)
            tar_path.unlink()  # Remove the tar file after extraction
        except Exception as e:
            log.warning(f"  ⚠️ Failed to extract {tar_path.name}: {e}")


def flatten_nested_directory(version_dir: Path, nested_name: str) -> None:
    """
    Move contents from a nested directory up to the version directory.

    Args:
        version_dir: The version directory (e.g., IRFs/prod3b-v2)
        nested_name: Name of the nested directory to flatten
    """
    nested_dir = version_dir / nested_name
    if nested_dir.exists() and nested_dir.is_dir():
        log.info(f"  📁 Flattening {nested_name}...")
        for item in nested_dir.iterdir():
            dest = version_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        nested_dir.rmdir()


def download_ctao_irfs(
    output_dir: str | Path = "./IRFs/CTAO",
    versions: list[str] | None = None,
    force: bool = False,
    keep_zip: bool = False,
) -> None:
    """
    Download CTA IRFs from Zenodo.

    Args:
        output_dir: Directory to save the IRFs (default: ./IRFs)
        versions: List of versions to download. If None, downloads all available versions.
                  Available versions: 'prod5-v0.1', 'prod3b-v2' [default]
        force: If True, re-download even if files already exist
        keep_zip: If True, keep the downloaded zip files after extraction
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if versions is None:
        versions = list(IRF_SOURCES.keys())

    # Validate versions
    invalid_versions = [v for v in versions if v not in IRF_SOURCES]
    if invalid_versions:
        raise ValueError(
            f"Invalid version(s): {invalid_versions}. "
            f"Available versions: {list(IRF_SOURCES.keys())}"
        )

    for version in versions:
        source = IRF_SOURCES[version]
        version_dir = output_dir / version
        zip_path = output_dir / source["filename"]

        # Check if already downloaded
        if version_dir.exists() and not force:
            log.info(
                f"✅ {version} already exists at {version_dir}, skipping (use --force to re-download)"
            )
            continue

        log.info(f"📥 Downloading {source['description']}...")

        # Download the zip file
        try:
            download_file(source["url"], zip_path, description=f"Downloading {version}")
        except Exception as e:
            log.error(f"❌ Failed to download {version}: {e}")
            continue

        # Create version directory
        version_dir.mkdir(parents=True, exist_ok=True)

        # Extract the zip file
        log.info(f"📦 Extracting {version}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(version_dir)

            # Handle nested directories (e.g., prod3b-v2 extracts to CTA-Prod3-Zenodo-main/)
            if version == "prod3b-v2":
                flatten_nested_directory(version_dir, "CTA-Prod3-Zenodo-main")

            # Extract any nested tar.gz files (both versions have these)
            extract_tar_files(version_dir)

            log.info(f"✅ Successfully installed {version} to {version_dir}")
        except zipfile.BadZipFile as e:
            log.error(f"❌ Failed to extract {version}: {e}")
            continue
        finally:
            # Clean up zip file unless requested to keep
            if not keep_zip and zip_path.exists():
                zip_path.unlink()

    log.info(f"🎉 IRF download complete! Files saved to {output_dir.resolve()}")


def verify_irfs(output_dir: str | Path = "./IRFs/CTAO") -> bool:
    """
    Verify that downloaded IRFs can be loaded properly using IRFHouse.

    Args:
        output_dir: Directory where IRFs are stored (default: ./IRFs/CTAO)

    Returns:
        True if all IRFs loaded successfully, False otherwise
    """
    from ..ctaoirf import IRFHouse

    output_dir = Path(output_dir)

    if not output_dir.exists():
        log.error(f"❌ IRF directory {output_dir} does not exist")
        return False

    log.info(f"🔍 Verifying IRFs in {output_dir.resolve()}...")

    try:
        irf_house = IRFHouse(base_directory=output_dir, check_irfs=False)
        all_found = irf_house.check_all_paths()
        if all_found:
            log.info("✅ IRF verification complete!")
        return all_found
    except Exception as e:
        log.error(f"❌ IRF verification failed: {e}")
        return False


def main() -> int:
    """CLI entry point for downloading IRFs."""
    parser = argparse.ArgumentParser(
        description="Download CTAO Instrument Response Functions (IRFs) from Zenodo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available IRF versions:
  prod5-v0.1   CTAO prod5 v0.1 IRFs (Alpha configuration) [default]
  prod3b-v2    CTAO prod3b v2 IRFs

Examples:
  # Download latest CTAO IRFs to ./IRFs/CTAO (default)
  sensipy-download-ctao-irfs

  # Download to a custom directory
  sensipy-download-ctao-irfs --output-dir /path/to/irfs/CTAO

  # Download only specific version(s)
  sensipy-download-ctao-irfs --versions prod5-v0.1 prod3b-v2

  # Force re-download even if files exist
  sensipy-download-ctao-irfs --force

  # Download and verify IRFs load correctly
  sensipy-download-ctao-irfs --verify

  # Verify existing IRFs without downloading
  sensipy-download-ctao-irfs --verify-only
        """,
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./IRFs/CTAO",
        help="Output directory for IRF files (default: ./IRFs)",
    )

    parser.add_argument(
        "-v",
        "--versions",
        nargs="+",
        choices=list(IRF_SOURCES.keys()),
        default=["prod5-v0.1"],
        help="Specific IRF versions to download (default: all)",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-download even if files already exist",
    )

    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip files after extraction",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_versions",
        help="List available IRF versions and exit",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify that IRFs can be loaded after download",
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing IRFs without downloading",
    )

    args = parser.parse_args()

    if args.list_versions:
        print("Available IRF versions:")
        for version, info in IRF_SOURCES.items():
            print(f"  {version:12s}  {info['description']}")
        return 0

    if args.verify_only:
        success = verify_irfs(output_dir=args.output_dir)
        return 0 if success else 1

    try:
        download_ctao_irfs(
            output_dir=args.output_dir,
            versions=args.versions,
            force=args.force,
            keep_zip=args.keep_zip,
        )

        if args.verify:
            success = verify_irfs(output_dir=args.output_dir)
            return 0 if success else 1

        return 0
    except Exception as e:
        log.error(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
