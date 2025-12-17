"""Generate plots for documentation.

This script generates all plots used in the documentation.
Add new plot generation functions here as needed.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from pathlib import Path
from typing import Optional

from sensipy.source import Source
from sensipy.sensitivity import Sensitivity
from sensipy.detectability import LookupData, create_heatmap_grid
from sensipy.data.create_mock_lookup import create_mock_lookup_table
from sensipy.util import get_data_path


class PlotGenerator:
    """Main class for generating documentation plots."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the plot generator.

        Args:
            output_dir: Directory to save plots. Defaults to docs/public.
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / "public"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_spectral_plots(self):
        """Generate plots for spectral models documentation."""
        # Load mock data
        mock_data_path = get_data_path("mock_data/GRB_42_mock.csv")
        source = Source(
            filepath=str(mock_data_path),
            min_energy=30 * u.GeV,
            max_energy=10 * u.TeV,
        )
        source.set_spectral_grid()

        # Create side-by-side figure for spectrum and lightcurve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Spectrum at a specific time
        time = 100 * u.s
        spectrum = source.get_spectrum(time=time)

        ax1.loglog(source.energy.to("TeV").value, spectrum.value, linewidth=2)
        ax1.set_xlabel("Energy [TeV]", fontsize=12)
        ax1.set_ylabel("dN/dE [cm⁻² s⁻¹ GeV⁻¹]", fontsize=12)
        ax1.set_title(f"Spectrum at t = {time.value:.0f} s", fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Lightcurve at a specific energy
        energy = 1 * u.TeV
        lightcurve = source.get_flux(energy=energy)

        ax2.loglog(source.time.value, lightcurve.value, linewidth=2)
        ax2.set_xlabel("Time [s]", fontsize=12)
        ax2.set_ylabel("dN/dE [cm⁻² s⁻¹ GeV⁻¹]", fontsize=12)
        ax2.set_title(f"Lightcurve at E = {energy.value:.1f} TeV", fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "spectrum_and_lightcurve.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # Plot 3: Spectral pattern visualization with larger fonts
        # Recreate the plot manually to have full control over font sizes
        source.set_spectral_grid()
        
        resolution = 100
        cutoff_flux = 1e-20 * u.Unit("1 / (cm2 s GeV)")
        
        loge = np.log10(source.energy.value)
        logt = np.log10(source.time.value)
        
        x = np.linspace(loge.min(), loge.max(), resolution + 1)[::-1]
        y = np.linspace(logt.min(), logt.max(), resolution + 1)
        
        points = []
        for e in x:
            for t in y:
                points.append([e, t])
        
        spectrum = source.SpectralGrid(points)
        cutoff_flux = cutoff_flux.to("1 / (cm2 s GeV)")
        spectrum[spectrum < cutoff_flux.value] = cutoff_flux.value
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            np.log10(spectrum).reshape(resolution + 1, resolution + 1),
            extent=(logt.min(), logt.max(), loge.min(), loge.max()),
            cmap="viridis",
            aspect="auto",
        )
        ax.set_xlabel("Log(t [s])", fontsize=16)
        ax.set_ylabel("Log(E [GeV])", fontsize=16)
        ax.tick_params(labelsize=14)
        
        # Add colorbar with larger font
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Log dN/dE [cm-2 s-1 GeV-1]", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "spectral_pattern.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        print("Generated spectral model plots:")
        print("  - spectrum_and_lightcurve.png")
        print("  - spectral_pattern.png")

    def _try_load_irf(self):
        """Try to load an IRF, return None if not available."""
        try:
            from sensipy.ctaoirf import IRFHouse
            
            # Try common IRF locations
            irf_paths = [
                Path("./IRFs/CTAO"),
                Path(__file__).parent.parent / "IRFs" / "CTAO",
                Path.home() / "IRFs" / "CTAO",
            ]
            
            for irf_path in irf_paths:
                if irf_path.exists():
                    try:
                        house = IRFHouse(base_directory=str(irf_path), check_irfs=False)
                        irf = house.get_irf(
                            site="south",
                            configuration="alpha",
                            zenith=20,
                            duration=1800,
                            azimuth="average",
                            version="prod5-v0.1",
                        )
                        return irf
                    except Exception:
                        continue
            
            return None
        except ImportError:
            return None

    def generate_sensitivity_plots(self):
        """Generate plots for sensitivity calculations documentation."""
        # Try to load IRF
        irf = self._try_load_irf()
        if irf is None:
            print("Skipping sensitivity plots: IRFs not found.")
            print("  To generate sensitivity plots, download IRFs using:")
            print("  uv run sensipy-download-ctao-irfs")
            return

        # Load mock source data
        mock_data_path = get_data_path("mock_data/GRB_42_mock.csv")
        source = Source(
            filepath=str(mock_data_path),
            min_energy=20 * u.GeV,
            max_energy=10 * u.TeV,
            ebl="franceschini",
        )
        source.set_spectral_grid()

        # Example 1: Basic Example
        sens_basic = Sensitivity(
            irf=irf,
            observatory="ctao_south",
            min_energy=20 * u.GeV,
            max_energy=10 * u.TeV,
            radius=3.0 * u.deg,
        )
        sens_basic.get_sensitivity_curve(source=source)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot sensitivity curve
        ax1.loglog(
            sens_basic.times.value,
            sens_basic.sensitivity_curve.value,
            linewidth=2,
            label="Differential sensitivity",
        )
        ax1.set_xlabel("Time [s]", fontsize=12)
        ax1.set_ylabel("Sensitivity [GeV cm⁻² s⁻¹]", fontsize=12)
        ax1.set_title("Basic Example: Sensitivity Curve", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)

        # Plot photon flux curve
        ax2.loglog(
            sens_basic.times.value,
            sens_basic.photon_flux_curve.value,
            linewidth=2,
            color="orange",
            label="Photon flux sensitivity",
        )
        ax2.set_xlabel("Time [s]", fontsize=12)
        ax2.set_ylabel("Photon Flux Sensitivity [cm⁻² s⁻¹]", fontsize=12)
        ax2.set_title("Basic Example: Photon Flux Curve", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "sensitivity_basic_example.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # Example 2: Custom Time Range
        sens_custom = Sensitivity(
            irf=irf,
            observatory="ctao_south",
            min_energy=20 * u.GeV,
            max_energy=10 * u.TeV,
            radius=3.0 * u.deg,
            min_time=10 * u.s,
            max_time=36000 * u.s,
            n_sensitivity_points=20,
        )
        sens_custom.get_sensitivity_curve(source=source)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot sensitivity curve
        ax1.loglog(
            sens_custom.times.value,
            sens_custom.sensitivity_curve.value,
            linewidth=2,
            marker="o",
            markersize=4,
            label="Differential sensitivity",
        )
        ax1.set_xlabel("Time [s]", fontsize=12)
        ax1.set_ylabel("Sensitivity [GeV cm⁻² s⁻¹]", fontsize=12)
        ax1.set_title("Custom Time Range: Sensitivity Curve", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)

        # Plot photon flux curve
        ax2.loglog(
            sens_custom.times.value,
            sens_custom.photon_flux_curve.value,
            linewidth=2,
            marker="o",
            markersize=4,
            color="orange",
            label="Photon flux sensitivity",
        )
        ax2.set_xlabel("Time [s]", fontsize=12)
        ax2.set_ylabel("Photon Flux Sensitivity [cm⁻² s⁻¹]", fontsize=12)
        ax2.set_title("Custom Time Range: Photon Flux Curve", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "sensitivity_custom_time.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # Example 3: Pre-computed Curves
        # Use the computed curves from the basic example as "pre-computed"
        times_precomputed = sens_basic.times
        sens_values_precomputed = sens_basic.sensitivity_curve
        photon_flux_values_precomputed = sens_basic.photon_flux_curve

        sens_precomputed = Sensitivity(
            irf=irf,
            observatory="ctao_south",
            min_energy=20 * u.GeV,
            max_energy=10 * u.TeV,
            radius=3.0 * u.deg,
            sensitivity_curve=sens_values_precomputed,
            photon_flux_curve=photon_flux_values_precomputed,
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot sensitivity curve
        ax1.loglog(
            sens_precomputed.times.value,
            sens_precomputed.sensitivity_curve.value,
            linewidth=2,
            linestyle="--",
            label="Pre-computed sensitivity",
        )
        ax1.set_xlabel("Time [s]", fontsize=12)
        ax1.set_ylabel("Sensitivity [GeV cm⁻² s⁻¹]", fontsize=12)
        ax1.set_title("Pre-computed Curves: Sensitivity Curve", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)

        # Plot photon flux curve
        ax2.loglog(
            sens_precomputed.times.value,
            sens_precomputed.photon_flux_curve.value,
            linewidth=2,
            linestyle="--",
            color="orange",
            label="Pre-computed photon flux",
        )
        ax2.set_xlabel("Time [s]", fontsize=12)
        ax2.set_ylabel("Photon Flux Sensitivity [cm⁻² s⁻¹]", fontsize=12)
        ax2.set_title("Pre-computed Curves: Photon Flux Curve", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "sensitivity_precomputed.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        print("Generated sensitivity plots:")
        print("  - sensitivity_basic_example.png")
        print("  - sensitivity_custom_time.png")
        print("  - sensitivity_precomputed.png")

    def generate_direct_sensitivity_plots(self):
        """Generate plots for direct sensitivity calculation examples."""
        # Try to load IRF
        irf = self._try_load_irf()
        if irf is None:
            print("Skipping direct sensitivity plots: IRFs not found.")
            return

        from gammapy.modeling.models import PowerLawSpectralModel

        # Create sensitivity calculator
        sens = Sensitivity(
            irf=irf,
            observatory="ctao_south",
            min_energy=20 * u.GeV,
            max_energy=10 * u.TeV,
            radius=3.0 * u.deg,
        )

        # Define a spectral model
        spectral_model = PowerLawSpectralModel(
            index=2.0,
            amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
            reference=1 * u.TeV,
        )

        # Example 1: Integral Sensitivity at 3 different times
        times = [100 * u.s, 1000 * u.s, 10000 * u.s]
        photon_fluxes = []
        energy_fluxes = []
        
        for time in times:
            result_integral = sens.get_sensitivity_from_model(
                t=time,
                spectral_model=spectral_model,
                sensitivity_type="integral",
            )
            photon_fluxes.append(result_integral["photon_flux"].to("cm-2 s-1").value)
            energy_fluxes.append(result_integral["energy_flux"].to("GeV cm-2 s-1").value)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot photon flux sensitivity vs time
        ax1.loglog(
            [t.value for t in times],
            photon_fluxes,
            linewidth=2,
            marker="o",
            markersize=8,
            label="Photon flux sensitivity",
            color="orange",
        )
        ax1.set_xlabel("Time [s]", fontsize=12)
        ax1.set_ylabel("Photon Flux Sensitivity [cm⁻² s⁻¹]", fontsize=12)
        ax1.set_title("Integral Sensitivity: Photon Flux", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)

        # Plot energy flux sensitivity vs time
        ax2.loglog(
            [t.value for t in times],
            energy_fluxes,
            linewidth=2,
            marker="s",
            markersize=8,
            label="Energy flux sensitivity",
            color="blue",
        )
        ax2.set_xlabel("Time [s]", fontsize=12)
        ax2.set_ylabel("Energy Flux Sensitivity [erg cm⁻² s⁻¹]", fontsize=12)
        ax2.set_title("Integral Sensitivity: Energy Flux", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "sensitivity_integral_example.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # Example 2: Differential Sensitivity at 3 different times
        # Get differential sensitivity at one energy bin (middle energy) for different times
        times = [100 * u.s, 1000 * u.s, 10000 * u.s]
        sensitivity_values = []
        energy_ref = 50 * u.GeV
        
        for time in times:
            result_differential = sens.get_sensitivity_from_model(
                t=time,
                spectral_model=spectral_model,
                sensitivity_type="differential",
            )
            
            # Get the middle energy bin
            if energy_ref is None:
                mid_idx = len(result_differential) // 2
                e_ref_col = result_differential["e_ref"]
                energy_ref_quantity = e_ref_col[mid_idx] * e_ref_col.unit
                energy_ref = energy_ref_quantity.to("TeV").value * u.TeV
            
            # Find the bin closest to energy_ref
            e_ref_col = result_differential["e_ref"]
            # Convert each value individually to avoid unit conversion issues
            e_ref_values = np.array([(val * e_ref_col.unit).to("TeV").value for val in e_ref_col])
            mid_idx = np.argmin(np.abs(e_ref_values - energy_ref.value))
            
            # E² dN/dE
            sensitivity_values.append((result_differential["e2dnde"][mid_idx] * u.Unit("erg cm-2 s-1")).to("GeV cm-2 s-1").value)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot differential sensitivity vs time
        ax.loglog(
            [t.value for t in times],
            sensitivity_values,
            linewidth=2,
            marker="o",
            markersize=8,
            label=f"Differential sensitivity at E = {energy_ref.to('GeV'):.2f}",
            color="red",
        )
        ax.set_xlabel("Time [s]", fontsize=12)
        ax.set_ylabel("E² dN/dE [GeV cm⁻² s⁻¹]", fontsize=12)
        ax.set_title("Differential Flux Sensitivity", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "sensitivity_differential_example.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        print("Generated direct sensitivity plots:")
        print("  - sensitivity_integral_example.png")
        print("  - sensitivity_differential_example.png")

    def generate_detectability_plots(self):
        """Generate plots for detectability analysis documentation."""
        import tempfile
        
        # Create temporary directory for mock data
        with tempfile.TemporaryDirectory() as tmpdir:
            
            # Create mock lookup table
            lookup_path = create_mock_lookup_table(
                n_events=100,
                output_filename="detectability_lookup.parquet",
                output_dir=tmpdir,
                use_random_metadata=True,
                seed=42,
            )
            
            # Example 1: Basic detectability heatmap
            data = LookupData(lookup_path)
            data.set_observation_times(np.logspace(1, np.log10(3600 + 0.1), 10, dtype=int))
            
            fig, ax = plt.subplots(figsize=(10, 8))
            data.plot(
                ax=ax,
                title="Source Detectability",
                return_ax=True,
            )
            plt.tight_layout()
            plt.savefig(
                self.output_dir / "detectability_basic.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
            
            # Example 2: Custom filtering
            data2 = LookupData(lookup_path)
            data2.set_filters(
                ("irf_site", "==", "south"),
                ("irf_zenith", "<=", 40),
            )
            data2.set_observation_times([round(i) for i in np.logspace(1, np.log10(7200), 10)])
            
            fig, ax = plt.subplots(figsize=(10, 8))
            data2.plot(
                ax=ax,
                title="Detectability: South Site, Low Zenith, Events 1-3",
                as_percent=True,
                color_scheme="viridis",
                return_ax=True,
            )
            plt.tight_layout()
            plt.savefig(
                self.output_dir / "detectability_filtered.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
            
            # Example 3: Grid of heatmaps
            observation_times = np.logspace(1, np.log10(3600), 10, dtype=int)
            
            data_north_z20 = LookupData(lookup_path)
            data_north_z20.set_filters(("irf_site", "==", "north"), ("irf_zenith", "==", 20))
            data_north_z20.set_observation_times(observation_times)
            
            data_north_z40 = LookupData(lookup_path)
            data_north_z40.set_filters(("irf_site", "==", "north"), ("irf_zenith", "==", 40))
            data_north_z40.set_observation_times(observation_times)
            
            data_south_z20 = LookupData(lookup_path)
            data_south_z20.set_filters(("irf_site", "==", "south"), ("irf_zenith", "==", 20))
            data_south_z20.set_observation_times(observation_times)
            
            data_south_z40 = LookupData(lookup_path)
            data_south_z40.set_filters(("irf_site", "==", "south"), ("irf_zenith", "==", 40))
            data_south_z40.set_observation_times(observation_times)
            
            fig, axes = create_heatmap_grid(
                [data_north_z20, data_north_z40, data_south_z20, data_south_z40],
                grid_size=(2, 2),
                title="Detectability Comparison: Site and Zenith Configurations",
                subtitles=[
                    "North, z20",
                    "North, z40",
                    "South, z20",
                    "South, z40",
                ],
                cmap="mako",
                square=True,
            )
            plt.savefig(
                self.output_dir / "detectability_grid.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
        
        print("Generated detectability plots:")
        print("  - detectability_basic.png")
        print("  - detectability_filtered.png")
        print("  - detectability_grid.png")

    def generate_spectrum_export_plots(self):
        """Generate plots for spectrum export examples with/without EBL."""
        # Load mock data
        mock_data_path = get_data_path("mock_data/GRB_42_mock.csv")
        
        # Create source with EBL model
        source_with_ebl = Source(
            filepath=str(mock_data_path),
            min_energy=20 * u.GeV,
            max_energy=10 * u.TeV,
            ebl="franceschini",
            z=1.0,
        )
        source_with_ebl.set_spectral_grid()
        source_with_ebl.fit_spectral_indices()
        
        # Create source without EBL for comparison
        source_no_ebl = Source(
            filepath=str(mock_data_path),
            min_energy=20 * u.GeV,
            max_energy=10 * u.TeV,
        )
        source_no_ebl.set_spectral_grid()
        source_no_ebl.fit_spectral_indices()
        
        time = 100 * u.s
        
        # Plot 1: Power law spectrum with/without EBL
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Without EBL
        powerlaw_no_ebl = source_no_ebl.get_powerlaw_spectrum(time, use_ebl=False)
        energy_plot = np.logspace(
            np.log10(source_no_ebl.min_energy.value),
            np.log10(source_no_ebl.max_energy.value),
            100
        ) * u.GeV
        flux_no_ebl = powerlaw_no_ebl(energy_plot)
        
        ax1.loglog(energy_plot.to("TeV").value, flux_no_ebl.value, linewidth=2, label="Without EBL")
        ax1.set_xlabel("Energy [TeV]", fontsize=12)
        ax1.set_ylabel("dN/dE [cm⁻² s⁻¹ GeV⁻¹]", fontsize=12)
        ax1.set_title("Power Law Spectrum (No EBL)", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        # With EBL
        powerlaw_with_ebl = source_with_ebl.get_powerlaw_spectrum(time, use_ebl=True)
        flux_with_ebl = powerlaw_with_ebl(energy_plot)
        
        ax2.loglog(energy_plot.to("TeV").value, flux_with_ebl.value, linewidth=2, label="With EBL", color="orange")
        ax2.set_xlabel("Energy [TeV]", fontsize=12)
        ax2.set_ylabel("dN/dE [cm⁻² s⁻¹ GeV⁻¹]", fontsize=12)
        ax2.set_title("Power Law Spectrum (With EBL)", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "powerlaw_spectrum_ebl_comparison.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        
        # Plot 2: Comparison plot showing EBL effect
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.loglog(
            energy_plot.to("TeV").value,
            flux_no_ebl.value,
            linewidth=2,
            label="Without EBL",
            linestyle="-",
        )
        ax.loglog(
            energy_plot.to("TeV").value,
            flux_with_ebl.value,
            linewidth=2,
            label="With EBL (franceschini)",
            linestyle="--",
            color="orange",
        )
        ax.set_xlabel("Energy [TeV]", fontsize=12)
        ax.set_ylabel("dN/dE [cm⁻² s⁻¹ GeV⁻¹]", fontsize=12)
        ax.set_title("Power Law Spectrum: EBL Absorption Effect", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "powerlaw_spectrum_ebl_effect.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        
        # Plot 3: Template spectrum with/without EBL
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Without EBL
        template_no_ebl = source_no_ebl.get_template_spectrum(time, use_ebl=False)
        flux_template_no_ebl = template_no_ebl(energy_plot)
        
        ax1.loglog(energy_plot.to("TeV").value, flux_template_no_ebl.value, linewidth=2, label="Without EBL")
        ax1.set_xlabel("Energy [TeV]", fontsize=12)
        ax1.set_ylabel("dN/dE [cm⁻² s⁻¹ GeV⁻¹]", fontsize=12)
        ax1.set_title("Template Spectrum (No EBL)", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        # With EBL
        template_with_ebl = source_with_ebl.get_template_spectrum(time, use_ebl=True)
        flux_template_with_ebl = template_with_ebl(energy_plot)
        
        ax2.loglog(energy_plot.to("TeV").value, flux_template_with_ebl.value, linewidth=2, label="With EBL", color="orange")
        ax2.set_xlabel("Energy [TeV]", fontsize=12)
        ax2.set_ylabel("dN/dE [cm⁻² s⁻¹ GeV⁻¹]", fontsize=12)
        ax2.set_title("Template Spectrum (With EBL)", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "template_spectrum_ebl_comparison.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        
        # Plot 4: Template comparison showing EBL effect
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.loglog(
            energy_plot.to("TeV").value,
            flux_template_no_ebl.value,
            linewidth=2,
            label="Without EBL",
            linestyle="-",
        )
        ax.loglog(
            energy_plot.to("TeV").value,
            flux_template_with_ebl.value,
            linewidth=2,
            label="With EBL (franceschini)",
            linestyle="--",
            color="orange",
        )
        ax.set_xlabel("Energy [TeV]", fontsize=12)
        ax.set_ylabel("dN/dE [cm⁻² s⁻¹ GeV⁻¹]", fontsize=12)
        ax.set_title("Template Spectrum: EBL Absorption Effect", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "template_spectrum_ebl_effect.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        
        print("Generated spectrum export plots:")
        print("  - powerlaw_spectrum_ebl_comparison.png")
        print("  - powerlaw_spectrum_ebl_effect.png")
        print("  - template_spectrum_ebl_comparison.png")
        print("  - template_spectrum_ebl_effect.png")

    def generate_all(self):
        """Generate all documentation plots."""
        print("Generating documentation plots...")
        print(f"Output directory: {self.output_dir}\n")

        self.generate_spectral_plots()
        self.generate_sensitivity_plots()
        self.generate_direct_sensitivity_plots()
        self.generate_detectability_plots()
        self.generate_spectrum_export_plots()

        print(f"\nAll plots saved to {self.output_dir}")


def main():
    """Main entry point for plot generation."""
    
    parser = argparse.ArgumentParser(
        description="Generate plots for documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--plot-type",
        nargs="?",
        default="all",
        choices=["all", "spectral", "sensitivity", "direct-sensitivity", "detectability", "spectrum-export"],
        help="Type of plot to generate (default: all)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: docs/public)",
    )
    
    args = parser.parse_args()
    
    generator = PlotGenerator(output_dir=args.output_dir)
    
    if args.plot_type == "all":
        generator.generate_all()
    elif args.plot_type == "spectral":
        generator.generate_spectral_plots()
    elif args.plot_type == "sensitivity":
        generator.generate_sensitivity_plots()
    elif args.plot_type == "direct-sensitivity":
        generator.generate_direct_sensitivity_plots()
    elif args.plot_type == "detectability":
        generator.generate_detectability_plots()
    elif args.plot_type == "spectrum-export":
        generator.generate_spectrum_export_plots()


if __name__ == "__main__":
    main()
