---
title: "Sensipy: simulate gamma-ray observations of transient astrophysical sources"
author: []
date: "6 February 2026"
header-includes:
  - |
    \AtBeginDocument{\author{\parbox{\textwidth}{\centering
    Jarred G. Green\textsuperscript{1,2}, Barbara Patricelli\textsuperscript{3,4}, Antonio Stamerra\textsuperscript{4,5}, Monica Seglar-Arroyo\textsuperscript{6}\\[0.5em]
    \footnotesize\textsuperscript{1}~Max-Planck-Institut für Physik, Garching, Germany\\
    \footnotesize\textsuperscript{2}~Technische Universität München, Germany\\
    \footnotesize\textsuperscript{3}~University of Pisa, Italy\\
    \footnotesize\textsuperscript{4}~INAF - Osservatorio Astronomico di Roma, Italy\\
    \footnotesize\textsuperscript{5}~Cherenkov Telescope Array Observatory, Bologna, Italy\\
    \footnotesize\textsuperscript{6}~Institut de Fisica d'Altes Energies (IFAE), Bellaterra (Barcelona), Spain}}}
bibliography: paper.bib
---
# Summary

We present `sensipy`, an open-source Python toolkit for simulating observations of transient astrophysical sources, particularly in the high-energy (HE, keV-GeV) and very-high-energy (VHE, GeV-TeV) gamma-ray ranges.
The most explosive events in our universe are often short-lived, emitting the bulk of their energy in a relatively narrow time window.
Due to often rapidly fading emission profiles, understanding how and when to observe these sources is crucial both to test theoretical predictions and efficiently optimize available telescope time.

The information extracted from the tools included in `sensipy` can be used to help astronomers investigate the detectability of sources considering various theoretical assumptions about their emission processes and mechanisms. This information can further help to justify the feasibility of proposed observations, estimate detection rates (events/year) for various classes of sources, and provide scheduling insight in realtime during gamma-ray and multi-messenger observational campaigns.

# Statement of need

The need for a toolkit like `sensipy` became clear while attempting to estimate the detectability of VHE counterparts to gravitational wave (GW) signals from binary neutron star mergers (BNS) with the upcoming Cherenkov Telescope Array Observatory (CTAO) [@patricelli_searching_2022; @green_chasing_2024]. During development, it became apparent that the included tools could be applied not only to VHE counterparts of BNS mergers, but also to other transient sources like gamma-ray bursts (GRBs), active galactic nuclei flares, novae, supernovae, and more.

Between GW, neutrino, optical, and space-based gamma-ray experiments, thousands of low-latency alerts are sent out to the greater community each year [@abac_gwtc-40_2025; @von_kienlin_fourth_2020; @abbasi_icecat-1_2023]. However, very few of these events actually result in detections in the VHE gamma-ray regime. This is due to many factors, including the rapid decay of fluxes, delay in telescope repointing, uncertainty on the sky localization of the source, and observatory duty cycles. In the face of these challenges, `sensipy` aims to help answer the following questions for gamma-ray astronomers interested in optimizing their follow-up campaigns:

- Given a theoretical emission model, what are the detection possibilities with a given instrument?
- How much observing time is needed to detect a source given a delay in starting observations?
- At what significance level is a source detectable given a certain observation time?
- How long does a source remain detectable after the onset of emission?
- How can intrinsic source properties (e.g. distance, flux) and observing conditions (e.g. latency, telescope pointing) affect detectability?
- How can these results for catalogs of simulated events inform follow-up strategies in realtime?

# Functionality

The two main inputs to any `sensipy` pipeline are:

- an instrument response function (IRF), which describes how a telescope performs under specific observing conditions.
- intrinsic time-dependent emission spectra for a source, which can be provided in either a FITS or CSV format.

Given these inputs, `sensipy` builds upon primitives provided by `astropy` and `gammapy` to provide the following main functionalities [@collaboration_astropy_2022; @donath_gammapy_2023]. In addition, mock datasets are provided with working code examples, and batteries are included for easy access to publicly-available IRFs, e.g. [@observatory_ctao_2021].  

## Sensitivity Curve Calculation with `sensipy.sensitivity`

Sensitivity curves represent the minimum flux needed to detect a source at a given significance (usually $5 \sigma$) given an exposure time $t_{exp}$. Such curves are often used to compare the performances of different instruments, and `sensipy` can produce them in two flavors: integral and differential sensitivity curves. The sensitivity itself depends heavily on the rapidly-changing spectral shape of an event, which itself may be highly affected by distance due to the extragalactic background light (EBL). All of these factors are automatically taken into account.

![A 2-D representation of the intrinsic time-dependent VHE gamma-ray flux for an example transient event (left) and the corresponding integral flux sensitivity of CTAO for a source with this spectrum (right). The sensitivity curves are calculated for different latencies ($t_L$) after the event onset.](figures/figure1.png)

## Simulating Observations with `sensipy.source`

This class addresses the fundamental question: if we begin observations with a latency of $t_L = X~\text{min}$ after an alert, what observation time is required in order to achieve a detection? In addition, the class can also determine the inverse: given an observation time, at what significance can a source be detected? Given that the user has already calculated the sensitivity curve for an event, `sensipy` can determine if the source is actually detectable, given $t_L$. When detectable, the exposure time necessary for detection is also calculated.

## Working with large catalogs with `sensipy.detectability`

`sensipy` can further estimate the overall detectability of entire classes of objects, given a catalog or survey of simulated events under various conditions. By performing and collating a large number of observation simulations for various events and latencies $t_L$, the toolkit can help produce visualizations which describe the optimal observing conditions for such events.

![Detectability heatmap produced with `sensipy`. Given a large catalog of transient events, this `sensipy` heatmap shows what fraction are potentially detectable given a specific observation time $t_{exp}$ and latency $t_L$ since the event onset.](figures/figure2.png)

## Realtime applications with `sensipy.followup`

Tables of observation times can also be used as lookup tables (LUTs) during telescope observations in order to plan observation campaigns. For example, the following workflow can be implemented within `sensipy`:

1. a catalog of simulated spectra is processed with the above pipeline considering various observation conditions, and a LUT is created
2. a transient alert arrives during normal telescope operation and telescopes begin observing the event position with a latency of $t_L$
3. the LUT is filtered and interpolated in realtime in order to quickly calculate an informed estimate on the exposure time needed for a detection

Such workflows based on `sensipy` modules are already being internally evaluated within the MAGIC, Large-Size Telescope (LST), and CTAO collaborations for follow-up of both GW and GRB alerts [e.g., @green_chasing_2024; @patricelli_searching_2022].

### Follow-ups of poorly localized events

In addition, the functions included in `sensipy.followup` may be used in tandem with scheduling software like `tilepy` for the realtime follow-up of poorly-localized events, including GRB, GW, and neutrino alerts [@seglar-arroyo_cross_2024]. These scheduling tools create an optimized list of telescope pointings on the sky, while `sensipy` is used simultaneously to optimize the exposure time needed at each new pointing.

![A follow-up coverage map for an example GW event (S250704ab). The ordering of pointings is calculated by `tilepy` and the optimal observing time at each pointing by `sensipy`.](figures/figure3.png)

# AI usage disclosure

AI tools (GitHub Copilot, Grammarly) were used to proofread documentation and pull requests as well as scaffold tests. Instances where functions or classes were primarily generated with assistance from GitHub Copilot (auto model) are explicitly marked in the Python docstrings. All AI-assisted suggestions were carefully reviewed and approved by the authors of this manuscript. AI tools were not used in the writing of this manuscript in any capacity.

# References
