---
title: 'Sensipy: simulate gamma-ray observations of transient astrophysical sources'
tags:
  - Python
  - astronomy
  - dynamics
  - galactic dynamics
  - milky way
authors:
  - name: Jarred G. Green
    orcid: 0000-0002-1130-6692
    # corresponding: true
    affiliation: "1, 2"
  - name: Barbara Patricelli
    orcid: 0000-0001-6709-0969
    affiliation: "3, 4"
  - name: Antonio Stamerra
    orcid: 0000-0002-9430-5264
    affiliation: "4, 5"
  - name: Monica Seglar-Arroyo
    orcid: 0000-0001-8654-409X
    affiliation: 6
affiliations:
 - name: Max-Planck-Institut für Physik, Boltzmannstr. 8, 85748 Garching, Germany
   index: 1
   ror: 0079jjr10
 - name: Technische Universität München, 85748 Garching, Germany
   index: 2
   ror: 02kkvpp62
 - name: University of Pisa, Largo B. Pontecorvo 3, 56127 Pisa, Italy
   index: 3
   ror: 03ad39j10
 - name: INAF - Osservatorio Astronomico di Roma, Via di Frascati 33, 00078, Monteporzio Catone, Italy
   index: 4
   ror: 02hnp4676
 - name: Cherenkov Telescope Array Observatory, Via Gobetti, 40129 Bologna, Italy
   index: 5
   ror: 05fb7v315
 - name: Institut de Física d’Altes Energies (IFAE), The Barcelona Institute of Science and Technology, Campus UAB, 08193 Bellaterra (Barcelona), Spain
   index: 6
   ror: 01sdrjx85
date: 02 December 2025
bibliography: paper.bib

header-includes:
  - \usepackage{xcolor}
---
# Summary

We present `sensipy`, an open-source Python toolkit for simulating observations of transient astrophysical sources, particularly in the high-energy (HE, keV-GeV) and very-high-energy (VHE, GeV-TeV) gamma-ray ranges.
The most explosive events in our universe are often short-lived, emitting the bulk of their energy in a relatively narrow time window.
Due to often rapidly fading emission profiles, understanding how and when to observe these sources is crucial both to test theoretical predictions and efficiently optimize available telescope time.

The information extracted from the tools included in `sensipy` can be used to help astronomers investigate the detectability of sources considering various theoretical assumptions about their emission processes and mechanisms. This information can further help to justify the feasibility of proposed observations, estimate detection rates (events/year) for various classes of sources, and provide scheduling insight in realtime during gamma-ray and multi-messenger observational campaigns.

# Statement of need

The need for a toolkit like `sensipy` became clear while attempting to estimate the detectability of VHE counterparts to gravitational wave (GW) signals from binary neutron star mergers (BNS) with the upcoming Cherenkov Telescope Array Observatory (CTAO). During development, it became apparent that the included tools could be applied not only to VHE counterparts of BNS mergers, but also to other transient sources like gamma-ray bursts (GRBs), active galactic nuclei flares, novae, supernovae, and more.

Between GW, neutrino, optical, and space-based gamma-ray experiments, thousands of low-latency alerts are sent out to the greater community each year [@abac_gwtc-40_2025; @von_kienlin_fourth_2020; @abbasi_icecat-1_2023]. However, very few of these events actually result in detections in the VHE gamma-ray regime. This is due to many factors, including the rapid decay of fluxes, delay in telescope repointing, uncertainty on the sky localization of the source, and observatory duty cycles. In the face of these challenges, `sensipy` aims to help answer the following questions for gamma-ray astronomers interested in optimizing their follow-up campaigns:

- Given a theoretical emission model, what are the detection possibilities with a given instrument?
- How much observing time is needed to detect a source given a delay in starting observations?
- At what significance level is a source detectable given a certain observation time?
- How long does a source remain detectable after the onset of emission?
- How can intrinsic source properties (e.g. distance, flux) and observing conditions (e.g. latency, telescope pointing) affect detectability?
- How can these results for catalogs of simulated events inform follow-up strategies in realtime?

# State of the field

Currently, `gammapy` is the largest player in the field of gamma-ray astrophysics, providing a substantial set of tools designed for the high-level analysis of data from many major future and currently-operating observatories [@donath_gammapy_2023]. We make use of an applicable set of `gammapy` and `astropy` primitives under the hood, such that experienced users do not have to learn new APIs when beginning work with `sensipy`. Given that, the decision to develop `sensipy` as a standalone package is twofold. Firstly, our goal is not to participate directly in the analysis of telescope data, but to provide a set of simulation tools which can help to plan observation campaigns for these observatories. Secondly, as we provide methods which can be integrated directly into telescope control systems, it is important to keep `sensipy` focused, lightweight, and modular.

# Software design

The two main inputs to any `sensipy` pipeline are:

- an instrument response function (IRF), which describes how a telescope performs under specific observing conditions
- intrinsic time-dependent emission spectra for a source, which can be provided in either a FITS or CSV format

Given these inputs, `sensipy` builds upon primitives provided by `astropy` and `gammapy` to provide the main functionalities outlined below [@collaboration_astropy_2022; @donath_gammapy_2023]. In addition, mock datasets are provided with working code examples, and batteries are included for easy access to publicly-available IRFs, e.g. [@observatory_ctao_2021].

## Design philosophy

Most users already come to `sensipy` with their own theoretical models at hand, so providing clear APIs with simple and speedy onboarding is front-of-mind during development. The documentation is focused on small and self-contained quick-start examples along with every feature, so that users can directly begin with code-blocks relevant to their problem.

In addition, because this package can also be built directly into telescope control software in order to provide realtime insights into observational campaigns, we chose to organize `sensipy` into a number of smaller modules that can be individually imported, with each module in turn only calling upon the bare set of primitives needed. Each of the modules and their functionalities are briefly described below.

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

### Follow-ups of poorly localized events

In addition, the functions included in `sensipy.followup` may be used in tandem with scheduling software like `tilepy` for the realtime follow-up of poorly-localized events, including GRB, GW, and neutrino alerts [@seglar-arroyo_cross_2024]. These scheduling tools create an optimized list of telescope pointings on the sky, while `sensipy` is used simultaneously to optimize the exposure time needed at each new pointing.

![A follow-up coverage map for an example GW event (S250704ab). The ordering of pointings is calculated by `tilepy` and the optimal observing time at each pointing by `sensipy`.](figures/figure3.png)

# Research impact statement

The `sensipy` package has been under development for the past four years, and in 2025 the community began to grow rapidly as the package reached maturity. The software has been adapted by members of the CTAO Collaboration for a number of applications, including the official evaluation of the prospects of GW follow-up campaigns with the observatory. Numerous talks at major conferences in the field have included contributions created with `sensipy` as part of the main results [@patricelli_searching_2022; @green_chasing_2024; @seglar-arroyo_icrc_2025; @seglar-arroyo_tevpa_2025]. In addition, independent researchers have also made use of the package in order to simulate observational campaigns with other observatories, including the Astrofisica con Specchi a Tecnologia Replicante Italiana (ASTRI) Mini-Array and the Large Array of imaging atmospheric Cherenkov Telescope (LACT) [@macera_detection_tevpa_2025]. Finally, workflows based upon the aforementioned `sensipy.followup` module, together with the `tilepy` package, are being internally tested within the Major Atmospheric Gamma Imaging Cherenkov (MAGIC) Telescopes and Large-Size Telescope (LST) collaborations for GW and GRB alert follow-ups.

# AI usage disclosure

AI tools (GitHub Copilot, Grammarly, Cursor Bugbot) were used to proofread documentation and pull requests as well as scaffold tests. Instances where functions or classes were primarily generated with assistance from GitHub Copilot (auto model) are explicitly marked in the Python docstrings. All AI-assisted suggestions were carefully reviewed and approved by the authors of this manuscript. AI tools were not used in the writing of this manuscript in any capacity.

# References
