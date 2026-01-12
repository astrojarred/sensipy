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
    affiliation: "3, 5"
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
 - name: INAF - Osservatorio Astronomico di Roma, Via di Frascati 33, 00078, Monteporzio Catone, Italy
   index: 3
   ror: 02hnp4676
 - name: University of Pisa, Largo B. Pontecorvo 3, 56127 Pisa, Italy
   index: 4
   ror: 03ad39j10
 - name: Cherenkov Telescope Array Observatory, Via Gobetti, 40129 Bologna, Italy
   index: 5
   ror: 05fb7v315
 - name: Institut de Fisica d’Altes Energies (IFAE), The Barcelona Institute of Science and Technology, Campus UAB, 08193 Bellaterra (Barcelona), Spain
   index: 6
   ror: 01sdrjx85
date: 02 December 2025
# bibliography: paper.bib
---
# Summary

We present `sensipy`, an open-source Python toolkit for simulating observations of transient astrophysical sources, particularly in the high-energy (HE, keV-GeV) and very-high-energy (VHE, GeV-TeV) gamma-ray sky.
The most explosive events in our universe are often short-lived, emitting the bulk of their energy in a relatively small time window.
Due to often rapidly fading emission profiles, understanding how and when to observe these sources is crucial to both test theoretical predictions and efficiently optimize the available telescope time.

The information extracted from the tools included in `sensipy` can be used to help astronomers investigate the detectability of sources considering different theoretical assumptions about their emission processes and mechanisms. This information can further help to justify the feasibility of proposed observations, estimate detection rates (events/year) for various classes of sources, and provide scheduling in realtime during gamma-ray and multi-messenger observational campaigns.

# Statement of need

The need for a toolkit like `sensipy` became clear when we were attempting to estimate the detectability of VHE counterparts to GW signals from binary neutron star mergers (BNS) with the upcoming Cherenkov Telescope Array Observatory (CTAO). While this toolklit began development with the aim to optimize a strategy for such joint detections with CTAO, the usefulness of the package became apparent and can be applied not only to VHE counterparts of BNS mergers, but also to other transient sources like GRBs, AGN flares, novae, supernovae, and more.

<!-- In its third observing run, the Advanced LIGO and Advanced Virgo observatories detected gravitational waves (GWs) from the binary neutron star (BNS) merger event GW170817.
1.7 seconds later, the Fermi and INTEGRAL observatories detected a short gamma-ray burst (GRB) in the same region of sky in the keV energy band, opening the gates to the field of multi-messenger astronomy.
In January 2019, the Fermi and Swift observatories detected a GRB and quickly alerted the community. Within one minute, the Major Atmospheric Gamma Imaging Cherenkov (MAGIC) telescopes pointed to the same position and provided the first evidence of TeV-band emission from a GRB. -->

Between GW, neutrino, optical, and orbiting gamma-ray experiments, thousands of low-latency alerts are sent out to the greater community each year. However, very few of these events actually result in detections in the VHE gamma-ray regime. This is due to many factors, including the rapid decay of fluxes, delay in telescope repointing, uncertainty on the sky localization of the source, and observatory duty cycles. In the face of these challenges, `sensipy` aims to help answer the following questions for gamma-ray astronomers interested in optimizing their follow-up campaigns:

- Given a theoretical emission model, what are the detection possibilities with a given instrument?
- How much observing time is needed to detect a source given you are delayed in starting observations?
- How much time after the onset of a given event does the source become undetectable?
- How can intrinsic source properties (eg distance, flux), and observing conditions (eg latency, telescope pointing) affect detectability?
- How can these results for catalogs of simulated events inform follow-up strategies in realtime?

# Functionality

The two main inputs to any `sensipy` pipeline are:

- an instrument response function (IRF), which describes how a telescope performs under specific observing conditions
- intrinsic time-dependent emission spectra for a source, which can be provided in either a FITS or CSV format.

Given these inputs, `sensipy` toolkit builds upon the primitives provided by `numpy`, `scipy`, `astropy`, and `gammapy` to provide the following main functionalities. In addition, mock datasets are provided with working code examples, and batteries are included for easy access to publicly-available IRFs.  

## Sensitivity Curve Calculation with `sensipy.sensitivity`

Sensitivity curves represent the minimum flux needed to detect a source at a given significance (usually $5 \sigma$) given an exposure time $t_{exp}$. Such curves are often used to compare the performances of different instruments, and `sensipy` can produce them in two flavors: integral and differential sensitivity curves. The sensitivity itself depends heavily on the rapidly-changing spectral shape of an event, which itself can be highly affected by distance due to the extragalactic background light (EBL). All of these factors are automatically taken into account.

[Plot of differential and integral sensitivity curves for CTAO calculated with sensipy]

## Simulating Observations with `sensipy.source`

This class addresses the fundamental question: if we begin observations with a latency of $t_L = X~\text{min}$ after an alert, what observation time is required in order to achieve a detection? Given that the user has already calculated the sensitivity curve for an event, `sensipy` can determine if the source is actually detectable, given $T_L$. When detectable, the exposure time necessary for detection is also calculated.

## Working with large catalogs with `sensipy.detectability`

`sensipy` can further estimate the overall detectability of entire classes of objects, given a catalog or survey of simulated events under various conditions. By performing and collating a large number of observation simulations for various events and latencies $t_L$, the toolkit can help produce visualizations which describe the optimal observing conditions for such events.

[Two example heatmap plots calculated with sensipy]

## Realtime applications with `sensipy.followup`

Tables of observation times can also be used as lookup tables (LUTs) during telescope observations in order to plan observation campaigns. For example, the following workflow can be implemented within `sensipy`:

1. a catalog of simulated spectra is processed with the above pipeline considering various observation conditions, and a LUT is created
2. a transient alert arrives during normal telescope operation and telescopes begin observing the event position with a latency of $t_L$
3. the LUT is filtered and interpolated in realtime in order to quickly calculate an informed estimate on the exposure time needed for a detection

Such workflows based on `sensipy` modules are already being internally evaluated within the MAGIC, Large-Size Telescope (LST), and CTAO collaborations for followup of both GW and GRB alerts.

## GW scheduling

In addition, the functions included in `sensipy.followup` may be used in tandem with scheduling software like `tilepy` for poorly-localized events. These scheduling tools create an optimized list of telescope pointings on the sky, while `sensipy` is used simultaneously to optimize the exposure time needed at each new pointing. It is in this context development of `sensipy` began within the CTAO collaboration.

[Show example of a GW pointing scheduling for a well-localized event, if possible overlay pointing durations on top of each pointing]

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below.

For a quick reference, the following citation commands can be used:

```md
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"
```

# Acknowledgements

We acknowledge contributions from some people

# Statement on AI

Let's say how we did not use generative AI to assist in the writing of this manuscript.

# References

