# sensipy

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Build Status](https://img.shields.io/github/actions/workflow/status/astrojarred/sensipy/ci.yml)](https://github.com/astrojarred/sensipy/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/sensipy.svg)](https://pypi.org/project/sensipy/)
[![GitHub Issues](https://img.shields.io/github/issues/astrojarred/sensipy.svg)](https://github.com/astrojarred/sensipy/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/astrojarred/sensipy.svg)](https://github.com/astrojarred/sensipy/pulls)
[![License](https://img.shields.io/github/license/astrojarred/sensipy)](LICENSE)

Sensipy is a Python toolkit for simulating gamma-ray follow-up observations of time-variable astrophysical sources. First built for CTAO-style analyses, it builds on top of [`gammapy`](https://gammapy.org/) and streamlines everything from IRF handling to exposure lookups.

The full documentation is available at [sensipy.vercel.app](https://sensipy.vercel.app/).

<img src="https://raw.githubusercontent.com/astrojarred/sensipy/main/images/sensipy-logo.svg" alt="Sensipy logo" style="max-width: min(60%, 800px); display: block; margin: 0 auto;">

## Features

- Sensitivity and exposure-time calculations powered by `gammapy`.
- Calculate differential or integral sensitivity curves for gamma-ray observatories using instrument response functions and spectral models.
- Easily simulate the observation time needed to detect a source with a given spectral model at a specified significance level.
- Follow-up utilities (`sensipy.followup`) for quick-look assessments.
- Ships with ready-to-use EBL models.
- CTAO-first workflow that loads IRFs, applies EBL absorption models, and simulates detectability curves.

## Installation

Sensipy requires Python 3.11+.

```bash
# Recommended: uv
uv add sensipy

# or pip
pip install sensipy
```

Using conda? Create/activate an environment, ensure `pip` is available, then run `pip install sensipy`.

Run the unit tests with `uv run pytest` (or `pytest` in any configured environment) to verify the installation.

## Documentation

Full documentation, tutorials, and the API reference live on [the docs](sensipy.vercel.app) (source in `docs/`). To work on the site locally:

```bash
cd docs
bun install
bun dev
```

Key sections include:

- Getting started guides for installation, setup, spectral/EBL models, and sensitivity calculations.
- Tutorials that walk through the full workflow (loading IRFs, simulating observations, running follow-ups).
- API reference material for core modules such as `ctaoirf`, `sensitivity`, `followup`, and `source`.

## Contributing

Contributions are welcome! Please:

- Open an issue for bugs or feature proposals.
- Keep changes tested via `uv run pytest` or `uv run pytest -n 4` for faster testing in parallel.
- Check for type errors with `uv run mypy src/sensipy`.
- Follow the existing code style and type-checking expectations (`ruff`, `mypy`, etc.).

## License

Sensipy is distributed under the [Apache 2.0 License](LICENSE).




