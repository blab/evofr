# `evofr`: Evolutionary forecasting for genetic variants

[![CI](https://github.com/blab/evofr/actions/workflows/ci.yaml/badge.svg)](https://github.com/blab/evofr/actions/workflows/ci.yaml)

## Overview

`evofr` is a Python package designed for evolutionary forecasting of genetic variants. 
This project provides tools aimed at modeling and predicting the evolution and prevalence of genetic variants over time. 
The package integrates various data handling, modeling, and plotting capabilities.

## Installation

The latest release of this package can be installed with:

```
pip install evofr
```


## Installing locally

The package can be built locally by running

```
poetry build
```

The package can then be installed from the resulting wheel file using

```
pip install <path-to-wheel>
```

## Releasing a new version

Bump the version in `pyproject.toml`, following [semantic versioning rules](https://semver.org/spec/v2.0.0.html), and push to the main branch.
[Create a new release on GitHub](https://github.com/blab/evofr/releases/new) using the new version number as the tag.
Locally, run `poetry build` to build the distribution package.

Setup [an API token to publish to PyPI](https://test.pypi.org/help/#apitoken), if you haven't before.
[Configure poetry to register your PyPI credentials](https://python-poetry.org/docs/repositories/#configuring-credentials).

Run `poetry publish` to upload the package to PyPI.
