# IGF python toolbox

## Objectives

This package contains a set of Python utilities for :

- easy data loading ;
- descriptive statistics iteration ;
- implementation of econometric methods.

## Organization

The repository is organized as follow :

- the `s3` module contains a set of functions for importing or exporting files in various formats from or to an s3 bucket on the Nubonyxia platform;
- the `stats_des` module contains functions to complement those integrated by default in pandas for descriptive statistics (addition of totals, weighted statistics, verification of statistical confidentiality, etc.).
- the `preprocessing` module contains a set of classes that can be integrated into a `sklearn.pipeline` to perform various data transformation operations
- the `model_selection` module contains functions for training a prediction model or estimating a regression model
- the `estimators` module contains  econometric models that can be integrated into a `sklearn.pipeline`
- the `utils` module contains  a set of utility functions on which other functions in this module depend, or for calculating weighted statistics, for example

## Installation

### Package and dependencies

```bash
git clone <repo_url>
pip install -e igf_toolbox
```

The package can then be used like any other Python package.

### Documentation

To visualize the documentation :

```
mkdocs build --port 5000
```

## License

The package is licensed under the MIT License.