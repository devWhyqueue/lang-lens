# lang-lens

Smart and fast language detection for Python.

## Project Structure

```
lang-lens/
├── data/                # Contains the WiLI-2018 dataset and the Khan preprocessed subset
├── experiments/         # Jupyter notebooks for experiments
├── langlens/            # The Python package
│   ├── configuration/   # App YAML config and logging config
│   ├── data.py          # Load (cleaned) data into splits
│   ├── evaluation.py    # Evaluate model (classification report, confusion matrix, PCA plot)
│   ├── vectorizer.py    # Vectorize text data
│   ├── main.py          # Click CLI interface
├── report/              # Milestone reports
│   ├── figures/         # Figures for the reports
│   ├── sources/         # Papers etc.
├── tests/               # Tests
├── pyproject.toml       # Defines dependencies and project configuration
```

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

To install the package, run:

```sh
poetry install
```

## Usage

To see available commands and usage, run:

```sh
poetry run langlens --help
```

## Running Tests

To execute the tests, run:

```sh
poetry run pytest tests/
```

