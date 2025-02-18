# lang-lens

Smart and fast language detection for Python.

## Project Structure

```
lang-lens/
├── data/                # Contains the dataset
├── langlens/            # The Python package
│   ├── baseline/        # Preprocessing, feature extraction, and classifier
│   ├── configuration/   # App YAML config and logging config
│   ├── data.py          # Load data into splits
│   ├── evaluation.py    # Evaluate model (classification report, confusion matrix, PCA plot)
│   ├── main.py          # Click CLI interface
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

