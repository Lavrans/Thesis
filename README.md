# Thesis

## Getting started

This project is built with python 3.10. To ensure compatability make sure you're using the correct version of python.

### Linux/macOS

Run the following command to setup a virtual environment and install the necessary dependencies

```code
make
```

### Windows

You probably need to set up the virtual environment and install the necessary dependencies yourself. All dependency requirements can be found in `requirements.txt`. If you're using pip, run the following command from your venv to install all of them at once:

```code
pip install -r requirements.txt
```

## Formatting/Linting

This project uses ruff and mypy (with pydantic) for formatting and linting. Their configs can be found in pyproject.toml and mypy.ini respectively.
