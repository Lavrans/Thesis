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

### Data preprocessing

1. Start with `code/data.ipynb`. This notebook walks you through the initial preprocessing of the dataset used in the project.

2. Download daily treasury yields from the [US treasury department](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2017) and put all files in a folder `./yield_data`. Run `python ./code/concat.py` to concat and make file `daily_treasury_rates.csv` needed for next step.

3. Run `python ./code/preprocess.py`. This will finalize the preprocessing stage by creating 3 files: `underlying.csv`, `hugging_v3.csv`, and `final_data.h5`. The data saved to `final_data.h5` is also filtered for empty data points and is the dataset that will be used for model training.






