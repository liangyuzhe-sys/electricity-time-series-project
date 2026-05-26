# Electricity Time Series Project

This repository contains a Python time-series analysis project for the German DE-AT-LU day-ahead electricity market from 2015-01-01 to 2018-09-30.

The project uses three modelling modules:

- `sarima.py`: SARIMA / SARIMAX load forecasting
- `var.py`: VAR, Granger causality, impulse response, and FEVD analysis
- `garch.py`: GARCH-family volatility modelling for day-ahead price differences

## Folder Structure

```text
electricity_time_series_project/
  README.md
  GITHUB.md
  .gitignore
  requirements.txt
  dataset/
    de_panel_daily.csv
    de_panel_hourly.csv
  code/
    main.py
    sarima.py
    var.py
    garch.py
  figures/
    *.png
```

Running the code will also create a `tables/` folder for CSV and TXT outputs.

## Setup

Use Python 3.10 or newer. A Conda environment is recommended.

```bash
pip install -r requirements.txt
```

## Run

Run all three models:

```bash
python code/main.py
```

Run selected models:

```bash
python code/main.py var garch
```

Run a single script directly:

```bash
python code/sarima.py
python code/var.py
python code/garch.py
```

Note: `sarima.py` is the slowest script because it estimates hourly SARIMA/SARIMAX models.

## Data Source

The cleaned datasets in `dataset/` are derived from Open Power System Data:

https://open-power-system-data.org/

The sample covers the DE-AT-LU bidding zone before the 2018 market split.
