# Restaurant Demand Forecasting

An end-to-end data science pipeline that forecasts a restaurant's delivery
and dine-in demand. It ingests order data (POS exports, Uber Eats, and
DoorDash), enriches it with weather and holiday signals, compares three
forecasting models, and renders the results as an interactive dashboard,
"The Demand Ledger."

Live dashboard: https://kelsonlam.github.io/restaurant-demand-forecast/

## Why this project

Restaurants run on thin margins and unpredictable foot traffic. Knowing how
busy tomorrow will be, broken down by channel, day, and hour, turns staffing
from a guess into a plan. This project builds that forecast from scratch:
cleaning raw order exports, engineering features (day of week, holidays,
weather), and evaluating multiple time-series models against a real holdout
before trusting any of them.

The current run uses a calibrated, clearly-labeled simulated dataset for an
upscale Knoxville, TN steakhouse (no restaurant publishes its full point-of-
sale history publicly), built from a public restaurant sales dataset and real
Knoxville weather. The pipeline is designed to run unchanged on real POS,
Uber Eats Manager, and DoorDash exports once available.

## Project structure

```
data/raw/         <- order exports go here (POS, Uber Eats, DoorDash)
data/processed/   <- cleaned tables the scripts produce
scripts/          <- the pipeline, numbered in run order
outputs/          <- charts, forecasts, staffing recommendations, dashboard
docs/             <- GitHub Pages copy of the dashboard (do not edit directly)
```

## Using real order data

**POS exports (dine-in / takeout):** export an order-level report (check
number, opened-at timestamp, check total) and save it to `data/raw/` with
`dinein` or `takeout` in the filename.

**Uber Eats:** log in to Uber Eats Manager, go to Reports, request an "Order
History" report over the widest date range available, and save the CSV to
`data/raw/` with `uber` in the filename.

**DoorDash:** log in to the DoorDash Merchant Portal, go to Financials or
Report Builder, export an order-level report, and save it to `data/raw/`
with `doordash` in the filename.

Column names are auto-detected, so exports from different report formats
usually work without edits. Aim for at least 6 to 12 months of history for
reliable seasonality detection.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Edit `scripts/config.py` to set the restaurant's latitude, longitude, and
state so the weather and holiday features are correct.

## Running the pipeline

From the `scripts/` folder, using `..\.venv\Scripts\python.exe`:

```
python 01_ingest_clean.py        # clean and merge exports into a daily sales table
python 02_enrich_features.py     # add weather, holidays, calendar, lag features
python 03_eda.py                 # trend, day-of-week, hourly, and weather charts
python 04_model.py               # train models, 14-day forecast, staffing plan
python 05_dashboard.py           # build the interactive dashboard
```

Or run everything at once: `run_all.bat`

To regenerate the simulated dataset used in this repo, run
`make_calibrated_data.py`.

## Models compared

- **Seasonal naive** (same weekday last week), the baseline to beat
- **SARIMA** with weekly seasonality
- **XGBoost** on calendar, lag, holiday, and weather features

Each is evaluated on the last 28 days held out, using MAE and MAPE. The
best-performing model produces a 14-day forecast, tiered into busy, normal,
and slow days for staffing decisions.

## The dashboard

"The Demand Ledger" is a single self-contained HTML file (no build step, no
external requests, fonts embedded inline) styled around the subject:
menu-style dot-leader stats, a service-rhythm heatmap shaped like a
reservation grid, and a 14-day forecast rendered as a kitchen ticket rail.
It supports a channel toggle (dine-in, takeout, Uber Eats, DoorDash), a
date-range brush, and a weekday/weekend view, and includes a plain-table
fallback for accessibility.
