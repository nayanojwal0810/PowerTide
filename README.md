# PowerTide

An end-to-end ERCOT electricity demand forecasting platform using EIA load data and weather forecasts, with automated data pipelines, MLflow experiment tracking, scheduled predictions, a FastAPI service, and a Streamlit dashboard.

> **Status:** Implementation in progress. The project foundation, packaging, and canonical forecast contract are currently established.

## Overview

PowerTide is designed to deliver automated, operational day-ahead electricity demand forecasts for the ERCOT region. It enforces a strict operational forecast contract, combining historical grid load observations published by the U.S. Energy Information Administration (EIA) with numerical weather predictions from ECMWF IFS HRES via Open-Meteo.

### Forecast Contract Specifications

- **Issue Time:** 07:00 UTC on day $D$.
- **Historical Demand Cutoff:** 04:00 UTC on day $D$ (accommodating operational publication lag).
- **Forecast Horizon:** Calendar day $D+1$ across 24 hourly intervals under the EIA hour-ending convention (01:00 UTC day $D+1$ through 00:00 UTC day $D+2$).
- **Lead Times:** +18 hours through +41 hours from the issue time.
- **Weather Inputs:** ECMWF IFS HRES 00:00 UTC Single Run.

## Architecture & Planned Workflow

1. **Ingestion & Caching:** Ingest historical hourly ERCOT demand from the EIA v2 API and numerical weather forecasts from Open-Meteo.
2. **Validation & Feature Engineering:** Validate timestamps, interval continuity, and schema integrity; construct point-in-time features strictly aligned to the 07:00 UTC issue cutoff.
3. **Model Training & Experiment Tracking:** Benchmark baselines against weather-augmented ML models with chronological backtesting and MLflow tracking.
4. **Scheduled Inference & Resilience:** Generate daily 24-hour forecasts atomically (`latest.json`), track pipeline health (`pipeline_status.json`), and gracefully handle missing or stale upstream data.
5. **Serving & Visualization:** Provide low-latency API access via FastAPI and an interactive operator dashboard via Streamlit.

## Tech Stack

- **Language:** Python 3.11+
- **Data Validation & Schemas:** Pydantic v2
- **Data Processing & ML:** pandas, NumPy, scikit-learn
- **Experiment Tracking:** MLflow
- **Application & UI:** FastAPI, Streamlit
- **Code Quality & Testing:** pytest, Ruff

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git

### Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/nayanojwal0810/PowerTide.git
cd PowerTide
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
pip install -e ".[dev]"
```

Configure environment variables:

```bash
cp .env.example .env
```

### Running Tests and Linters

```bash
pytest -q
ruff check .
ruff format --check .
```
