# EarthInsurance

Tornado risk prediction with XGBoost using satellite embeddings and contextual spatiotemporal features. Ships a Flask API and a React + Leaflet map UI.

## Quickstart

- Python deps
  - `pip install -r requirements.txt`
  - If install fails, ensure at least: `earthengine-api`, `numpy`, `pandas`, `xgboost`, `flask`, `flask-cors`, `scikit-learn`.

- Earth Engine (embeddings)
  - `earthengine authenticate`
  - Optional project (matches code default): `earthengine set_project gen-lang-client-0546266030`
  - Quick test (prints a 64-d vector): `python Backend/model.py`

- Backend (Flask)
  - `python Backend/api.py`
  - Check: open `http://localhost:5000/api/health`

- Frontend (React + Vite)
  - `cd Frontend && npm install`
  - `npm run dev`
  - Open the shown URL (e.g., `http://localhost:5173`) and click on the map. Backend defaults to `http://localhost:5000`.

## What It Does

- Predicts tornado occurrence probability and EF-scale magnitude distribution (EF0–EF5).
- Computes a combined risk score and human-readable labels.
- Visualizes results on an interactive map with color-coded markers.

## API (Cheat Sheet)

- POST `/api/calculate-risk`
  - Body:
    ```json
    { "latitude": 35.47, "longitude": -97.52, "time_utc": "2025-05-02 14:30:00+00:00" }
    ```
    `time_utc` optional; ISO with `Z` also accepted.
  - Returns: `risk_score` (0–1), `risk_level`, `probability`, `magnitude` (0–5), `magnitude_probs` (EF0..EF5), `tornado_damage`, `ef_label`.

Other endpoints: `/api/health`, `/api/batch-calculate-risk`, `/api/risk-zones`, `/api/predict-detailed`.

## Train Models (Optional)

- Probability (occurrence):
  ```
  python Backend/tornado_probability.py train --train path/to/train.csv --test path/to/test.csv --outdir Backend/models_prob
  ```
- Magnitude (EF classifier):
  ```
  python Backend/tornado_damage.py train --train path/to/train_i.csv --test path/to/test_i.csv --outdir Backend/models_damage
  ```

CSV schema: `lat, lon, time_utc, f1..f64` (+ `label_occ` for probability, `label_magn` for magnitude). Time format: `YYYY-MM-DD HH:MM:SS+HH:MM` or `...Z`.

## Repo Layout

- `Backend/api.py` Flask API
- `Backend/risk_inference.py` model loading + features
- `Backend/models_prob/` and `Backend/models_damage/` pretrained models
- `Frontend/` React app (Vite + Leaflet)

## Python Files Overview

- `Backend/api.py`: Flask API exposing health, single/batch risk, zones, and detailed inference endpoints; starts the server.
- `Backend/risk_inference.py`: Loads XGBoost models, builds geo/time features, and returns probability and EF magnitude predictions.
- `Backend/tornado_probability.py`: Trains the tornado occurrence classifier and saves the model/metadata (CLI usage).
- `Backend/tornado_damage.py`: Trains the EF magnitude classifier and saves the model/metadata (CLI usage).
- `Backend/tornado_loss.py`: Computes a unitless risk score R = p * v(EF) using a simple MDR table by occupancy (residential/commercial/industrial); includes a small CLI for single‑point scoring.
- `Backend/generate_features_from_events.py`: Generates positives/negatives from `events.csv` and samples AlphaEarth embeddings via GEE (exports to Drive/GCS/local).
- `Backend/generate_features_from_events_nopandas.py`: Minimal no‑pandas generator that writes `features.csv` (~40% positives) for quick tests.
- `Backend/model.py`: Earth Engine example that fetches a 64‑dim AlphaEarth embedding at a given point.
- `Backend/test.py`: Prototype pipeline to build a training CSV from events and random negatives using AlphaEarth sampling.
- `Backend/test_tornado.py`: Utility to load, filter (2017–2024), and reshape tornado track data for exploration.
- `embedding_match.py`: Helper to return a standardized `{embedding, lat, lon, time_utc}` record from AlphaEarth for a lat/lon/date.
- `extract_alphaearth_local.py`: Local‑only AlphaEarth extractor; samples yearly embeddings and saves to CSV with paged downloads.
- `Backend/extract_alphaearth_local.py`: Variant of the local AlphaEarth extractor scoped to the Backend folder.

## Troubleshooting

- Start backend before frontend; verify `/api/health`.
- Ensure model JSONs + `preprocess.json` exist in `Backend/models_*`.
- If ports change, update fetch URL in `Frontend/src/App.jsx`.
