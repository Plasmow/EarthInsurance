# EarthInsurance

Tornado risk prediction with XGBoost using satellite embeddings and contextual spatiotemporal features. Ships a Flask API and a React + Leaflet map UI.

## Quickstart

- Backend (Flask)
  - `pip install -r requirements.txt`
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

## Troubleshooting

- Start backend before frontend; verify `/api/health`.
- Ensure model JSONs + `preprocess.json` exist in `Backend/models_*`.
- If ports change, update fetch URL in `Frontend/src/App.jsx`.
