# EarthInsurance

Tornado risk prediction using XGBoost with satellite embeddings and contextual features.

## Data schema

Training CSV must include:

- E0..E63: 64-d embedding floats (from AlphaEarth)
- lat, lon: decimal degrees
- period_start: date string (YYYY-MM-DD). Optional: period_end (for duration)
- flood_occurred: 0/1 flag
- flood_intensity: optional numeric (if missing, a proxy is computed from `flood_depth_m` and `flood_duration_days` when available)
- tornado_occurred: 0/1 label (target)
- tornado_intensity: numeric label (target)

Additional numeric columns are automatically included as features.

## Quick start

0) Install dependencies (Windows cmd):

```
py -m pip install -r requirements.txt
```
1) Authenticate to GoogleEarth (first request access to the owner to manage API requests)

```
earthengine authenticate
```



2) Train with a synthetic dataset (for a quick test):

```
py tornado_risk_model.py train --synthetic --outdir models
```

This prints AUC, average precision, and RMSE, and writes models to `models/`.

3) Predict for a single example:

```
py tornado_risk_model.py predict --modeldir models \
	--embedding "0.1,0.2,0.3, ... 64 values ..." \
	--lat 35.1 --lon -97.5 --start 2024-05-01 --end 2024-05-15 --flood --flood_intensity 1.2
```

Alternatively you can pass a `.npy` file with `--embedding @path/to/vec.npy`.

## Notes

- Classification and regression are trained separately (occurrence probability and intensity).
- Temporal seasonality is encoded with cyclical month features; optional duration is included if `period_end` is provided.
- Class imbalance is handled via `scale_pos_weight`.
- Model files: `models/tornado_classifier.json`, `models/tornado_regressor.json`, and preprocessing metadata in `models/preprocess.json`.