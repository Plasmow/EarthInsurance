import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Schema: lat, lon, time_utc, f1..f64, label_occ, label_int_f, label_int_wind_ms

EMBEDDING_DIM = 64
EMBED_COLS = [f"f{i}" for i in range(1, EMBEDDING_DIM + 1)]


def _parse_time_strict(s: Union[str, datetime]) -> datetime:
    """
    Parse strictly the format 'YYYY-MM-DD HH:MM:SS+HH:MM', convert to naive UTC.
    """
    if isinstance(s, datetime):
        return s.astimezone(timezone.utc).replace(tzinfo=None) if s.tzinfo else s
    raw = str(s).strip()
    if not raw:
        raise ValueError("Empty time string")
    try:
        dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S%z")
    except ValueError:
        if raw.endswith('Z'):
            dt = datetime.strptime(raw[:-1] + "+00:00", "%Y-%m-%d %H:%M:%S%z")
        else:
            raise ValueError(f"Expected 'YYYY-MM-DD HH:MM:SS+HH:MM', got '{s}'")
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _cyc_features(ts: datetime) -> Tuple[float, float, float, float, float, float]:
    m_angle = 2.0 * np.pi * (ts.month - 1) / 12.0
    m_sin, m_cos = float(np.sin(m_angle)), float(np.cos(m_angle))
    doy = ts.timetuple().tm_yday
    d_angle = 2.0 * np.pi * (doy - 1) / 365.25
    d_sin, d_cos = float(np.sin(d_angle)), float(np.cos(d_angle))
    h_angle = 2.0 * np.pi * ts.hour / 24.0
    h_sin, h_cos = float(np.sin(h_angle)), float(np.cos(h_angle))
    return m_sin, m_cos, d_sin, d_cos, h_sin, h_cos


def _validate_columns(df: pd.DataFrame) -> None:
    required = ["lat", "lon", "time_utc"] + EMBED_COLS + ["label_occ", "label_int_f", "label_int_wind_ms"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")


def _build_X(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    _validate_columns(df)
    t_parsed = df["time_utc"].apply(_parse_time_strict)
    feats = list(zip(*t_parsed.apply(_cyc_features)))
    month_sin, month_cos, doy_sin, doy_cos, hour_sin, hour_cos = feats
    geo_df = pd.DataFrame({
        "lat": pd.to_numeric(df["lat"], errors="coerce").fillna(0.0),
        "lon": pd.to_numeric(df["lon"], errors="coerce").fillna(0.0),
        "month_sin": month_sin,
        "month_cos": month_cos,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        # Optional: include occurrence label as feature for intensity (if known in training rows)
        "occ_label": df["label_occ"].astype(int),
    })
    emb_df = df[EMBED_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = pd.concat([emb_df.reset_index(drop=True), geo_df.reset_index(drop=True)], axis=1)
    return X, list(X.columns)


def _load_train_test(train_csv: str, test_csv: str):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    _validate_columns(train_df)
    _validate_columns(test_df)
    X_train, feature_names = _build_X(train_df)
    X_test, _ = _build_X(test_df)
    y_f_train = pd.to_numeric(train_df["label_int_f"], errors="coerce").fillna(0.0)
    y_f_test = pd.to_numeric(test_df["label_int_f"], errors="coerce").fillna(0.0)
    y_w_train = pd.to_numeric(train_df["label_int_wind_ms"], errors="coerce").fillna(0.0)
    y_w_test = pd.to_numeric(test_df["label_int_wind_ms"], errors="coerce").fillna(0.0)
    return X_train, y_f_train, y_w_train, X_test, y_f_test, y_w_test, feature_names


def _tree_method(use_gpu: bool) -> str:
    return "gpu_hist" if use_gpu else "hist"


def train_damage(
    train_csv: str,
    test_csv: str,
    outdir: str = "models_damage",
    use_gpu: bool = False,
    random_state: int = 42,
) -> Dict[str, float]:
    os.makedirs(outdir, exist_ok=True)
    (X_tr, y_f_tr, y_w_tr, X_te, y_f_te, y_w_te, feature_names) = _load_train_test(train_csv, test_csv)

    reg_f = XGBRegressor(
        n_estimators=5000,
        max_depth=9,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2.0,
        reg_lambda=1.5,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method=_tree_method(use_gpu),
        random_state=random_state,
        n_jobs=0,
    )
    # Train without early stopping for broader xgboost version compatibility
    reg_f.fit(X_tr, y_f_tr, eval_set=[(X_te, y_f_te)], verbose=False)

    reg_w = XGBRegressor(
        n_estimators=5000,
        max_depth=9,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2.0,
        reg_lambda=1.5,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method=_tree_method(use_gpu),
        random_state=random_state,
        n_jobs=0,
    )
    reg_w.fit(X_tr, y_w_tr, eval_set=[(X_te, y_w_te)], verbose=False)

    rmse_f = float(np.sqrt(mean_squared_error(y_f_te, reg_f.predict(X_te))))
    rmse_w = float(np.sqrt(mean_squared_error(y_w_te, reg_w.predict(X_te))))

    reg_f.save_model(os.path.join(outdir, "tornado_int_f_xgb.json"))
    reg_w.save_model(os.path.join(outdir, "tornado_int_wind_xgb.json"))
    with open(os.path.join(outdir, "preprocess.json"), "w", encoding="utf-8") as f:
        json.dump({
            "feature_names": feature_names,
            "embedding_dim": EMBEDDING_DIM,
            "time_format": "YYYY-MM-DD HH:MM:SS+HH:MM",
            "created_utc": datetime.utcnow().isoformat() + "Z",
        }, f, indent=2)

    return {"rmse_int_f": rmse_f, "rmse_int_wind": rmse_w}


def _load_models(model_dir: str):
    reg_f = XGBRegressor()
    reg_w = XGBRegressor()
    reg_f.load_model(os.path.join(model_dir, "tornado_int_f_xgb.json"))
    reg_w.load_model(os.path.join(model_dir, "tornado_int_wind_xgb.json"))
    with open(os.path.join(model_dir, "preprocess.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return reg_f, reg_w, meta["feature_names"]


def _build_single_row(embedding: List[float], lat: float, lon: float, time_utc: Union[str, datetime]) -> pd.DataFrame:
    if len(embedding) != EMBEDDING_DIM:
        raise ValueError(f"Embedding length must be {EMBEDDING_DIM}")
    ts = _parse_time_strict(time_utc)
    m_sin, m_cos, d_sin, d_cos, h_sin, h_cos = _cyc_features(ts)
    row: Dict[str, float] = {f"f{i+1}": float(embedding[i]) for i in range(EMBEDDING_DIM)}
    row.update({
        "lat": float(lat),
        "lon": float(lon),
        "month_sin": m_sin,
        "month_cos": m_cos,
        "doy_sin": d_sin,
        "doy_cos": d_cos,
        "hour_sin": h_sin,
        "hour_cos": h_cos,
        "occ_label": 0.0,  # placeholder when predicting intensity alone
    })
    return pd.DataFrame([row])


def predict_damage(model_dir: str, embedding: List[float], lat: float, lon: float, time_utc: Union[str, datetime]) -> Dict[str, float]:
    reg_f, reg_w, feature_names = _load_models(model_dir)
    X = _build_single_row(embedding=embedding, lat=lat, lon=lon, time_utc=time_utc)
    X = X.reindex(columns=feature_names, fill_value=0.0)
    return {
        "intensity_f": float(reg_f.predict(X)[0]),
        "intensity_wind_ms": float(reg_w.predict(X)[0]),
    }


def main():
    import argparse
    import sys

    # Auto-run training if no arguments: looks for train.csv/test.csv in CWD or ./data
    if len(sys.argv) == 1:
        candidates = [
            (os.path.join(os.getcwd(), "train.csv"), os.path.join(os.getcwd(), "test.csv")),
            (os.path.join(os.getcwd(), "data", "train.csv"), os.path.join(os.getcwd(), "data", "test.csv")),
        ]
        for tr, te in candidates:
            if os.path.exists(tr) and os.path.exists(te):
                print(f"[auto] Training (damage) with {tr} and {te} ...")
                metrics = train_damage(
                    train_csv=tr,
                    test_csv=te,
                    outdir="models_damage",
                    use_gpu=False,
                    random_state=42,
                )
                print(json.dumps(metrics, indent=2))
                return
        print("No arguments and train.csv/test.csv not found in CWD or ./data.")

    parser = argparse.ArgumentParser(description="XGBoost tornado intensity (F-scale and wind) prediction")
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train", help="Train regressors with train/test CSVs")
    p_train.add_argument("--train", required=True)
    p_train.add_argument("--test", required=True)
    p_train.add_argument("--outdir", default="models_damage")
    p_train.add_argument("--gpu", action="store_true")
    p_train.add_argument("--seed", type=int, default=42)

    p_pred = sub.add_parser("predict", help="Predict intensities for one example")
    p_pred.add_argument("--modeldir", default="models_damage")
    p_pred.add_argument("--embedding", required=True, help="Comma-separated 64 floats or @path")
    p_pred.add_argument("--lat", type=float, required=True)
    p_pred.add_argument("--lon", type=float, required=True)
    p_pred.add_argument("--time", required=True, help='"YYYY-MM-DD HH:MM:SS+HH:MM"')

    args = parser.parse_args()
    if args.cmd == "train":
        metrics = train_damage(
            train_csv=args.train,
            test_csv=args.test,
            outdir=args.outdir,
            use_gpu=bool(args.gpu),
            random_state=args.seed,
        )
        print(json.dumps(metrics, indent=2))
    else:
        emb_arg = args.embedding
        if emb_arg.startswith("@") and os.path.exists(emb_arg[1:]):
            path = emb_arg[1:]
            if path.endswith(".npy"):
                embedding = np.load(path).astype(float).tolist()
            else:
                with open(path, "r", encoding="utf-8") as f:
                    txt = f.read().replace("\n", " ").replace("\t", " ")
                embedding = [float(x) for x in txt.replace(",", " ").split() if x]
        else:
            embedding = [float(x) for x in emb_arg.split(",")]
        res = predict_damage(
            model_dir=args.modeldir,
            embedding=embedding,
            lat=args.lat,
            lon=args.lon,
            time_utc=args.time,
        )
        print(json.dumps(res))


if __name__ == "__main__":
    main()
