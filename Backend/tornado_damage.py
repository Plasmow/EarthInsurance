import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from xgboost import XGBClassifier

# Expected schema: lat, lon, time_utc, f1..f64, magnitude



EMBEDDING_DIM = 64
EMBED_COLS = [f"f{i}" for i in range(1, EMBEDDING_DIM + 1)]


def _parse_time_strict(s: Union[str, datetime]) -> datetime:
    """
    Parse a strict UTC-aware timestamp and return a naive UTC datetime.

    Accepted formats:
      - 'YYYY-MM-DD HH:MM:SS+HH:MM' (space)
      - 'YYYY-MM-DDTHH:MM:SS+HH:MM' (ISO-8601 with 'T')
      - 'YYYY-MM-DD HH:MM:SSZ' or 'YYYY-MM-DDTHH:MM:SSZ' (Z = UTC)
    """
    if isinstance(s, datetime):
        return s.astimezone(timezone.utc).replace(tzinfo=None) if s.tzinfo else s
    raw = str(s).strip()
    if not raw:
        raise ValueError("Empty time string")
    norm = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    patterns = [
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%z",
    ]
    last_exc = None
    for p in patterns:
        try:
            dt = datetime.strptime(norm, p)
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        except ValueError as exc:
            last_exc = exc
            continue
    raise ValueError(
        f"Expected one of ['YYYY-MM-DD HH:MM:SS+HH:MM', 'YYYY-MM-DDTHH:MM:SS+HH:MM', '...Z'], got '{s}'"
    ) from last_exc





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
    # Ensure the minimal required columns are present for magnitude classification
    required = ["lat", "lon", "time_utc"] + EMBED_COLS + ["magnitude"]
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
    y_train = pd.to_numeric(train_df["magnitude"], errors="coerce")
    y_test = pd.to_numeric(test_df["magnitude"], errors="coerce")
    # Enforce non-negativity in training labels before transform
#    y_train = y_train.clip(lower=0.0)
#    y_test = y_test.clip(lower=0.0)
    return X_train, y_train, X_test, y_test, feature_names


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
    (X_tr, y_tr, X_te, y_te, feature_names) = _load_train_test(train_csv, test_csv)

    # Prepare multiclass labels; infer classes present in training set
    y_tr = pd.to_numeric(y_tr, errors="coerce").fillna(0.0).astype(int).clip(lower=0)
    y_te = pd.to_numeric(y_te, errors="coerce").fillna(0.0).astype(int).clip(lower=0)
    present_classes = np.unique(y_tr.values)
    present_classes = np.sort(present_classes)
    num_class = int(len(present_classes))
    if num_class < 2:
        raise ValueError("Need at least 2 classes present in training labels for multiclass.")

    # Compute class frequencies for weighting using only PRESENT classes
    # Avoid including absent classes, which can distort normalization and collapse training.
    classes = present_classes  # only classes observed in y_tr
    freq = np.array([(y_tr == c).sum() for c in classes], dtype=float)
    # Inverse-frequency weights, normalized so average weight ~1 over present classes
    inv = 1.0 / np.maximum(freq, 1.0)
    class_weights = inv / inv.mean()
    cw_map = {int(c): float(w) for c, w in zip(classes, class_weights)}
    sample_weight = np.array([cw_map.get(int(v), 1.0) for v in y_tr], dtype=float)
    
    
    
    
    

    clf = XGBClassifier(
        n_estimators=5000,
        max_depth=9,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2.0,
        reg_lambda=1.5,
        objective="multi:softprob",
        num_class=num_class,
        eval_metric="mlogloss",
        tree_method=_tree_method(use_gpu),
        random_state=random_state,
        n_jobs=-1,
    )
    # Train without early stopping for broader XGBoost version compatibility
    clf.fit(X_tr, y_tr, sample_weight=sample_weight, eval_set=[(X_te, y_te)], verbose=False)

    proba = clf.predict_proba(X_te)
    y_pred = np.asarray(proba).argmax(axis=1)
    acc = float(accuracy_score(y_te, y_pred))
    macro_f1 = float(f1_score(y_te, y_pred, average="macro"))
    try:
        ll = float(log_loss(y_te, proba, labels=present_classes.tolist()))
    except Exception:
        ll = float(log_loss(y_te, proba))

    clf.save_model(os.path.join(outdir, "tornado_magnitude_cls_xgb.json"))
    with open(os.path.join(outdir, "preprocess.json"), "w", encoding="utf-8") as f:
        json.dump({
            "feature_names": feature_names,
            "embedding_dim": EMBEDDING_DIM,
            "time_format": "YYYY-MM-DD HH:MM:SS+HH:MM",
            "task": "multiclass",
            "class_labels": present_classes.tolist(),
            "created_utc": datetime.utcnow().isoformat() + "Z",
        }, f, indent=2)

    return {"accuracy": acc, "macro_f1": macro_f1, "log_loss": ll, "class_weights": {int(c): float(class_weights[i]) for i,c in enumerate(classes)}}


def _load_model(model_dir: str):
    clf = XGBClassifier()
    # New filename for classifier; keep fallback to old if needed
    model_candidates = [
        os.path.join(model_dir, "tornado_magnitude_cls_xgb.json"),
        os.path.join(model_dir, "tornado_magnitude_xgb.json"),
    ]
    for mp in model_candidates:
        if os.path.exists(mp):
            clf.load_model(mp)
            break
    with open(os.path.join(model_dir, "preprocess.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return clf, meta["feature_names"]


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
    })
    return pd.DataFrame([row])


def predict_damage(model_dir: str, embedding: List[float], lat: float, lon: float, time_utc: Union[str, datetime]) -> Dict[str, float]:
    # Backward-compatible function name; now returns probability vector over magnitudes 0..5
    clf, feature_names = _load_model(model_dir)
    X = _build_single_row(embedding=embedding, lat=lat, lon=lon, time_utc=time_utc)
    X = X.reindex(columns=feature_names, fill_value=0.0)
    proba = clf.predict_proba(X)[0]
    return {"magnitude_probs": [float(p) for p in proba.tolist()]}


def main():
    import argparse
    import sys

    # Auto-run training if no arguments: looks for train_dmg.csv/test_dmg.csv in CWD or ./data
    if len(sys.argv) == 1:
        candidates = [
            (os.path.join(os.getcwd(), "train_dmg.csv"), os.path.join(os.getcwd(), "test_dmg.csv")),
            (os.path.join(os.getcwd(), "data", "train_dmg.csv"), os.path.join(os.getcwd(), "data", "test_dmg.csv")),
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
        print("No arguments and train_dmg.csv/test_dmg.csv not found in CWD or ./data.")

    parser = argparse.ArgumentParser(description="XGBoost tornado magnitude prediction (training only; inference via risk_inference)")
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train", help="Train regressors with train/test CSVs")
    p_train.add_argument("--train", required=True)
    p_train.add_argument("--test", required=True)
    p_train.add_argument("--outdir", default="models_damage")
    p_train.add_argument("--gpu", action="store_true")
    p_train.add_argument("--seed", type=int, default=42)

    # No predict CLI; use risk_inference.py for inference

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
        parser.print_help()


if __name__ == "__main__":
    main()
