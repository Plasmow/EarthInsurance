import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from xgboost import XGBClassifier

# Schema attendu: lat, lon, time_utc, f1..f64, label

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

    # Normalize trailing Z to +00:00 (UTC)
    norm = raw[:-1] + "+00:00" if raw.endswith("Z") else raw

    # Try multiple strict patterns
    patterns = [
        "%Y-%m-%d %H:%M:%S%z",   # space
        "%Y-%m-%dT%H:%M:%S%z",   # ISO T
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
    """
    Vérifie la présence des colonnes minimales.

    Colonnes requises pour le modèle de probabilité:
      - lat, lon, time_utc, f1..f64, label

    """
    required = ["lat", "lon", "time_utc"] + EMBED_COLS + ["label"]
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
    y_train = train_df["label"].astype(int)
    y_test = test_df["label"].astype(int)
    return X_train, y_train, X_test, y_test, feature_names


def _scale_pos_weight(y) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos == 0:
        return 1.0
    return max(1.0, neg / pos)


def _tree_method(use_gpu: bool) -> str:
    return "gpu_hist" if use_gpu else "hist"


def train_probability(
    train_csv: str,
    test_csv: str,
    outdir: str = "models_prob",
    use_gpu: bool = False,
    random_state: int = 42,
) -> Dict[str, float]:
    os.makedirs(outdir, exist_ok=True)
    data = _load_train_test(train_csv, test_csv)  # returns (X_train, y_train, X_test, y_test, feature_names)
    X_tr, y_tr, X_te, y_te, feature_names = data

    clf = XGBClassifier(
        n_estimators=5000,
        max_depth=9,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.85,
        min_child_weight=2.0,
        reg_lambda=1.5,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method=_tree_method(use_gpu),
        scale_pos_weight=_scale_pos_weight(y_tr),
        random_state=random_state,
        n_jobs=0,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    prob = clf.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_te, prob))
    ap = float(average_precision_score(y_te, prob))
    acc = float(accuracy_score(y_te, (prob >= 0.5).astype(int)))

    clf.save_model(os.path.join(outdir, "tornado_prob_xgb.json"))
    with open(os.path.join(outdir, "preprocess.json"), "w", encoding="utf-8") as f:
        json.dump({
            "feature_names": feature_names,
            "embedding_dim": EMBEDDING_DIM,
            "time_format": "YYYY-MM-DD HH:MM:SS+HH:MM",
            "created_utc": datetime.utcnow().isoformat() + "Z",
        }, f, indent=2)

    return {"auc": auc, "average_precision": ap, "accuracy": acc}


def _load_model(model_dir: str):
    clf = XGBClassifier()
    clf.load_model(os.path.join(model_dir, "tornado_prob_xgb.json"))
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


def predict_probability(model_dir: str, embedding: List[float], lat: float, lon: float, time_utc: Union[str, datetime]) -> float:
    clf, feature_names = _load_model(model_dir)
    X = _build_single_row(embedding=embedding, lat=lat, lon=lon, time_utc=time_utc)
    X = X.reindex(columns=feature_names, fill_value=0.0)
    return float(clf.predict_proba(X)[:, 1][0])


def main():
    import argparse
    import sys

    if len(sys.argv) == 1:
        candidates = [
            (os.path.join(os.getcwd(), "train.csv"), os.path.join(os.getcwd(), "test.csv")),
            (os.path.join(os.getcwd(), "data", "train.csv"), os.path.join(os.getcwd(), "data", "test.csv")),
        ]
        for tr, te in candidates:
            if os.path.exists(tr) and os.path.exists(te):
                print(f"[auto] Training with {tr} and {te} ...")
                metrics = train_probability(
                    train_csv=tr,
                    test_csv=te,
                    outdir="models_prob",
                    use_gpu=False,
                    random_state=42,
                )
                print(json.dumps(metrics, indent=2))
                return
        print("No arguments and train.csv/test.csv not found in CWD or ./data.")

    parser = argparse.ArgumentParser(description="XGBoost probability of tornado occurrence (training only; inference via risk_inference)")
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train", help="Train classifier with train/test CSVs")
    p_train.add_argument("--train", required=True)
    p_train.add_argument("--test", required=True)
    p_train.add_argument("--outdir", default="models_prob")
    p_train.add_argument("--gpu", action="store_true")
    p_train.add_argument("--seed", type=int, default=42)

    # No predict CLI; use risk_inference.py for inference

    args = parser.parse_args()
    if args.cmd == "train":
        metrics = train_probability(
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


