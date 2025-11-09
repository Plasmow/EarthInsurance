"""
High-level inference helpers for EarthInsurance models.

Usage example:

from risk_inference import predict_probability, predict_damage, predict_all
p = predict_probability(embedding, lat, lon, time_utc)
d = predict_damage(embedding, lat, lon, time_utc)
all_res = predict_all(embedding, lat, lon, time_utc)

- embedding: list[float] of length 64
- lat, lon: float
- time_utc: str in 'YYYY-MM-DD HH:MM:SS+HH:MM' (or with trailing 'Z')

This module loads the trained weights from models_prob/ and models_damage/ by default,
recomputes the exact feature set (embedding + geospatial + cyclical time features), and
returns model outputs.
"""
from __future__ import annotations
import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb

EMBEDDING_DIM = 64
STRICT_FORMAT = "%Y-%m-%d %H:%M:%S%z"


def _parse_time_strict(s: Union[str, datetime]) -> datetime:
    if isinstance(s, datetime):
        return s.astimezone(timezone.utc).replace(tzinfo=None) if s.tzinfo else s
    raw = str(s).strip()
    if not raw:
        raise ValueError("Empty time string")
    try:
        dt = datetime.strptime(raw, STRICT_FORMAT)
    except ValueError as exc:
        if raw.endswith('Z'):
            dt = datetime.strptime(raw[:-1] + '+00:00', STRICT_FORMAT)
        else:
            raise ValueError(f"Expected 'YYYY-MM-DD HH:MM:SS+HH:MM', got '{s}'") from exc
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _cyc_features(ts: datetime) -> Tuple[float, float, float, float, float, float]:
    import math
    m_angle = 2.0 * math.pi * (ts.month - 1) / 12.0
    d_angle = 2.0 * math.pi * (ts.timetuple().tm_yday - 1) / 365.25
    h_angle = 2.0 * math.pi * ts.hour / 24.0
    return (
        math.sin(m_angle), math.cos(m_angle),
        math.sin(d_angle), math.cos(d_angle),
        math.sin(h_angle), math.cos(h_angle),
    )


def _build_row(embedding: List[float], lat: float, lon: float, time_utc: Union[str, datetime]) -> pd.DataFrame:
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


# Cached loaders
_prob_cache = None  # (clf, feature_names)
_dmg_cache = None   # (reg_f, reg_w, feature_names)


def _resolve_dir(model_dir: str) -> str:
    p = Path(model_dir)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return str(p)


def _load_prob(model_dir: str = "models_prob"):
    global _prob_cache
    if _prob_cache is not None:
        return _prob_cache
    base = _resolve_dir(model_dir)
    model_path = os.path.join(base, "tornado_prob_xgb.json")
    meta_path = os.path.join(base, "preprocess.json")
    clf = XGBClassifier()
    clf.load_model(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    _prob_cache = (clf, meta["feature_names"])
    return _prob_cache


def _load_dmg(model_dir: str = "models_damage"):
    global _dmg_cache
    if _dmg_cache is not None:
        return _dmg_cache
    base = _resolve_dir(model_dir)
    m_path = os.path.join(base, "tornado_magnitude_xgb.json")
    meta_path = os.path.join(base, "preprocess.json")
    reg_m = XGBRegressor(); reg_m.load_model(m_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    _dmg_cache = (reg_m, meta["feature_names"])
    return _dmg_cache


def predict_probability(
    embedding: List[float],
    lat: float,
    lon: float,
    time_utc: Union[str, datetime],
    model_prob_dir: str = "models_prob",
) -> float:
    clf, feature_names = _load_prob(model_prob_dir)
    X = _build_row(embedding, lat, lon, time_utc).reindex(columns=feature_names, fill_value=0.0)
    # Prefer using the Booster directly (robust after load_model), fallback to sklearn predict_proba
    try:
        booster = clf.get_booster()
        dmat = xgb.DMatrix(X)
        pred = booster.predict(dmat)
        # binary:logistic -> (n_samples,) or (n_samples, 1); if unexpected 2D with >1, take class 1
        if isinstance(pred, np.ndarray):
            if pred.ndim == 2:
                if pred.shape[1] == 1:
                    return float(pred[0, 0])
                return float(pred[0, 1])
            return float(pred[0])
        return float(np.asarray(pred).ravel()[0])
    except Exception:
        # Some xgboost versions lose sklearn wrapper attrs after load_model; set minimal attrs to enable predict_proba
        if not hasattr(clf, "n_classes_"):
            clf.n_classes_ = 2
        if not hasattr(clf, "classes_"):
            clf.classes_ = np.array([0, 1], dtype=np.int64)
        return float(clf.predict_proba(X)[:, 1][0])





essential_damage_keys = ("tornado_magnitude")


def predict_damage(
    embedding: List[float],
    lat: float,
    lon: float,
    time_utc: Union[str, datetime],
    model_damage_dir: str = "models_damage",
) -> Dict[str, float]:
    reg_m, feature_names = _load_dmg(model_damage_dir)
    X = _build_row(embedding, lat, lon, time_utc).reindex(columns=feature_names, fill_value=0.0)
    return {
        "tornado_magnitude": float(reg_m.predict(X)[0]),
    }


def predict_all(
    embedding: List[float],
    lat: float,
    lon: float,
    time_utc: Union[str, datetime],
    model_prob_dir: str = "models_prob",
    model_damage_dir: str = "models_damage",
) -> Dict[str, float]:
    p = predict_probability(embedding, lat, lon, time_utc, model_prob_dir)
    out = {"probability": p}
    try:
        d = predict_damage(embedding, lat, lon, time_utc, model_damage_dir)
        out.update(d)
    except Exception:
        # Damage models optional; if not present, return probability only
        pass
    return out



embedding = np.random.rand(64).astype(float).tolist()


d = predict_probability(
embedding=embedding,
lat=35.47,
lon=-97.52,
time_utc="2025-05-02 14:30:00+00:00",
model_prob_dir="models_prob",
)

m=predict_damage(
embedding=embedding,
lat=35.47,
lon=-97.52,
time_utc="2025-05-02 14:30:00+00:00",
model_damage_dir="models_damage",
)

print(d) # float probability value


print(m) # {"tornado_magnitude": float}