"""High-level inference helpers for EarthInsurance models.

Now supports:
    - Occurrence probability (binary classifier)
    - Magnitude distribution (multiclass 0..5) OR legacy regression magnitude

Usage example:
    from risk_inference import predict_probability, predict_damage, predict_all
    p = predict_probability(embedding, lat, lon, time_utc)  # float in [0,1]
    d = predict_damage(embedding, lat, lon, time_utc)       # {"magnitude_probs": [...]} or legacy magnitude
    all_res = predict_all(embedding, lat, lon, time_utc)    # merged dict

Inputs:
    - embedding: list[float] length 64
    - lat, lon: float
    - time_utc: 'YYYY-MM-DD HH:MM:SS+HH:MM' (or trailing 'Z')

If only a legacy regression magnitude model exists, we convert its output to a one-hot
probability vector.
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


_prob_cache = None    # (clf, feature_names)
_magn_cache = None    # (model, feature_names, class_labels, model_type)


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


def _load_magnitude(model_dir: str = "models_damage"):
    global _magn_cache
    if _magn_cache is not None:
        return _magn_cache
    base = _resolve_dir(model_dir)
    meta_path = os.path.join(base, "preprocess.json")
    cls_path = os.path.join(base, "tornado_magnitude_cls_xgb.json")
    reg_path = os.path.join(base, "tornado_magnitude_xgb.json")  # legacy regression
    if not os.path.exists(meta_path):
        raise FileNotFoundError("Missing preprocess.json for magnitude model")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = meta.get("feature_names")
    class_labels = meta.get("class_labels", [0,1,2,3,4,5])
    if os.path.exists(cls_path):
        model = XGBClassifier(); model.load_model(cls_path)
        _magn_cache = (model, feature_names, class_labels, "cls")
    elif os.path.exists(reg_path):
        model = XGBRegressor(); model.load_model(reg_path)
        _magn_cache = (model, feature_names, class_labels, "reg")
    else:
        raise FileNotFoundError("No magnitude model found (expected classifier or legacy regressor)")
    return _magn_cache


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





def predict_damage(
    embedding: List[float],
    lat: float,
    lon: float,
    time_utc: Union[str, datetime],
    model_damage_dir: str = "models_damage",
) -> Dict[str, float]:
    model, feature_names, class_labels, model_type = _load_magnitude(model_damage_dir)
    X = _build_row(embedding, lat, lon, time_utc).reindex(columns=feature_names, fill_value=0.0)
    if model_type == "cls":
        try:
            booster = model.get_booster(); proba = booster.predict(xgb.DMatrix(X))
            probs = np.asarray(proba)[0]
        except Exception:
            probs = np.asarray(model.predict_proba(X))[0]
        probs = probs / probs.sum() if probs.sum() > 0 else probs
        return {"magnitude_probs": [float(p) for p in probs.tolist()]}
    # legacy regression fallback
    try:
        booster = model.get_booster(); mag_val = float(np.asarray(booster.predict(xgb.DMatrix(X))).ravel()[0])
    except Exception:
        mag_val = float(np.asarray(model.predict(X)).ravel()[0])
    mag_cls = int(np.clip(round(mag_val), 0, len(class_labels)-1))
    one_hot = [0.0]*len(class_labels); one_hot[mag_cls] = 1.0
    return {"magnitude_probs": one_hot, "magnitude": float(mag_cls)}


def predict_all(
    embedding: List[float],
    lat: float,
    lon: float,
    time_utc: Union[str, datetime],
    model_prob_dir: str = "models_prob",
    model_damage_dir: str = "models_damage",
) -> Dict[str, float]:
    p = predict_probability(embedding, lat, lon, time_utc, model_prob_dir)
    out: Dict[str, float] = {"probability": p}
    try:
        d = predict_damage(embedding, lat, lon, time_utc, model_damage_dir)
        out.update(d)
        if "magnitude_probs" in d:
            probs = np.asarray(d["magnitude_probs"]).ravel()
            if probs.size > 0:
                out["magnitude"] = int(np.argmax(probs))
    except Exception:
        pass
    return out



# No module-level test execution; import-only module.