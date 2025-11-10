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
from random import sample
from typing import Dict, List, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb
from embedding_match import get_alphaearth_record

EMBEDDING_DIM = 64
STRICT_FORMAT = "%Y-%m-%d %H:%M:%S%z"  # kept for backward compat (space variant)


def _parse_time_strict(s: Union[str, datetime]) -> datetime:
    """Parse UTC-aware timestamp to naive UTC datetime.

    Accepts:
      - 'YYYY-MM-DD HH:MM:SS+HH:MM'
      - 'YYYY-MM-DDTHH:MM:SS+HH:MM'
      - same with trailing 'Z' meaning UTC
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
    """Resolve a model directory relative to this file, with parent fallback.

    Priority:
      1) <this_dir>/<model_dir>
      2) <this_dir>/../<model_dir>  (repo-level models_* when running from Backend/)
    """
    base_dir = Path(__file__).resolve().parent
    candidates = []
    p = Path(model_dir)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(base_dir / p)
        candidates.append(base_dir.parent / p)
    for c in candidates:
        if c.exists():
            return str(c)
    # fallback to first candidate even if missing (will surface clearer error later)
    return str(candidates[0])


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
    base_dir = Path(__file__).resolve().parent
    repo_dir = base_dir.parent
    candidates = [
        base_dir / model_dir,           # Backend/models_damage
        repo_dir / model_dir,           # repo-level models_damage
    ]
    # Resolve meta and model paths for both locations
    locations = []
    for b in candidates:
        b = b.resolve()
        meta = b / "preprocess.json"
        cls = b / "tornado_magnitude_cls_xgb.json"
        reg = b / "tornado_magnitude_xgb.json"
        locations.append((b, meta, cls, reg))
    # Prefer classifier if present in any location; else use first available regressor
    chosen = None
    for b, meta, cls, reg in locations:
        if cls.exists():
            chosen = (b, meta, cls, "cls")
            break
    if chosen is None:
        for b, meta, cls, reg in locations:
            if reg.exists():
                chosen = (b, meta, reg, "reg")
                break
    if chosen is None:
        raise FileNotFoundError("No magnitude model found (expected classifier or legacy regressor)")
    base, meta_path, model_path, mtype = chosen
    if not os.path.exists(meta_path):
        raise FileNotFoundError("Missing preprocess.json for magnitude model")
    with open(str(meta_path), "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = meta.get("feature_names")
    class_labels = meta.get("class_labels", [0,1,2,3,4,5])
    if mtype == "cls":
        model = XGBClassifier(); model.load_model(str(model_path))
        _magn_cache = (model, feature_names, class_labels, "cls")
    else:
        model = XGBRegressor(); model.load_model(str(model_path))
        _magn_cache = (model, feature_names, class_labels, "reg")
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
        # Prefer sklearn wrapper predict_proba to avoid potential DMatrix feature alignment quirks
        try:
            probs = np.asarray(model.predict_proba(X))[0]
        except Exception:
            booster = model.get_booster(); proba = booster.predict(xgb.DMatrix(X))
            probs = np.asarray(proba)[0]
        probs = probs / probs.sum() if probs.sum() > 0 else probs
        return {"magnitude_probs": [float(p) for p in probs.tolist()]}
    # legacy regression fallback -> smooth probabilistic distribution around predicted magnitude
    try:
        booster = model.get_booster(); mag_val = float(np.asarray(booster.predict(xgb.DMatrix(X))).ravel()[0])
    except Exception:
        mag_val = float(np.asarray(model.predict(X)).ravel()[0])
    # Gaussian smoothing across discrete classes
    k = np.arange(len(class_labels), dtype=float)
    sigma = 0.85  # spread; tune if needed
    weights = np.exp(-0.5 * ((k - mag_val) / max(sigma, 1e-6))**2)
    probs = weights / weights.sum() if weights.sum() > 0 else weights
    mag_cls = int(np.clip(np.argmax(probs), 0, len(class_labels)-1))
    return {"magnitude_probs": [float(p) for p in probs.tolist()], "magnitude": float(mag_cls)}


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





def manual_example():  # pragma: no cover (optional helper)
    embedding=get_alphaearth_record(
        lat=37.43685,
        lon=-91.9,
        when="2023-05-02"
    )["embedding"]

    res = predict_all(embedding, 37.43685, -91.9, "2023-05-02 00:00:00+00:00")
    print({
        "source": "manual_example",
        "input": {"lat": 37.43685, "lon": -91.9, "time_utc": "2023-05-02"},
        "result": res
    })

if __name__ == "__main__":  # only run when executed directly, not on import
    manual_example()