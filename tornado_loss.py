#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
risk_point_scorer.py
--------------------
Given a predictor (lat, lon, t) -> (p, i) with i in EF0..EF5 (or 0..5),
compute the risk score R = p * v(i) where v(i) is an MDR from a simple EF table.

API:
    scorer = RiskScorer(predictor)
    result = scorer.score(lat, lon, date, occupancy="residential")
    # result = {"p": ..., "i": "EFk", "v": ..., "R": ..., ...}

No money/value here. Later you can do: expected_loss = result["R"] * chosen_value
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
from datetime import datetime

# ---- MDR (Mean Damage Ratio) table per occupancy and EF class (MVP simple) ----
MDR_TABLE: Dict[str, Dict[str, float]] = {
    "residential": {"EF0": 0.01,  "EF1": 0.05, "EF2": 0.15, "EF3": 0.35, "EF4": 0.65, "EF5": 0.90},
    "commercial":  {"EF0": 0.005, "EF1": 0.04, "EF2": 0.12, "EF3": 0.30, "EF4": 0.60, "EF5": 0.85},
    "industrial":  {"EF0": 0.005, "EF1": 0.03, "EF2": 0.10, "EF3": 0.25, "EF4": 0.55, "EF5": 0.80},
}

def _norm_occ(x: Optional[str]) -> str:
    if not x: return "residential"
    s = x.strip().lower()
    if s in ("res","residential","house","housing"): return "residential"
    if s in ("com","commercial","retail","office","public","service"): return "commercial"
    if s in ("ind","industrial","factory","warehouse"): return "industrial"
    return "residential"

def _norm_ef(i) -> str:
    s = str(i).strip().upper()
    if s.startswith("EF"):
        n = s[2:]
    else:
        n = s
    try:
        k = max(0, min(5, int(n)))
    except:
        k = 0
    return f"EF{k}"

@dataclass
class RiskScorer:
    """
    Wrap your ML predictor into a simple scorer:
      predictor(lat, lon, date) -> (p, i)  # p in [0,1], i in {EF0..EF5 or 0..5}
    """
    predictor: Callable[[float, float, datetime], Tuple[float, str|int]]
    mdr_table: Dict[str, Dict[str, float]] = None
    default_occupancy: str = "residential"

    def __post_init__(self):
        if self.mdr_table is None:
            self.mdr_table = MDR_TABLE
        self.default_occupancy = _norm_occ(self.default_occupancy)

    def score(self, lat: float, lon: float, date: datetime, occupancy: Optional[str] = None) -> Dict[str, float|str]:
        # 1) Call your model
        p, i = self.predictor(lat, lon, date)
        if not (0.0 <= float(p) <= 1.0):
            raise ValueError("Predictor returned p outside [0,1].")
        ef = _norm_ef(i)

        # 2) Choose occupancy (single flag for MDR table)
        occ = _norm_occ(occupancy or self.default_occupancy)

        # 3) Vulnerability from table and risk score
        v = float(self.mdr_table.get(occ, self.mdr_table["residential"]).get(ef, 0.01))
        R = float(p) * v

        return {
            "lat": float(lat),
            "lon": float(lon),
            "date": date.isoformat(),
            "occupancy": occ,
            "p": float(p),
            "i": ef,
            "v": v,     # MDR(i)
            "R": R      # p * v(i)  (unitless; multiply by any value later)
        }

# ---------------- Demo (replace with your real model) ----------------
def _demo_predictor(lat: float, lon: float, date: datetime):
    # Example only: you will replace this with your real (lat,lon,t)->(p,i)
    # Here we just fabricate a seasonal p and a class:
    month = date.month
    p = 0.12 if month in (4,5) else (0.08 if month in (3,6) else 0.03)
    i = "EF3" if p >= 0.12 else ("EF2" if p >= 0.08 else "EF1")
    return p, i

if __name__ == "__main__":
    import json, argparse
    ap = argparse.ArgumentParser(description="Compute R = p * v(EF) for a single point.")
    ap.add_argument("--lat", type=float, required=True, default=36.8)
    ap.add_argument("--lon", type=float, required=True, default=-97.5)
    ap.add_argument("--date", type=str, required=True, help="YYYY-MM-DD", default="2024-05-15")
    ap.add_argument("--occupancy", type=str, default="residential",
                    help="residential|commercial|industrial")
    args = ap.parse_args()

    scorer = RiskScorer(predictor=_demo_predictor, default_occupancy=args.occupancy)
    result = scorer.score(args.lat, args.lon, datetime.fromisoformat(args.date), occupancy=args.occupancy)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Later in your pipeline, multiply by any chosen price/value:
    # expected_loss = result["R"] * chosen_value_eur
