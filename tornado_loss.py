#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tornado_loss_mvp.py
-------------------
MVP "point API" pour estimer la perte attendue d'une tornade à partir de (lat, lon, date).

Formule:
    E[L] = p * value_eur * MDR(i)
où:
    - p : probabilité de tornade (fournie par ton modèle via 'predictor')
    - i : intensité en classe EF0..EF5 (fournie par 'predictor')
    - value_eur : valeur assurée exposée, estimée ici via un proxy simple:
        value ≈ population_buffer * (m2/personne) * (coût_moyen_USD/m2) * (taux_assuré) * FX(USD→EUR)
      (avec fallback constant si GEE/WorldPop indisponible)
    - MDR(i) : ratio moyen de dommage (table fixe pour MVP)
    
Ce fichier fournit :
    - TornadoLossEstimator: classe avec méthode estimate(lat, lon, date, occupancy)
    - value_provider_population_proxy(): provider basé sur WorldPop via GEE (si dispo)
    - value_provider_constant(): fallback constant par type d’occupation
    - Un main de démonstration en CLI

Dépendances:
    - Optionnel: earthengine-api (si tu veux activer le provider WorldPop)
        pip install earthengine-api
    - Aucune dépendance obligatoire côté runtime (hors standard lib)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
from datetime import datetime
import math
import json
import logging

# ========================
# Paramètres "moyenne USA"
# ========================

# Taux fixe USD→EUR pour MVP (pas de FX temps réel pour rester simple)
FX_USD_TO_EUR = 0.93

# Coûts moyens de reconstruction US par m² (approx., toutes taxes/soft costs non incluses)
# NB: valeurs indicatives pour MVP; à ajuster si tu as de meilleures sources internes.
COST_USD_PER_M2 = {
    "residential": 1900.0,   # ~175–200 $/ft²
    "commercial":  1700.0,   # dépend bcp du type; valeur moyenne
    "industrial":  1400.0,   # shells métalliques/entrepôts plus "légers"
}

# Conversion en EUR
COST_EUR_PER_M2 = {k: v * FX_USD_TO_EUR for k, v in COST_USD_PER_M2.items()}

# Surface "m² par personne" (proxy simple)
M2_PER_PERSON = {
    "residential": 36.0,    # approx. 35–40 m²/pers.
    "commercial":  10.0,    # surface utile ramenée au staff/clients
    "industrial":   8.0,
}

# Part réellement assurée (MVP)
INSURED_SHARE = 0.60

# Rayon du buffer pour l'estimation population (m)
DEFAULT_POP_BUFFER_M = 200

# ========================
# Table MDR (vulnérabilité)
# ========================

# Valeurs "HAZUS-like" simplifiées pour MVP, monotones, par classe EF
MDR_TABLE: Dict[str, Dict[str, float]] = {
    "residential": {"EF0":0.01,"EF1":0.05,"EF2":0.15,"EF3":0.35,"EF4":0.65,"EF5":0.90},
    "commercial":  {"EF0":0.005,"EF1":0.04,"EF2":0.12,"EF3":0.30,"EF4":0.60,"EF5":0.85},
    "industrial":  {"EF0":0.005,"EF1":0.03,"EF2":0.10,"EF3":0.25,"EF4":0.55,"EF5":0.80},
}

# ========================
# Helpers normalisation
# ========================

def _norm_occ(x: Optional[str]) -> str:
    """Normalise l'occupancy en {residential,commercial,industrial}."""
    if not x:
        return "residential"
    s = x.strip().lower()
    if s in ("res","residential","house","housing"): return "residential"
    if s in ("com","commercial","retail","office","public"): return "commercial"
    if s in ("ind","industrial","factory","warehouse"): return "industrial"
    return "residential"

def _norm_ef(i: str|int) -> str:
    """Normalise la classe EF en 'EF0'..'EF5'."""
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

def _apply_policy(loss_gross: float,
                  deductible_eur: float = 0.0,
                  limit_eur: Optional[float] = None,
                  coinsurance: float = 1.0) -> float:
    """Applique franchise, limite et coassurance (MVP)."""
    d = max(0.0, float(deductible_eur))
    c = max(0.0, min(1.0, float(coinsurance)))
    covered = max(loss_gross - d, 0.0)
    if limit_eur is not None and limit_eur >= 0.0:
        covered = min(covered, float(limit_eur))
    return c * covered

# ========================
# Providers de valeur
# ========================

def value_provider_constant(lat: float, lon: float, date: datetime, occupancy: str = "residential") -> float:
    """
    Fallback immédiat: renvoie une valeur *constante* par type d'occupancy,
    calibrée grossièrement à partir de coûts moyens US et d'une "taille" implicite.
    Utile si GEE/WorldPop n'est pas dispo.
    """
    occ = _norm_occ(occupancy)
    # "Taille" implicite (m²) pour donner un ordre de grandeur
    IMPLIED_AREA_M2 = {
        "residential": 180.0,   # ~ 1900 ft²
        "commercial":  1000.0,  # petite surface commerciale
        "industrial":  1500.0,  # entrepôt modeste
    }
    area = IMPLIED_AREA_M2[occ]
    cost_m2 = COST_EUR_PER_M2.get(occ, COST_EUR_PER_M2["residential"])
    return area * cost_m2 * INSURED_SHARE

def value_provider_population_proxy(lat: float,
                                    lon: float,
                                    date: datetime,
                                    occupancy: str = "residential",
                                    radius_m: int = DEFAULT_POP_BUFFER_M) -> float:
    """
    Provider "data-driven" léger:
        value ≈ pop_buffer × m2/person × cost_per_m2(USA) × insured_share.
    Utilise WorldPop depuis GEE si disponible, sinon retombe sur value_provider_constant.
    """
    try:
        import ee  # earthengine-api
        try:
            ee.Initialize()
        except Exception:
            ee.Authenticate()  # peut ouvrir un flow; si impossible, on catchera plus bas
            ee.Initialize()

        occ = _norm_occ(occupancy)
        pt = ee.Geometry.Point([lon, lat])
        buf = pt.buffer(radius_m)

        # Choisit l'année WorldPop la plus proche, bornée [2000, 2025]
        year = min(max(int(date.year), 2000), 2025)
        col = ee.ImageCollection('WorldPop/GP/100m/pop').filter(ee.Filter.eq('year', year))
        img = ee.Image(col.first())

        # Somme de population dans le buffer (gens)
        pop_sum = img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=buf,
            scale=100,
            maxPixels=1e9
        ).getNumber('population').getInfo()
        if pop_sum is None:
            raise RuntimeError("WorldPop returned None")

        m2pp = M2_PER_PERSON.get(occ, M2_PER_PERSON["residential"])
        cost = COST_EUR_PER_M2.get(occ, COST_EUR_PER_M2["residential"])
        value_eur = float(pop_sum) * m2pp * cost * INSURED_SHARE
        return value_eur

    except Exception as e:
        logging.warning(f"[value_provider_population_proxy] fallback constant due to: {e}")
        return value_provider_constant(lat, lon, date, occupancy)

# ========================
# Estimateur principal
# ========================

@dataclass
class TornadoLossEstimator:
    """
    Estime la perte attendue pour (lat, lon, date).

    Tu fournis:
      - predictor(lat, lon, date) -> (p, i)   # p in [0,1], i in {EF0..EF5 ou 0..5}
      - value_provider(lat, lon, date, occupancy) -> value_eur (EUR)
    Optionnel:
      - occupancy par défaut (ou un provider d'occupancy si tu préfères)
      - paramètres de police (franchise, limite, coassurance)

    Méthode:
      estimate(lat, lon, date, occupancy="residential") -> dict
    """
    predictor: Callable[[float, float, datetime], Tuple[float, str|int]]
    value_provider: Callable[[float, float, datetime, str], float] = value_provider_population_proxy
    default_occupancy: str = "residential"
    deductible_eur: float = 0.0
    limit_eur: Optional[float] = None
    coinsurance: float = 1.0
    mdr_table: Dict[str, Dict[str, float]] = None

    def __post_init__(self):
        if self.mdr_table is None:
            self.mdr_table = MDR_TABLE
        self.default_occupancy = _norm_occ(self.default_occupancy)

    def estimate(self, lat: float, lon: float, date: datetime, occupancy: Optional[str] = None) -> Dict[str, float|str]:
        # 1) Inputs du modèle d'aléa
        p, i = self.predictor(lat, lon, date)  # p \in [0,1], i classe EF
        ef = _norm_ef(i)

        # 2) Occupancy (défaut si non fourni)
        occ = _norm_occ(occupancy or self.default_occupancy)

        # 3) Valeur exposée (EUR)
        value_eur = float(self.value_provider(lat, lon, date, occ))

        # 4) Vulnérabilité (MDR)
        mdr = float(self.mdr_table.get(occ, self.mdr_table["residential"]).get(ef, 0.01))

        # 5) Pertes
        loss_gross = value_eur * mdr
        loss_net = _apply_policy(loss_gross, self.deductible_eur, self.limit_eur, self.coinsurance)
        expected_loss = float(p) * loss_net

        return {
            "lat": float(lat),
            "lon": float(lon),
            "date": date.isoformat(),
            "occupancy": occ,
            "p": float(p),
            "i": ef,
            "value_eur": value_eur,
            "mdr": mdr,
            "loss_gross_eur": loss_gross,
            "loss_net_eur": loss_net,
            "expected_loss_eur": expected_loss,
            "params": {
                "cost_eur_per_m2": COST_EUR_PER_M2.get(occ),
                "m2_per_person": M2_PER_PERSON.get(occ),
                "insured_share": INSURED_SHARE,
                "usd_to_eur_fx": FX_USD_TO_EUR,
                "pop_buffer_m": DEFAULT_POP_BUFFER_M
            }
        }

# ========================
# Démo CLI
# ========================

def _demo_predictor(lat: float, lon: float, date: datetime) -> Tuple[float, str]:
    """
    Stub de prédicteur: remplace par ton modèle ML réel.
    Retourne une proba p et une classe EF.
    """
    # Exemples très simples selon le mois (juste pour la démo)
    month = date.month
    if month in (4, 5):    # saisonnalité fictive
        return 0.15, "EF3"
    if month in (3, 6):
        return 0.10, "EF2"
    return 0.04, "EF1"

def _main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    ap = argparse.ArgumentParser(description="MVP Expected Tornado Loss for a single (lat, lon, date)")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--occupancy", type=str, default="residential", help="residential|commercial|industrial")
    ap.add_argument("--deductible", type=float, default=0.0)
    ap.add_argument("--limit", type=float, default=None)
    ap.add_argument("--coinsurance", type=float, default=1.0)
    args = ap.parse_args()

    date = datetime.fromisoformat(args.date)

    est = TornadoLossEstimator(
        predictor=_demo_predictor,  # 👉 remplace par ton modèle
        value_provider=value_provider_population_proxy,  # essaie GEE; sinon fallback constant
        default_occupancy=args.occupancy,
        deductible_eur=args.deductible,
        limit_eur=args.limit,
        coinsurance=args.coinsurance
    )

    result = est.estimate(args.lat, args.lon, date, occupancy=args.occupancy)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    _main()
