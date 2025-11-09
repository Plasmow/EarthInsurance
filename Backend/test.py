#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Génère un CSV d'entraînement pour prédiction de tornades avec magnitude.

POSITIFS: Vecteurs AlphaEarth de l'année AVANT chaque tornade de events.csv
NÉGATIFS: Points aléatoires aux USA (même distribution temporelle)

Format de sortie: lat, lon, time_utc, f1, f2, ..., f64, label, magnitude
  label = 1 (tornade), 0 (pas de tornade)
  magnitude = 0-5 (échelle EF) pour les tornades, 0 pour les points négatifs
"""

import ee
import sys
import csv
import random
import math
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import List, Optional

# Configuration AlphaEarth
COLLECTION_ID = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
BAND_PREFIX = 'A'  # Les bandes sont A00, A01, ..., A63
DIMS = 64
SCALE_M = 30.0

# Bounding box USA
US_LAT_MIN, US_LAT_MAX = 24.5, 49.5
US_LON_MIN, US_LON_MAX = -125.0, -66.0

def log(msg):
    """Affiche un message avec timestamp."""
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)

def init_gee():
    """Initialise Google Earth Engine."""
    log("Initialisation GEE...")
    try:
        ee.Initialize()
        log("✅ GEE initialisé")
    except Exception:
        log("Authentification nécessaire...")
        ee.Authenticate()
        ee.Initialize(project='gen-lang-client-0546266030')
        log("✅ Authentification réussie")

def parse_datetime(s: str) -> Optional[datetime]:
    """Parse une date UTC."""
    s = (s or "").strip()
    if not s:
        return None
    
    try:
        if "T" in s:
            if s.endswith("Z"):
                return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
            dt = datetime.fromisoformat(s)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        else:
            return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None

def parse_magnitude(mag_str: str) -> int:
    """
    Parse la magnitude d'une tornade depuis la colonne TOR_F_SCALE ou similaire.
    Retourne un entier entre 0 et 5 (échelle EF).
    """
    if not mag_str or mag_str.strip() == "":
        return 0
    
    mag_str = mag_str.strip().upper()
    
    # Enlever les préfixes comme "EF", "F", etc.
    if mag_str.startswith("EF"):
        mag_str = mag_str[2:]
    elif mag_str.startswith("F"):
        mag_str = mag_str[1:]
    
    try:
        mag = int(mag_str)
        # Clamp entre 0 et 5
        return max(0, min(5, mag))
    except ValueError:
        return 0

def load_events(csv_path: str):
    """Charge les événements de events.csv avec leur magnitude."""
    log(f"Chargement de {csv_path}...")
    
    try:
        with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fields = reader.fieldnames or []
    except FileNotFoundError:
        print(f"❌ Fichier non trouvé: {csv_path}")
        sys.exit(1)
    
    # Détection des colonnes
    fields_lc = {f.strip().lower(): f for f in fields}
    
    def get_col(candidates):
        for c in candidates:
            if c.lower() in fields_lc:
                return fields_lc[c.lower()]
        return None
    
    col_slat = get_col(['start_lat', 'start latitude', 'slat']) or 'Start_Lat'
    col_slon = get_col(['start_lon', 'start longitude', 'slon']) or 'Start_Lon'
    col_elat = get_col(['end_lat', 'end latitude', 'elat']) or 'End_Lat'
    col_elon = get_col(['end_lon', 'end longitude', 'elon']) or 'End_Lon'
    col_date = get_col(['begin_time_utc', 'date', 'datetime', 'time']) or 'Date'
    col_magnitude = get_col(['tor_f_scale', 'magnitude', 'ef_scale', 'f_scale']) or 'TOR_F_SCALE'
    
    log(f"Colonnes détectées:")
    log(f"  - Latitude: {col_slat}")
    log(f"  - Longitude: {col_slon}")
    log(f"  - Date: {col_date}")
    log(f"  - Magnitude: {col_magnitude}")
    
    # Parser les événements
    events = []
    magnitude_counts = defaultdict(int)
    
    for r in rows:
        try:
            slat = float(r[col_slat]) if r.get(col_slat, "") else None
            slon = float(r[col_slon]) if r.get(col_slon, "") else None
            elat = float(r[col_elat]) if r.get(col_elat, "") else None
            elon = float(r[col_elon]) if r.get(col_elon, "") else None
            dt = parse_datetime(r.get(col_date, ""))
            magnitude = parse_magnitude(r.get(col_magnitude, ""))
            
            if None in (slat, slon, elat, elon) or dt is None:
                continue
            
            # Point central de l'événement
            lat = (slat + elat) / 2.0
            lon = (slon + elon) / 2.0
            
            magnitude_counts[magnitude] += 1
            
            events.append({
                'lat': lat,
                'lon': lon,
                'time': dt,
                'year': dt.year,
                'label': 1,  # Positif
                'magnitude': magnitude
            })
        except Exception as e:
            continue
    
    log(f"✅ {len(events)} événements chargés")
    log(f"Distribution des magnitudes:")
    for mag in sorted(magnitude_counts.keys()):
        log(f"  EF{mag}: {magnitude_counts[mag]} ({magnitude_counts[mag]/len(events)*100:.1f}%)")
    
    return events

def generate_negatives(n_neg: int, events: List[dict], seed: int = 123):
    """Génère des points négatifs aléatoires (magnitude=0)."""
    log(f"Génération de {n_neg} négatifs...")
    
    random.seed(seed)
    
    # Distribution temporelle des positifs
    if events:
        min_year = min(e['year'] for e in events)
        max_year = max(e['year'] for e in events)
    else:
        min_year, max_year = 2018, 2021
    
    negatives = []
    for _ in range(n_neg):
        lat = random.uniform(US_LAT_MIN, US_LAT_MAX)
        lon = random.uniform(US_LON_MIN, US_LON_MAX)
        year = random.randint(min_year, max_year)
        
        # Date aléatoire dans l'année
        dt = datetime(year, 1, 1, tzinfo=timezone.utc)
        dt += timedelta(days=random.randint(0, 364))
        
        negatives.append({
            'lat': lat,
            'lon': lon,
            'time': dt,
            'year': year,
            'label': 0,  # Négatif
            'magnitude': 0  # Pas de tornade = magnitude 0
        })
    
    log(f"✅ {len(negatives)} négatifs générés")
    return negatives

def get_year_mosaic(year: int):
    """Récupère la mosaïque AlphaEarth pour une année."""
    col = ee.ImageCollection(COLLECTION_ID)
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, 'year')
    
    filtered = col.filterDate(start, end)
    
    # Mosaïque pour combiner toutes les tuiles
    img = filtered.mosaic()
    
    # Fallback si année non disponible
    img = ee.Image(ee.Algorithms.If(
        filtered.size().gt(0),
        img,
        col.filterDate(
            start.advance(-3, 'year'),
            end.advance(3, 'year')
        ).mosaic()
    ))
    
    return ee.Image(img)

def sample_points_by_year(all_points: List[dict], lookback_years: int = 1):
    """
    Échantillonne tous les points, groupés par année.
    
    Pour chaque point, utilise l'image de (year - lookback_years).
    Par défaut lookback_years=1 → image de l'année précédente.
    """
    log(f"Groupement des points par année (lookback={lookback_years})...")
    
    # Grouper par année d'échantillonnage (year - lookback)
    points_by_sample_year = defaultdict(list)
    for idx, pt in enumerate(all_points):
        sample_year = pt['year'] - lookback_years
        points_by_sample_year[sample_year].append((idx, pt))
    
    log(f"Années à échantillonner: {sorted(points_by_sample_year.keys())}")
    
    # Échantillonner année par année
    results = {}
    
    for sample_year in sorted(points_by_sample_year.keys()):
        year_points = points_by_sample_year[sample_year]
        log(f"\n📅 Année {sample_year}: {len(year_points)} points")
        
        # Récupérer l'image
        img = get_year_mosaic(sample_year)
        
        # Vérifier les bandes
        band_names = img.bandNames().getInfo()
        log(f"   Bandes disponibles: {band_names[:5]}... ({len(band_names)} total)")
        
        # Sélectionner les bandes A00-A63 et renommer en f1-f64
        band_list = [f'{BAND_PREFIX}{i:02d}' for i in range(DIMS)]
        img = img.select(band_list).rename([f'f{i}' for i in range(1, DIMS + 1)])
        
        # Créer FeatureCollection
        features = []
        for idx, pt in year_points:
            features.append(ee.Feature(
                ee.Geometry.Point([pt['lon'], pt['lat']]),
                {'idx': idx}
            ))
        
        fc = ee.FeatureCollection(features)
        
        # Échantillonner
        log(f"   Échantillonnage...")
        sampled = img.sampleRegions(
            collection=fc,
            scale=SCALE_M,
            geometries=False,
            tileScale=4
        )
        
        # Récupérer les résultats
        log(f"   Téléchargement...")
        sampled_list = sampled.getInfo()
        
        if sampled_list and 'features' in sampled_list:
            n_results = len(sampled_list['features'])
            log(f"   ✅ {n_results} résultats reçus")
            
            for feat in sampled_list['features']:
                props = feat.get('properties', {})
                idx = props.get('idx')
                if idx is not None:
                    results[idx] = props
        else:
            log(f"   ⚠️  Aucun résultat")
    
    return results

def write_training_csv(all_points: List[dict], results: dict, output_path: str):
    """Écrit le CSV d'entraînement avec magnitude."""
    log(f"\nÉcriture de {output_path}...")
    
    # Header: lat, lon, time_utc, f1, f2, ..., f64, label, magnitude
    header = ['lat', 'lon', 'time_utc'] + [f'f{i}' for i in range(1, DIMS + 1)] + ['label', 'magnitude']
    
    valid_count = 0
    invalid_count = 0
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for idx, pt in enumerate(all_points):
            props = results.get(idx, {})
            
            # Extraire les features
            features = []
            for i in range(1, DIMS + 1):
                val = props.get(f'f{i}')
                if val is not None:
                    try:
                        features.append(float(val))
                    except:
                        features.append(float('nan'))
                else:
                    features.append(float('nan'))
            
            # Vérifier si on a des données valides
            n_valid = sum(1 for f in features if f == f)  # f==f est False pour NaN
            
            if n_valid > 0:
                valid_count += 1
            else:
                invalid_count += 1
            
            # Écrire la ligne avec magnitude
            row = [
                f"{pt['lat']:.6f}",
                f"{pt['lon']:.6f}",
                pt['time'].strftime('%Y-%m-%dT%H:%M:%SZ')
            ] + [f"{f:.6f}" if f == f else "" for f in features] + [pt['label'], pt['magnitude']]
            
            writer.writerow(row)
    
    log(f"✅ Terminé!")
    log(f"   Points valides: {valid_count}/{len(all_points)}")
    log(f"   Points sans données: {invalid_count}/{len(all_points)}")

def main():
    """Fonction principale."""
    print("\n" + "="*80)
    print("  GÉNÉRATION CSV D'ENTRAÎNEMENT - Prédiction de Tornades avec Magnitude")
    print("="*80)
    
    # Configuration
    EVENTS_CSV = 'data/events.csv'
    OUTPUT_CSV = 'data/training2.csv'
    LOOKBACK_YEARS = 1  # Utiliser l'image de l'année AVANT l'événement
    NEG_RATIO = 1.5  # 1.5x plus de négatifs que de positifs
    
    # 1. Initialisation
    init_gee()
    
    # 2. Charger les événements positifs
    positives = load_events(EVENTS_CSV)
    
    if not positives:
        print("❌ Aucun événement valide trouvé")
        sys.exit(1)
    
    # 3. Générer les négatifs
    n_neg = int(len(positives) * NEG_RATIO)
    negatives = generate_negatives(n_neg, positives)
    
    # 4. Combiner tous les points
    all_points = positives + negatives
    log(f"\n📊 Total: {len(all_points)} points ({len(positives)} positifs, {len(negatives)} négatifs)")
    
    # 5. Échantillonner avec AlphaEarth
    results = sample_points_by_year(all_points, lookback_years=LOOKBACK_YEARS)
    
    # 6. Écrire le CSV
    write_training_csv(all_points, results, OUTPUT_CSV)
    
    # 7. Résumé final
    print("\n" + "="*80)
    print("✅ GÉNÉRATION TERMINÉE")
    print("="*80)
    print(f"Fichier de sortie: {OUTPUT_CSV}")
    print(f"Total de points: {len(all_points)}")
    print(f"Positifs (tornades): {len(positives)}")
    print(f"Négatifs (random): {len(negatives)}")
    print(f"Lookback: {LOOKBACK_YEARS} an(s)")
    print(f"Nouvelle colonne: magnitude (0-5)")
    print()

if __name__ == "__main__":
    main()