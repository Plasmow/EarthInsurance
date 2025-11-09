#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Génère un CSV à partir de events.csv avec les vecteurs AlphaEarth.

Pour chaque tornade dans events.csv:
  - Récupère le vecteur AlphaEarth de l'année AVANT l'événement
  - Extrait le label (1 = tornade) et la magnitude (EF_Scale)
  
Format de sortie: lat, lon, time_utc, f1...f64, label, magnitude
"""

import ee
import sys
import csv
from datetime import datetime, timezone
from collections import defaultdict
from typing import List, Dict, Optional

# Configuration AlphaEarth
COLLECTION_ID = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
BAND_PREFIX = 'A'
DIMS = 64
SCALE_M = 30.0

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
            # Format: YYYY-MM-DD HH:MM:SS
            return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None

def parse_magnitude(mag_val) -> int:
    """
    Parse la magnitude d'une tornade.
    Retourne un entier entre 0 et 5 (échelle EF).
    """
    if mag_val is None or str(mag_val).strip() == "":
        return 0
    
    try:
        # Essayer de convertir directement en int
        mag = int(float(mag_val))
        # Clamp entre 0 et 5
        return max(0, min(5, mag))
    except (ValueError, TypeError):
        # Si c'est une chaîne comme "EF2", "F3", etc.
        mag_str = str(mag_val).strip().upper()
        if mag_str.startswith("EF"):
            mag_str = mag_str[2:]
        elif mag_str.startswith("F"):
            mag_str = mag_str[1:]
        
        try:
            mag = int(mag_str)
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
    
    log(f"Colonnes disponibles: {fields}")
    
    # Détection flexible des colonnes
    fields_lc = {f.strip().lower(): f for f in fields}
    
    def get_col(candidates):
        for c in candidates:
            if c.lower() in fields_lc:
                return fields_lc[c.lower()]
        return None
    
    col_slat = get_col(['start_lat', 'start latitude', 'slat', 'latitude', 'lat'])
    col_slon = get_col(['start_lon', 'start longitude', 'slon', 'longitude', 'lon'])
    col_elat = get_col(['end_lat', 'end latitude', 'elat'])
    col_elon = get_col(['end_lon', 'end longitude', 'elon'])
    col_date = get_col(['date', 'begin_time_utc', 'datetime', 'time'])
    
    # Pour la magnitude, essayer plusieurs colonnes
    col_magnitude = get_col(['magnitude', 'ef_scale', 'tor_f_scale', 'f_scale', 'scale'])
    
    log(f"\nColonnes détectées:")
    log(f"  - Start Lat: {col_slat}")
    log(f"  - Start Lon: {col_slon}")
    log(f"  - End Lat: {col_elat}")
    log(f"  - End Lon: {col_elon}")
    log(f"  - Date: {col_date}")
    log(f"  - Magnitude: {col_magnitude}")
    
    if not all([col_slat, col_slon, col_date]):
        print(f"❌ Colonnes essentielles manquantes!")
        sys.exit(1)
    
    # Parser les événements
    events = []
    magnitude_counts = defaultdict(int)
    skipped = 0
    
    for r in rows:
        try:
            slat = float(r[col_slat]) if r.get(col_slat, "") else None
            slon = float(r[col_slon]) if r.get(col_slon, "") else None
            
            # Si on a End_Lat/End_Lon, calculer le point central
            if col_elat and col_elon:
                elat = float(r[col_elat]) if r.get(col_elat, "") else slat
                elon = float(r[col_elon]) if r.get(col_elon, "") else slon
            else:
                elat = slat
                elon = slon
            
            dt = parse_datetime(r.get(col_date, ""))
            
            # Parser la magnitude
            if col_magnitude:
                magnitude = parse_magnitude(r.get(col_magnitude, ""))
            else:
                magnitude = 0
            
            if None in (slat, slon, elat, elon) or dt is None:
                skipped += 1
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
                'label': 1,  # Toujours 1 pour une tornade
                'magnitude': magnitude
            })
        except Exception as e:
            skipped += 1
            continue
    
    log(f"\n✅ {len(events)} événements chargés ({skipped} ignorés)")
    log(f"\nDistribution des magnitudes:")
    for mag in sorted(magnitude_counts.keys()):
        log(f"  EF{mag}: {magnitude_counts[mag]} ({magnitude_counts[mag]/len(events)*100:.1f}%)")
    
    return events

def get_year_mosaic(year: int):
    """Récupère la mosaïque AlphaEarth pour une année."""
    col = ee.ImageCollection(COLLECTION_ID)
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, 'year')
    
    filtered = col.filterDate(start, end)
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
    Échantillonne tous les points avec AlphaEarth, groupés par année.
    
    Pour chaque point, utilise l'image de (year - lookback_years).
    """
    log(f"\nGroupement des points par année (lookback={lookback_years})...")
    
    # Grouper par année d'échantillonnage
    points_by_sample_year = defaultdict(list)
    for idx, pt in enumerate(all_points):
        sample_year = pt['year'] - lookback_years
        points_by_sample_year[sample_year].append((idx, pt))
    
    log(f"Années à échantillonner: {sorted(points_by_sample_year.keys())}")
    
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

def write_output_csv(all_points: List[dict], results: Dict, output_path: str):
    """
    Écrit le CSV de sortie.
    
    Format: lat, lon, time_utc, f1...f64, label, magnitude
    """
    log(f"\nÉcriture de {output_path}...")
    
    # Header
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
            
            # Écrire la ligne avec label et magnitude
            row = [
                f"{pt['lat']:.6f}",
                f"{pt['lon']:.6f}",
                pt['time'].strftime('%Y-%m-%dT%H:%M:%SZ')
            ] + [f"{f:.6f}" if f == f else "" for f in features] + [pt['label'], pt['magnitude']]
            
            writer.writerow(row)
    
    log(f"\n✅ Terminé!")
    log(f"   Points valides: {valid_count}/{len(all_points)}")
    log(f"   Points sans données: {invalid_count}/{len(all_points)}")

def main():
    """Fonction principale."""
    print("\n" + "="*80)
    print("  GÉNÉRATION CSV depuis events.csv - Vecteurs AlphaEarth")
    print("="*80)
    
    # Configuration
    EVENTS_CSV = 'data/events.csv'
    OUTPUT_CSV = 'data/events_with_vectors.csv'
    LOOKBACK_YEARS = 1  # Utiliser l'image de l'année AVANT l'événement
    
    # 1. Initialisation
    init_gee()
    
    # 2. Charger les événements
    events = load_events(EVENTS_CSV)
    
    if not events:
        print("❌ Aucun événement valide trouvé")
        sys.exit(1)
    
    log(f"\n📊 Total: {len(events)} événements (tornades)")
    
    # 3. Échantillonner avec AlphaEarth
    results = sample_points_by_year(events, lookback_years=LOOKBACK_YEARS)
    
    # 4. Écrire le CSV
    write_output_csv(events, results, OUTPUT_CSV)
    
    # 5. Résumé final
    print("\n" + "="*80)
    print("✅ GÉNÉRATION TERMINÉE")
    print("="*80)
    print(f"Fichier d'entrée: {EVENTS_CSV}")
    print(f"Fichier de sortie: {OUTPUT_CSV}")
    print(f"Total de tornades: {len(events)}")
    print(f"Lookback: {LOOKBACK_YEARS} an(s)")
    print(f"\nFormat: lat, lon, time_utc, f1...f64, label, magnitude")
    print(f"  - label: 1 (tornade)")
    print(f"  - magnitude: 0-5 (échelle EF)")
    print()

if __name__ == "__main__":
    main()