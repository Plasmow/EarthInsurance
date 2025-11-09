#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Génère un CSV de points ALÉATOIRES avec les vecteurs AlphaEarth.
VERSION OPTIMISÉE: Une requête par année (3000 pts/an).

Pour chaque point aléatoire:
  - Coordonnées (lat, lon) aléatoires sur les USA
  - Date aléatoire en 2017 ou 2023 (3000 points chacun)
  - Récupère le vecteur AlphaEarth de l'année AVANT
  - label = 0 (pas de tornade)
  - magnitude = 0
  
Format de sortie: lat, lon, time_utc, f1...f64, label, magnitude
"""

import ee
import sys
import csv
import random
import os
from datetime import datetime, timezone
from collections import defaultdict
from typing import List, Dict

# Configuration AlphaEarth
COLLECTION_ID = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
BAND_PREFIX = 'A'
DIMS = 64
SCALE_M = 30.0

# Configuration du dataset aléatoire
NUM_RANDOM_POINTS = 15000  # Nombre de points à générer (3000 par année)
LAT_MIN, LAT_MAX = 25.0, 49.0  # USA continental
LON_MIN, LON_MAX = -125.0, -66.0
ALLOWED_YEARS = [2017,2018,2019,2020,2021,2022, 2023,2024]  # Années spécifiques pour la génération

# Configuration des batches - pas utilisé maintenant, une requête par année
BATCH_SIZE = 500  # Traiter 500 points à la fois
POINTS_PER_YEAR_REQUEST = 100  # Points à échantillonner par requête GEE

def log(msg):
    """Affiche un message avec timestamp."""
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)

def init_gee():
    """Initialise Google Earth Engine."""
    log("Initialisation GEE...")
    try:
        ee.Initialize(project='gen-lang-client-0546266030')
        log("✅ GEE initialisé")
    except Exception:
        log("Authentification nécessaire...")
        ee.Authenticate()
        ee.Initialize(project='gen-lang-client-0546266030')
        log("✅ Authentification réussie")

def generate_random_points(n: int) -> List[dict]:
    """
    Génère n points aléatoires sur le territoire US.
    Équilibre exactement les points entre les années.
    
    Returns:
        Liste de dicts avec lat, lon, time, year, label, magnitude
    """
    log(f"\nGénération de {n} points aléatoires...")
    log(f"  Zone: lat [{LAT_MIN}, {LAT_MAX}], lon [{LON_MIN}, {LON_MAX}]")
    log(f"  Années: {ALLOWED_YEARS}")
    
    points = []
    year_counts = defaultdict(int)
    
    # Calculer le nombre de points par année (équilibré)
    points_per_year = n // len(ALLOWED_YEARS)
    log(f"  Points par année: {points_per_year}")
    
    for year in ALLOWED_YEARS:
        for i in range(points_per_year):
            # Coordonnées aléatoires
            lat = random.uniform(LAT_MIN, LAT_MAX)
            lon = random.uniform(LON_MIN, LON_MAX)
            
            # Date aléatoire dans cette année
            month = random.randint(1, 12)
            day = random.randint(1, 28)  # Simplification pour éviter les jours invalides
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            
            dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
            
            year_counts[year] += 1
            
            points.append({
                'lat': lat,
                'lon': lon,
                'time': dt,
                'year': year,
                'label': 0,  # 0 = pas de tornade
                'magnitude': 0  # 0 = pas de magnitude
            })
    
    log(f"✅ {len(points)} points générés")
    log(f"\nDistribution par année:")
    for year in sorted(year_counts.keys()):
        log(f"  {year}: {year_counts[year]} points ({year_counts[year]/len(points)*100:.1f}%)")
    
    return points

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

def sample_points_by_year_batched(all_points: List[dict], lookback_years: int = 1):
    """
    Échantillonne tous les points avec AlphaEarth.
    UNE SEULE requête par année pour éviter les complications.
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
        
        # Récupérer et préparer l'image
        img = get_year_mosaic(sample_year)
        
        # Vérifier les bandes
        band_names = img.bandNames().getInfo()
        log(f"   Bandes disponibles: {band_names[:5] if band_names else '[]'}... ({len(band_names)} total)")
        
        # Si pas de bandes disponibles, sauter cette année
        if not band_names or len(band_names) == 0:
            log(f"   ⚠️  Année {sample_year} sans données AlphaEarth - ignorée")
            continue
        
        # Sélectionner les bandes A00-A63 et renommer en f1-f64
        band_list = [f'{BAND_PREFIX}{i:02d}' for i in range(DIMS)]
        img = img.select(band_list).rename([f'f{i}' for i in range(1, DIMS + 1)])
        
        # Créer FeatureCollection avec TOUS les points de cette année
        log(f"   Création de la FeatureCollection...")
        features = []
        for idx, pt in year_points:
            features.append(ee.Feature(
                ee.Geometry.Point([pt['lon'], pt['lat']]),
                {'idx': idx}
            ))
        
        fc = ee.FeatureCollection(features)
        
        # UNE SEULE requête d'échantillonnage pour toute l'année
        log(f"   Échantillonnage de {len(year_points)} points en une requête...")
        try:
            sampled = img.sampleRegions(
                collection=fc,
                scale=SCALE_M,
                geometries=False,
                tileScale=4
            )
            
            # Récupérer les résultats
            log(f"   Téléchargement des résultats...")
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
        except Exception as e:
            log(f"   ❌ Erreur lors de l'échantillonnage: {str(e)}")
            continue
    
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
            
            # Écrire la ligne avec label=0 et magnitude=0
            row = [
                f"{pt['lat']:.6f}",
                f"{pt['lon']:.6f}",
                pt['time'].strftime('%Y-%m-%dT%H:%M:%SZ')
            ] + [f"{f:.6f}" if f == f else "" for f in features] + [pt['label'], pt['magnitude']]
            
            writer.writerow(row)
    
    log(f"\n✅ Terminé!")
    log(f"   Points valides: {valid_count}/{len(all_points)} ({valid_count/len(all_points)*100:.1f}%)")
    log(f"   Points sans données: {invalid_count}/{len(all_points)} ({invalid_count/len(all_points)*100:.1f}%)")

def main():
    """Fonction principale."""
    print("\n" + "="*80)
    print("  GÉNÉRATION CSV - Points ALÉATOIRES avec Vecteurs AlphaEarth")
    print("  VERSION OPTIMISÉE: 3000 pts/an, 1 requête par année")
    print("="*80)
    
    # Configuration
    OUTPUT_CSV = 'data/random_points_with_vectors.csv'
    LOOKBACK_YEARS = 1  # Utiliser l'image de l'année AVANT
    
    # Créer le dossier data si nécessaire
    os.makedirs('data', exist_ok=True)
    
    # 1. Initialisation
    init_gee()
    
    # 2. Générer les points aléatoires
    points = generate_random_points(NUM_RANDOM_POINTS)
    
    log(f"\n📊 Total: {len(points)} points aléatoires")
    log(f"📦 Configuration: UNE requête par année (2 requêtes au total)")
    
    # 3. Échantillonner avec AlphaEarth (par batches)
    results = sample_points_by_year_batched(points, lookback_years=LOOKBACK_YEARS)
    
    # 4. Écrire le CSV
    write_output_csv(points, results, OUTPUT_CSV)
    
    # 5. Résumé final
    print("\n" + "="*80)
    print("✅ GÉNÉRATION TERMINÉE")
    print("="*80)
    print(f"Fichier de sortie: {OUTPUT_CSV}")
    print(f"Total de points: {len(points)}")
    print(f"Zone géographique: USA continental")
    print(f"Années: {ALLOWED_YEARS}")
    print(f"Lookback: {LOOKBACK_YEARS} an(s)")
    print(f"\nFormat: lat, lon, time_utc, f1...f64, label, magnitude")
    print(f"  - label: 0 (pas de tornade)")
    print(f"  - magnitude: 0")
    print()

if __name__ == "__main__":
    main()