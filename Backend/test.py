#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fusionne les datasets de tornades et de points aléatoires.
Filtre les lignes avec embeddings vides.

Input:
  - data/events_with_vectors.csv (tornades, label=1)
  - data/random_points_with_vectors.csv (points aléatoires, label=0)

Output:
  - data/combined_dataset.csv (dataset fusionné et filtré)
"""

import pandas as pd
import numpy as np
from datetime import datetime

def log(msg):
    """Affiche un message avec timestamp."""
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)

def load_and_check(csv_path: str, dataset_name: str):
    """Charge un CSV et affiche des stats."""
    log(f"\nChargement de {csv_path}...")
    
    df = pd.read_csv(csv_path)
    
    log(f"  Forme: {df.shape}")
    log(f"  Colonnes: {list(df.columns[:5])}... (total: {len(df.columns)})")
    
    # Compter les lignes avec des valeurs manquantes dans les features
    feature_cols = [col for col in df.columns if col.startswith('f')]
    
    # Vérifier combien de lignes ont au moins un NaN dans les features
    rows_with_nan = df[feature_cols].isna().any(axis=1).sum()
    rows_without_nan = len(df) - rows_with_nan
    
    log(f"  Lignes valides (avec embeddings): {rows_without_nan}")
    log(f"  Lignes invalides (sans embeddings): {rows_with_nan}")
    
    # Stats sur les labels
    if 'label' in df.columns:
        log(f"  Distribution labels:")
        for label, count in df['label'].value_counts().items():
            log(f"    label={label}: {count} ({count/len(df)*100:.1f}%)")
    
    # Stats sur les magnitudes si c'est le dataset de tornades
    if 'magnitude' in df.columns and dataset_name == "Tornades":
        log(f"  Distribution magnitudes:")
        for mag, count in df['magnitude'].value_counts().sort_index().items():
            log(f"    EF{mag}: {count} ({count/len(df)*100:.1f}%)")
    
    return df

def filter_valid_embeddings(df, dataset_name: str):
    """Filtre les lignes avec embeddings valides."""
    log(f"\nFiltrage de {dataset_name}...")
    
    # Colonnes de features
    feature_cols = [col for col in df.columns if col.startswith('f')]
    
    # Garder seulement les lignes sans NaN dans les features
    df_filtered = df.dropna(subset=feature_cols)
    
    removed = len(df) - len(df_filtered)
    log(f"  Lignes gardées: {len(df_filtered)}/{len(df)}")
    log(f"  Lignes supprimées: {removed} ({removed/len(df)*100:.1f}%)")
    
    return df_filtered

def combine_datasets(df_tornades, df_random):
    """Fusionne les deux datasets."""
    log(f"\nFusion des datasets...")
    
    df_combined = pd.concat([df_tornades, df_random], ignore_index=True)
    
    log(f"  Taille combinée: {len(df_combined)} lignes")
    log(f"    Tornades (label=1): {(df_combined['label'] == 1).sum()}")
    log(f"    Points aléatoires (label=0): {(df_combined['label'] == 0).sum()}")
    
    # Mélanger les lignes
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    log(f"  Dataset mélangé (random_state=42)")
    
    return df_combined

def main():
    """Fonction principale."""
    print("\n" + "="*80)
    print("  FUSION DES DATASETS - Tornades + Points Aléatoires")
    print("="*80)
    
    # Chemins des fichiers
    TORNADES_CSV = 'data/events_with_vectors.csv'
    RANDOM_CSV = 'data/random_points_with_vectors.csv'
    OUTPUT_CSV = 'data/combined_dataset.csv'
    
    # 1. Charger les datasets
    df_tornades = load_and_check(TORNADES_CSV, "Tornades")
    df_random = load_and_check(RANDOM_CSV, "Points aléatoires")
    
    # 2. Filtrer les embeddings vides
    df_tornades_filtered = filter_valid_embeddings(df_tornades, "Tornades")
    df_random_filtered = filter_valid_embeddings(df_random, "Points aléatoires")
    
    # 3. Fusionner
    df_combined = combine_datasets(df_tornades_filtered, df_random_filtered)
    
    # 4. Sauvegarder
    log(f"\nÉcriture de {OUTPUT_CSV}...")
    df_combined.to_csv(OUTPUT_CSV, index=False)
    
    # 5. Résumé final
    print("\n" + "="*80)
    print("✅ FUSION TERMINÉE")
    print("="*80)
    print(f"Fichier d'entrée 1: {TORNADES_CSV}")
    print(f"Fichier d'entrée 2: {RANDOM_CSV}")
    print(f"Fichier de sortie: {OUTPUT_CSV}")
    print(f"\nDataset final:")
    print(f"  Total: {len(df_combined)} lignes")
    print(f"  Tornades (label=1): {(df_combined['label'] == 1).sum()}")
    print(f"  Points négatifs (label=0): {(df_combined['label'] == 0).sum()}")
    print(f"  Ratio: {(df_combined['label'] == 0).sum() / (df_combined['label'] == 1).sum():.2f}:1 (négatif:positif)")
    
    # Distribution des magnitudes
    if 'magnitude' in df_combined.columns:
        print(f"\nDistribution des magnitudes (tornades uniquement):")
        tornades_only = df_combined[df_combined['label'] == 1]
        for mag in sorted(tornades_only['magnitude'].unique()):
            count = (tornades_only['magnitude'] == mag).sum()
            print(f"  EF{mag}: {count} ({count/len(tornades_only)*100:.1f}%)")
    
    # Distribution par année
    print(f"\nDistribution par année:")
    df_combined['year'] = pd.to_datetime(df_combined['time_utc']).dt.year
    for year in sorted(df_combined['year'].unique()):
        count = (df_combined['year'] == year).sum()
        tornades_count = ((df_combined['year'] == year) & (df_combined['label'] == 1)).sum()
        random_count = ((df_combined['year'] == year) & (df_combined['label'] == 0)).sum()
        print(f"  {year}: {count} lignes ({tornades_count} tornades, {random_count} négatifs)")
    
    print()

if __name__ == "__main__":
    main()