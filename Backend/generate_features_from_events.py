#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch feature extraction from events.csv using Google Satellite Embedding (AlphaEarth) on GEE.

- Lit data/events.csv, construit les positifs (milieu du segment, horodaté UTC).
- Génère des négatifs aléatoires hors buffer 3 km des événements, et hors fenêtre ±3h (au pas 1h) des événements.
- Concatène (positifs + négatifs) → FeatureCollection avec props {lat, lon, time_utc, year, split}.
- Pour chaque année présente → sampleRegions (64D) côté serveur → Export table (Drive ou GCS).

Dépendances : earthengine-api (ee.Authenticate puis ee.Initialize()).
"""

import sys, os, csv, argparse, time, math, random
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import ee  # pip install earthengine-api

COLLECTION_ID = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
US_LAT_MIN, US_LAT_MAX = 24.5, 49.5
US_LON_MIN, US_LON_MAX = -125.0, -66.0

# Bande(s) explicites: embedding_0..embedding_63
EMB_BANDS = [f'embedding_{i}' for i in range(64)]

def log(msg: str):
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--events', default='data/events.csv', help='Chemin vers events.csv')
    p.add_argument('--out', default='data/features.csv', help='Chemin CSV local si --export local')
    p.add_argument('--dims', type=int, default=64, help='Nb de dimensions à exporter (<=64)')
    p.add_argument('--scale', type=float, default=30.0, help='Scale échantillonnage (m)')
    p.add_argument('--export', choices=['drive','gcs','local'], default='drive',
                   help='Cible export: Google Drive (par défaut), GCS ou local (lent)')
    p.add_argument('--gcs-bucket', default='', help='Bucket GCS si --export gcs')
    p.add_argument('--task-prefix', default='features_alphaearth', help='Préfixe des tâches')
    p.add_argument('--watch', action='store_true', help='Suivre la complétion des tâches EE')
    p.add_argument('--poll-seconds', type=int, default=5, help='Période de polling des tâches')
    p.add_argument('--page-size', type=int, default=2000, help='Taille page pour mode local')
    return p.parse_args()

def safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None

def parse_time_utc(s: str) -> Optional[datetime]:
    s = (s or '').strip()
    if not s:
        return None
    try:
        if 'T' in s:
            if s.endswith('Z'):
                return datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(timezone.utc)
            dt = datetime.fromisoformat(s)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        return datetime.strptime(s, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    except Exception:
        try:
            dt = datetime.fromisoformat(s)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

def main():
    args = parse_args()

    # ---- EE init
    log('Initializing Earth Engine…')
    try:
        ee.Initialize(project='gen-lang-client-0546266030')
    except Exception:
        ee.Authenticate()
        ee.Initialize(project='gen-lang-client-0546266030')
    log('EE ready.')

    # ---- Lecture events.csv
    log(f'Loading events from {args.events} …')
    try:
        with open(args.events, 'r', encoding='utf-8-sig', newline='') as f:
            rdr = csv.DictReader(f)
            rows = list(rdr)
            fns = [c.strip() for c in (rdr.fieldnames or [])]
    except FileNotFoundError:
        print(f'ERROR: not found: {args.events}', file=sys.stderr)
        sys.exit(1)
    if not rows:
        print('ERROR: events.csv empty.', file=sys.stderr)
        sys.exit(1)

    # Détection colonnes
    lc = {c.lower(): c for c in fns}
    def get_col(cands: List[str], default=None):
        for c in cands:
            if c.lower() in lc:
                return lc[c.lower()]
        return default

    col_slat = get_col(['start_lat','start latitude','start_latitude','slat']) or 'Start_Lat'
    col_slon = get_col(['start_lon','start longitude','start_longitude','slon']) or 'Start_Lon'
    col_elat = get_col(['end_lat','end latitude','end_latitude','elat']) or 'End_Lat'
    col_elon = get_col(['end_lon','end longitude','end_longitude','elon']) or 'End_Lon'
    col_date = get_col(['begin_time_utc','date','datetime','time','start_time']) or 'Date'

    # ---- Positifs (milieu du segment)
    pos_feats = []
    years_local = set()
    for r in rows:
        slat, slon = safe_float(r.get(col_slat)), safe_float(r.get(col_slon))
        elat, elon = safe_float(r.get(col_elat)), safe_float(r.get(col_elon))
        dt = parse_time_utc(r.get(col_date))
        if None in (slat, slon, elat, elon) or dt is None:
            continue
        lat = (slat + elat)/2.0
        lon = (slon + elon)/2.0
        y = dt.year
        years_local.add(y)
        props = {
            'lat': lat,
            'lon': lon,
            'time_utc': dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            'year': y,
            'split': 'pos'
        }
        pos_feats.append(ee.Feature(ee.Geometry.Point([lon, lat]), props))
    pos_fc = ee.FeatureCollection(pos_feats)
    n_pos = int(pos_fc.size().getInfo() or 0)
    if n_pos == 0:
        print('ERROR: No valid events after parsing.', file=sys.stderr)
        sys.exit(1)
    log(f'Parsed positives: {n_pos}')

    # ---- Négatifs (hors buffer 3 km, temps hors ±3h)
    # Géométrie sûre
    line_feats = []
    for r in rows:
        slat, slon = safe_float(r.get(col_slat)), safe_float(r.get(col_slon))
        elat, elon = safe_float(r.get(col_elat)), safe_float(r.get(col_elon))
        if None in (slat, slon, elat, elon):
            continue
        line_feats.append(ee.Feature(ee.Geometry.LineString([[slon, slat],[elon, elat]])))
    lines_fc = ee.FeatureCollection(line_feats) if line_feats else ee.FeatureCollection([])
    buffered_union = lines_fc.geometry().buffer(3000)  # 3 km
    us_geom = ee.Geometry.Rectangle([US_LON_MIN, US_LAT_MIN, US_LON_MAX, US_LAT_MAX], None, False)
    safe_region = us_geom.difference(buffered_union, 1)

    # Fenêtre temporelle globale
    pos_times = [parse_time_utc(r.get(col_date)) for r in rows if parse_time_utc(r.get(col_date)) is not None]
    min_dt = min(pos_times)
    max_dt = max(pos_times)
    min_date = ee.Date(min_dt.isoformat())
    span_hours = max(0, int((max_dt - min_dt).total_seconds() // 3600))

    # Heures interdites (±3h) autour des événements
    forbidden = set()
    for dt in pos_times:
        off = int((dt - min_dt).total_seconds() // 3600)
        for dh in range(-3, 4):
            h = off + dh
            if 0 <= h <= span_hours:
                forbidden.add(h)
    allowed = [h for h in range(span_hours + 1) if h not in forbidden] or list(range(span_hours + 1))
    allowed_list = ee.List(allowed)
    allowed_size = allowed_list.size()

    # Nombre de négatifs pour ~40% de positifs
    total_target = int(round(n_pos / 0.4))
    n_neg = max(total_target - n_pos, 0)
    log(f'Generating ~{n_neg} negatives…')

    neg_fc = ee.FeatureCollection.randomPoints(safe_region, n_neg, 123)
    def _neg_props(f):
        coords = f.geometry().coordinates()  # [lon, lat]
        # index déterministe à partir des coords → heure autorisée
        seed = ee.Number(coords.get(0)).multiply(1000000).abs().add(
               ee.Number(coords.get(1)).multiply(1000000).abs()).toInt()
        idx = seed.mod(allowed_size)
        off_h = ee.Number(allowed_list.get(idx))
        dt = min_date.advance(off_h, 'hour')
        return f.set({
            'lat': coords.get(1),
            'lon': coords.get(0),
            'time_utc': dt.format("YYYY-MM-dd'T'HH:mm:ss'Z'"),
            'year': ee.Number.parse(dt.format('YYYY')),
            'split': 'neg'
        })
    neg_fc = neg_fc.map(_neg_props)

    all_fc = pos_fc.merge(neg_fc)

    # ---- Échantillonnage par année
    years = sorted(list(years_local))
    if not years:
        years = list(range(min_dt.year, max_dt.year + 1))
    dims = max(1, min(int(args.dims), 64))
    selectors = ['lat','lon','time_utc','split'] + [f'f{i}' for i in range(1, dims+1)]

    tasks = []
    for y in years:
        start = ee.Date.fromYMD(y, 1, 1)
        end   = start.advance(1, 'year')
        img = ee.ImageCollection(COLLECTION_ID).filterDate(start, end).first()
        # fallback ±2 ans si image absente
        img = ee.Image(ee.Algorithms.If(
            img, img,
            ee.ImageCollection(COLLECTION_ID)
              .filterDate(start.advance(-2, 'year'), end.advance(2, 'year'))
              .sort('system:time_start').first()
        ))
        img = ee.Image(img)

        # Sélection explicite des bandes d'embedding et renommage f1..f{dims}
        emb = img.select(EMB_BANDS[:dims])
        new_names = [f'f{i}' for i in range(1, dims+1)]
        emb = emb.rename(new_names)

        year_fc = all_fc.filter(ee.Filter.eq('year', y))
        # Important: géométries inutiles pour la sortie → geometries=False
        samp = emb.sampleRegions(collection=year_fc, scale=args.scale, geometries=False, tileScale=2)

        if args.export == 'local':
            # (Lent) récupération paginée locale
            total = int(samp.size().getInfo() or 0)
            log(f'Year {y}: {total} rows → local CSV {args.out}')
            write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
            with open(args.out, 'a', encoding='utf-8', newline='') as fcsv:
                w = csv.writer(fcsv)
                if write_header:
                    w.writerow(selectors)
                page, offset = args.page_size, 0
                while offset < total:
                    batch = ee.FeatureCollection(samp.toList(page, offset))
                    feats = (batch.getInfo() or {}).get('features', [])
                    for feat in feats:
                        props = feat.get('properties', {})
                        w.writerow([props.get(k, '') for k in selectors])
                    offset += len(feats)
            log(f'Year {y}: local write done.')
        elif args.export == 'gcs':
            if not args.gcs_bucket:
                print('ERROR: --gcs-bucket is required for GCS export', file=sys.stderr)
                sys.exit(1)
            task = ee.batch.Export.table.toCloudStorage(
                collection=samp,
                description=f'{args.task_prefix}_y{y}',
                bucket=args.gcs_bucket,
                fileNamePrefix=f'{args.task_prefix}/features_y{y}',
                fileFormat='CSV',
                selectors=selectors
            )
            task.start()
            tasks.append(task)
            log(f'Started GCS export for year {y} → task {task.id}')
        else:
            task = ee.batch.Export.table.toDrive(
                collection=samp,
                description=f'{args.task_prefix}_y{y}',
                fileNamePrefix=f'{args.task_prefix}_y{y}',
                fileFormat='CSV',
                selectors=selectors
            )
            task.start()
            tasks.append(task)
            log(f'Started Drive export for year {y} → task {task.id}')

    if args.export == 'local':
        log('All local pages written.')
        return

    if tasks:
        log('Exports started. Check the Tasks tab.')
        if args.watch:
            total = len(tasks); done_prev = -1
            try:
                while True:
                    done = 0; failed = 0
                    for t in tasks:
                        try:
                            st = (t.status() or {}).get('state','UNKNOWN')
                        except Exception:
                            st = 'UNKNOWN'
                        if st == 'COMPLETED': done += 1
                        elif st in ('FAILED','CANCELLED'): failed += 1
                    if done != done_prev:
                        log(f'Progress: {done}/{total} completed, {failed} failed')
                        done_prev = done
                    if done + failed >= total:
                        break
                    time.sleep(max(1, int(args.poll_seconds)))
                log(f'All exports finished. Completed={done}, Failed={failed}')
            except KeyboardInterrupt:
                log('Stopped watching; tasks keep running on EE.')

if __name__ == '__main__':
    main()
