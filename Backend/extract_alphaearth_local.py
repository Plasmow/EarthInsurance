
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch feature extraction from events.csv using Google Satellite Embedding (AlphaEarth) on GEE.
Local-only version: NO Drive/GCS exports. Uses server-side sampleRegions per year,
then downloads in local pages with tqdm progress bars and detailed logs.

- Lit data/events.csv, construit les positifs (milieu du segment, horodaté UTC).
- Génère des négatifs aléatoires hors buffer 3 km des événements, et hors fenêtre +/-3h (au pas 1h).
- Concatène (positifs + négatifs) -> FeatureCollection {lat, lon, time_utc, year, split}.
- Pour chaque année -> sampleRegions(64D) -> filtre notNull(f1) -> téléchargement paginé local.

Dépendances :
  pip install earthengine-api tqdm
  ee.Authenticate() puis ee.Initialize(project='votre-project-id')
"""

import sys, os, csv, argparse, time, math
from datetime import datetime, timezone
from typing import List, Optional

import ee  # pip install earthengine-api
from tqdm import tqdm  # pip install tqdm

COLLECTION_ID = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
US_LAT_MIN, US_LAT_MAX = 24.5, 49.5
US_LON_MIN, US_LON_MAX = -125.0, -66.0
EMB_BANDS = [f'embedding_{i}' for i in range(64)]  # bandes explicites (ancien schéma)

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)

def parse_args():
    p = argparse.ArgumentParser(description="AlphaEarth local extractor (no Drive/GCS).")
    p.add_argument('--events', default='data/events.csv', help='Chemin vers events.csv')
    p.add_argument('--out', default='data/features.csv', help='Chemin CSV de sortie (unique, append)')
    p.add_argument('--dims', type=int, default=64, help='Dimensions a exporter (<=64)')
    p.add_argument('--scale', type=float, default=30.0, help='Scale (m) pour sampleRegions')
    p.add_argument('--page-size', type=int, default=1000, help='Taille de page pour le rapatriement local')
    p.add_argument('--project', default=None, help='Project ID GEE pour ee.Initialize(project=...)')
    p.add_argument('--neg-ratio-pos', type=float, default=0.6, help='Fraction de negatifs desiree (=1-pos). Ex: 0.6 -> 40% pos')
    p.add_argument('--seed', type=int, default=123, help='Seed pour points aleatoires negatifs')
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
    # Plusieurs formats courants
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass
    # ISO 8601 generique
    try:
        if s.endswith('Z'):
            return datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(timezone.utc)
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None

def init_ee(project: Optional[str]):
    log('Initializing Earth Engine…')
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as e:
        log('EE init failed, attempting interactive auth…')
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    log('EE ready.')

def load_events(path: str):
    log(f'Loading events from {path} …')
    try:
        with open(path, 'r', encoding='utf-8-sig', newline='') as f:
            rdr = csv.DictReader(f)
            rows = list(rdr)
            fields = [c.strip() for c in (rdr.fieldnames or [])]
    except FileNotFoundError:
        print(f'ERROR: not found: {path}', file=sys.stderr)
        sys.exit(1)
    if not rows:
        print('ERROR: events.csv empty.', file=sys.stderr)
        sys.exit(1)
    return rows, fields

def pick_cols(fields: List[str]):
    lc = {c.lower(): c for c in fields}
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
    return col_slat, col_slon, col_elat, col_elon, col_date

def build_pos_fc(rows, col_slat, col_slon, col_elat, col_elon, col_date):
    feats = []
    years = set()
    ok = 0
    for r in rows:
        slat, slon = safe_float(r.get(col_slat)), safe_float(r.get(col_slon))
        elat, elon = safe_float(r.get(col_elat)), safe_float(r.get(col_elon))
        dt = parse_time_utc(r.get(col_date))
        if None in (slat, slon, elat, elon) or dt is None:
            continue
        lat = (slat + elat)/2.0
        lon = (slon + elon)/2.0
        y = dt.year
        years.add(y)
        props = {
            'lat': lat, 'lon': lon,
            'time_utc': dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            'year': y, 'split': 'pos'
        }
        feats.append(ee.Feature(ee.Geometry.Point([lon, lat]), props))
        ok += 1
    fc = ee.FeatureCollection(feats)
    log(f'Parsed positives: {ok}')
    if ok == 0:
        print('ERROR: No valid events after parsing.', file=sys.stderr)
        sys.exit(1)
    return fc, years

def build_safe_region_and_times(rows, col_slat, col_slon, col_elat, col_elon, col_date):
    # Buffer 3 km autour des segments
    line_feats = []
    pos_times = []
    for r in rows:
        slat, slon = safe_float(r.get(col_slat)), safe_float(r.get(col_slon))
        elat, elon = safe_float(r.get(col_elat)), safe_float(r.get(col_elon))
        dt = parse_time_utc(r.get(col_date))
        if None not in (slat, slon, elat, elon):
            line_feats.append(ee.Feature(ee.Geometry.LineString([[slon, slat],[elon, elat]])))
        if dt is not None:
            pos_times.append(dt)
    if not pos_times:
        print('ERROR: No valid event times.', file=sys.stderr); sys.exit(1)
    lines_fc = ee.FeatureCollection(line_feats) if line_feats else ee.FeatureCollection([])
    buffered_union = lines_fc.geometry().buffer(3000)
    us_geom = ee.Geometry.Rectangle([US_LON_MIN, US_LAT_MIN, US_LON_MAX, US_LAT_MAX], None, False)
    safe_region = us_geom.difference(buffered_union, 1)
    min_dt, max_dt = min(pos_times), max(pos_times)
    return safe_region, min_dt, max_dt

def build_neg_fc(n_neg: int, safe_region, min_dt, max_dt, seed: int):
    span_hours = max(0, int((max_dt - min_dt).total_seconds() // 3600))
    allowed = list(range(span_hours + 1)) if span_hours > 0 else [0]
    allowed_list = ee.List(allowed)
    allowed_size = allowed_list.size()
    min_date = ee.Date(min_dt.isoformat())

    log(f'Generating negatives: {n_neg} (seed={seed})')
    neg_fc = ee.FeatureCollection.randomPoints(safe_region, n_neg, seed)
    def _neg_props(f):
        coords = f.geometry().coordinates()  # [lon, lat]
        idx = (ee.Number(coords.get(0)).multiply(1000000).abs()
               .add(ee.Number(coords.get(1)).multiply(1000000).abs())
               .toInt().mod(allowed_size))
        off_h = ee.Number(allowed_list.get(idx))
        dt = min_date.advance(off_h, 'hour')
        return f.set({
            'lat': coords.get(1),
            'lon': coords.get(0),
            'time_utc': dt.format("YYYY-MM-dd'T'HH:mm:ss'Z'"),
            'year': ee.Number.parse(dt.format('YYYY')),
            'split': 'neg'
        })
    return neg_fc.map(_neg_props)

def sample_year_to_csv(year: int, all_fc: ee.FeatureCollection, out_csv: str, dims: int, scale_m: float, page_size: int):
    start = ee.Date.fromYMD(year, 1, 1)
    end   = start.advance(1, 'year')
    img = ee.ImageCollection(COLLECTION_ID).filterDate(start, end).first()
    img = ee.Image(ee.Algorithms.If(
        img, img,
        ee.ImageCollection(COLLECTION_ID)
          .filterDate(start.advance(-2,'year'), end.advance(2,'year'))
          .sort('system:time_start').first()
    ))
    img = ee.Image(img)
    # Certaines versions de la collection fournissent des bandes 'embedding_0..63',
    # d'autres 'A00..A63'. On sélectionne dynamiquement selon les bandes présentes.
    band_names = img.bandNames()
    has_embedding = band_names.contains('embedding_0')
    has_A00 = band_names.contains('A00')

    emb_select_embed = img.select(EMB_BANDS[:dims])
    a_bands = [f"A{str(i).zfill(2)}" for i in range(dims)]
    emb_select_A = img.select(a_bands)

    emb_img = ee.Image(
        ee.Algorithms.If(
            has_embedding,
            emb_select_embed,
            ee.Algorithms.If(has_A00, emb_select_A, img)
        )
    )

    emb = emb_img.rename([f'f{i}' for i in range(1, dims+1)])

    year_fc = all_fc.filter(ee.Filter.eq('year', year))
    samp = emb.sampleRegions(collection=year_fc, scale=scale_m, geometries=False, tileScale=2)
    samp = samp.filter(ee.Filter.notNull(['f1']))

    total = int(samp.size().getInfo() or 0)
    log(f'Year {year}: {total} rows to fetch -> {out_csv}')
    if total == 0:
        return 0

    selectors = ['lat','lon','time_utc','split'] + [f'f{i}' for i in range(1, dims+1)]
    write_header = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    if os.path.dirname(out_csv):
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'a', encoding='utf-8', newline='') as fcsv:
        w = csv.writer(fcsv)
        if write_header:
            w.writerow(selectors)

        offset = 0
        pbar = tqdm(total=total, desc=f'Year {year}', unit='row')
        def fetch_batch(fc, size, start_idx, tries=3):
            last_err = None
            for k in range(tries):
                try:
                    batch = ee.FeatureCollection(fc.toList(size, start_idx))
                    data = (batch.getInfo() or {}).get('features', [])
                    return data
                except Exception as e:
                    last_err = e
                    time.sleep(2 * (k+1))
            raise last_err

        while offset < total:
            feats = fetch_batch(samp, page_size, offset)
            if not feats:
                break
            for feat in feats:
                props = feat.get('properties', {})
                w.writerow([props.get(k, '') for k in selectors])
            offset += len(feats)
            pbar.update(len(feats))
        pbar.close()
    log(f'Year {year}: done.')
    return total

def main():
    args = parse_args()
    dims = max(1, min(int(args.dims), 64))
    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

    init_ee(args.project)
    rows, fields = load_events(args.events)
    col_slat, col_slon, col_elat, col_elon, col_date = pick_cols(fields)

    pos_fc, years_local = build_pos_fc(rows, col_slat, col_slon, col_elat, col_elon, col_date)
    safe_region, min_dt, max_dt = build_safe_region_and_times(rows, col_slat, col_slon, col_elat, col_elon, col_date)

    n_pos = int(pos_fc.size().getInfo() or 0)
    pos_frac = 1.0 - float(args.neg_ratio_pos)
    pos_frac = max(0.05, min(pos_frac, 0.95))
    target_total = int(round(n_pos / pos_frac))
    n_neg = max(target_total - n_pos, 0)
    log(f'Negatives planned: {n_neg} (target pos fraction ~{int(pos_frac*100)}%)')

    neg_fc = build_neg_fc(n_neg, safe_region, min_dt, max_dt, args.seed)
    all_fc = pos_fc.merge(neg_fc)

    years = sorted(list(years_local)) or list(range(min_dt.year, max_dt.year+1))
    log(f'Years to process: {years}')

    total_rows = 0
    for y in years:
        total_rows += sample_year_to_csv(y, all_fc, args.out, dims, args.scale, args.page_size)
    log(f'All done. Total rows written: {total_rows} -> {args.out}')

if __name__ == '__main__':
    main()
