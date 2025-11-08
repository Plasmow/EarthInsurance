#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a spatiotemporal tornado dataset from features and historical events.

Inputs (CSV in current directory by default):
 - features.csv: columns [lat, lon, time_utc, f1..f64]
 - events.csv:   columns [start_lat, start_lon, end_lat, end_lon, EF, begin_time_utc, end_time_utc]

Outputs:
 - train.csv (80%) and test.csv (20%) saved to the current directory.

Behavior:
 - All timestamps are converted to datetime64[ns, UTC].
 - For each feature point (lat, lon, time_utc), we search for tornado events
   whose active time overlaps the point time within ±3 hours and whose line
   segment (start->end) is within 3 km geodesic distance of the point.
 - Labels:
     label_occ: 1 if a matching event exists, else 0
     label_int_ef: EF of nearest matching tornado (NaN if none)
     label_int_wind_ms: median wind (m/s) per EF mapping (NaN if none)
 - Dataset is rebalanced to ~35% positives via over/under-sampling.
 - Final dataset is split into 80/20 train/test.

Notes:
 - CRS is EPSG:4326 (WGS84). Distances are computed in meters using a local
   Azimuthal Equidistant projection (pyproj) centered at the query point to
   preserve metrics locally, without changing the stored CRS of the data.
 - Requires: pandas, geopandas, shapely, numpy, sklearn, pyproj
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from pyproj import CRS, Transformer


# ---------------------------------------------
# Utilities
# ---------------------------------------------

def to_utc_datetime(series: pd.Series, colname: str) -> pd.Series:
    """Coerce a column to timezone-aware UTC datetimes (datetime64[ns, UTC]).

    - Accepts strings with or without tz; naive strings assumed UTC.
    - Raises a clear error if conversion fails.
    """
    # Support mixed formats (e.g., 'YYYY-MM-DD HH:MM:SS' and ISO8601 with 'Z').
    # pandas >= 2.0 supports format='mixed'.
    try:
        dt = pd.to_datetime(series, utc=True, errors="raise", format="mixed")
    except Exception:
        # Fallback to general parser if mixed not available/failed
        try:
            dt = pd.to_datetime(series, utc=True, errors="raise")
        except Exception as exc:
            raise ValueError(f"Failed to parse datetime column '{colname}': {exc}") from exc
    return dt


def normalize_ef(value) -> Optional[int]:
    """Normalize EF values which may be provided as integers or strings like 'EF3'.
    Returns an integer EF (0..5) or None if missing/unparseable.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        s = value.strip().upper()
        if s.startswith("EF"):
            s = s[2:]
        try:
            return int(s)
        except Exception:
            return None
    try:
        return int(value)
    except Exception:
        return None


def ef_to_median_ms(ef: Optional[int]) -> Optional[float]:
    """Map EF to median wind speed in m/s. Returns None if EF is None/out of range.
    EF0 = 33.5, EF1 = 43.5, EF2 = 55, EF3 = 67.5, EF4 = 82, EF5 = 95
    """
    mapping: Dict[int, float] = {
        0: 33.5,
        1: 43.5,
        2: 55.0,
        3: 67.5,
        4: 82.0,
        5: 95.0,
    }
    if ef is None:
        return None
    return mapping.get(int(ef))


def distance_point_to_segment_m(lat: float, lon: float, line_ll: LineString) -> float:
    """Compute geodesic-like distance (meters) from a lat/lon point to a small
    line segment expressed in lon/lat (EPSG:4326), using a local Azimuthal
    Equidistant projection centered at the point. This preserves metric distances
    locally without changing the data CRS.
    """
    # Local Azimuthal Equidistant CRS centered at the query point
    aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs("EPSG:4326", aeqd, always_xy=True)

    # Project line endpoints to the local metric CRS
    xs, ys = [], []
    for (lon_i, lat_i) in list(line_ll.coords):
        x_i, y_i = transformer.transform(lon_i, lat_i)
        xs.append(x_i)
        ys.append(y_i)
    line_m = LineString(list(zip(xs, ys)))

    # Project the point (will be ~ (0,0) by construction)
    px, py = transformer.transform(lon, lat)
    pt_m = Point(px, py)

    return float(pt_m.distance(line_m))


def compute_labels_for_point(
    lat: float,
    lon: float,
    t_utc: pd.Timestamp,
    events: pd.DataFrame,
    time_window_hours: int = 3,
    dist_threshold_m: float = 3000.0,
) -> Tuple[int, Optional[int], Optional[float]]:
    """Compute (label_occ, label_int_ef, label_int_wind_ms) for one point.

    - Filter events whose active window overlaps [t - H, t + H].
    - Spatial prefilter: bounding-box around event segment expanded by ~3 km.
    - Precise distance: shortest distance from point to event line segment in meters.
    - If any event within threshold, select the nearest and return labels.
    """
    # Time overlap filter
    start_w = t_utc - pd.Timedelta(hours=time_window_hours)
    end_w = t_utc + pd.Timedelta(hours=time_window_hours)
    cand = events[(events["end_time_utc"] >= start_w) & (events["begin_time_utc"] <= end_w)]
    if cand.empty:
        return 0, None, None

    # Spatial bounding-box prefilter (~3 km ~ 0.027 deg latitude)
    delta_lat = 0.03  # conservative margin (~3.3 km)
    # Prevent division by zero near poles; cap denominator
    cos_lat = max(math.cos(math.radians(lat)), 0.1)
    delta_lon = delta_lat / cos_lat
    cand = cand[
        (lat >= cand["min_lat"] - delta_lat)
        & (lat <= cand["max_lat"] + delta_lat)
        & (lon >= cand["min_lon"] - delta_lon)
        & (lon <= cand["max_lon"] + delta_lon)
    ]
    if cand.empty:
        return 0, None, None

    # Precise distances to each candidate segment
    dists = []
    for idx, row in cand.iterrows():
        line = row["geometry"]  # LineString in lon/lat
        try:
            d_m = distance_point_to_segment_m(lat, lon, line)
        except Exception:
            # If projection fails for some odd reason, skip this candidate
            continue
        dists.append((idx, d_m))

    if not dists:
        return 0, None, None

    # Find nearest candidate within threshold
    nearest_idx, nearest_d = min(dists, key=lambda x: x[1])
    if nearest_d <= dist_threshold_m:
        ef_raw = cand.loc[nearest_idx, "EF_norm"]
        ef_int = int(ef_raw) if ef_raw is not None else None
        wind_ms = ef_to_median_ms(ef_int)
        return 1, ef_int, wind_ms

    return 0, None, None


def balance_dataset(df: pd.DataFrame, target_pos_ratio: float, random_state: int = 42) -> pd.DataFrame:
    """Rebalance rows so that positives are about target_pos_ratio (e.g., 0.35).

    Strategy:
      - Keep all rows in the minority class; resample the other side accordingly.
      - If negatives are too many, undersample negatives.
      - If positives are too few, oversample positives with replacement.
    """
    if not (0 < target_pos_ratio < 1):
        raise ValueError("target_pos_ratio must be in (0,1)")

    pos = df[df["label_occ"] == 1]
    neg = df[df["label_occ"] == 0]
    n_pos, n_neg = len(pos), len(neg)

    if n_pos == 0 and n_neg == 0:
        return df.copy()
    if n_pos == 0:
        # No positives; cannot satisfy target; return a small undersample of negatives
        return neg.sample(min(len(neg), 10000), random_state=random_state).reset_index(drop=True)
    if n_neg == 0:
        # All positives; just return them
        return pos.reset_index(drop=True)

    # Desired negatives given current positives to reach target ratio
    desired_neg = int(round(n_pos * (1 - target_pos_ratio) / target_pos_ratio))

    if desired_neg <= n_neg:
        # Undersample negatives to target
        neg_bal = neg.sample(n=desired_neg, random_state=random_state, replace=False)
        pos_bal = pos
    else:
        # Not enough negatives relative to positives; oversample positives to match
        desired_pos = int(round(n_neg * target_pos_ratio / (1 - target_pos_ratio)))
        pos_bal = resample(pos, replace=True, n_samples=desired_pos, random_state=random_state)
        neg_bal = neg

    out = pd.concat([pos_bal, neg_bal], axis=0, ignore_index=True)
    # Shuffle rows for good measure
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Build labeled tornado dataset from features and events.")
    parser.add_argument("--features", default="features.csv", help="Path to features.csv")
    parser.add_argument("--events", default="events.csv", help="Path to events.csv")
    parser.add_argument("--dist_km", type=float, default=3.0, help="Distance threshold in km (default: 3.0)")
    parser.add_argument("--time_window_hours", type=int, default=3, help="Time window ±hours (default: 3)")
    parser.add_argument("--target_pos_ratio", type=float, default=0.35, help="Target positive ratio (default: 0.35)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for sampling/splitting")
    args = parser.parse_args()

    # ---------------------------------------------
    # Load CSVs
    # ---------------------------------------------
    try:
        f_df = pd.read_csv(args.features)
    except FileNotFoundError:
        print(f"ERROR: Could not find features file: {args.features}", file=sys.stderr)
        sys.exit(1)
    try:
        e_df_raw = pd.read_csv(args.events)
    except FileNotFoundError:
        print(
            f"ERROR: Could not find events file: {args.events}. Please add it before running.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---------------------------------------------
    # Validate required columns
    # ---------------------------------------------
    feat_required = {"lat", "lon", "time_utc"}
    missing_f = feat_required - set(f_df.columns)
    if missing_f:
        print(f"ERROR: features.csv missing columns: {sorted(missing_f)}", file=sys.stderr)
        sys.exit(1)

    # ---------------------------------------------
    # Normalize events schema (case-insensitive, alternate names)
    # Accepts e.g. Start_Lat/Start_Lon/End_Lat/End_Lon/Date/EF_Scale/Magnitude
    # If end time is missing, default it to begin time (instantaneous event)
    # ---------------------------------------------

    def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        cols_map = {str(c).strip().lower(): c for c in df.columns}
        for cand in candidates:
            key = cand.strip().lower()
            if key in cols_map:
                return cols_map[key]
        return None

    def build_events_df(df_raw: pd.DataFrame) -> pd.DataFrame:
        # Identify columns in a case-insensitive fashion
        col_slat = find_col(df_raw, ["start_lat", "start latitude", "start_latitude", "slat", "start lat"]) or find_col(df_raw, ["start_lat"])
        col_slon = find_col(df_raw, ["start_lon", "start longitude", "start_longitude", "slon", "start lon"]) or find_col(df_raw, ["start_lon"])
        col_elat = find_col(df_raw, ["end_lat", "end latitude", "end_latitude", "elat", "end lat"]) or find_col(df_raw, ["end_lat"])
        col_elon = find_col(df_raw, ["end_lon", "end longitude", "end_longitude", "elon", "end lon"]) or find_col(df_raw, ["end_lon"])

        # Time columns: prefer explicit begin/end; otherwise, use a single 'date'
        col_begin = find_col(df_raw, ["begin_time_utc", "begin_time", "start_time", "date", "datetime", "time"])  # type: ignore
        col_end = find_col(df_raw, ["end_time_utc", "end_time", "stop_time", "finish_time"])  # type: ignore

        # EF columns
        col_ef = find_col(df_raw, ["ef", "ef_scale", "efscale", "magnitude"])

        missing = [name for name, col in (
            ("start_lat", col_slat),
            ("start_lon", col_slon),
            ("end_lat", col_elat),
            ("end_lon", col_elon),
            ("begin_time_utc/date", col_begin),
        ) if col is None]
        if missing:
            raise ValueError(f"events.csv missing needed columns (case-insensitive): {missing}")

        # Build normalized dataframe with expected columns
        df = pd.DataFrame({
            "start_lat": pd.to_numeric(df_raw[col_slat], errors="coerce"),
            "start_lon": pd.to_numeric(df_raw[col_slon], errors="coerce"),
            "end_lat": pd.to_numeric(df_raw[col_elat], errors="coerce"),
            "end_lon": pd.to_numeric(df_raw[col_elon], errors="coerce"),
        })

        # Time handling
        begin_parsed = to_utc_datetime(df_raw[col_begin], col_begin)
        if col_end is not None:
            end_parsed = to_utc_datetime(df_raw[col_end], col_end)
        else:
            # Default end_time to begin_time if no explicit end is available
            end_parsed = begin_parsed.copy()

        df["begin_time_utc"] = begin_parsed
        df["end_time_utc"] = end_parsed

        # EF normalization (optional)
        if col_ef is not None:
            df["EF"] = df_raw[col_ef]
        else:
            df["EF"] = np.nan

        # Drop rows with missing essential values
        df = df.dropna(subset=["start_lat", "start_lon", "end_lat", "end_lon", "begin_time_utc", "end_time_utc"]).reset_index(drop=True)
        return df

    try:
        e_df = build_events_df(e_df_raw)
    except Exception as exc:
        print(f"ERROR: Failed to normalize events.csv schema: {exc}", file=sys.stderr)
        sys.exit(1)

    # ---------------------------------------------
    # Normalize dtypes and timestamps to UTC
    # ---------------------------------------------
    # Coerce numeric columns for stability
    for c in ["lat", "lon"]:
        f_df[c] = pd.to_numeric(f_df[c], errors="coerce")
    f_df["time_utc"] = to_utc_datetime(f_df["time_utc"], "time_utc")

    for c in ["start_lat", "start_lon", "end_lat", "end_lon"]:
        e_df[c] = pd.to_numeric(e_df[c], errors="coerce")
    e_df["begin_time_utc"] = to_utc_datetime(e_df["begin_time_utc"], "begin_time_utc")
    e_df["end_time_utc"] = to_utc_datetime(e_df["end_time_utc"], "end_time_utc")

    # Drop any rows with missing essential numeric coordinates or timestamps
    f_df = f_df.dropna(subset=["lat", "lon", "time_utc"]).reset_index(drop=True)
    e_df = e_df.dropna(
        subset=["start_lat", "start_lon", "end_lat", "end_lon", "begin_time_utc", "end_time_utc"]
    ).reset_index(drop=True)

    # ---------------------------------------------
    # Build GeoDataFrames in EPSG:4326 (WGS84)
    # ---------------------------------------------
    f_gdf = gpd.GeoDataFrame(
        f_df.copy(), geometry=gpd.points_from_xy(f_df["lon"], f_df["lat"]), crs="EPSG:4326"
    )

    # Event geometry as line segment from (start_lon, start_lat) to (end_lon, end_lat)
    e_geom = [
        LineString([(lon1, lat1), (lon2, lat2)])
        for lon1, lat1, lon2, lat2 in zip(
            e_df["start_lon"], e_df["start_lat"], e_df["end_lon"], e_df["end_lat"]
        )
    ]
    e_gdf = gpd.GeoDataFrame(e_df.copy(), geometry=e_geom, crs="EPSG:4326")

    # Normalize EF to integers 0..5 where possible
    e_gdf["EF_norm"] = e_gdf["EF"].apply(normalize_ef)

    # Precompute simple bounding boxes to speed up spatial prefiltering
    e_gdf["min_lat"] = e_gdf[["start_lat", "end_lat"]].min(axis=1)
    e_gdf["max_lat"] = e_gdf[["start_lat", "end_lat"]].max(axis=1)
    e_gdf["min_lon"] = e_gdf[["start_lon", "end_lon"]].min(axis=1)
    e_gdf["max_lon"] = e_gdf[["start_lon", "end_lon"]].max(axis=1)

    # ---------------------------------------------
    # Compute labels per feature row
    # ---------------------------------------------
    dist_m = float(args.dist_km) * 1000.0
    labels_occ = []
    labels_ef = []
    labels_wind = []

    # Iterate per point; for large datasets consider chunking or multiprocessing
    for i, row in f_gdf.iterrows():
        lat = float(row["lat"])  # already validated numeric
        lon = float(row["lon"])
        t = pd.Timestamp(row["time_utc"]).tz_convert("UTC")

        occ, ef, wind = compute_labels_for_point(
            lat=lat,
            lon=lon,
            t_utc=t,
            events=e_gdf,
            time_window_hours=int(args.time_window_hours),
            dist_threshold_m=dist_m,
        )
        labels_occ.append(occ)
        labels_ef.append(ef if ef is not None else np.nan)
        labels_wind.append(wind if wind is not None else np.nan)

        # Light progress indicator every 10k rows
        if (i + 1) % 10000 == 0:
            print(f"Labeled {i+1} rows...")

    f_gdf["label_occ"] = labels_occ
    f_gdf["label_int_ef"] = labels_ef
    f_gdf["label_int_wind_ms"] = labels_wind

    # ---------------------------------------------
    # Balance dataset to target positive ratio
    # ---------------------------------------------
    labeled_df = pd.DataFrame(f_gdf.drop(columns=["geometry"]))
    balanced_df = balance_dataset(labeled_df, target_pos_ratio=float(args.target_pos_ratio), random_state=int(args.random_state))

    # ---------------------------------------------
    # Train/test split and save
    # ---------------------------------------------
    train_df, test_df = train_test_split(
        balanced_df,
        test_size=0.20,
        random_state=int(args.random_state),
        stratify=balanced_df["label_occ"],
    )

    # Save CSVs in the current directory
    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    # Quick summary
    pos_ratio = float((balanced_df["label_occ"] == 1).mean()) if len(balanced_df) > 0 else 0.0
    print(
        f"Done. Wrote train.csv ({len(train_df)} rows) and test.csv ({len(test_df)} rows). "
        f"Balanced positive ratio ~ {pos_ratio:.3f}."
    )


if __name__ == "__main__":
    main()
