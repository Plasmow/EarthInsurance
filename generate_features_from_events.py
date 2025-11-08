#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate features.csv from events.csv with ~40% positives.

Positives: one point per tornado event (near segment midpoint, time at event midpoint).
Negatives: random US points (no event overlap by time: year 2000), to reach 60% share.

Outputs: features.csv with columns: lat, lon, time_utc, f1..f64
"""

import sys
import numpy as np
import pandas as pd


US_LAT_MIN, US_LAT_MAX = 24.5, 49.5
US_LON_MIN, US_LON_MAX = -125.0, -66.0


def main():
    events_path = "events.csv"
    features_path = "features.csv"
    rng = np.random.RandomState(123)

    try:
        e_df = pd.read_csv(events_path)
    except FileNotFoundError:
        print("ERROR: events.csv not found.", file=sys.stderr)
        sys.exit(1)

    # Validate required columns
    required = {
        "start_lat",
        "start_lon",
        "end_lat",
        "end_lon",
        "begin_time_utc",
        "end_time_utc",
    }
    missing = required - set(e_df.columns)
    if missing:
        print(f"ERROR: events.csv missing columns: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    # Parse times (assume UTC or naive interpreted as UTC)
    e_df["begin_time_utc"] = pd.to_datetime(e_df["begin_time_utc"], utc=True, errors="coerce")
    e_df["end_time_utc"] = pd.to_datetime(e_df["end_time_utc"], utc=True, errors="coerce")

    # Drop invalid rows
    e_df = e_df.dropna(
        subset=[
            "start_lat",
            "start_lon",
            "end_lat",
            "end_lon",
            "begin_time_utc",
            "end_time_utc",
        ]
    ).reset_index(drop=True)

    # Build positive samples: midpoint position and midpoint time
    lat_mid = (pd.to_numeric(e_df["start_lat"]) + pd.to_numeric(e_df["end_lat"])) / 2.0
    lon_mid = (pd.to_numeric(e_df["start_lon"]) + pd.to_numeric(e_df["end_lon"])) / 2.0
    time_mid = e_df["begin_time_utc"] + (e_df["end_time_utc"] - e_df["begin_time_utc"]) / 2

    pos_df = pd.DataFrame({
        "lat": lat_mid,
        "lon": lon_mid,
        "time_utc": time_mid.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
    })

    n_pos = len(pos_df)
    # Target 40% positives => total N = n_pos / 0.4, negatives = total - n_pos
    total_n = int(round(n_pos / 0.4))
    n_neg = max(total_n - n_pos, 0)

    # Generate negatives inside US bbox with times in year 2000 to avoid ±3h overlaps
    neg_lat = rng.uniform(US_LAT_MIN, US_LAT_MAX, size=n_neg)
    neg_lon = rng.uniform(US_LON_MIN, US_LON_MAX, size=n_neg)
    # Random times within the year 2000
    # Generate days 0..365 and seconds 0..86400 uniformly
    days = rng.randint(0, 366, size=n_neg)
    secs = rng.randint(0, 24 * 3600, size=n_neg)
    base = pd.Timestamp("2000-01-01T00:00:00Z")
    neg_times = (base + pd.to_timedelta(days, unit="D") + pd.to_timedelta(secs, unit="s")).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    neg_df = pd.DataFrame({
        "lat": neg_lat,
        "lon": neg_lon,
        "time_utc": neg_times,
    })

    # Build 64 synthetic feature columns f1..f64 (uniform [0,1))
    def add_features(df: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
        feats = rng.rand(len(df), 64)
        for i in range(64):
            df[f"f{i+1}"] = feats[:, i]
        return df

    pos_df = add_features(pos_df, rng)
    neg_df = add_features(neg_df, rng)

    # Combine and shuffle
    all_df = pd.concat([pos_df, neg_df], axis=0, ignore_index=True)
    all_df = all_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Save
    all_df.to_csv(features_path, index=False)
    print(
        f"Wrote {features_path} with {len(all_df)} rows: positives={n_pos} (~{n_pos/len(all_df):.2%}), negatives={n_neg}."
    )


if __name__ == "__main__":
    main()

