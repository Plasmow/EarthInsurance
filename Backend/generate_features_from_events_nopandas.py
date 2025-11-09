#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate features.csv from events.csv (~40% positives) without external libs.

Positives: one point per event at segment midpoint, time=begin_time_utc.
Negatives: random U.S. points with time in year 2000 (no overlap by ±3h).

Writes features.csv with columns: lat, lon, time_utc, f1..f64
"""

import csv
import os
import random
import sys


US_LAT_MIN, US_LAT_MAX = 24.5, 49.5
US_LON_MIN, US_LON_MAX = -125.0, -66.0


def frand():
    return random.random()


def main():
    random.seed(123)
    events_path = "events.csv"
    out_path = "features.csv"

    if not os.path.exists(events_path):
        print("ERROR: events.csv not found.", file=sys.stderr)
        sys.exit(1)

    positives = []
    with open(events_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # Build a case-insensitive field lookup
        fields = { (name or "").strip().lower(): name for name in (reader.fieldnames or []) }

        def find_col(candidates):
            for cand in candidates:
                key = cand.lower()
                if key in fields:
                    return fields[key]
            return None

        col_start_lat = find_col(["start_lat", "start latitude", "start_latitude", "start_lat" , "slat"])
        col_start_lon = find_col(["start_lon", "start longitude", "start_longitude", "start_lon" , "slon"])
        col_end_lat = find_col(["end_lat", "end latitude", "end_latitude", "end_lat" , "elat"])
        col_end_lon = find_col(["end_lon", "end longitude", "end_longitude", "end_lon" , "elon"])
        col_time = find_col(["begin_time_utc", "date", "datetime", "time", "begin_time", "start_time", "starttime"])

        missing = [
            name for name, col in (
                ("start_lat", col_start_lat),
                ("start_lon", col_start_lon),
                ("end_lat", col_end_lat),
                ("end_lon", col_end_lon),
                ("begin_time_utc/date", col_time),
            ) if col is None
        ]
        if missing:
            print(f"ERROR: events.csv missing needed columns (case-insensitive match): {missing}", file=sys.stderr)
            sys.exit(1)

        for row in reader:
            try:
                slat_raw = row.get(col_start_lat, "")
                slon_raw = row.get(col_start_lon, "")
                elat_raw = row.get(col_end_lat, "")
                elon_raw = row.get(col_end_lon, "")
                slat = float(slat_raw) if slat_raw != "" else None
                slon = float(slon_raw) if slon_raw != "" else None
                elat = float(elat_raw) if elat_raw != "" else None
                elon = float(elon_raw) if elon_raw != "" else None
            except Exception:
                continue
            if None in (slat, slon, elat, elon):
                continue
            lat = (slat + elat) / 2.0
            lon = (slon + elon) / 2.0
            # Use begin_time_utc string directly; build script will parse as UTC
            t = row.get(col_time) or ""
            if not t:
                continue
            positives.append((lat, lon, t))

    n_pos = len(positives)
    if n_pos == 0:
        print("ERROR: No valid events found to build positives.", file=sys.stderr)
        sys.exit(1)

    total = int(round(n_pos / 0.4))
    n_neg = max(total - n_pos, 0)

    # Prepare CSV header
    headers = ["lat", "lon", "time_utc"] + [f"f{i}" for i in range(1, 65)]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        # Write positive rows (one per event)
        for (lat, lon, t) in positives:
            feats = [frand() for _ in range(64)]
            writer.writerow([f"{lat:.6f}", f"{lon:.6f}", t] + [f"{v:.6f}" for v in feats])

        # Write negative rows
        for _ in range(n_neg):
            lat = random.uniform(US_LAT_MIN, US_LAT_MAX)
            lon = random.uniform(US_LON_MIN, US_LON_MAX)
            # Year 2000 time to avoid any overlap with 2017-2025 events (±3h)
            t = "2000-06-15T12:00:00Z"
            feats = [frand() for _ in range(64)]
            writer.writerow([f"{lat:.6f}", f"{lon:.6f}", t] + [f"{v:.6f}" for v in feats])

    print(f"Wrote {out_path} with {n_pos} positives (~40%) and {n_neg} negatives.")


if __name__ == "__main__":
    main()
