"""Precompute planting deadlines on the NASA POWER grid covering Ohio.

NASA POWER's meteorology is on a 0.5 deg (lat) x 0.625 deg (lon) grid, so we sample at that
spacing; each cell is one independent POWER data point. Output: data/ohio_grid.json
Usage: .venv/bin/python scripts/build_grid.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.data import daily_records, fetch_power  # noqa: E402
from app.rules import Climate, planting_deadlines  # noqa: E402

# Rough Ohio outline (lon, lat) used to keep only cells whose centre is in or near the state.
OHIO = [(-84.82, 41.70), (-83.45, 41.73), (-82.68, 41.68), (-81.10, 42.05), (-80.52, 41.98), (-80.52, 40.64),
        (-80.66, 40.40), (-80.86, 39.76), (-81.25, 39.39), (-81.75, 39.18), (-82.00, 38.88), (-82.57, 38.40),
        (-83.03, 38.73), (-83.67, 38.63), (-84.31, 38.99), (-84.82, 39.10)]


def inside(lon: float, lat: float, poly=OHIO) -> bool:
    hit = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def main() -> None:
    lats = [38.5 + 0.5 * i for i in range(8)]          # 38.5 .. 42.0
    lons = [-85.0 + 0.625 * j for j in range(8)]       # -85.0 .. -80.625
    cells = []
    for lat in lats:
        for lon in lons:
            # keep the cell if any corner or the centre falls in Ohio
            pts = [(lon, lat), (lon - 0.31, lat - 0.25), (lon + 0.31, lat + 0.25), (lon - 0.31, lat + 0.25), (lon + 0.31, lat - 0.25)]
            if not any(inside(x, y) for x, y in pts):
                continue
            c = Climate(daily_records(fetch_power(lat, lon)))
            d = planting_deadlines(c, lat)
            cells.append({"lat": lat, "lon": lon, **{k: {"p90": v["p90"], "p80": v["p80"], "p50": v["p50"]} for k, v in d["options"].items()}})
            print(f"{lat:.2f},{lon:.3f}  rye p90 {cells[-1]['cereal_rye']['p90']}  wheat p90 {cells[-1]['winter_wheat']['p90']}", flush=True)
    out = {"cell_deg": [0.5, 0.625], "cells": cells, "outline": OHIO}
    (ROOT / "data" / "ohio_grid.json").write_text(json.dumps(out))
    print(len(cells), "cells written")


if __name__ == "__main__":
    main()
