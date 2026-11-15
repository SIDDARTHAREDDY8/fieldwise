"""Sample corn/soybean field points in every Ohio county from the USDA Cropland Data Layer (CDL).

A point qualifies for census year Y if it was corn or soybean in year Y AND in year Y+1, so a
field that went to winter wheat (green all winter) is excluded: winter greenness at such a
point means a cover crop (or weeds), which is what the census counts.

Output: data/sample_points.json  {year: [{fips, lat, lon}, ...]}
Usage:  .venv/bin/python scripts/sample_points.py [points_per_county]
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
CDL_DIR = ROOT / "data" / "cdl"
CDL_DIR.mkdir(parents=True, exist_ok=True)
SVC = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"
CORN_SOY = (1, 5)  # CDL classes: 1 corn, 5 soybeans
TO_WGS = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)


def cdl_path(year: int, fips: str) -> Path:
    p = CDL_DIR / f"CDL_{year}_{fips}.tif"
    if p.exists() and p.stat().st_size > 1000:
        return p
    if os.getenv("CDL_OFFLINE"):  # use only maps already downloaded
        raise RuntimeError("not cached")
    last = None
    for attempt in range(4):  # the CropScape service is slow at times: retry with backoff
        try:
            xml = httpx.get(SVC, params={"year": year, "fips": fips}, timeout=180).text
            url = re.search(r"<returnURL>(.*?)</returnURL>", xml).group(1)
            tmp = p.with_suffix(".part")
            with httpx.stream("GET", url, timeout=300) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
            tmp.rename(p)
            return p
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"CDL {year} {fips}: {last}")


def sample(year: int, fips: str, n: int, seed: int) -> list[dict]:
    try:
        a, b = cdl_path(year, fips), cdl_path(year + 1, fips)
    except Exception as e:  # noqa: BLE001 - skip one county rather than lose the whole run
        if not os.getenv("CDL_OFFLINE"):
            print("  skipped", year, fips, str(e)[:100], flush=True)
        return []
    with rasterio.open(a) as ra, rasterio.open(b) as rb:
        A, B = ra.read(1), rb.read(1)
        if A.shape != B.shape or ra.transform != rb.transform:
            return []  # grids differ: skip rather than misalign
        ok = np.isin(A, CORN_SOY) & np.isin(B, CORN_SOY)
        # keep pixels whose 3x3 neighbourhood is all corn/soy, i.e. inside a field, away from edges
        inner = ok.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                inner &= np.roll(np.roll(ok, dy, 0), dx, 1)
        rows, cols = np.nonzero(inner)
        if len(rows) == 0:
            return []
        rng = random.Random(seed)
        idx = rng.sample(range(len(rows)), min(n, len(rows)))
        pts = []
        for i in idx:
            x, y = ra.xy(rows[i], cols[i])
            lon, lat = TO_WGS.transform(x, y)
            pts.append({"fips": fips, "lat": round(lat, 5), "lon": round(lon, 5)})
        return pts


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    census = json.loads((ROOT / "data" / "census_cover.json").read_text())
    out = {}
    for year in (2017, 2022):
        fipss = [f for f, v in census.items() if str(year) in v]
        with ThreadPoolExecutor(4) as ex:
            res = list(ex.map(lambda f: sample(year, f, n, int(f) * 10000 + year), fipss))
        out[str(year)] = [p for r in res for p in r]
        print(year, len(out[str(year)]), "points in", sum(1 for r in res if r), "counties", flush=True)
    (ROOT / "data" / "sample_points.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
