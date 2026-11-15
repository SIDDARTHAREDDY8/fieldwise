"""Winter greenness from NASA Harmonized Landsat Sentinel-2 (HLS v2.0, 30 m).

Searches NASA's CMR-STAC catalog for HLS scenes, then reads only the pixels needed from each
cloud-optimised GeoTIFF (red, near-infrared and the Fmask quality layer) and computes NDVI for
clear, snow-free observations. Reading needs a free NASA Earthdata login
(EARTHDATA_USERNAME / EARTHDATA_PASSWORD in .env).
"""
from __future__ import annotations

import os
import stat
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import httpx
import rasterio
from dotenv import load_dotenv
from pyproj import Transformer
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
CACHE = ROOT / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
STAC = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD/search"
BANDS = {"HLSL30_2.0": ("B04", "B05"), "HLSS30_2.0": ("B04", "B8A")}  # (red, nir)
# Fmask bits: 1 cloud, 2 adjacent to cloud/shadow, 3 cloud shadow, 4 snow/ice, 5 water
BAD_BITS = 0b00111110


def gdal_env() -> dict:
    user, pw = os.getenv("EARTHDATA_USERNAME"), os.getenv("EARTHDATA_PASSWORD")
    if not user or not pw:
        raise RuntimeError("Add EARTHDATA_USERNAME and EARTHDATA_PASSWORD to fieldwise/.env (free: urs.earthdata.nasa.gov)")
    netrc = CACHE / "netrc"
    if not netrc.exists():
        netrc.write_text(f"machine urs.earthdata.nasa.gov login {user} password {pw}\n")
        netrc.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return {
        "GDAL_HTTP_NETRC": "YES", "GDAL_HTTP_NETRC_FILE": str(netrc),
        "GDAL_HTTP_COOKIEFILE": str(CACHE / "cookies.txt"), "GDAL_HTTP_COOKIEJAR": str(CACHE / "cookies.txt"),
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "TIF",
        "GDAL_HTTP_MAX_RETRY": "4", "GDAL_HTTP_RETRY_DELAY": "1", "VSI_CACHE": "TRUE",
    }


def search(bbox: tuple[float, float, float, float], start: date, end: date, max_cloud: int = 75) -> list[dict]:
    """HLS scenes over bbox (lon_min, lat_min, lon_max, lat_max) between start and end."""
    body = {"collections": list(BANDS), "bbox": list(bbox),
            "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z", "limit": 250}
    scenes, url = [], STAC
    with httpx.Client(timeout=60) as cli:
        while url:
            r = cli.post(url, json=body) if url == STAC else cli.get(url)
            r.raise_for_status()
            j = r.json()
            for f in j.get("features", []):
                cc = f["properties"].get("eo:cloud_cover")
                if cc is not None and cc > max_cloud:
                    continue
                red, nir = BANDS[f["collection"]]
                a = f["assets"]
                if red not in a or nir not in a or "Fmask" not in a:
                    continue
                scenes.append({"id": f["id"], "date": f["properties"]["datetime"][:10], "tile": f["id"].split(".")[2],
                               "bbox": f.get("bbox"), "red": a[red]["href"], "nir": a[nir]["href"],
                               "fmask": a["Fmask"]["href"], "cloud": cc})
            # next page is a GET link that already carries the query
            url = next((l["href"] for l in j.get("links", []) if l.get("rel") == "next"), None)
    return scenes


def read_scene(scene: dict, points: list[tuple[float, float]]) -> list[float | None]:
    """NDVI at each (lon, lat) for one scene; None if outside, cloudy, shadowed, snowy or water."""
    out: list[float | None] = [None] * len(points)
    with rasterio.Env(**gdal_env()):
        with rasterio.open(scene["fmask"]) as fm:
            tr = Transformer.from_crs("EPSG:4326", fm.crs, always_xy=True)
            idx = []
            for i, (lon, lat) in enumerate(points):
                x, y = tr.transform(lon, lat)
                r, c = fm.index(x, y)
                if 0 <= r < fm.height and 0 <= c < fm.width:
                    q = int(fm.read(1, window=Window(c, r, 1, 1))[0, 0])
                    if q != 255 and not (q & BAD_BITS):
                        idx.append((i, r, c))
            if not idx:
                return out
        with rasterio.open(scene["red"]) as rd, rasterio.open(scene["nir"]) as nr:
            for i, r, c in idx:
                red = float(rd.read(1, window=Window(c, r, 1, 1))[0, 0])
                nir = float(nr.read(1, window=Window(c, r, 1, 1))[0, 0])
                if red <= -9999 or nir <= -9999 or red + nir <= 0:
                    continue
                out[i] = round((nir - red) / (nir + red), 4)
    return out


def winter_series(lon: float, lat: float, first_winter: int, last_winter: int, workers: int = 16,
                  per_winter: int = 10) -> list[dict]:
    """Clear-sky NDVI observations Dec 1 - Mar 31 for each winter starting in first..last (Dec of year Y).
    Uses the `per_winter` least-cloudy scenes of each winter, spread across the season."""
    d = 0.002
    jobs = []
    for y in range(first_winter, last_winter + 1):
        sc = search((lon - d, lat - d, lon + d, lat + d), date(y, 12, 1), date(y + 1, 3, 31), max_cloud=50)
        seen, picked = set(), []
        for s in sorted(sc, key=lambda s: (s["cloud"] if s["cloud"] is not None else 100)):
            if s["date"] in seen:  # one view per day (overlapping tiles duplicate it)
                continue
            seen.add(s["date"])
            picked.append(s)
            if len(picked) >= per_winter:
                break
        jobs += [(y, s) for s in picked]

    def run(job):
        y, s = job
        try:
            v = read_scene(s, [(lon, lat)])[0]
        except Exception:  # noqa: BLE001 - one unreadable scene shouldn't sink the series
            v = None
        return {"winter": y, "date": s["date"], "ndvi": v, "sensor": "Landsat" if "L30" in s["id"] else "Sentinel-2"}

    with ThreadPoolExecutor(workers) as ex:
        obs = list(ex.map(run, jobs))
    return sorted((o for o in obs if o["ndvi"] is not None), key=lambda o: o["date"])


def winter_score(ndvis: list[float]) -> float | None:
    """Robust 'how green was it this winter': the second-highest clear NDVI (needs >= 2 clear looks)."""
    if len(ndvis) < 2:
        return None
    return sorted(ndvis)[-2]
