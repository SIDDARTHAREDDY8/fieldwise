"""Data access: NASA POWER daily history and USDA soil survey for a point."""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import httpx

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
# T2M_MAX/T2M_MIN: air temperature (C); PRECTOTCORR: precipitation (mm/day);
# GWETTOP: surface soil wetness (0-1); TS: earth skin temperature (C)
POWER_PARAMS = ["T2M_MAX", "T2M_MIN", "T2M", "PRECTOTCORR", "GWETTOP", "TS"]
SDA_URL = "https://SDMDataAccess.sc.egov.usda.gov/Tabular/post.rest"


def _cache_get(key: str):
    p = CACHE / f"{hashlib.sha1(key.encode()).hexdigest()}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _cache_put(key: str, value) -> None:
    p = CACHE / f"{hashlib.sha1(key.encode()).hexdigest()}.json"
    p.write_text(json.dumps(value))


def fetch_power(lat: float, lon: float, start_year: int = 2001, end_year: int = 2025) -> dict:
    """Return {param: {YYYYMMDD: value}} from NASA POWER (AG community)."""
    lat, lon = round(lat, 2), round(lon, 2)
    key = f"power:{lat}:{lon}:{start_year}:{end_year}"
    cached = _cache_get(key)
    if cached:
        return cached
    params = {
        "parameters": ",".join(POWER_PARAMS),
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": f"{start_year}0101",
        "end": f"{end_year}1231",
        "format": "JSON",
    }
    r = httpx.get(POWER_URL, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]
    _cache_put(key, data)
    return data


def fetch_power_current(lat: float, lon: float, today: date) -> dict:
    """This year's NASA POWER daily data so far (near-real-time, a few days behind). Cached per day."""
    lat, lon = round(lat, 2), round(lon, 2)
    key = f"power-now:{lat}:{lon}:{today.isoformat()}"
    cached = _cache_get(key)
    if cached:
        return cached
    params = {"parameters": ",".join(POWER_PARAMS), "community": "AG", "latitude": lat, "longitude": lon,
              "start": f"{today.year}0101", "end": today.strftime("%Y%m%d"), "format": "JSON"}
    r = httpx.get(POWER_URL, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]
    _cache_put(key, data)
    return data


def daily_records(power: dict) -> dict[date, dict]:
    """Pivot POWER output into {date: {param: value}} and drop fill values (-999)."""
    out: dict[date, dict] = {}
    for param, series in power.items():
        for ymd, v in series.items():
            d = date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]))
            out.setdefault(d, {})[param] = None if v is None or v <= -998 else v
    return out


def _sda(q: str) -> list[dict]:
    r = httpx.post(SDA_URL, json={"format": "JSON+COLUMNNAME", "query": q}, timeout=30)
    r.raise_for_status()
    rows = r.json().get("Table", [])
    return [dict(zip(rows[0], row)) for row in rows[1:]] if len(rows) > 1 else []


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch_soil(lat: float, lon: float) -> dict | None:
    """Dominant soil at the point from USDA NRCS Soil Data Access (SSURGO), with its topsoil properties."""
    key = f"soil2:{round(lat, 4)}:{round(lon, 4)}"
    cached = _cache_get(key)
    if cached:
        return cached
    try:
        comps = _sda(
            "SELECT TOP 5 mu.muname, mu.farmlndcl, c.cokey, c.compname, c.comppct_r, c.drainagecl, c.hydgrp, c.slope_r, c.erocl "
            "FROM mapunit mu INNER JOIN component c ON c.mukey = mu.mukey "
            "WHERE mu.mukey IN (SELECT * FROM SDA_Get_Mukey_from_intersection_with_WktWgs84("
            f"'point({lon} {lat})')) ORDER BY c.comppct_r DESC")
    except Exception:  # noqa: BLE001
        return None
    if not comps:
        return None
    # Prefer the largest real soil component (skip "Urban land", "Water" etc. with no drainage class)
    soil = next((c for c in comps if c.get("drainagecl")), comps[0])
    result = {
        "map_unit": soil.get("muname"),
        "component": soil.get("compname"),
        "drainage": soil.get("drainagecl"),
        "hydrologic_group": soil.get("hydgrp"),
        "slope_pct": _f(soil.get("slope_r")),
        "farmland": soil.get("farmlndcl"),
        "erosion": soil.get("erocl"),
    }
    try:
        hz = _sda("SELECT hzdept_r, hzdepb_r, om_r, ph1to1h2o_r, awc_r, claytotal_r, sandtotal_r, silttotal_r "
                  f"FROM chorizon WHERE cokey = '{int(soil['cokey'])}' ORDER BY hzdept_r")
    except Exception:  # noqa: BLE001
        hz = []
    if hz:
        top = hz[0]
        result.update({"organic_matter_pct": _f(top.get("om_r")), "ph": _f(top.get("ph1to1h2o_r")),
                       "clay_pct": _f(top.get("claytotal_r")), "sand_pct": _f(top.get("sandtotal_r")), "silt_pct": _f(top.get("silttotal_r"))})
        # plant-available water in the top 100 cm (root zone), in inches
        water_cm = 0.0
        for h in hz:
            a, b, awc = _f(h.get("hzdept_r")), _f(h.get("hzdepb_r")), _f(h.get("awc_r"))
            if a is None or b is None or awc is None or a >= 100:
                continue
            water_cm += awc * (min(b, 100) - a)
        result["water_holding_in"] = round(water_cm / 2.54, 1) if water_cm else None
    _cache_put(key, result)
    return result


def fetch_forecast(lat: float, lon: float, today: date) -> dict:
    """7-day forecast (plus the last 2 days) for the field from Open-Meteo, cached per day."""
    lat, lon = round(lat, 3), round(lon, 3)
    key = f"forecast:{lat}:{lon}:{today.isoformat()}"
    cached = _cache_get(key)
    if cached:
        return cached
    r = httpx.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon, "timezone": "America/New_York", "forecast_days": 7, "past_days": 2,
        "daily": "precipitation_sum,precipitation_probability_max,temperature_2m_max,temperature_2m_min",
        "temperature_unit": "fahrenheit", "precipitation_unit": "inch"}, timeout=30)
    r.raise_for_status()
    data = r.json()["daily"]
    _cache_put(key, data)
    return data
