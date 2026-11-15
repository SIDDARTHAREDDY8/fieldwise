"""Crop-yield model: what would corn, soybeans and wheat yield on this field, given its weather?

Yield = state trend (better seed and practice over time) + county level (soil and management)
        + weather effect (machine-learned from NASA POWER weather against USDA county yields).

The weather effect is a gradient-boosted tree model trained on 2001-2025 Ohio county yields.
To answer a farmer, we replay each of the field's 25 past seasons of NASA weather on today's
trend and county level: "if a season like 2012 came again, what would wheat yield here?"
"""
from __future__ import annotations

import json
import math
import pickle
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "data" / "yield_model.pkl"


def _days(recs, start: date, end: date):
    d = start
    while d <= end:
        r = recs.get(d)
        if r:
            yield r
        d += timedelta(days=1)


def _sum(recs, start, end, f):
    return sum(f(r) for r in _days(recs, start, end) if all(r.get(k) is not None for k in ("T2M_MAX", "T2M_MIN", "PRECTOTCORR")))


def _mean(recs, start, end, key):
    v = [r[key] for r in _days(recs, start, end) if r.get(key) is not None]
    return sum(v) / len(v) if v else float("nan")


def _gdd(base, cap):
    return lambda r: max(0.0, (min(r["T2M_MAX"], cap) + max(r["T2M_MIN"], base)) / 2 - base)


rain = lambda r: r["PRECTOTCORR"]
heat = lambda r: max(0.0, r["T2M_MAX"] - 30.0)          # degree-days above 30 C (86 F): heat stress
wet = lambda r: 1.0 if r["PRECTOTCORR"] >= 10 else 0.0   # days with ~0.4 in of rain or more

FEATURES = {
    "corn": ["gdd_season", "heat_jul_aug", "rain_may", "rain_jun", "rain_jul", "rain_aug", "wet_days_spring", "soil_wet_jul_aug", "tmax_jul"],
    "soybeans": ["gdd_season", "heat_jul_aug", "rain_may", "rain_jun", "rain_jul", "rain_aug", "rain_sep", "wet_days_spring", "soil_wet_aug"],
    "wheat": ["wet_days_fall", "gdd_fall", "tmin_winter", "cold_days", "rain_fall", "rain_apr", "rain_may", "rain_jun", "wet_days_may", "tmax_jun", "gdd_spring"],
}

LABELS = {  # plain words for each weather factor (when below typical, when above), used to explain bad years
    "gdd_season": ("a cool summer", "a hot summer"), "heat_jul_aug": ("a mild summer", "summer heat stress"),
    "rain_may": ("a dry May", "a wet May"), "rain_jun": ("a dry June", "a wet June"), "rain_jul": ("a dry July", "a wet July"),
    "rain_aug": ("a dry August", "a wet August"), "rain_sep": ("a dry September", "a wet September"),
    "wet_days_spring": ("a dry spring", "a wet planting season"), "soil_wet_jul_aug": ("dry summer soil", "waterlogged summer soil"),
    "soil_wet_aug": ("dry August soil", "waterlogged August soil"), "tmax_jul": ("a cool July", "a hot July"),
    "wet_days_fall": ("a dry fall", "a wet planting fall"), "gdd_fall": ("a cold fall", "a warm fall"),
    "tmin_winter": ("a bitter cold snap", "a mild winter"), "cold_days": ("a mild winter", "many hard-freeze nights"),
    "rain_fall": ("a dry fall", "a wet fall"), "rain_apr": ("a dry April", "a wet April"),
    "wet_days_may": ("a dry May", "a wet May (head scab weather)"), "tmax_jun": ("a cool June", "a hot June"),
    "gdd_spring": ("a cold spring", "a warm spring"),
}


def season_features(recs: dict, crop: str, y: int) -> dict | None:
    """Weather features for the season harvested in year y. None if the data doesn't cover it."""
    if crop == "wheat":
        if date(y - 1, 10, 1) not in recs or date(y, 7, 1) not in recs:
            return None
        return {
            "wet_days_fall": _sum(recs, date(y - 1, 9, 25), date(y - 1, 11, 10), wet),
            "gdd_fall": _sum(recs, date(y - 1, 10, 1), date(y - 1, 12, 31), _gdd(0, 30)),
            "tmin_winter": min((r["T2M_MIN"] for r in _days(recs, date(y - 1, 12, 1), date(y, 2, 28)) if r.get("T2M_MIN") is not None), default=float("nan")),
            "cold_days": _sum(recs, date(y - 1, 12, 1), date(y, 2, 28), lambda r: 1.0 if r["T2M_MIN"] <= -15 else 0.0),
            "rain_fall": _sum(recs, date(y - 1, 10, 1), date(y - 1, 11, 30), rain),
            "rain_apr": _sum(recs, date(y, 4, 1), date(y, 4, 30), rain),
            "rain_may": _sum(recs, date(y, 5, 1), date(y, 5, 31), rain),
            "rain_jun": _sum(recs, date(y, 6, 1), date(y, 6, 30), rain),
            "wet_days_may": _sum(recs, date(y, 5, 1), date(y, 5, 31), wet),
            "tmax_jun": _mean(recs, date(y, 6, 1), date(y, 6, 30), "T2M_MAX"),
            "gdd_spring": _sum(recs, date(y, 3, 1), date(y, 5, 31), _gdd(0, 30)),
        }
    if date(y, 9, 30) not in recs:
        return None
    f = {
        "gdd_season": _sum(recs, date(y, 5, 1), date(y, 9, 30), _gdd(10, 30)),
        "heat_jul_aug": _sum(recs, date(y, 7, 1), date(y, 8, 31), heat),
        "rain_may": _sum(recs, date(y, 5, 1), date(y, 5, 31), rain),
        "rain_jun": _sum(recs, date(y, 6, 1), date(y, 6, 30), rain),
        "rain_jul": _sum(recs, date(y, 7, 1), date(y, 7, 31), rain),
        "rain_aug": _sum(recs, date(y, 8, 1), date(y, 8, 31), rain),
        "rain_sep": _sum(recs, date(y, 9, 1), date(y, 9, 30), rain),
        "wet_days_spring": _sum(recs, date(y, 4, 15), date(y, 5, 31), wet),
        "soil_wet_jul_aug": _mean(recs, date(y, 7, 1), date(y, 8, 31), "GWETTOP"),
        "soil_wet_aug": _mean(recs, date(y, 8, 1), date(y, 8, 31), "GWETTOP"),
        "tmax_jul": _mean(recs, date(y, 7, 1), date(y, 7, 31), "T2M_MAX"),
    }
    return {k: f[k] for k in FEATURES[crop]}


def nearest_county(lat: float, lon: float) -> str:
    counties = json.loads((ROOT / "data" / "ohio_counties.json").read_text())
    return min(counties, key=lambda f: (counties[f]["lat"] - lat) ** 2 + ((counties[f]["lon"] - lon) * math.cos(math.radians(lat))) ** 2)


_MODEL = None


def load():
    global _MODEL
    if _MODEL is None and MODEL_PATH.exists():
        _MODEL = pickle.loads(MODEL_PATH.read_bytes())
    return _MODEL


def field_yields(recs: dict, lat: float, lon: float, target_year: int) -> dict | None:
    """Replay each past season's weather at this field on target_year's trend and county level."""
    m = load()
    if not m:
        return None
    fips = nearest_county(lat, lon)
    out = {"county": None, "crops": {}, "skill": m["skill"]}
    for crop, cm in m["crops"].items():
        level = float(cm["trend_a"] + cm["trend_b"] * target_year + cm["county_offset"].get(fips, 0.0))
        out["county"] = cm["county_names"].get(fips) or out["county"]
        per = []
        if not cm.get("use_weather", True):
            # No weather skill (wheat): replay this county's actual USDA results, detrended to today.
            res = cm["resid"].get(fips) or {}
            per = [{"year": int(y), "yield": round(level + float(r), 1), "why": None} for y, r in sorted(res.items())]
        for y in ([] if per or not cm.get("use_weather", True) else range(2001, 2026)):
            f = season_features(recs, crop, y)
            if not f or any(isinstance(v, float) and math.isnan(v) for v in f.values()):
                continue
            x = [[f[k] for k in FEATURES[crop]]]
            eff = float(cm["model"].predict(x)[0])
            # why: which factor pulled this season down most (difference from that factor's typical value)
            contrib = {}
            for k in FEATURES[crop]:
                x2 = [[(cm["typical"][k] if j == k else f[j]) for j in FEATURES[crop]]]
                contrib[k] = eff - float(cm["model"].predict(x2)[0])
            worst = min(contrib, key=contrib.get)
            why = LABELS[worst][0 if f[worst] < cm["typical"][worst] else 1] if contrib[worst] < -2 else None
            per.append({"year": y, "yield": round(level + eff, 1), "why": why})
        ys = sorted(p["yield"] for p in per)
        if not ys:
            continue
        q = lambda p: ys[min(len(ys) - 1, max(0, round(p * (len(ys) - 1))))]
        out["crops"][crop] = {"typical": round(q(0.5)), "bad_year": round(q(0.1)), "good_year": round(q(0.9)),
                              "county_avg_recent": cm["county_recent"].get(fips), "per_year": per,
                              "method": "weather model" if cm.get("use_weather", True) else "county history"}
    return out
