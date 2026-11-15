"""Train the crop-yield model on USDA NASS Ohio county yields (2001-2025) and NASA POWER weather.

Each county is matched to its NASA POWER grid cell (0.5 x 0.625 deg). For each crop:
  yield = trend(year) + county offset + f(weather), f = gradient-boosted trees on the weather anomaly.
Validation is leave-one-year-out: the whole year (every county) is held out, so the model never sees
the weather of the season it is scored on. We compare against trend + county offset alone.

Output: data/yield_model.pkl, data/yield_validation.json
Usage: .venv/bin/python scripts/train_yield.py
"""
from __future__ import annotations

import json
import pickle
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.data import daily_records, fetch_power  # noqa: E402
from app.yieldmodel import FEATURES, MODEL_PATH, season_features  # noqa: E402

LATS = [38.5 + 0.5 * i for i in range(8)]
LONS = [-85.0 + 0.625 * j for j in range(8)]


def cell_of(lat, lon):
    return min(LATS, key=lambda a: abs(a - lat)), min(LONS, key=lambda b: abs(b - lon))


def fit_level(rows):
    """Linear state trend over years + each county's average offset from it."""
    yrs = np.array([r["year"] for r in rows], float)
    ys = np.array([r["yield"] for r in rows], float)
    b, a = np.polyfit(yrs, ys, 1)
    off = defaultdict(list)
    for r in rows:
        off[r["fips"]].append(r["yield"] - (a + b * r["year"]))
    return a, b, {f: float(np.mean(v)) for f, v in off.items()}


def gbm():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=25, l2_regularization=1.0, random_state=0)


def main() -> None:
    counties = json.loads((ROOT / "data" / "ohio_counties.json").read_text())
    yields = json.loads((ROOT / "data" / "county_yields.json").read_text())
    cells = {}
    feats = {}
    for fips, c in counties.items():
        cell = cell_of(c["lat"], c["lon"])
        if cell not in cells:
            cells[cell] = daily_records(fetch_power(*cell))
            print("weather for cell", cell, flush=True)
    model = {"crops": {}, "skill": {}}
    report = {}
    for crop, rows in yields.items():
        data = []
        for r in rows:
            c = counties.get(r["fips"])
            if not c:
                continue
            key = (cell_of(c["lat"], c["lon"]), crop, r["year"])
            if key not in feats:
                feats[key] = season_features(cells[key[0]], crop, r["year"])
            f = feats[key]
            if f and not any(np.isnan(v) for v in f.values()):
                data.append({**r, "x": [f[k] for k in FEATURES[crop]]})
        years = sorted({r["year"] for r in data})
        err_w, err_b, anom, pred_anom = [], [], [], []
        for y in years:  # leave one whole season out
            tr = [r for r in data if r["year"] != y]
            te = [r for r in data if r["year"] == y]
            a, b, off = fit_level(tr)
            lvl = lambda r: a + b * r["year"] + off.get(r["fips"], 0.0)
            m = gbm().fit(np.array([r["x"] for r in tr]), np.array([r["yield"] - lvl(r) for r in tr]))
            p = m.predict(np.array([r["x"] for r in te]))
            for r, pa in zip(te, p):
                base = lvl(r)
                err_b.append(abs(r["yield"] - base)); err_w.append(abs(r["yield"] - base - pa))
                anom.append(r["yield"] - base); pred_anom.append(pa)
        anom, pred_anom = np.array(anom), np.array(pred_anom)
        r2 = 1 - np.sum((anom - pred_anom) ** 2) / np.sum((anom - anom.mean()) ** 2)
        # state-level check: does the model call the good and bad years?
        by_year_true, by_year_pred = defaultdict(list), defaultdict(list)
        i = 0
        for y in years:
            for r in [r for r in data if r["year"] == y]:
                by_year_true[y].append(anom[i]); by_year_pred[y].append(pred_anom[i]); i += 1
        yt = [st.mean(by_year_true[y]) for y in years]; yp = [st.mean(by_year_pred[y]) for y in years]
        r_year = float(np.corrcoef(yt, yp)[0, 1])
        worst = sorted(zip(yt, years))[:3]
        report[crop] = {"county_years": len(data), "years": f"{years[0]}-{years[-1]}",
                        "mae_trend_only": round(st.mean(err_b), 1), "mae_with_weather": round(st.mean(err_w), 1),
                        "r2_weather_effect": round(float(r2), 2), "r_statewide_year": round(r_year, 2),
                        "worst_years": [{"year": y, "actual_vs_trend": round(t, 1), "model_vs_trend": round(yp[years.index(y)], 1)} for t, y in worst]}
        print(crop, json.dumps(report[crop]), flush=True)

        a, b, off = fit_level(data)
        X = np.array([r["x"] for r in data])
        m = gbm().fit(X, np.array([r["yield"] - (a + b * r["year"] + off[r["fips"]]) for r in data]))
        recent = defaultdict(list)
        for r in data:
            if r["year"] >= 2019:
                recent[r["fips"]].append(r["yield"])
        model["crops"][crop] = {"model": m, "trend_a": float(a), "trend_b": float(b), "county_offset": off,
                                "typical": {k: float(np.median(X[:, j])) for j, k in enumerate(FEATURES[crop])},
                                "county_names": {r["fips"]: r["county"] for r in data},
                                "county_recent": {f: round(st.mean(v)) for f, v in recent.items()}}
        resid = defaultdict(dict)
        for r in data:
            resid[r["fips"]][r["year"]] = round(r["yield"] - (a + b * r["year"] + off[r["fips"]]), 1)
        model["crops"][crop]["resid"] = dict(resid)
        model["crops"][crop]["use_weather"] = bool(report[crop]["r2_weather_effect"] >= 0.1)
        model["skill"][crop] = {"use_weather": model["crops"][crop]["use_weather"], **{k: report[crop][k] for k in ("mae_trend_only", "mae_with_weather", "r2_weather_effect", "r_statewide_year")}}
    MODEL_PATH.write_bytes(pickle.dumps(model))
    (ROOT / "data" / "yield_validation.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
