"""Validate FieldWise's spring field-work model against USDA NASS observations for Ohio.

USDA NASS Crop Progress reports, every week, how many days were "suitable for fieldwork"
in Ohio (judged by county extension staff and farm reporters). FieldWise estimates the same
quantity from NASA POWER daily rainfall. For each spring (weeks ending Apr 16 - May 16,
2001-2025) we sum USDA's days and the model's workable days (averaged over six Ohio farm
regions) and measure agreement.

Several day-definitions were tried and all are reported, together with a time split
(2001-2013 vs 2014-2025), so the chosen rule is not just tuned to one period.

Usage:  .venv/bin/python scripts/validate.py   (needs NASS_API_KEY in .env)
Key (free): https://quickstats.nass.usda.gov/api
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from datetime import date, timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.data import daily_records, fetch_power  # noqa: E402
from app.rules import Climate  # noqa: E402

OHIO_POINTS = [  # farm regions: NW, N-central, W-central, central, SW, NE
    (41.30, -84.10), (41.05, -82.90), (40.40, -84.30), (40.10, -83.20), (39.40, -84.56), (40.85, -81.75),
]
NASS = "https://quickstats.nass.usda.gov/api/api_GET/"


def usda_days_suitable(key: str) -> dict[int, list[tuple[date, float]]]:
    params = {"key": key, "commodity_desc": "FIELDWORK", "statisticcat_desc": "DAYS SUITABLE",
              "agg_level_desc": "STATE", "state_alpha": "OH", "year__GE": "2001", "format": "JSON"}
    r = httpx.get(NASS, params=params, timeout=60)
    r.raise_for_status()
    weeks: dict[int, list[tuple[date, float]]] = {}
    for row in r.json()["data"]:
        we = date.fromisoformat(row["week_ending"][:10])
        if we.year <= 2025 and date(we.year, 4, 16) <= we <= date(we.year, 5, 16):
            weeks.setdefault(we.year, []).append((we, float(row["Value"])))
    return weeks


def main() -> None:
    key = os.getenv("NASS_API_KEY")
    if not key:
        sys.exit("Set NASS_API_KEY in fieldwise/.env (free key: https://quickstats.nass.usda.gov/api)")
    weeks = usda_days_suitable(key)
    usda = {y: round(sum(v for _, v in w), 1) for y, w in weeks.items()}
    climates = [Climate(daily_records(fetch_power(lat, lon))) for lat, lon in OHIO_POINTS]

    def wet_cut(c: Climate, q: float) -> float:
        v = sorted(r["GWETTOP"] for d, r in c.rec.items() if d.month in (4, 5) and r.get("GWETTOP") is not None)
        return v[int(q * (len(v) - 1))]

    rules = {
        "rain_only (used)": lambda c, d, cut: c.is_workable(d),
        "rain + soil wetness < p95": lambda c, d, cut: c.is_workable(d) and (c.get(d, "GWETTOP") or 0) < cut[0.95],
        "rain + soil wetness < p85": lambda c, d, cut: c.is_workable(d) and (c.get(d, "GWETTOP") or 0) < cut[0.85],
        "rain + soil wetness < p75": lambda c, d, cut: c.is_workable(d) and (c.get(d, "GWETTOP") or 0) < cut[0.75],
    }
    cuts = [{q: wet_cut(c, q) for q in (0.75, 0.85, 0.95)} for c in climates]
    years = sorted(usda)
    early, late = [y for y in years if y <= 2013], [y for y in years if y > 2013]
    results, model_used = {}, None
    for name, fn in rules.items():
        model = {y: round(st.mean(sum(fn(c, we - timedelta(days=k), cut) for we, _ in weeks[y] for k in range(7))
                                  for c, cut in zip(climates, cuts)), 1) for y in years}
        r = lambda ys: round(st.correlation([model[y] for y in ys], [usda[y] for y in ys]), 2)
        results[name] = {"r_all": r(years), "r_2001_2013": r(early), "r_2014_2025": r(late),
                         "mae_days": round(st.mean(abs(model[y] - usda[y]) for y in years), 1)}
        if model_used is None:
            model_used = model
    out = {
        "what": "Spring days suitable for fieldwork in Ohio (weeks ending Apr 16 - May 16)",
        "years": years,
        "model_workable_days": [model_used[y] for y in years],
        "usda_days_suitable": [usda[y] for y in years],
        "pearson_r": results["rain_only (used)"]["r_all"],
        "n": len(years),
        "rules_tested": results,
        "wettest_model_year": min(years, key=lambda y: model_used[y]),
        "wettest_usda_year": min(years, key=lambda y: usda[y]),
    }
    (ROOT / "data" / "validation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(results, indent=2))
    print("wettest springs  model:", sorted(years, key=lambda y: model_used[y])[:4], " USDA:", sorted(years, key=lambda y: usda[y])[:4])


if __name__ == "__main__":
    main()
