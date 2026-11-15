"""Download Ohio county crop yields (USDA NASS Quick Stats) and county centroids (Census gazetteer).

Output: data/county_yields.json  {crop: [{fips, county, year, yield}]}, data/ohio_counties.json
Usage: .venv/bin/python scripts/yields.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CROPS = {
    "corn": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
    "soybeans": "SOYBEANS - YIELD, MEASURED IN BU / ACRE",
    "wheat": "WHEAT, WINTER - YIELD, MEASURED IN BU / ACRE",
}


def main() -> None:
    counties = {}
    for line in (ROOT / "data" / "raw" / "2023_Gaz_counties_national.txt").read_text().splitlines()[1:]:
        f = line.split("\t")
        if f[0] == "OH":
            counties[f[1]] = {"name": f[3].replace(" County", ""), "lat": float(f[8]), "lon": float(f[9].strip())}
    (ROOT / "data" / "ohio_counties.json").write_text(json.dumps(counties, indent=0))

    out = {}
    for crop, short in CROPS.items():
        p = {"key": os.environ["NASS_API_KEY"], "source_desc": "SURVEY", "short_desc": short, "state_alpha": "OH",
             "agg_level_desc": "COUNTY", "year__GE": "2001", "year__LE": "2025", "reference_period_desc": "YEAR", "format": "JSON"}
        r = httpx.get("https://quickstats.nass.usda.gov/api/api_GET/", params=p, timeout=120)
        r.raise_for_status()
        rows = []
        for d in r.json()["data"]:
            if not d.get("county_ansi"):
                continue  # "other (combined) counties"
            try:
                v = float(d["Value"].replace(",", ""))
            except ValueError:
                continue
            rows.append({"fips": "39" + d["county_ansi"], "county": d["county_name"].title(), "year": int(d["year"]), "yield": v})
        out[crop] = rows
        print(crop, len(rows), "county-years,", len({r["fips"] for r in rows}), "counties", file=sys.stderr)
    (ROOT / "data" / "county_yields.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
