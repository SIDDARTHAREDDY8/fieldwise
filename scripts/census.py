"""USDA Census of Agriculture: cover-crop acres and harvested cropland by Ohio county, 2017 and 2022.

Output: data/census_cover.json  {fips: {name, 2017: {cover, cropland, pct}, 2022: {...}}}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
API = "https://quickstats.nass.usda.gov/api/api_GET/"
SERIES = {
    "cover": "PRACTICES, LAND USE, CROPLAND, COVER CROP PLANTED, (EXCL CRP) - ACRES",
    "cropland": "AG LAND, CROPLAND, HARVESTED - ACRES",
}


def fetch(short_desc: str, year: int) -> dict[str, tuple[str, float]]:
    p = {"key": os.environ["NASS_API_KEY"], "source_desc": "CENSUS", "short_desc": short_desc, "state_alpha": "OH",
         "agg_level_desc": "COUNTY", "year": str(year), "domain_desc": "TOTAL", "format": "JSON"}
    r = httpx.get(API, params=p, timeout=120)
    r.raise_for_status()
    out = {}
    for row in r.json()["data"]:
        v = row["Value"].replace(",", "").strip()
        if not v.replace(".", "").isdigit():  # "(D)" = withheld to protect a farm's privacy
            continue
        out[row["state_fips_code"] + row["county_code"]] = (row["county_name"].title(), float(v))
    return out


def main() -> None:
    res: dict = {}
    for year in (2017, 2022):
        cov, crop = fetch(SERIES["cover"], year), fetch(SERIES["cropland"], year)
        for fips, (name, c) in cov.items():
            if fips in crop and crop[fips][1] > 0:
                res.setdefault(fips, {"name": name})[str(year)] = {"cover": c, "cropland": crop[fips][1], "pct": round(100 * c / crop[fips][1], 2)}
    (ROOT / "data" / "census_cover.json").write_text(json.dumps(res, indent=1))
    both = [f for f, v in res.items() if "2017" in v and "2022" in v]
    print(len(res), "counties;", len(both), "with both years")
    top = sorted(both, key=lambda f: -res[f]["2022"]["pct"])[:5]
    for f in top:
        print(f, res[f]["name"], res[f]["2017"]["pct"], "->", res[f]["2022"]["pct"], "% of harvested cropland")


if __name__ == "__main__":
    main()
