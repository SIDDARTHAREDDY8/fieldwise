"""Second attempt at the county check, using the approach from published cover-crop mapping:

* window: early spring (Mar 1 - Apr 30 of the year after the census year), when cover crops green
  up before they are killed and corn/soybean fields are still bare;
* relative greenness: each point's NDVI minus the median NDVI of all sampled points in the same
  image, which removes weather, snow-melt and sun-angle differences between images and regions;
* point score: its highest relative greenness over images where >= 15 points were clear.

County share = % of points whose score exceeds a threshold, compared with the USDA census
cover-crop % (2017). The threshold is tuned on half the counties and tested on the other half.
Attempt 1 (Dec-Mar absolute NDVI) found no relationship (r = -0.09); both results are reported.

Usage: .venv/bin/python scripts/validate_cover2.py [points_per_county]
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.hls import read_scene, search  # noqa: E402

YEAR = 2017
OHIO_BBOX = (-84.82, 38.40, -80.52, 41.98)
THRESHOLDS = [round(0.02 * i, 2) for i in range(1, 16)]  # relative NDVI 0.02 .. 0.30


def main() -> None:
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    census = json.loads((ROOT / "data" / "census_cover.json").read_text())
    allp = json.loads((ROOT / "data" / "sample_points.json").read_text())[str(YEAR)]
    per = defaultdict(list)
    for p in allp:
        if len(per[p["fips"]]) < cap:
            per[p["fips"]].append(p)
    pts = [p for v in per.values() for p in v]
    ck = ROOT / "data" / f"hls_spring_{YEAR}.json"
    state = json.loads(ck.read_text()) if ck.exists() else {"scenes": {}}
    scenes = [s for s in search(OHIO_BBOX, date(YEAR + 1, 3, 1), date(YEAR + 1, 4, 30), max_cloud=40) if s["id"] not in state["scenes"]]
    print(f"{len(scenes)} spring scenes for {len(pts)} points in {len(per)} counties", flush=True)
    t0 = time.time()

    def job(s):
        b = s["bbox"]
        idx = [i for i, p in enumerate(pts) if b and b[0] <= p["lon"] <= b[2] and b[1] <= p["lat"] <= b[3]]
        if not idx:
            return s["id"], {}
        try:
            vals = read_scene(s, [(pts[i]["lon"], pts[i]["lat"]) for i in idx])
        except Exception:  # noqa: BLE001
            return None, {}
        return s["id"], {str(i): v for i, v in zip(idx, vals) if v is not None}

    with ThreadPoolExecutor(10) as ex:
        futs = [ex.submit(job, s) for s in scenes]
        for n, f in enumerate(as_completed(futs), 1):
            sid, vals = f.result()
            if sid:
                state["scenes"][sid] = vals
            if n % 20 == 0 or n == len(futs):
                ck.write_text(json.dumps(state))
                print(f"  {n}/{len(futs)} scenes, {time.time() - t0:.0f}s", flush=True)

    # relative greenness per point
    rel = defaultdict(list)
    for sid, vals in state["scenes"].items():
        if len(vals) < 15:
            continue
        med = st.median(vals.values())
        for i, v in vals.items():
            rel[int(i)].append(v - med)
    score = {i: max(v) for i, v in rel.items() if len(v) >= 2}

    def share(thr, only):
        by = defaultdict(list)
        for i, p in enumerate(pts):
            if i in score and p["fips"] in only:
                by[p["fips"]].append(score[i] >= thr)
        return {f: 100 * sum(v) / len(v) for f, v in by.items() if len(v) >= 10}

    def corr(thr, only):
        sh = share(thr, only)
        fs = [f for f in sh if str(YEAR) in census.get(f, {})]
        if len(fs) < 3 or len({round(sh[f], 6) for f in fs}) < 2:
            return float("-inf"), len(fs), sh
        return st.correlation([sh[f] for f in fs], [census[f][str(YEAR)]["pct"] for f in fs]), len(fs), sh

    fs = sorted(per)
    halves = (set(fs[0::2]), set(fs[1::2]))
    res = {}
    for k, (tr, te) in enumerate((halves, halves[::-1])):
        best = max(THRESHOLDS, key=lambda t: corr(t, tr)[0])
        res[f"fold{k + 1}"] = {"threshold": best, "r_train": round(corr(best, tr)[0], 2), "r_test": round(corr(best, te)[0], 2), "n_test": corr(best, te)[1]}
    thr = round(st.mean(v["threshold"] for v in res.values()), 2)
    r_all, n_all, sh = corr(thr, set(fs))
    out = {"attempt": 2, "method": "spring Mar-Apr, NDVI relative to same-image median", "year": YEAR,
           "folds": res, "threshold": thr, "r_heldout_mean": round(st.mean(v["r_test"] for v in res.values()), 2),
           "r_all": round(r_all, 2), "n_counties": n_all, "points": len(pts),
           "counties": {f: {"name": census[f]["name"], "satellite_pct": round(sh[f], 1), "census_pct": census[f][str(YEAR)]["pct"]} for f in sh if str(YEAR) in census.get(f, {})},
           "attempt1": "Dec-Mar absolute NDVI, 10 points/county: r = -0.09 on held-out counties (no relationship)"}
    (ROOT / "data" / "cover_validation2.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "counties"}, indent=1))


if __name__ == "__main__":
    main()
