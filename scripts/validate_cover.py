"""County-level check: does NASA HLS winter greenness on corn/soy fields track USDA census cover-crop acres?

For census year Y (2017, 2022) we look at the winter after Y's harvest (Dec 1 Y - Mar 31 Y+1) at
sampled corn/soy points (scripts/sample_points.py), compute each point's winter greenness score
(app.hls.winter_score) and the share of points per county above a threshold, then correlate that
share with the census "% of harvested cropland planted to cover crops".

The threshold is chosen on one census year and tested on the other, so the reported agreement
is out-of-sample. Progress is checkpointed (data/hls_points_<year>.json) so the run can resume.

Usage: .venv/bin/python scripts/validate_cover.py [max_points_per_county]
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
from app.hls import read_scene, search, winter_score  # noqa: E402

OHIO_BBOX = (-84.82, 38.40, -80.52, 41.98)
THRESHOLDS = [round(0.20 + 0.02 * i, 2) for i in range(16)]  # 0.20 .. 0.50


def in_bbox(lon, lat, b):
    return b and b[0] <= lon <= b[2] and b[1] <= lat <= b[3]


def collect(year: int, pts: list[dict]) -> dict[int, list[float]]:
    ck = ROOT / "data" / f"hls_points_{year}.json"
    state = json.loads(ck.read_text()) if ck.exists() else {"done": [], "obs": {}}
    obs = defaultdict(list, {int(k): v for k, v in state["obs"].items()})
    done = set(state["done"])
    scenes = [s for s in search(OHIO_BBOX, date(year, 12, 1), date(year + 1, 3, 31), max_cloud=40) if s["id"] not in done]
    print(f"{year}: {len(scenes)} scenes to read for {len(pts)} points", flush=True)
    t0 = time.time()

    def job(s):
        idx = [i for i, p in enumerate(pts) if in_bbox(p["lon"], p["lat"], s["bbox"])]
        if not idx:
            return s["id"], []
        try:
            vals = read_scene(s, [(pts[i]["lon"], pts[i]["lat"]) for i in idx])
        except Exception as e:  # noqa: BLE001
            print("  skip", s["id"], str(e)[:80], flush=True)
            return None, []
        return s["id"], [(i, v) for i, v in zip(idx, vals) if v is not None]

    with ThreadPoolExecutor(10) as ex:
        futs = [ex.submit(job, s) for s in scenes]
        for n, f in enumerate(as_completed(futs), 1):
            sid, res = f.result()
            if sid:
                done.add(sid)
            for i, v in res:
                obs[i].append(v)
            if n % 10 == 0 or n == len(futs):
                ck.write_text(json.dumps({"done": sorted(done), "obs": obs}))
                print(f"  {n}/{len(futs)} scenes, {time.time() - t0:.0f}s", flush=True)
    return obs


def county_share(pts, obs, thr):
    by = defaultdict(list)
    for i, p in enumerate(pts):
        s = winter_score(obs.get(i, []))
        if s is not None:
            by[p["fips"]].append(s >= thr)
    return {f: 100 * sum(v) / len(v) for f, v in by.items() if len(v) >= 5}


def main() -> None:
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    census = json.loads((ROOT / "data" / "census_cover.json").read_text())
    allpts = json.loads((ROOT / "data" / "sample_points.json").read_text())
    data = {}
    years = [y for y in (2017, 2022) if len({p["fips"] for p in allpts.get(str(y), [])}) >= 10]
    for year in years:
        per = defaultdict(list)
        for p in allpts[str(year)]:
            if len(per[p["fips"]]) < cap:
                per[p["fips"]].append(p)
        pts = [p for v in per.values() for p in v]
        data[year] = (pts, collect(year, pts))

    def corr(year, thr, only=None):
        pts, obs = data[year]
        sh = county_share(pts, obs, thr)
        fs = [f for f in sh if str(year) in census.get(f, {}) and (only is None or f in only)]
        if len(fs) < 3 or len({round(sh[f], 6) for f in fs}) < 2:
            return float("-inf"), len(fs), sh
        return st.correlation([sh[f] for f in fs], [census[f][str(year)]["pct"] for f in fs]), len(fs), sh

    results = {}
    if len(years) == 2:  # tune on one census year, test on the other
        for train, test in ((2017, 2022), (2022, 2017)):
            best = max(THRESHOLDS, key=lambda t: corr(train, t)[0])
            r_tr, _, _ = corr(train, best)
            r_te, n_te, _ = corr(test, best)
            results[f"train{train}_test{test}"] = {"threshold": best, "r_train": round(r_tr, 2), "r_test": round(r_te, 2), "n_counties": n_te}
        thr = results["train2017_test2022"]["threshold"]
    else:  # one census year: tune on half the counties, test on the other half
        y = years[0]
        fs = sorted({p["fips"] for p in data[y][0]})
        halves = (set(fs[0::2]), set(fs[1::2]))
        for k, (tr, te) in enumerate((halves, halves[::-1])):
            best = max(THRESHOLDS, key=lambda t: corr(y, t, tr)[0])
            r_tr, _, _ = corr(y, best, tr)
            r_te, n_te, _ = corr(y, best, te)
            results[f"{y}_fold{k + 1}"] = {"threshold": best, "r_train": round(r_tr, 2), "r_test": round(r_te, 2), "n_counties": n_te}
        thr = round(st.mean(v["threshold"] for v in results.values()), 2)
        r_all, n_all, _ = corr(y, thr)
        results["train2017_test2022"] = {"threshold": thr, "r_test": round(st.mean(v["r_test"] for k, v in results.items() if "fold" in k), 2),
                                         "r_all": round(r_all, 2), "n_counties": n_all, "note": f"{y} only; held-out half of counties"}
    out = {"results": results, "threshold_used": thr, "counties": {}}
    for year in years:
        _, _, sh = corr(year, thr)
        for f, v in sh.items():
            if str(year) in census.get(f, {}):
                out["counties"].setdefault(f, {"name": census[f]["name"]})[str(year)] = {"satellite_pct": round(v, 1), "census_pct": census[f][str(year)]["pct"]}
    (ROOT / "data" / "cover_validation.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
