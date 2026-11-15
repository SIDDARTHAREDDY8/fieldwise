"""Rotation and cover-crop fit engine.

For every year of NASA POWER history it replays the farm calendar (harvest, planting,
freeze, spring field work) and checks whether each option would have worked that year.
Thresholds are rule-of-thumb values from Midwest extension guidance; they are kept in
one place (ASSUMPTIONS) so they are visible in the UI and easy to adjust.
"""
from __future__ import annotations

import statistics as st
from dataclasses import dataclass, field
from datetime import date, timedelta

ASSUMPTIONS = {
    "soy_harvest": "10-01",          # typical SW Ohio soybean harvest
    "corn_harvest": "10-15",         # typical SW Ohio corn harvest
    "corn_plant": "05-01",           # target corn planting date
    "wheat_harvest": "07-05",        # typical SW Ohio wheat harvest
    "killing_freeze_c": -2.0,        # 28 F: ends growth of frost-sensitive covers
    "hard_freeze_c": -4.0,           # ~25 F: ends fall growth for hardy small grains
    "rye_min_gdd4": 100,             # C-days above 4 C needed for cereal rye to establish
    "oats_min_days": 42,             # oats/radish need ~6 weeks before killing freeze
    "oats_min_gdd4": 350,
    "wheat_min_gdd0": 250,           # C-days above 0 C before mid-Dec for tillering
    "wheat_coldkill_c": -18.0,       # 0 F without snow cover risks winterkill
    "dcsoy_min_days": 90,            # frost-free days needed after wheat harvest
    "dcsoy_min_gdd10": 1000,         # C-days above 10 C for double-crop soybean
    "summer_cover_rain_mm": 25,      # rain in 3 weeks after seeding to germinate
    "spring_min_workable_days": 8,   # workable days Apr 10 - May 10 to terminate cover + plant
    "heavy_rain_mm": 25,             # erosive rain event (~1 inch)
    "flyon_rain_mm": 12.7,           # seed broadcast into a standing crop needs ~0.5 in of rain
    "flyon_rain_days": 7,            #   within about a week to sprout before it dries out or is eaten
    "flyon_window_days": 7,          #   seeded any day within +/- 1 week of the target, ahead of a forecast rain
    "flyon_soy_lead_days": 21,       # soybean leaf-yellowing: ~3 weeks before soybean harvest
    "flyon_corn_lead_days": 30,      # corn at black layer / leaf drop: ~4 weeks before corn harvest
}

# Qualitative agronomic ratings (0-5) summarised from extension literature (MCCC, OSU
# Agronomy Guide, SARE "Managing Cover Crops Profitably"). They are not computed from data.
BENEFITS = {
    "baseline": {"soil": 1, "nitrogen": 0, "pest_break": 1, "income": 4, "ease": 5},
    "rye_after_soy": {"soil": 5, "nitrogen": 1, "pest_break": 2, "income": 3, "ease": 3},
    "rye_after_corn": {"soil": 4, "nitrogen": 1, "pest_break": 2, "income": 3, "ease": 4},
    "oats_radish_after_soy": {"soil": 3, "nitrogen": 2, "pest_break": 2, "income": 3, "ease": 4},
    "wheat": {"soil": 4, "nitrogen": 1, "pest_break": 4, "income": 3, "ease": 3},
    "wheat_dcsoy": {"soil": 4, "nitrogen": 1, "pest_break": 4, "income": 4, "ease": 2},
    "wheat_summer_cover": {"soil": 5, "nitrogen": 3, "pest_break": 4, "income": 3, "ease": 2},
}


def _md(year: int, mmdd: str) -> date:
    m, d = mmdd.split("-")
    return date(year, int(m), int(d))


@dataclass
class Climate:
    rec: dict  # {date: {param: value}}
    years: list[int] = field(default_factory=list)

    def __post_init__(self):
        ys = sorted({d.year for d in self.rec})
        self.years = ys

    def get(self, d: date, p: str):
        v = self.rec.get(d, {}).get(p)
        return v

    def days(self, start: date, end: date):
        d = start
        while d <= end:
            if d in self.rec:
                yield d, self.rec[d]
            d += timedelta(days=1)

    def gdd(self, start: date, end: date, base: float) -> float:
        return sum(max(0.0, (r["T2M"] or 0) - base) for _, r in self.days(start, end) if r.get("T2M") is not None)

    def rain(self, start: date, end: date) -> float:
        return sum(r.get("PRECTOTCORR") or 0 for _, r in self.days(start, end))

    def first_fall_freeze(self, year: int, thresh: float) -> date | None:
        for d, r in self.days(date(year, 8, 1), date(year, 12, 31)):
            if r.get("T2M_MIN") is not None and r["T2M_MIN"] <= thresh:
                return d
        return None

    def last_spring_freeze(self, year: int, thresh: float) -> date | None:
        last = None
        for d, r in self.days(date(year, 1, 1), date(year, 6, 30)):
            if r.get("T2M_MIN") is not None and r["T2M_MIN"] <= thresh:
                last = d
        return last

    def spring_wet_threshold(self) -> float:
        vals = sorted(r["GWETTOP"] for d, r in self.rec.items() if d.month in (4, 5) and r.get("GWETTOP") is not None)
        return vals[int(0.75 * (len(vals) - 1))] if vals else 1.0

    def is_workable(self, d: date) -> bool:
        """Dry enough for field work: <2 mm rain today and <8 mm over the previous two days.

        Validated against USDA NASS weekly "days suitable for fieldwork" for Ohio, Apr 10 - May 16,
        2001-2025 (r = 0.74; see scripts/validate.py). Adding a GWETTOP soil-wetness cutoff
        lowered agreement, so the rule is rain-only.
        """
        p0 = self.get(d, "PRECTOTCORR") or 0
        p12 = (self.get(d - timedelta(days=1), "PRECTOTCORR") or 0) + (self.get(d - timedelta(days=2), "PRECTOTCORR") or 0)
        return d in self.rec and p0 < 2 and p12 < 8

    def workable_days(self, year: int, start="04-10", end="05-10", wet_thresh: float | None = None) -> int:
        return sum(1 for d, _ in self.days(_md(year, start), _md(year, end)) if self.is_workable(d))


def fly_free_date(lat: float) -> str:
    """Approximate Ohio Hessian fly-safe wheat seeding date: ~Sep 22 in the north to ~Oct 5 in the south."""
    north, south = 41.7, 38.5
    frac = min(1.0, max(0.0, (north - lat) / (north - south)))
    d = date(2001, 9, 22) + timedelta(days=round(13 * frac))
    return d.strftime("%m-%d")


def _summ(per_year: list[dict]) -> dict:
    n = len(per_year)
    ok = sum(1 for y in per_year if y["ok"])
    return {"years": n, "success_years": ok, "fit": ok / n if n else 0.0}


def evaluate(c: Climate, lat: float, a: dict | None = None) -> dict:
    a = {**ASSUMPTIONS, **(a or {})}
    years = c.years
    # fall options need the fall of year Y and the spring of Y+1
    fall_years = [y for y in years if y + 1 in years]
    wet = c.spring_wet_threshold()
    workable = {y: c.workable_days(y, wet_thresh=wet) for y in years}
    fly = fly_free_date(lat)
    options = []

    # --- Baseline corn-soy: exposure of bare soil to erosive rain Nov-Apr
    per = []
    for y in fall_years:
        heavy = sum(1 for _, r in c.days(date(y, 11, 1), date(y + 1, 4, 30)) if (r.get("PRECTOTCORR") or 0) >= a["heavy_rain_mm"])
        per.append({"year": y, "ok": True, "metric": heavy, "label": "erosive rain days on bare soil"})
    options.append({
        "id": "baseline", "kind": "baseline", "name": "Corn–soybean, no cover (current)",
        "per_year": per, **_summ(per),
        "metric_name": f"Rain days ≥{a['heavy_rain_mm']} mm on bare soil (Nov–Apr)", "threshold": None,
        "risks": [f"Soil is bare Nov–Apr; on average {st.mean(p['metric'] for p in per):.1f} erosive rain days hit it each winter."],
    })

    # --- Cereal rye after soybean / corn
    for oid, harvest_key, next_crop in (("rye_after_soy", "soy_harvest", "corn"), ("rye_after_corn", "corn_harvest", "soybean")):
        per = []
        for y in fall_years:
            plant = _md(y, a[harvest_key]) + timedelta(days=3)
            g = c.gdd(plant, date(y, 12, 31), 4.0)
            spring_ok = workable.get(y + 1, 0) >= a["spring_min_workable_days"]
            per.append({"year": y, "ok": g >= a["rye_min_gdd4"], "metric": round(g), "spring_workable": workable.get(y + 1), "spring_ok": spring_ok})
        risky = [p["year"] + 1 for p in per if not p["spring_ok"]]
        risks = []
        if next_crop == "corn" and risky:
            risks.append(f"Wet springs ({len(risky)} of {len(per)} years, e.g. {', '.join(map(str, risky[-3:]))}) left too few dry days to kill the rye 2 weeks before corn planting. Kill it early, or put rye ahead of soybeans instead.")
        elif risky:
            risks.append(f"Wet springs ({len(risky)} of {len(per)} years) make rye hard to kill on time, but soybeans can be planted green into rye, so the risk is lower.")
        options.append({
            "id": oid, "kind": "cover",
            "name": f"Cereal rye cover after {'soybean' if harvest_key == 'soy_harvest' else 'corn'} harvest",
            "per_year": per, **_summ(per), "spring_risk_years": risky,
            "metric_name": "Growing degree days (base 4 °C) from seeding to Dec 31", "threshold": a["rye_min_gdd4"],
            "risks": risks,
        })

    # --- Oats + radish (winter-killed) after soybean
    per = []
    for y in fall_years:
        plant = _md(y, a["soy_harvest"]) + timedelta(days=3)
        frz = c.first_fall_freeze(y, a["killing_freeze_c"]) or date(y, 12, 31)
        days_ = (frz - plant).days
        g = c.gdd(plant, frz, 4.0)
        per.append({"year": y, "ok": days_ >= a["oats_min_days"] and g >= a["oats_min_gdd4"], "metric": round(g), "days_before_freeze": days_})
    options.append({
        "id": "oats_radish_after_soy", "kind": "cover", "name": "Oats + radish (winter-killed) after soybean",
        "per_year": per, **_summ(per),
        "metric_name": "GDD (base 4 °C) from seeding to first killing freeze", "threshold": a["oats_min_gdd4"],
        "risks": ["Needs ~6 weeks of growth before a killing freeze; after an Oct 1 soybean harvest it often runs out of season. Aerial or interseeding in early September fixes this."],
    })

    # --- Third crop: winter wheat after soybean
    per = []
    for y in fall_years:
        plant = max(_md(y, a["soy_harvest"]) + timedelta(days=2), _md(y, fly))
        g = c.gdd(plant, date(y, 12, 15), 0.0)
        cold = sum(1 for _, r in c.days(date(y, 12, 1), date(y + 1, 2, 28)) if (r.get("T2M_MIN") or 0) <= a["wheat_coldkill_c"])
        per.append({"year": y, "ok": g >= a["wheat_min_gdd0"], "metric": round(g), "cold_days": cold, "plant": plant.isoformat()})
    cold_years = [p["year"] + 1 for p in per if p["cold_days"] >= 2]
    wheat_per = per
    options.append({
        "id": "wheat", "kind": "rotation", "name": "Add winter wheat: corn → soybean → wheat",
        "per_year": per, **_summ(per),
        "metric_name": f"GDD (base 0 °C) from seeding (after fly-safe date {fly}) to Dec 15", "threshold": a["wheat_min_gdd0"],
        "risks": ([f"Winterkill risk: ≥2 nights at or below 0 °F in {len(cold_years)} winters ({', '.join(map(str, cold_years[-3:]))}) — snow cover usually protects."] if cold_years else [])
        + ["Wheat earns less per acre than corn; the payoff is yield boost to the next corn/soy crop and a summer window for a second crop or cover."],
    })

    # --- Wheat + double-crop soybean
    per = []
    for y in years:
        plant = _md(y, a["wheat_harvest"]) + timedelta(days=3)
        frz = c.first_fall_freeze(y, -1.0)
        if not frz:
            continue
        g = c.gdd(plant, frz, 10.0)
        days_ = (frz - plant).days
        rain = c.rain(plant, plant + timedelta(days=60))
        per.append({"year": y, "ok": days_ >= a["dcsoy_min_days"] and g >= a["dcsoy_min_gdd10"], "metric": round(g), "days_before_frost": days_, "rain_60d": round(rain)})
    dry = [p["year"] for p in per if p["rain_60d"] < 120]
    options.append({
        "id": "wheat_dcsoy", "kind": "rotation", "name": "Wheat + double-crop soybean after wheat harvest",
        "per_year": per, **_summ(per),
        "metric_name": "GDD (base 10 °C) from July seeding to first frost", "threshold": a["dcsoy_min_gdd10"],
        "risks": [f"Dry July–August ({len(dry)} of {len(per)} years had under 4.7 inches of rain in the 60 days after planting) cuts double-crop yields."] if dry else [],
    })

    # --- Wheat + summer cover (oats/radish/clover mix) after wheat harvest
    per = []
    for y in years:
        plant = _md(y, a["wheat_harvest"]) + timedelta(days=20)
        rain = c.rain(plant, plant + timedelta(days=21))
        per.append({"year": y, "ok": rain >= a["summer_cover_rain_mm"], "metric": round(rain)})
    options.append({
        "id": "wheat_summer_cover", "kind": "rotation", "name": "Wheat + summer cover mix (oats/radish/clover) after wheat",
        "per_year": per, **_summ(per),
        "metric_name": "Rain (mm) in the 3 weeks after late-July seeding", "threshold": a["summer_cover_rain_mm"],
        "risks": ["Germination depends on late-July rain; in dry years seed into moisture or wait for a forecast rain."],
    })

    for o in options:
        o["benefits"] = BENEFITS[o["id"]]
    return {"options": options, "assumptions": a, "fly_free_date": fly, "workable_by_year": workable, "wet_threshold": round(wet, 3)}


def _plant_ok(c: Climate, option: str, plant: date, a: dict, fly: str) -> bool | None:
    """Would `option` seeded on `plant` have worked that season? None if data is missing."""
    y = plant.year
    if option == "cereal_rye":
        if date(y, 12, 31) not in c.rec:
            return None
        return c.gdd(plant, date(y, 12, 31), 4.0) >= a["rye_min_gdd4"]
    if option == "oats_radish":
        frz = c.first_fall_freeze(y, a["killing_freeze_c"]) or date(y, 12, 31)
        return (frz - plant).days >= a["oats_min_days"] and c.gdd(plant, frz, 4.0) >= a["oats_min_gdd4"]
    if option == "winter_wheat":
        if plant < _md(y, fly):
            return False  # Hessian fly risk before the fly-safe date
        return c.gdd(plant, date(y, 12, 15), 0.0) >= a["wheat_min_gdd0"]
    if option == "double_crop_soy":
        frz = c.first_fall_freeze(y, -1.0)
        if not frz:
            return None
        return (frz - plant).days >= a["dcsoy_min_days"] and c.gdd(plant, frz, 10.0) >= a["dcsoy_min_gdd10"]
    raise ValueError(option)


DEADLINE_OPTIONS = {
    "cereal_rye": ("Cereal rye (cover)", "09-01", "12-15"),
    "oats_radish": ("Oats + radish (cover)", "08-01", "11-15"),
    "winter_wheat": ("Winter wheat (third crop)", "09-15", "12-01"),
    "double_crop_soy": ("Double-crop soybean after wheat", "06-15", "08-15"),
}


def planting_deadlines(c: Climate, lat: float, today: date | None = None, a: dict | None = None) -> dict:
    """For each option, the share of past years in which seeding on each calendar day worked,
    and the latest day that still worked in >=90%, >=80% and >=50% of years."""
    a = {**ASSUMPTIONS, **(a or {})}
    fly = fly_free_date(lat)
    years = [y for y in c.years if date(y, 12, 31) in c.rec]
    mid = years[len(years) // 2]  # split into an early and a recent half
    out = {}
    for key, (label, start, end) in DEADLINE_OPTIONS.items():
        curve, early_curve, late_curve = [], [], []
        d = _md(2001, start)
        while d <= _md(2001, end):
            res = {y: _plant_ok(c, key, date(y, d.month, d.day), a, fly) for y in years}
            res = {y: r for y, r in res.items() if r is not None}
            md = d.strftime("%m-%d")
            if res:
                curve.append({"md": md, "share": round(sum(res.values()) / len(res), 3), "n": len(res)})
                e = [r for y, r in res.items() if y < mid]
                l_ = [r for y, r in res.items() if y >= mid]
                if e and l_:
                    early_curve.append({"md": md, "share": sum(e) / len(e)})
                    late_curve.append({"md": md, "share": sum(l_) / len(l_)})
            d += timedelta(days=1)

        def latest(th: float, cv=curve):
            ok = [p["md"] for p in cv if p["share"] >= th]
            return ok[-1] if ok else None

        def shift_days(th: float):
            e, l_ = latest(th, early_curve), latest(th, late_curve)
            if not e or not l_:
                return None
            return (date.fromisoformat(f"2001-{l_}") - date.fromisoformat(f"2001-{e}")).days

        first_ok = next((p["md"] for p in curve if p["share"] >= 0.8), None)
        entry = {"label": label, "curve": curve, "p90": latest(0.9), "p80": latest(0.8), "p50": latest(0.5), "opens": first_ok,
                 "shift": {"early_years": f"{years[0]}–{mid - 1}", "recent_years": f"{mid}–{years[-1]}",
                           "early_p80": latest(0.8, early_curve), "recent_p80": latest(0.8, late_curve), "days": shift_days(0.8)}}
        if today:
            md = today.strftime("%m-%d")
            hit = next((p for p in curve if p["md"] == md), None)
            entry["today"] = {"md": md, "share": hit["share"] if hit else (0.0 if md > curve[-1]["md"] else None)}
        out[key] = entry
    return {"options": out, "fly_free_date": fly, "years": f"{years[0]}–{years[-1]}"}


def fly_on(c: Climate, a: dict | None = None) -> dict:
    """Late harvest? Cereal rye broadcast (by plane, drone or high-clearance seeder) into the standing
    crop weeks before harvest. It works when ~0.5 in of rain falls within a week to sprout the seed
    and the rye still gets enough fall growth. Replayed for every past season."""
    a = {**ASSUMPTIONS, **(a or {})}
    years = [y for y in c.years if date(y, 12, 31) in c.rec]
    out = {}
    for crop, hk, lead in (("soybeans", "soy_harvest", "flyon_soy_lead_days"), ("corn", "corn_harvest", "flyon_corn_lead_days")):
        per = []
        for y in years:
            # the farmer can fly it on any day in a 2-week window, timed ahead of a forecast rain
            target = _md(y, a[hk]) - timedelta(days=a[lead])
            tries = [target + timedelta(days=k) for k in range(-a["flyon_window_days"], a["flyon_window_days"] + 1)]
            seed = max(tries, key=lambda d: c.rain(d, d + timedelta(days=a["flyon_rain_days"] - 1)))
            rain = c.rain(seed, seed + timedelta(days=a["flyon_rain_days"] - 1))
            g = c.gdd(seed, date(y, 12, 31), 4.0)
            drilled = c.gdd(_md(y, a[hk]) + timedelta(days=3), date(y, 12, 31), 4.0) >= a["rye_min_gdd4"]
            per.append({"year": y, "ok": rain >= a["flyon_rain_mm"] and g >= a["rye_min_gdd4"], "rain_in": round(rain / 25.4, 2), "drilled_ok": drilled})
        n = len(per)
        out[crop] = {"seed_md": (_md(2001, a[hk]) - timedelta(days=a[lead])).strftime("%m-%d"), "harvest_md": a[hk],
                     "per_year": per, "years": n, "success_years": sum(p["ok"] for p in per),
                     "fit": sum(p["ok"] for p in per) / n if n else 0.0,
                     "drilled_fit": sum(p["drilled_ok"] for p in per) / n if n else 0.0}
    return out


def this_season(hist: Climate, now: dict, today: date, a: dict | None = None) -> dict | None:
    """Compare this year's fall (and spring) so far with the same dates in every past year."""
    a = {**ASSUMPTIONS, **(a or {})}
    days = sorted(d for d, r in now.items() if r.get("T2M") is not None)
    if not days:
        return None
    last = days[-1]
    y = today.year
    cur = Climate(now)
    out: dict = {"data_through": last.isoformat()}

    def window_stats(c: Climate, yr: int, start: str, end_md: str):
        s, e = _md(yr, start), _md(yr, end_md)
        temps = [r["T2M"] for _, r in c.days(s, e) if r.get("T2M") is not None]
        return (st.mean(temps) if temps else None, c.rain(s, e), c.gdd(s, e, 4.0))

    if last >= date(y, 9, 15):
        end_md = last.strftime("%m-%d")
        t_now, r_now, g_now = window_stats(cur, y, "09-01", end_md)
        past = [window_stats(hist, yy, "09-01", end_md) for yy in hist.years]
        t_norm = st.mean(p[0] for p in past if p[0] is not None)
        r_norm = st.mean(p[1] for p in past)
        g_norm = st.mean(p[2] for p in past)
        frz = cur.first_fall_freeze(y, a["killing_freeze_c"])
        hist_frz = [hist.first_fall_freeze(yy, a["killing_freeze_c"]) for yy in hist.years]
        norm_doy = st.mean(f.timetuple().tm_yday for f in hist_frz if f)
        warmer = sum(1 for p in past if p[0] is not None and p[0] < t_now)
        out["fall"] = {
            "since": "Sep 1", "temp_anomaly_c": round(t_now - t_norm, 1), "rain_pct_of_normal": round(100 * r_now / r_norm) if r_norm else None,
            "gdd4_now": round(g_now), "gdd4_normal": round(g_norm),
            "warmer_than_n_of": [warmer, len(past)],
            "first_freeze": frz.isoformat() if frz else None,
            "first_freeze_vs_normal_days": (frz.timetuple().tm_yday - round(norm_doy)) if frz else None,
            "normal_first_freeze": (date(y, 1, 1) + timedelta(days=round(norm_doy) - 1)).isoformat(),
        }
    if last >= date(y, 5, 10):
        w_now = cur.workable_days(y)
        w_hist = sorted(hist.workable_days(yy) for yy in hist.years)
        out["spring"] = {"workable_days": w_now, "median_past": w_hist[len(w_hist) // 2],
                         "wetter_than_n_of": [sum(1 for w in w_hist if w > w_now), len(w_hist)]}
    return out


def climate_summary(c: Climate) -> dict:
    ys = [y for y in c.years if y + 1 in c.years]
    first = [c.first_fall_freeze(y, -2.0) for y in ys]
    last = [c.last_spring_freeze(y, -2.0) for y in c.years]
    doy = lambda d: d.timetuple().tm_yday
    first_doys = [doy(d) for d in first if d]
    last_doys = [doy(d) for d in last if d]
    gdd10 = {y: round(c.gdd(date(y, 5, 1), date(y, 9, 30), 10.0)) for y in c.years}
    precip = {y: round(c.rain(date(y, 1, 1), date(y, 12, 31))) for y in c.years}
    heavy = {y: sum(1 for _, r in c.days(date(y, 1, 1), date(y, 12, 31)) if (r.get("PRECTOTCORR") or 0) >= 25) for y in c.years}
    spring_wet = {y: round(st.mean(r["GWETTOP"] for _, r in c.days(date(y, 4, 1), date(y, 5, 31)) if r.get("GWETTOP") is not None), 3) for y in c.years}

    def trend(series: dict) -> float:
        xs, vs = list(series.keys()), list(series.values())
        mx, mv = st.mean(xs), st.mean(vs)
        den = sum((x - mx) ** 2 for x in xs)
        return sum((x - mx) * (v - mv) for x, v in zip(xs, vs)) / den if den else 0.0

    as_date = lambda n: (date(2001, 1, 1) + timedelta(days=round(n) - 1)).strftime("%b %d")
    return {
        "years": f"{c.years[0]}–{c.years[-1]}",
        "avg_first_fall_freeze": as_date(st.mean(first_doys)) if first_doys else None,
        "avg_last_spring_freeze": as_date(st.mean(last_doys)) if last_doys else None,
        "avg_frost_free_days": round(st.mean(first_doys) - st.mean(last_doys)) if first_doys and last_doys else None,
        "avg_annual_precip_mm": round(st.mean(precip.values())),
        "heavy_rain_days_per_year": round(st.mean(heavy.values()), 1),
        "heavy_rain_trend_per_decade": round(trend(heavy) * 10, 1),
        "growing_gdd10_may_sep": round(st.mean(gdd10.values())),
        "first_freeze_trend_days_per_decade": round(trend({y: doy(d) for y, d in zip(ys, first) if d}) * 10, 1) if first_doys else None,
        "series": {"precip": precip, "heavy_rain_days": heavy, "spring_wetness": spring_wet, "gdd10": gdd10,
                   "first_freeze_doy": {y: doy(d) for y, d in zip(ys, first) if d}},
    }
