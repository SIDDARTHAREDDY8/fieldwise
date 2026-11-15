"""FieldWise API: rotation and cover-crop fit for a field, from NASA POWER history."""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from datetime import date

from .data import daily_records, fetch_forecast, fetch_power, fetch_power_current, fetch_soil
from .yieldmodel import field_yields
from .rules import Climate, climate_summary, evaluate, fly_on, planting_deadlines, this_season

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

app = FastAPI(title="FieldWise")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

WEIGHTS = {  # priority -> weights for (weather fit, soil benefit, farm economics)
    "soil": (0.40, 0.45, 0.15),
    "risk": (0.60, 0.20, 0.20),
    "profit": (0.40, 0.15, 0.45),
}
CURRENT = {"corn_soy": "baseline"}


def score(option: dict, priority: str) -> dict:
    b = option["benefits"]
    fit = option["fit"]
    spring = option.get("spring_risk_years") or []
    if option["id"] == "rye_after_soy" and option["years"]:
        fit = max(0.0, fit - 0.5 * len(spring) / option["years"])  # rye ahead of corn is the risky case
    soil = (b["soil"] + b["nitrogen"] + b["pest_break"]) / 15
    econ = (b["income"] + b["ease"]) / 10
    wf, ws, we = WEIGHTS.get(priority, WEIGHTS["soil"])
    total = 100 * (wf * fit + ws * soil + we * econ)
    if option["fit"] < 0.5:
        total *= 0.6  # rarely works here: push it down regardless of benefits
    return {"score": round(total), "weather_fit": round(fit, 2), "soil_benefit": round(soil, 2), "economics": round(econ, 2)}


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/analyze")
def analyze(lat: float, lon: float, priority: str = "soil", soy_harvest: str | None = None, corn_harvest: str | None = None):
    if not (24 <= lat <= 50 and -125 <= lon <= -66):
        raise HTTPException(400, "FieldWise is tuned for the continental US Corn Belt; pick a point in the US.")
    try:
        power = fetch_power(lat, lon)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"NASA POWER request failed: {e}") from e
    recs = daily_records(power)
    clim = Climate(recs)
    overrides = {k: v for k, v in {"soy_harvest": soy_harvest, "corn_harvest": corn_harvest}.items() if v}
    result = evaluate(clim, lat, overrides)
    soil = fetch_soil(lat, lon)
    notes = []
    if soil and soil.get("drainage") and "poorly" in soil["drainage"].lower():
        notes.append(f"Soil is {soil['drainage'].lower()}: wet springs hit harder here — favor rye before soybeans and tile-drained fields for wheat.")
    if soil and soil.get("slope_pct") and soil["slope_pct"] >= 6:
        notes.append(f"Slope ~{soil['slope_pct']:.0f}%: bare winter soil is especially erosion-prone, so living cover pays off more.")
    for o in result["options"]:
        o["scores"] = score(o, priority)
    ranked = sorted((o for o in result["options"] if o["id"] != "baseline"), key=lambda o: -o["scores"]["score"])
    return {
        "location": {"lat": round(lat, 4), "lon": round(lon, 4)},
        "priority": priority,
        "soil": soil,
        "soil_notes": notes,
        "climate": climate_summary(clim),
        "baseline": next(o for o in result["options"] if o["id"] == "baseline"),
        "ranked": ranked,
        "assumptions": result["assumptions"],
        "workable_by_year": result["workable_by_year"],
        "deadlines": planting_deadlines(clim, lat, date.today()),
        "this_season": _season(clim, lat, lon),
        "fly_on": fly_on(clim, overrides),
        "yields": _yields(recs, lat, lon),
        "today": date.today().isoformat(),
        "sources": [
            "NASA POWER daily (T2M, T2M_MIN, T2M_MAX, PRECTOTCORR, GWETTOP, TS), 2001–2025",
            "USDA NRCS Soil Data Access (SSURGO) dominant soil component",
            "Thresholds: Midwest Cover Crops Council, Ohio State Agronomy Guide, SARE Managing Cover Crops Profitably",
        ],
    }


def _yields(recs, lat, lon):
    try:
        return field_yields(recs, lat, lon, date.today().year + 1)
    except Exception:  # noqa: BLE001
        return None


@app.get("/api/week")
def week(lat: float, lon: float):
    """The next 7 days on this field: which days are dry enough to work, freezes, and rain to germinate seed."""
    today = date.today()
    try:
        f = fetch_forecast(lat, lon, today)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Forecast unavailable: {e}") from e
    rain = [x or 0.0 for x in f["precipitation_sum"]]
    days = []
    for i, d in enumerate(f["time"]):
        if d < today.isoformat():
            continue
        prev2 = sum(rain[max(0, i - 2):i])
        days.append({"date": d, "rain_in": round(rain[i], 2), "rain_chance": f["precipitation_probability_max"][i],
                     "high_f": round(f["temperature_2m_max"][i]), "low_f": round(f["temperature_2m_min"][i]),
                     # same rule as the 25-year field-day model (USDA-checked): < 2 mm today, < 8 mm the two days before
                     "workable": rain[i] < 0.08 and prev2 < 0.31})
    return {"days": days, "source": "Open-Meteo forecast (NOAA GFS/HRRR and other models)"}


def _season(clim: Climate, lat: float, lon: float):
    try:
        today = date.today()
        return this_season(clim, daily_records(fetch_power_current(lat, lon, today)), today)
    except Exception:  # noqa: BLE001 - live data is a bonus; never block the main analysis
        return None


def _cover_threshold() -> tuple[float, dict | None]:
    p = ROOT / "data" / "cover_validation.json"
    if p.exists():
        v = json.loads(p.read_text())
        return v["threshold_used"], v["results"]
    return 0.30, None


@app.get("/api/verify")
def verify(lat: float, lon: float, first: int = 2016, last: int = 2025):
    """Satellite check: was this spot green over each winter? NASA HLS 30 m, clear-sky NDVI Dec-Mar."""
    from .hls import winter_score, winter_series

    key = f"verify:{round(lat, 4)}:{round(lon, 4)}:{first}:{last}"
    from .data import _cache_get, _cache_put

    cached = _cache_get(key)
    if cached:
        return cached
    try:
        obs = winter_series(lon, lat, first, min(last, date.today().year - 1))
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    thr, val = _cover_threshold()
    winters = []
    for y in range(first, min(last, date.today().year - 1) + 1):
        vals = [o["ndvi"] for o in obs if o["winter"] == y]
        s = winter_score(vals)
        winters.append({"winter": f"{y}–{str(y + 1)[2:]}", "year": y, "clear_views": len(vals), "score": s,
                        "verdict": None if s is None else ("green cover" if s >= thr else "bare")})
    out = {"location": {"lat": lat, "lon": lon}, "threshold": thr, "validation": val, "winters": winters, "observations": obs,
           "source": "NASA HLS v2.0 (Landsat 8/9 + Sentinel-2), 30 m, Fmask-screened clear-sky NDVI, Dec 1 – Mar 31"}
    _cache_put(key, out)
    return out


@app.get("/api/ohio-grid")
def ohio_grid():
    p = ROOT / "data" / "ohio_grid.json"
    return json.loads(p.read_text()) if p.exists() else {"available": False}


@app.get("/api/validation")
def validation():
    p = ROOT / "data" / "validation.json"
    return json.loads(p.read_text()) if p.exists() else {"available": False}


class ExplainIn(BaseModel):
    analysis: dict


def _template_explanation(a: dict) -> str:
    parts = []
    for kind, lead in (("cover", "Keeping corn and soybeans? Your best cover crop is"), ("rotation", "Want a third crop? Your best option is")):
        opts = [o for o in a["ranked"] if o["kind"] == kind]
        if not opts:
            continue
        top = opts[0]
        name = top["name"][0].lower() + top["name"][1:]
        s = f"{lead} {name}. It worked {top['success_years']} of the last {top['years']} years on this field."
        if top["risks"]:
            s += " Watch out: " + top["risks"][0]
        parts.append(s)
    return "\n\n".join(parts)


@app.post("/api/explain")
def explain(body: ExplainIn):
    a = body.analysis
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return {"source": "template", "text": _template_explanation(a)}
    slim = {
        "soil": a.get("soil"), "soil_notes": a.get("soil_notes"), "priority": a.get("priority"),
        "climate": {k: v for k, v in a.get("climate", {}).items() if k != "series"},
        "baseline_risks": a.get("baseline", {}).get("risks"),
        "options": [{k: o[k] for k in ("kind", "name", "success_years", "years", "risks", "scores", "metric_name", "threshold")} for o in a.get("ranked", [])],
    }
    prompt = (
        "You are an agronomy extension advisor talking to an Ohio corn-soybean farmer. Using ONLY the computed "
        "results below (25 years of NASA POWER weather replayed against the farm calendar), write a short "
        "recommendation in plain language with two short paragraphs: (a) if they keep corn-soybean, the best "
        "cover crop; (b) if they add a third crop, the best rotation. For each: why, citing the years-out-of-N "
        "numbers, and the main risk with how to manage it. Max 140 words, no headings, no invented numbers.\n\n"
        + json.dumps(slim)
    )
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=os.getenv("FIELDWISE_MODEL", "claude-sonnet-5-5"),
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        return {"source": "ai", "text": text}
    except Exception as e:  # noqa: BLE001
        return {"source": "template", "text": _template_explanation(a), "error": str(e)[:200]}
