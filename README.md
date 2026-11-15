# FieldWise

**Should this field get a cover crop or a third crop?** FieldWise replays 25 years of NASA
weather and soil-wetness history on a farmer's own calendar to show how often each option
would have actually worked on *their* field.

NASA Space Apps Challenge 2026 — *Adapting Farms with NASA Data* (Cincinnati local event).

## What's new here
- **Your last safe planting date.** For each option FieldWise tries every seeding day across 25 seasons
  and reports the latest date that still worked in 9 of 10 years (e.g. "Plant cereal rye by Oct 24"),
  plus how often seeding *today* would have worked.
- **Ohio deadline map.** The same calculation on every NASA POWER grid cell over Ohio
  (`scripts/build_grid.py`). It shows Lake Erie's shoreline buying ~10 extra fall days and northwest
  Ohio closing first.
- **Checked against USDA.** The spring field-day model tracks USDA's weekly "days suitable for
  fieldwork" for Ohio (r = 0.74, 2001–2025).

- **What each crop would make on your field (machine learning).** A gradient-boosted model learned how
  NASA POWER weather moves Ohio county yields (USDA NASS, 2001–2025, ~5,300 county-years). FieldWise replays
  your field's last 25 seasons of weather at today's yield levels: typical yield, a 1-in-10 bad year, and why
  the worst years were bad ("2012: a dry June"). See the validation table below.
- **Should you add wheat?** Combines the planting odds, the yields and your own prices and costs
  (editable on the page) into one answer: corn–soybean vs corn–soybean–wheat (± double-crop soybeans), $/acre/year.
- **This week on your field.** 7-day forecast turned into field days (the same USDA-checked rule) and the best
  seeding window ahead of a soaking rain.
- **Your plan for the next 12 months.** A dated to-do list (cost-share call, seed order, seeding, spring kill)
  plus the crop-insurance rule for killing cover crops in Ohio (USDA zone 4).
- **Harvest running late?** Replays seeding cereal rye into the standing crop (plane, drone or high-clearance
  seeder) ahead of a half-inch rain, and compares it with drilling after your own harvest date. On the demo
  field with a Nov 10 corn harvest, drilled rye worked 1 year in 10; flown on in early October it worked 8 in 10.
- **Is your fall changing?** The first hard freeze for every fall since 2001 on your field, with its trend, how far the
  last safe rye date has moved (2013–2025 vs 2001–2012), and the trend in 1-inch rain days. On the demo field the
  safe rye date now comes 5 days earlier.
- **Take it with you.** One button puts the plan dates into a phone calendar (.ics, with a 2-day reminder); another copies
  a link that reopens the same field and dates, for sharing with a crop adviser or Soil and Water office.
- **Your soil.** Topsoil organic matter, pH and root-zone water-holding capacity from SSURGO.

## How it works
1. Click a field on the map, pick a priority (soil health / low weather risk / income) and harvest dates.
2. The backend pulls NASA POWER daily data (2001–2025) for that point and USDA SSURGO soil.
3. For every year it simulates the farm calendar: harvest → seeding → growing degree days →
   killing freeze → next spring's workable field days.
4. Options are scored on weather fit (years that worked), soil benefit and farm economics:
   - **A. Keep corn–soybean + cover crop:** cereal rye after soybean, cereal rye after corn, oats + radish after soybean
   - **B. Add a third crop:** winter wheat, wheat + double-crop soybean, wheat + summer cover mix
5. An optional AI layer (Claude) turns the computed results into a plain-language recommendation;
   all numbers come from the data, not the model.

## Try it on the web
**https://siddarthareddy8.github.io/fieldwise/** (Ohio only).

GitHub Pages can't run Python, so `scripts/build_pages.py` precomputes the full analysis for all 54 NASA POWER
grid cells over Ohio at three harvest-date presets (early, typical, late) into `docs/`. The page snaps your pin to
its grid cell and the nearest preset, and says so on screen. NASA POWER weather is the same anywhere inside one
cell, so the dates and odds match the full app for that cell; soil comes from the cell center. The 7-day forecast
is fetched live from Open-Meteo in your browser. For your exact field and harvest dates, run it locally:

## Run
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python fastapi "uvicorn[standard]" httpx anthropic python-dotenv scikit-learn numpy
cp .env.example .env   # optional keys
.venv/bin/uvicorn app.main:app --port 8077
```

## Validation
`scripts/validate.py` compares the model's spring workable days (from NASA POWER rainfall,
averaged over six Ohio farm regions) with USDA NASS Crop Progress **days suitable for fieldwork**
in Ohio, mid-April to mid-May, 2001–2025:

| Day rule | r (all) | r 2001–13 | r 2014–25 | mean error |
|---|---|---|---|---|
| **rain only (used)** | **0.74** | 0.79 | 0.91 | 3.1 days |
| rain + soil wetness < p95 | 0.68 | 0.74 | 0.90 | 3.1 days |
| rain + soil wetness < p85 | 0.59 | 0.68 | 0.77 | 3.5 days |
| rain + soil wetness < p75 | 0.51 | 0.60 | 0.69 | 4.1 days |

The model's three wettest springs (2019, 2011, 2002) are all in USDA's four wettest.

### Yield model (`scripts/yields.py`, `scripts/train_yield.py`)
Yield = state trend + county level + weather effect (gradient-boosted trees on NASA POWER season weather:
growing degree days, heat above 86 °F, monthly rain, wet planting days, soil wetness, winter cold).
Leave-one-year-out test: every county in the held-out year is predicted without seeing that year.

| Crop | County-years | Error, trend + county only | Error with NASA weather | Weather effect explained | Statewide good/bad-year match |
|---|---|---|---|---|---|
| Corn | 1,965 | 16.2 bu/a | **13.4 bu/a** | 31% | r = 0.74 |
| Soybeans | 1,873 | 4.6 bu/a | **4.2 bu/a** | 17% | r = 0.55 |
| Winter wheat | 1,429 | 7.0 bu/a | 7.0 bu/a | 3% | r = 0.40 |

Weather did not predict wheat yields (disease and planting date matter more than our weather features
capture), so for wheat FieldWise uses the county's actual USDA record, detrended, instead of the model.
The model misses the size of extreme years (2002 and 2012 corn), so the "bad year" figures are on the mild side.

## Experimental: satellite record of winter cover
`app/hls.py` reads NASA HLS v2.0 (Landsat 8/9 + Sentinel-2, 30 m) for a field: clear-sky,
Fmask-screened NDVI for every winter since 2016 (needs a free Earthdata login in `.env`).

We tried to validate it statewide against USDA Census of Agriculture cover-crop acres
(2017; 27 Ohio counties, the ones whose USDA Cropland Data Layer maps downloaded before the
CropScape service went down), sampling corn/soy fields that stayed corn/soy the next year:

| Attempt | Method | Points | r on held-out counties |
|---|---|---|---|
| 1 (`scripts/validate_cover.py`) | Dec–Mar peak NDVI | 270 | −0.09 |
| 2 (`scripts/validate_cover2.py`) | Mar–Apr NDVI relative to same-image median | 810 | −0.21 |

Neither found a county-level relationship, so the app labels the satellite panel
**experimental** and treats it as a record of winter greenness, not proof of a cover crop.
Likely reasons: snow and colder winters in the counties with the most cover crops, spring
weeds, a small county sample, and census acres that include fields we didn't sample.
Field-level ground truth is the next step.

## Data sources
- NASA POWER daily: T2M, T2M_MIN, T2M_MAX, PRECTOTCORR, GWETTOP, TS
- NASA GIBS map layers: MODIS Terra NDVI (8-day), SMAP L4 surface soil moisture, MODIS true color
- USDA NRCS Soil Data Access (SSURGO)
- USDA NASS Quick Stats: Ohio county yields for corn, soybeans, winter wheat (model training); crop progress (validation)
- US Census county centroids (matching counties to NASA POWER cells)
- Open-Meteo 7-day forecast (this week's field days)
- Agronomic thresholds: Midwest Cover Crops Council, Ohio State Agronomy Guide, SARE *Managing Cover Crops Profitably*

## Use of Super Intelligence (SI) / AI
- **Code:** the application code (backend, rule engine, UI, validation script) was written with the help of
  Claude (Anthropic) as an AI coding assistant, then reviewed and tested by the author.
- **Research:** an AI assistant helped summarise past Space Apps winners and the challenge pages.
- **In-app text:** the plain-language recommendation can be generated at runtime by Claude from the computed
  results; the UI labels it as AI-written. All numbers come from NASA/USDA data, not from the model.
- No AI-generated images, video or audio are used. No NASA logos or branding are used.

## License
MIT — see `LICENSE`.

## Libraries
FastAPI, Uvicorn, HTTPX, scikit-learn, NumPy, Anthropic Python SDK, python-dotenv, Leaflet, OpenStreetMap and Esri imagery tiles.
