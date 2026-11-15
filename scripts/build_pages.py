"""Build the GitHub Pages version of FieldWise into docs/.

GitHub Pages only serves static files, so this precomputes the full analysis for every NASA POWER
grid cell over Ohio (the same 54 cells as scripts/build_grid.py) at three harvest-date presets.
The page snaps a clicked field to its grid cell and preset. NASA POWER weather is the same anywhere
inside one cell, so the dates and odds match the live app for that cell; soil comes from the cell
center instead of the exact pin. The 7-day forecast is fetched live from Open-Meteo in the browser.

    .venv/bin/python scripts/build_pages.py
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.main import WEIGHTS, _template_explanation, analyze, score  # noqa: E402

DOCS = ROOT / "docs"
SNAP = DOCS / "snap"
PRESETS = {  # name -> (soybean harvest, corn harvest)
    "early": ("09-25", "10-05"),
    "typical": ("10-01", "10-15"),
    "late": ("10-20", "11-05"),
}


def build_one(i: int, cell: dict, name: str, soy: str, corn: str) -> str:
    out = SNAP / f"{i}-{name}.json"
    if out.exists():
        return f"{i}-{name} cached"
    a = analyze(cell["lat"], cell["lon"], "soil", soy, corn)
    opts = [a["baseline"], *a["ranked"]]
    by = {}
    for p in WEIGHTS:
        sc = {o["id"]: score(o, p) for o in opts}
        ranked = sorted((o for o in a["ranked"]), key=lambda o: -sc[o["id"]]["score"])
        by[p] = {"order": [o["id"] for o in ranked], "scores": sc,
                 "explain": _template_explanation({**a, "ranked": [{**o, "scores": sc[o["id"]]} for o in ranked]})}
    a["by_priority"] = by
    a["preset"] = {"name": name, "soy_harvest": soy, "corn_harvest": corn}
    out.write_text(json.dumps(a, separators=(",", ":")))
    return f"{i}-{name} ok"


def main() -> None:
    grid = json.loads((ROOT / "data" / "ohio_grid.json").read_text())
    SNAP.mkdir(parents=True, exist_ok=True)
    # preset-major order: the two workers never fetch the same cell at once (the file cache isn't locked)
    jobs = [(i, c, n, *PRESETS[n]) for n in PRESETS for i, c in enumerate(grid["cells"])]
    t = time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:  # 8 GB Mac: keep it light
        for k, msg in enumerate(ex.map(lambda j: build_one(*j), jobs), 1):
            print(f"[{k}/{len(jobs)}] {msg} ({time.time() - t:.0f}s)", flush=True)
    (SNAP / "index.json").write_text(json.dumps({
        "built": date.today().isoformat(), "cell_deg": grid["cell_deg"],
        "cells": [{"i": i, "lat": c["lat"], "lon": c["lon"]} for i, c in enumerate(grid["cells"])],
        "outline": grid.get("outline"), "presets": PRESETS}, separators=(",", ":")))
    html = (ROOT / "static" / "index.html").read_text().replace("<html lang=\"en\">", "<html lang=\"en\" data-static=\"1\">", 1)
    assert 'data-static="1"' in html, "index.html <html> tag changed; update build_pages.py"
    (DOCS / "index.html").write_text(html)
    (DOCS / ".nojekyll").write_text("")
    print("done", sum(f.stat().st_size for f in SNAP.iterdir()) // 1024, "KB")


if __name__ == "__main__":
    main()
