#!/usr/bin/env python3
"""Web app for the India 3D map, the world risk map and the damage-grade model.

Run from the project folder:
    python3 start.py
    # or
    uvicorn app:app --host 0.0.0.0 --port 8000

Pages
  GET  /                 India in 3D: state outlines, national hazard grid, epicentres,
                         and Assam's Overture footprints at building level
  GET  /place/{id}       retired per-state pages, redirect to /
  GET  /2d               world risk map
  GET  /2d/assam         Assam hazard detail + Gorkha damage table
  GET  /docs             API documentation
  GET  /3d, /lab         redirect to /
  GET  /globe            redirect to /, /world redirects to /2d

API
  GET  /api/health       status
  GET  /api/model        model metrics and protocol
  GET  /api/metrics      final model metrics (accuracy, macro-F1, per class)
  GET  /api/options      form fields with allowed values and defaults
  POST /api/predict      damage grade and probabilities for one building
  GET  /api/examples     model run on a random sample of real held-out survey records
  GET  /api/results      model comparison and final confusion matrix
  GET  /api/areas        area summaries on the held-out test split
  GET  /api/districts    Nepal district boundaries (for the archive map)
  GET  /api/hazard       hazard method, thresholds and cell counts
  GET  /api/events       M5+ earthquake events (GeoJSON)
  GET  /api/damage-zones Gorkha 2015 damage zones by anonymised area
  GET  /api/india/status       states, in-India epicentres, hazard cells, per-state rows
  GET  /api/india/hazard       national 0.25 degree hazard grid
  GET  /api/india/events       catalogued events that fall inside an Indian outline
  GET  /api/india/state/{id}   one state: outline, events, largest event
  GET  /api/places             the places registry and its counts
  GET  /api/places/{id}        hazard grid + epicentres for one place
  GET  /api/world/status       what the world risk map has loaded
  GET  /api/world/countries    Natural Earth 1:110m countries as lon/lat rings
  GET  /api/world/cities       GeoNames cities, population >= 100,000
  GET  /api/world/events       global M5+ / M6+ catalogue (live USGS, bundled fallback)
  GET  /api/world/hazard       world hazard grid computed from that catalogue
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
sys.path.insert(0, ROOT)
import india as ind  # noqa: E402  (whole-India layer for the 3D map at /)
import model as dm  # noqa: E402  (provides DamageModel used by the saved bundle)
import places as pl  # noqa: E402  (the 3D places view and its hazard grids)
import world as wd  # noqa: E402  (world-scale data for the /2d risk map)

MODEL_PATH = os.path.join(ROOT, "models", "damage_model.joblib")
DATA = os.path.join(ROOT, "data")
PUBLIC = os.path.join(ROOT, "public")
HAZARD = os.path.join(PUBLIC, "static", "hazard")

app = FastAPI(
    title="Worldwide earthquake hazard and building damage",
    version="1.2",
    description=(
        "A 3D map of the whole of India with its real state outlines, hazard grid and "
        "epicentres plus Assam's building footprints; a 2D world risk map; and a "
        "building damage-grade model."
    ),
    # /docs is our own human-readable page, so the generated Swagger UI is turned off.
    docs_url=None,
    redoc_url=None,
)
app.mount("/static", StaticFiles(directory=os.path.join(PUBLIC, "static")), name="static")


def _read_json(name: str) -> Any:
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        raise HTTPException(503, detail=f"{name} missing.")
    with open(path) as f:
        return json.load(f)


def _read_hazard(name: str) -> Any:
    path = os.path.join(HAZARD, name)
    if not os.path.exists(path):
        raise HTTPException(
            503, detail=f"{name} missing. Run scripts/build_hazard.py first."
        )
    with open(path) as f:
        return json.load(f)


_bundle = None
_options = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(503, detail="Model file not found.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def get_options():
    global _options
    if _options is None:
        _options = _read_json("options.json")
    return _options


def _coerce(field: dict, value: Any) -> Any:
    name, kind = field["name"], field["kind"]
    if kind == "number":
        v = float(value)
        if not (field["min"] <= v <= field["max"]):
            raise ValueError(f"{name} must be between {field['min']:g} and {field['max']:g}")
        return v
    if kind == "geo":
        v = int(value)
        if not (field["min"] <= v <= field["max"]):
            raise ValueError(f"{name} must be between {field['min']} and {field['max']}")
        return v
    if kind == "binary":
        v = int(value)
        if v not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1")
        return v
    if kind == "choice":
        v = str(value)
        if v not in field["values"]:
            raise ValueError(f"{name} must be one of the listed values")
        return v
    raise ValueError(f"unknown field kind for {name}")


def build_frame(payload: dict) -> pd.DataFrame:
    fields = get_options()["fields"]
    missing = [f["name"] for f in fields if f["name"] not in payload]
    if missing:
        raise HTTPException(422, detail={"missing": missing})
    row = {}
    errors = {}
    for f in fields:
        try:
            row[f["name"]] = _coerce(f, payload[f["name"]])
        except (ValueError, TypeError) as e:
            errors[f["name"]] = str(e)
    if errors:
        raise HTTPException(422, detail={"invalid": errors})
    return pd.DataFrame([row])


GRADE_LABELS = {1: "Low damage", 2: "Medium damage", 3: "Near-complete destruction"}
GRADE_HELP = {
    1: "Minor or no structural damage expected from this survey profile.",
    2: "Moderate damage: repair likely needed before reuse.",
    3: "Severe damage: the building may be unsafe or beyond repair.",
}


# ---------------------------------------------------------------- pages
@app.get("/")
def index():
    # The 3D page is the whole country. Assam keeps its building footprints inside it.
    return FileResponse(os.path.join(PUBLIC, "index.html"))


@app.get("/place/{pid}")
def place_page(pid: str):
    # The per-state 3D pages are retired; the country view contains them.
    return RedirectResponse("/", status_code=302)


@app.get("/2d")
def hazard_2d():
    # the risk map is the whole world
    return FileResponse(os.path.join(PUBLIC, "world.html"))


@app.get("/2d/assam")
def hazard_2d_assam():
    return FileResponse(os.path.join(PUBLIC, "2d.html"))


@app.get("/globe")
def globe():
    # the globe is retired; the 3D page is the places view
    return RedirectResponse("/", status_code=302)


@app.get("/world")
def world_2d():
    return RedirectResponse("/2d", status_code=301)


@app.get("/docs")
def docs():
    return HTMLResponse(DOCS_HTML)


@app.get("/3d")
def assam_3d_redirect():
    return RedirectResponse("/", status_code=301)


@app.get("/lab")
def lab_redirect():
    return RedirectResponse("/", status_code=301)


# ---------------------------------------------------------------- places API
@app.get("/api/places")
def api_places():
    return pl.list_places()


@app.get("/api/places/{pid}")
def api_place(pid: str):
    try:
        return pl.place(pid)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown place {pid!r}")


# ---------------------------------------------------------------- model API
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "model_loaded": os.path.exists(MODEL_PATH),
        "hazard_ready": os.path.exists(os.path.join(HAZARD, "hazard_grid.geojson")),
        "world_ready": os.path.exists(os.path.join(PUBLIC, "static", "world", "countries.json")),
    }


@app.get("/api/model")
def model_info():
    return _read_json("model_info.json")


@app.get("/api/metrics")
def metrics():
    return _read_json("metrics.json")


@app.get("/api/options")
def options():
    return get_options()


_predict_cache: dict[str, dict] = {}
_CACHE_MAX = 2048


@app.post("/api/predict")
def predict(payload: dict[str, Any]):
    key = json.dumps(payload, sort_keys=True, default=str)
    if key in _predict_cache:
        return JSONResponse(_predict_cache[key], headers={"X-Cache": "hit"})
    bundle = get_bundle()
    frame = build_frame(payload)
    proba = bundle.predict_proba(frame)[0]
    grade = int(proba.argmax()) + 1
    out = {
        "damage_grade": grade,
        "label": GRADE_LABELS[grade],
        "guidance": GRADE_HELP[grade],
        "probabilities": {str(k + 1): round(float(proba[k]), 4) for k in range(3)},
        "model": "damage-model-v1 (LightGBM + XGBoost)",
        "disclaimer": (
            "Statistical estimate from the 2015 Gorkha survey data. It is not a structural "
            "assessment and not a safety certificate. Use a qualified engineer for decisions."
        ),
    }
    if len(_predict_cache) >= _CACHE_MAX:
        _predict_cache.pop(next(iter(_predict_cache)))
    _predict_cache[key] = out
    return JSONResponse(out, headers={"X-Cache": "miss"})


_examples = None


def get_examples():
    global _examples
    if _examples is None:
        _examples = _read_json("examples.json")
    return _examples


@app.get("/api/examples")
def examples(n: int = Query(30, ge=5, le=100), seed: int = Query(0, ge=0, le=10**9)):
    """Run the saved model on a seeded random sample of real held-out survey records."""
    ex = get_examples()
    rows = ex["rows"]
    n = min(n, len(rows))
    pick = np.random.default_rng(seed).choice(len(rows), size=n, replace=False)
    fields = get_options()["fields"]
    walls_labels = {
        f["name"]: f["label"].replace("Has superstructure ", "")
        for f in fields
        if f["name"].startswith("has_superstructure_") and not f["name"].endswith("_other")
    }
    frames, meta = [], []
    for i in pick:
        payload = rows[i]["payload"]
        frames.append({f["name"]: _coerce(f, payload[f["name"]]) for f in fields})
        meta.append(rows[i])
    proba = get_bundle().predict_proba(pd.DataFrame(frames))
    out_rows, confusion = [], [[0, 0, 0] for _ in range(3)]
    for k, m in enumerate(meta):
        pred = int(proba[k].argmax()) + 1
        actual = int(m["actual"])
        confusion[actual - 1][pred - 1] += 1
        p = m["payload"]
        out_rows.append({
            "record": m["record"],
            "actual": actual,
            "predicted": pred,
            "correct": pred == actual,
            "confidence": round(float(proba[k][pred - 1]), 4),
            "probabilities": {str(j + 1): round(float(proba[k][j]), 4) for j in range(3)},
            "inputs": {
                "floors": p["count_floors_pre_eq"],
                "age": p["age"],
                "plinth_pct": p["area_percentage"],
                "height_units": p["height_percentage"],
                "roof": p["roof_type"],
                "walls": [walls_labels[k2] for k2 in walls_labels if int(p.get(k2, 0)) == 1],
            },
        })
    correct = sum(r["correct"] for r in out_rows)
    per_grade = {}
    for g in range(3):
        total = sum(confusion[g])
        per_grade[str(g + 1)] = round(confusion[g][g] / total, 4) if total else None
    return {
        "n": n,
        "seed": seed,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else None,
        "confusion": confusion,
        "recall_by_grade": per_grade,
        "test_rows": ex["test_rows"],
        "test_accuracy": ex["test_accuracy"],
        "majority_accuracy": ex["majority_accuracy"],
        "rows": out_rows,
        "note": (
            "These are real survey records from the held-out 20% test split. The model sees only "
            "the survey inputs. The dataset has no coordinates, so these records are not placed "
            "on the map."
        ),
    }


@app.get("/api/areas")
def areas():
    return _read_json("areas.json")


@app.get("/api/districts")
def districts():
    return _read_json("districts.json")


@app.get("/api/results")
def results():
    return _read_json("results.json")


# ---------------------------------------------------------------- hazard API
@app.get("/api/hazard")
def hazard():
    return _read_hazard("hazard_meta.json")


@app.get("/api/events")
def events():
    return _read_hazard("events_m5plus.geojson")


@app.get("/api/damage-zones")
def damage_zones():
    return _read_hazard("damage_zones.json")


# ---------------------------------------------------------------- India API
@app.get("/api/india/status")
def india_status():
    """Whole-country summary: states, in-India epicentres, hazard cells, footprints."""
    return ind.status()


@app.get("/api/india/hazard")
def india_hazard():
    """National 0.25 degree hazard grid, clipped to Indian land."""
    return ind.hazard_national()


@app.get("/api/india/events")
def india_events(min_mag: float = Query(4.0, ge=1.0, le=9.9)):
    """Only the catalogued events that fall inside an Indian state outline."""
    rows = [e for e in ind.in_india_events() if e["mag"] >= min_mag]
    return {"count": len(rows), "min_mag": min_mag, "events": rows}


@app.get("/api/india/state/{sid}")
def india_state(sid: str):
    """One state or union territory: outline, its epicentres, its largest event."""
    try:
        return ind.state_detail(sid)
    except KeyError:
        raise HTTPException(404, detail=f"No state with id {sid!r}.")


# ---------------------------------------------------------------- world API
@app.get("/api/world/status")
def world_status():
    return wd.status()


@app.get("/api/world/countries")
def world_countries():
    return wd.countries()


@app.get("/api/world/countries.geojson")
def world_countries_geojson():
    return wd.countries_geojson()


@app.get("/api/world/cities")
def world_cities(min_pop: int = Query(100_000, ge=0), limit: int = Query(5000, ge=1, le=20000)):
    rows = [c for c in wd.cities()["cities"] if c["pop"] >= min_pop]
    rows.sort(key=lambda c: -c["pop"])
    return {"count": len(rows[:limit]), "min_pop": min_pop, "cities": rows[:limit]}


@app.get("/api/world/events")
def world_events(min_mag: float = Query(5.5, ge=1.0, le=9.9),
                 limit: int = Query(6000, ge=1, le=40000)):
    return wd.events_payload(min_mag=min_mag, limit=limit)


@app.get("/api/world/hazard")
def world_hazard(refresh: bool = False):
    grid = wd.hazard_grid(force=refresh)
    return {"cells": grid["cells"], "meta": grid["meta"]}


# ---------------------------------------------------------------- docs page
DOCS_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>API documentation · Worldwide earthquake hazard</title>
<style>
  :root { --ink:#1c2530; --ink2:#3b4856; --muted:#6b7785; --line:#dfe5ec; --navy:#14213d; --amber:#b7791f; --gold:#d69e2e; }
  * { box-sizing: border-box; }
  body { margin:0; background:#f6f8fb; color:var(--ink); font:15px/1.65 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif; }
  header { background:linear-gradient(135deg,#14213d,#1f3358); color:#fff; padding:26px 24px; }
  .wrap { max-width: 980px; margin: 0 auto; }
  header .wrap { display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap; }
  h1 { margin:0; font:700 24px Georgia,serif; }
  header p { margin:6px 0 0; color:rgba(255,255,255,.78); font-size:13.5px; }
  nav a { color:#fff; text-decoration:none; border:1px solid rgba(255,255,255,.35); border-radius:9px; padding:6px 11px; font-size:13px; margin-left:6px; }
  nav a:hover { background:rgba(255,255,255,.12); }
  main { padding: 24px; }
  h2 { font:700 19px Georgia,serif; color:var(--navy); margin:28px 0 10px; }
  .ep { background:#fff; border:1px solid var(--line); border-radius:12px; padding:14px 16px; margin:10px 0; }
  .ep code.path { font:600 14px ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--navy); }
  .verb { display:inline-block; font:700 11px ui-monospace,monospace; letter-spacing:.04em; padding:3px 7px; border-radius:6px; margin-right:8px; vertical-align:2px; }
  .get { background:#e6f0ff; color:#1a4f9c; }
  .post { background:#e9f7ee; color:#22683f; }
  .ep p { margin:8px 0 0; color:var(--ink2); font-size:14px; }
  pre { background:#0e1726; color:#dbe4f0; padding:12px 14px; border-radius:10px; overflow:auto; font:12.5px/1.55 ui-monospace,Menlo,monospace; }
  .note { background:#fff8e6; border:1px solid #f0d9a8; border-radius:10px; padding:12px 14px; color:#6b5220; font-size:13.5px; }
  table { width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line); border-radius:12px; overflow:hidden; font-size:13.5px; }
  th,td { text-align:left; padding:9px 12px; border-bottom:1px solid var(--line); vertical-align:top; }
  th { background:#eef3f9; color:var(--navy); font-size:12.5px; text-transform:uppercase; letter-spacing:.03em; }
  td code { font:12.5px ui-monospace,Menlo,monospace; }
  footer { color:var(--muted); font-size:12.5px; padding:26px 24px 40px; }
</style>
</head>
<body>
<header><div class="wrap">
  <div><h1>API documentation</h1><p>India in 3D and the world risk map · FastAPI</p></div>
  <nav><a href="/">India 3D</a><a href="/2d">World risk map</a></nav>
</div></header>
<main class="wrap">
  <p>Every endpoint returns JSON except the pages below. The machine-readable schema is at
  <a href="/openapi.json"><code>/openapi.json</code></a>; this page is the human-readable reference.</p>

  <h2>Pages</h2>
  <div class="ep"><span class="verb get">GET</span><code class="path">/</code><p><b>India in 3D.</b> The whole country on one map: 36 state and union-territory outlines, the national 0.25° hazard grid (2,651 cells), 1,463 catalogued M4+ epicentres, and Assam's 13,302,299 Overture building footprints streamed as 0.1° tiles. Click a state for its record and to score a building profile; zoom into Assam and click a building for its damage grade. The Example panel scores 30 real held-out survey records. Layers can be switched off from the map.</p></div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/place/{id}</code><p><b>Retired.</b> The per-state 3D pages (<code>kutch</code>, <code>koyna</code>) redirect to <code>/</code>, which contains every state. The place data they used is still served at <code>/api/places</code>.</p></div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/2d</code><p><b>World risk map.</b> 2° hazard cells, the global M5+ catalogue with a magnitude slider, country outlines and the strongest events.</p></div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/2d/assam</code><p><b>Assam hazard detail.</b> The regional 0.25° grid, M5+ events and the 2015 Gorkha damage table for 31 anonymised areas.</p></div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/docs</code><p>This page. <code>/globe</code> redirects to <code>/</code> and <code>/world</code> to <code>/2d</code>.</p></div>

  <h2>Model</h2>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/health</code><p>Status: <code>ok</code>, <code>model_loaded</code>, <code>hazard_ready</code>.</p></div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/model</code><p>Model name, version, accuracy, macro-F1, per-class F1 and the split protocol.</p></div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/metrics</code><p>Full final metrics: ensemble and LightGBM-only, per class, confusion matrix, best iterations.</p></div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/options</code><p>The 38 survey form fields with <code>kind</code>, <code>min</code>, <code>max</code>, allowed <code>values</code> and <code>default</code>.</p></div>
  <div class="ep"><span class="verb post">POST</span><code class="path">/api/predict</code>
    <p>Damage grade for one building. Send all 38 fields from <code>/api/options</code> as a JSON object.</p>
    <pre>curl -s -X POST localhost:8000/api/predict \\
  -H 'Content-Type: application/json' \\
  -d '{"geo_level_1_id":6,"geo_level_2_id":120,"geo_level_3_id":1223,
       "count_floors_pre_eq":2,"age":15,"area_percentage":8,
       "height_percentage":6,"land_surface_condition":"t",
       "foundation_type":"r","roof_type":"n","ground_floor_type":"v",
       "other_floor_type":"q","position":"s","plan_configuration":"d",
       "has_superstructure_adobe_mud":1,"count_families":1}'</pre>
    <pre>{"damage_grade":2,"label":"Medium damage",
 "probabilities":{"1":0.1284,"2":0.6317,"3":0.2399},
 "model":"damage-model-v1 (LightGBM + XGBoost)","disclaimer":"..."}</pre>
    <p>Responses are cached; <code>X-Cache: hit</code> marks a repeat. Invalid input returns 422 with the offending
    field names.</p>
  </div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/examples</code>
    <p>Runs the saved model on a seeded sample of real held-out records and compares with the recorded grade.</p>
    <table><tr><th>Query</th><th>Type</th><th>Default</th><th>Meaning</th></tr>
      <tr><td><code>n</code></td><td>int 5–100</td><td>30</td><td>Sample size</td></tr>
      <tr><td><code>seed</code></td><td>int ≥ 0</td><td>0</td><td>Same seed, same rows</td></tr></table>
  </div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/results</code><p>Model comparison table (baselines, boosting, shipped ensemble) and the final confusion matrix.</p></div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/areas</code><p>31 anonymised areas on the held-out split: actual and predicted grade shares, accuracy, risk score.</p></div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/districts</code><p>Nepal district boundaries as GeoJSON (77 features, used by the archive map).</p></div>

  <h2>Hazard</h2>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/hazard</code><p>Method, tercile thresholds, cell counts and the disclaimer for the 2D hazard layer.</p></div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/events</code><p>153 magnitude 5+ events in the map box as a GeoJSON FeatureCollection of points (mag, time, depth, place).</p></div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/damage-zones</code><p>Gorkha 2015 damage by anonymised area: zone, buildings surveyed, grade-3 share, accuracy, risk score.</p></div>

  <h2>India</h2>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/india/status</code><p>Whole-country summary: <code>states</code> (36), <code>events</code> (1,463 catalogued M4+ events located inside an Indian outline), <code>events_catalogue</code> (18,501, the size of the bundled India-region catalogue), <code>events_m5</code> (158), <code>cells</code> (2,651), <code>footprint_states</code> (<code>["assam"]</code>), <code>footprint_buildings</code> (13,302,299), the hazard method and disclaimer, and <code>per_state</code> — one row per state with <code>events</code>, <code>events_m5</code>, <code>largest</code> and <code>has_footprints</code>.</p>
    <pre>{"states":36,"events":1463,"events_catalogue":18501,"events_m5":158,"largest":7.7,
 "cells":2651,"footprint_states":["assam"],
 "per_state":[{"id":"jammu-kashmir","name":"Jammu &amp; Kashmir","events":500,"events_m5":55,"largest":7.6}, …]}</pre>
  </div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/india/hazard</code><p>The national hazard grid: 0.25° cells over Indian land, each <code>[lon, lat, class, log10 energy, events, largest magnitude]</code> with class <code>r</code>/<code>y</code>/<code>g</code> from the terciles, plus <code>meta</code> with the method, thresholds (red 6.48, yellow 5.22 log10 J) and disclaimer. The energy field uses every event in the bundled catalogue, including neighbouring countries, so the border belts are not cut off.</p></div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/india/events</code>
    <p>Only the catalogued events that fall inside an Indian state outline, strongest first.</p>
    <table><tr><th>Query</th><th>Type</th><th>Default</th><th>Meaning</th></tr>
      <tr><td><code>min_mag</code></td><td>float</td><td>4.0</td><td>Magnitude floor</td></tr></table>
    <pre>{"count":1463,"min_mag":4.0,
 "events":[{"lon":70.232,"lat":23.419,"mag":7.7,"depth":16.0,"time":"2001-01-26",
            "place":"17 km NW of Bhachau, India"}, …]}</pre>
  </div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/india/state/{id}</code><p>One state or territory: its <code>bbox</code>, <code>centre</code>, <code>rings</code> (lon/lat outlines), its <code>events</code>, and <code>meta</code> with the counts and the largest event. Ids are the slugs in <code>states.json</code> — <code>assam</code>, <code>gujarat</code>, <code>jammu-kashmir</code>, <code>andaman</code>, … An unknown id returns 404.</p></div>

  <h2>Places (retired pages, data still served)</h2>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/places</code><p>The places in the 3D view, with what each one has: <code>assam</code> (13.3 million building footprints) and the two catalogue-only places, <code>kutch</code> and <code>koyna</code>.</p></div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/places/{id}</code><p>One place: its box, the hazard grid computed inside that box, and its epicentres. Assam has no computed grid here (it keeps its own 0.25° regional grid on <code>/2d/assam</code>).</p>
    <pre>{"id":"kutch","name":"Kutch, Gujarat","bbox":[68.4,22.0,72.2,24.9],
 "meta":{"events":150,"events_m5":16,"cells_total":320,"largest":{"mag":7.7,"time":"2001-01-26"}},
 "cells":[[70.2,23.35,"r",10.17,12,7.7]],
 "events":[{"lon":70.232,"lat":23.419,"mag":7.7,"time":"2001-01-26","place":"17 km NW of Bhachau, India"}]}</pre>
  </div>

  <h2>World</h2>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/world/status</code><p>What the world pages have loaded: countries, cities, catalogue coverage (global or bundled), hazard cells and the detail build.</p></div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/world/countries</code><p>177 Natural Earth 1:110m countries as lon/lat rings (10,583 points, 176 KB), plus licence and source.</p></div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/world/countries.geojson</code><p>The same countries as a GeoJSON FeatureCollection of MultiPolygons, for Leaflet.</p></div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/world/cities</code>
    <p>GeoNames cities, largest first.</p>
    <table><tr><th>Query</th><th>Type</th><th>Default</th><th>Meaning</th></tr>
      <tr><td><code>min_pop</code></td><td>int</td><td>100000</td><td>Population floor</td></tr>
      <tr><td><code>limit</code></td><td>int 1–20000</td><td>5000</td><td>Rows returned</td></tr></table>
  </div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/world/events</code>
    <p>The earthquake catalogue: live USGS FDSN query M5.5+ since 2000 when the host has outbound access, otherwise the bundled global catalogue (M5.0+ 2016–2025 and M6.0+ since 1900, 30,648 events). The response says which one you got.</p>
    <table><tr><th>Query</th><th>Type</th><th>Default</th><th>Meaning</th></tr>
      <tr><td><code>min_mag</code></td><td>float</td><td>5.5</td><td>Magnitude floor</td></tr>
      <tr><td><code>limit</code></td><td>int</td><td>6000</td><td>Events returned, strongest first</td></tr></table>
    <pre>{"count":1717,"coverage":"global","source":"USGS earthquake catalogue, packaged by Live Atlas",
 "events":[{"lon":-73.4,"lat":-36.5,"mag":9.5,"depth":25,"time":"1960-05-22","place":"1960 Great Chilean Earthquake"}]}</pre>
  </div>
  <div class="ep"><span class="verb get">GET</span><code class="path">/api/world/hazard</code><p>2° hazard grid computed from that catalogue: 6,000 cells with class, score, event count and largest magnitude, plus the method, tercile thresholds and disclaimer. <code>?refresh=true</code> recomputes.</p></div>

  <h2>Static files</h2>
  <table>
    <tr><th>Path</th><th>Contents</th></tr>
    <tr><td><code>/static/india/states.json</code></td><td>36 state and union-territory outlines, 29,503 points, 395 KB (built by <code>scripts/build_india_states.py</code>)</td></tr>
    <tr><td><code>/static/india/india.js</code></td><td>The country view: state fills and borders, hazard columns, epicentres, Assam tiles, state sheet</td></tr>
    <tr><td><code>/static/assam3d/tiles/</code></td><td>837 footprint tiles, 0.1° each, 13,302,299 buildings (Overture, ODbL)</td></tr>
    <tr><td><code>/static/world/countries.json</code></td><td>177 countries as lon/lat rings (Natural Earth 1:110m)</td></tr>
    <tr><td><code>/static/world/cities.json</code></td><td>4,442 cities, population 100,000+ (GeoNames)</td></tr>
    <tr><td><code>data/quakes/world_catalogue.json</code></td><td>30,648 events, compact integer encoding, CC BY 4.0</td></tr>
  </table>

  <h2>Notes</h2>
  <div class="note">
    <b>One country, one page.</b> The 3D view is the whole of India; Assam is the only state with
    packaged building footprints, so it is the only one that opens at building level. Every other state
    is drawn from the earthquake record — nothing is invented.<br><br>
    <b>Not a safety certificate.</b> Grades are statistical estimates from the 2015 Gorkha survey.
    On the world pages the prediction depends on the building profile you enter, not on the coordinates,
    because the training survey has no coordinates.
    The hazard layer ranks where M4+ events have clustered since 2000; it is not a forecast and not a
    design-level seismic hazard map.
  </div>
</main>
<footer class="wrap">FastAPI app · served by <code>uvicorn app:app</code> · build tag world-hazard 2026-11-01</footer>
</body>
</html>
"""
