# Assam building damage and earthquake hazard

A single web app with three pages:

| Page | What it does |
|---|---|
| `/` | **3D building map of Assam.** 13,302,299 Overture footprints streamed as 0.1° binary tiles. Click a building to get a damage grade from the model. The **Example** button scores 30 real held-out survey records and compares them with the recorded grade. |
| `/2d` | **Earthquake hazard map.** 912 hazard cells (red / yellow / green) from the USGS M4+ catalogue, 153 magnitude 5+ events, and the 2015 Gorkha damage table for 31 anonymised areas. |
| `/docs` | **API documentation.** Every endpoint, with an example request and response. |

The damage model predicts the 2015 Gorkha earthquake damage grade of a building
(1 = low, 2 = medium, 3 = near-complete destruction) from the DrivenData
*Richter's Predictor* survey data. It is a probability-averaged LightGBM + XGBoost
ensemble.

## Results (held-out 20% test split, 52,121 buildings)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Naive Bayes (baseline) | 69.85% | 0.662 |
| Naive Bayes (improved) | 70.38% | 0.667 |
| XGBoost | 75.02% | 0.701 |
| LightGBM | 75.32% | 0.708 |
| **LightGBM + XGBoost (shipped)** | **75.68%** | **0.710** |
| Base paper P4 (Chen & Zhang 2022, SMOTE-XGBoost) | not reported | 0.729 (own split) |

The 85% accuracy target was **not** reached. P4 is reported on its own split and
preprocessing, so its figure is not directly comparable. Details:
`research/results/report.md` and `research/results_boost/report.md`.

## Run it

Easiest — one click:

- **Windows:** double-click `START_HERE.bat`
- **macOS:** double-click `START_HERE.command` (first time: right-click → Open)
- **Any OS:** `python3 start.py`

`start.py` installs the requirements if they are missing, starts the server on
port 8000 and opens the browser.

Manual:

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

Docker:

```bash
docker build -t damage-risk-site .
docker run -p 8000:8000 damage-risk-site
```

The image honours `$PORT` when the host sets it (Render, Railway, Fly and
similar). `libgomp1` is installed in the image because LightGBM needs it.

> **Python version:** the pinned `xgboost==3.4.1` requires Python 3.12 or newer.
> On 3.11, install `xgboost==3.2.0` instead — the bundle loads with either.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Status, model-loaded and hazard-ready flags |
| GET | `/api/model` | Model name, accuracy, macro-F1, protocol |
| GET | `/api/metrics` | Full final metrics and confusion matrix |
| GET | `/api/options` | The 38 survey fields, with ranges and defaults |
| POST | `/api/predict` | Damage grade and class probabilities for one building |
| GET | `/api/examples?n=&seed=` | Saved model on a seeded sample of real held-out records |
| GET | `/api/results` | Model comparison table |
| GET | `/api/areas` | 31 anonymised areas on the held-out split |
| GET | `/api/districts` | Nepal district boundaries (GeoJSON) |
| GET | `/api/hazard` | Hazard method, thresholds, cell counts |
| GET | `/api/events` | M5+ events (GeoJSON) |
| GET | `/api/damage-zones` | Gorkha 2015 damage by anonymised area |

```bash
curl -s localhost:8000/api/health
curl -s "localhost:8000/api/examples?n=10&seed=3"
```

`POST /api/predict` needs all 38 fields from `/api/options`:

```bash
curl -s -X POST localhost:8000/api/predict -H 'Content-Type: application/json' \
  -d '{"geo_level_1_id":6,"geo_level_2_id":120,"geo_level_3_id":1223,"count_floors_pre_eq":2,
       "age":15,"area_percentage":8,"height_percentage":6,"land_surface_condition":"t",
       "foundation_type":"r","roof_type":"n","ground_floor_type":"v","other_floor_type":"q",
       "position":"s","plan_configuration":"d","has_superstructure_adobe_mud":1,"count_families":1}'
```

Invalid or missing fields return `422` naming the offending fields. Repeats are
cached and marked `X-Cache: hit`.

## Layout

```
app.py                     FastAPI app: pages, static files, API
model.py                   DamageModel class used by the saved bundle
start.py                   one-click launcher
models/damage_model.joblib  trained ensemble (LightGBM + XGBoost), about 25 MB
data/                      areas, districts, examples, metrics, options, results
data/quakes/               USGS M4+ catalogue, 2000–2026 (3.3 MB)
public/index.html          3D map page        (/)
public/2d.html             2D hazard page     (/2d)
public/static/assam3d/     viewer.js, boundary, three.js, tiles (837 × .bin)
public/static/hazard/      hazard grid, M5+ events, meta, damage zones
public/static/shared/      shared.js
public/static/vendor/      Leaflet 1.9.4
scripts/                   rebuild hazard data, damage zones, examples
research/                  training experiments (reference only)
tests/                     browser checks (Playwright, 15 checks)
archive/                   earlier project page and lab (not served)
```

## Rebuild the derived data

```bash
python scripts/build_hazard.py         # hazard grid + M5+ events + meta (needs shapely)
python scripts/build_damage_zones.py   # damage zones from data/areas.json
```

The model itself, `data/examples.json` and `data/areas.json` are built from the
DrivenData competition files (`train_values.csv`, `train_labels.csv`), which are
not redistributed here. Obtain them from the competition page under its terms:

```bash
python model.py train --values train_values.csv --labels train_labels.csv --out models
python research/damage_nb_compare.py --values train_values.csv --labels train_labels.csv --out research/results
python research/damage_boost.py      --values train_values.csv --labels train_labels.csv --out research/results_boost
```

## Tests

```bash
python -m playwright install chromium
pytest tests/test_app.py -v
```

15 checks: the three pages, the 3D loader and tile total, the Example panel, the
hazard cells and event layer, the 31-row damage table and its sorting, the docs
page, and the API (`health`, `predict`, `422` validation, redirects, sideways
scroll). See `tests/README.md`.

## Method notes

**Hazard layer.** Gutenberg–Richter energy `10^(1.5M)` per 0.25° cell, Gaussian
smoothed (σ = 0.35°), then split into terciles of the score across cells inside
Assam: red = top third, yellow = middle, green = bottom. It ranks where past
M4+ events clustered. It is **not** a forecast and **not** a seismic design map.

**Damage model.** Statistical estimate from 2015 survey records. It is **not** a
structural assessment or a safety certificate. The survey data has no
coordinates, so survey buildings cannot be placed on the Assam map; the 3D map
shows real footprints and density, not true heights (almost none are recorded,
so the rest are estimated at 4 m).

## Data and sources

- Building footprints: Overture Maps Foundation, release 2026-10-21.0 (ODbL).
  Boundary: Overture divisions, region IN-AS.
- Earthquake catalogue: USGS, M4.0+ 2000-01-01 to 2026-10-26.
- Survey data: DrivenData *Richter's Predictor* (2015 Nepal Earthquake Open Data
  Portal).
- Administrative boundaries: OCHA / Survey Department of Nepal, COD-AB, via HDX.
- Leaflet 1.9.4 and three.js 0.160 are vendored, so the pages work without a CDN.

## Project team

Venkata Sai Nikhil (CB.AI.U4CPS26013), Venkata Nishitha (CB.AI.U4CPS26014),
Guttha Lekhana (CB.AI.U4CPS26015), Harish Karteek (CB.AI.U4CPS26016).
