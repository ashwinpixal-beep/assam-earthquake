# Tests

Browser checks for the site, written with Playwright (sync API) and pytest.
There are 15 checks covering the three pages, the 3D viewer, the 2D hazard map
and the API.

## Setup

```bash
pip install -r requirements.txt
pip install pytest playwright
python -m playwright install chromium
```

## Run

From the project root:

```bash
pytest tests/test_app.py -v
```

The tests start their own server on port **8011** and stop it afterwards. To
test a server that is already running:

```bash
EXTERNAL_SERVER=1 BASE_URL=http://127.0.0.1:8000 pytest tests/test_app.py -v
```

## The 15 checks

| # | Test | What it asserts |
|---|---|---|
| 1 | `test_01_home_returns_200` | `/` is 200 and the title says "Assam in 3D" |
| 2 | `test_02_home_has_page_nav` | The page switcher links to `/`, `/2d`, `/docs` |
| 3 | `test_03_3d_reports_building_total` | The building counter starts with 13 (13,302,299) |
| 4 | `test_04_3d_loader_clears` | The loader clears and a WebGL canvas exists |
| 5 | `test_05_example_panel_runs_model` | Example panel opens, 30 rows, 3×3 confusion matrix |
| 6 | `test_06_2d_page_and_map` | `/2d` is 200 and Leaflet has initialised |
| 7 | `test_07_hazard_cells_drawn` | 912 hazard cells and more than 100 paths on the map |
| 8 | `test_08_events_layer_toggles` | The M5+ events layer turns off and back on |
| 9 | `test_09_damage_table_has_31_areas` | The Gorkha table has 31 rows |
| 10 | `test_10_damage_table_sorts` | Sorting by grade-3 share puts the maximum first |
| 11 | `test_11_docs_page_lists_endpoints` | `/docs` documents the API paths |
| 12 | `test_12_api_health` | `ok`, `model_loaded`, `hazard_ready` are all true |
| 13 | `test_13_api_predict` | `predict` returns a grade 1–3 with probabilities summing to 1 |
| 14 | `test_14_api_predict_rejects_bad_input` | An out-of-range field returns 422 |
| 15 | `test_15_redirects_and_scrollbars` | `/3d` and `/lab` redirect to `/`, docs don't scroll sideways |

## Notes

- Check 3 and 5 need the tile files in `public/static/assam3d/tiles/` and the
  model in `models/damage_model.joblib`. Without the tiles the 3D page still
  loads, but the building counter stays at zero.
- Checks 3–5 wait up to 45–60 s for the first tiles to stream.
- Headless Chromium needs WebGL; the fixture launches with
  `--use-gl=angle --enable-webgl`. On a machine without GPU support, run with a
  headed browser or skip checks 4 and 5.
- On failure, capture a screenshot for context:
  `pytest tests/test_app.py --screenshot=only-on-failure` is not supported by
  pytest-playwright unless it is installed; the fixture here is plain Playwright.
