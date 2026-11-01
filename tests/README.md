# Tests

Browser and API checks for the site, written with Playwright (sync API) and
pytest. There are 24 checks covering the India 3D page (national counts, a state
sheet, the layer toggles), the Example panel, the world risk map, the redirects
and the whole API.

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

## The 24 checks

| # | Test | What it asserts |
|---|---|---|
| 1 | `test_01_home_is_india_in_3d` | `/` is the India 3D page, title and `#info h1` |
| 2 | `test_02_page_nav` | The nav is India (`/`) and the world risk map (`/2d`) |
| 3 | `test_03_old_routes_redirect` | `/globe`, `/3d` → `/`; `/world` → `/2d`; `/place/kutch` and `/place/koyna` → `/` |
| 4 | `test_04_national_counts` | 36 states, 1,463 epicentres, 2,651 hazard cells, 13,302,299 buildings |
| 5 | `test_05_loader_clears_and_canvas_draws` | The loader clears, a WebGL canvas exists, `window.__india3d` is set |
| 6 | `test_06_example_panel_scores_real_records` | 30 rows, 3×3 confusion matrix, 75.68% held-out figure |
| 6b | `test_06b_state_sheet_from_the_dock` | Gujarat: 151 events, 18 M5+, M7.7, three probability bars, 37 picker options |
| 6c | `test_06c_layer_toggles` | Hazard and epicentre layers switch off and back on |
| 7 | `test_07_world_2d_page_draws` | `/2d` draws 6,000 cells and the events, stats fill in |
| 8 | `test_08_world_2d_layers_toggle` | The events layer leaves and returns |
| 9 | `test_09_world_2d_magnitude_filter` | Raising the floor to M7 cuts the plotted events |
| 10 | `test_10_docs_page_lists_endpoints` | `/docs` documents `/api/india/*` and the world API |
| 11 | `test_11_api_health` | `ok`, `model_loaded`, `hazard_ready` |
| 12 | `test_12_api_predict` | Grade 1–3, probabilities sum to 1 |
| 13 | `test_13_api_predict_rejects_bad_input` | Out-of-range field returns 422 |
| 14 | `test_14_examples_api_matches_the_panel` | 30 rows, 3×3 confusion, diagonal equals `correct` |
| 15 | `test_15_world_status` | Countries, cities, catalogue and grid counts |
| 16 | `test_16_world_countries_geojson` | 177 countries, India/Nepal/Japan/Chile present |
| 17 | `test_17_world_events_catalogue` | Magnitude floor honoured, coordinates in range |
| 18 | `test_18_world_hazard_grid` | 6,000 cells with tercile classes |
| 19 | `test_19_places_api` | Assam/Kutch/Koyna, Bhuj M7.7 on 2001-01-26, boxes contain the cells |
| 20 | `test_20_india_api` | Status, hazard grid and a state detail agree with each other; unknown id is 404; `states.json` totals 29,503 points |
| 20b | `test_20b_india_events_list` | 158 events at M5+, 1,463 at M4+, all inside India, strongest is M7.7 |
| 21 | `test_21_docs_and_scrollbars` | `/docs`, `/2d`, `/`, `/2d/assam` don't scroll sideways |

## Notes

- Check 6 needs `data/examples.json`; checks 4, 6b and 20 need the India catalogue
  (`data/quakes/india_quakes_m4_2000_2026.csv`) and `public/static/india/states.json`.
- The 3D checks (4, 5, 6b, 6c) need WebGL. The fixture launches Chromium with
  `--use-gl=angle --use-angle=swiftshader --enable-webgl
  --enable-unsafe-swiftshader`, and a `webgl` fixture skips those checks when the
  browser still has no working GL — so the suite passes on a headless container
  without a GPU, and the page shows a "3D needs WebGL" note (with a link to the 2D
  map) instead of hanging on its loader. The Example panel is a separate module,
  so it still works in that state.
- The API checks 11–20b need no browser; they can be run against any live server
  with `EXTERNAL_SERVER=1 BASE_URL=...`.
- On failure, capture a screenshot for context:
  `pytest tests/test_app.py --screenshot=only-on-failure` is not supported by
  pytest-playwright unless it is installed; the fixture here is plain Playwright.
