# Tests

Browser checks for the site, written with Playwright (sync API) and pytest.
There are 21 checks covering the Assam 3D page, the other 3D places (Kutch,
Koyna), the world risk map, the Example panel and the API.

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

## The 21 checks

| # | Test | What it asserts |
|---|---|---|
| 1 | `test_01_home_is_assam_in_3d` | `/` is the Assam 3D page |
| 2 | `test_02_assam_page_nav` | Places nav (Assam, Kutch, Koyna) plus the 2D and docs links |
| 3 | `test_03_old_routes_redirect` | `/globe` → `/`, `/world` → `/2d`, `/3d` → `/` |
| 4 | `test_04_assam_reports_building_total` | The building counter starts with 13 (13,302,299) |
| 5 | `test_05_assam_loader_clears` | The loader clears and a WebGL canvas exists |
| 6 | `test_06_example_panel_scores_real_records` | 30 rows, 3×3 confusion matrix, 75.68% held-out figure |
| 7 | `test_07_world_2d_page_draws` | `/2d` draws 6,000 cells and the events, stats fill in |
| 8 | `test_08_world_2d_layers_toggle` | The events layer leaves and returns |
| 9 | `test_09_world_2d_magnitude_filter` | Raising the floor to M7 cuts the plotted events |
| 10 | `test_10_docs_page_lists_endpoints` | `/docs` documents the pages and the API, including `/api/places` |
| 11 | `test_11_api_health` | `ok`, `model_loaded`, `hazard_ready` |
| 12 | `test_12_api_predict` | Grade 1–3, probabilities sum to 1 |
| 13 | `test_13_api_predict_rejects_bad_input` | Out-of-range field returns 422 |
| 14 | `test_14_examples_api_matches_the_panel` | 30 rows, 3×3 confusion, diagonal equals `correct` |
| 15 | `test_15_world_status` | Countries, cities, catalogue and grid counts |
| 16 | `test_16_world_countries_geojson` | 177 countries, India/Nepal/Japan/Chile present |
| 17 | `test_17_world_events_catalogue` | Magnitude floor honoured, coordinates in range |
| 18 | `test_18_world_hazard_grid` | 6,000 cells with tercile classes |
| 19 | `test_19_places_api` | Assam/Kutch/Koyna, Bhuj M7.7 on 2001-01-26, boxes contain the cells |
| 20 | `test_20_place_page_renders` | `/place/kutch` renders its grid, epicentres and prediction |
| 21 | `test_21_docs_and_scrollbars` | `/docs`, `/2d` and a place page don't scroll sideways |

## Notes

- Check 6 needs `data/examples.json`; check 20 needs the USGS India CSV in `data/quakes/`.
- The 3D checks (4, 5 and 20) need WebGL. The fixture launches
  Chromium with `--use-gl=angle --use-angle=swiftshader --enable-webgl
  --enable-unsafe-swiftshader`, and a `webgl` fixture skips those checks when the
  browser still has no working GL — so the suite passes on a headless container
  without a GPU, and the 3D pages show a "3D needs WebGL" note (with links to the
  2D pages) instead of hanging on their loader. The Example panel is a separate
  module, so it still works in that state.
- On failure, capture a screenshot for context:
  `pytest tests/test_app.py --screenshot=only-on-failure` is not supported by
  pytest-playwright unless it is installed; the fixture here is plain Playwright.
