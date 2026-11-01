# Tests

Browser checks for the site, written with Playwright (sync API) and pytest.
There are 20 checks covering the two world pages, the retired Assam routes, the
3D globe, the 2D world map, the Example panel and the API.

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

## The 20 checks

| # | Test | What it asserts |
|---|---|---|
| 1 | `test_01_root_redirects_to_globe` | `/` lands on `/globe` |
| 2 | `test_02_page_nav_lists_both_world_views` | Both pages link to `/globe`, `/world`, `/docs` only |
| 3 | `test_03_assam_pages_are_retired` | `/2d` lands on `/world`; no Assam links left on the globe |
| 4 | `test_04_globe_reports_its_data` | 177 countries, event count, 6,000 hazard cells |
| 5 | `test_05_globe_renders_a_canvas` | Loader clears, WebGL canvas plus city labels |
| 6 | `test_06_globe_click_predicts` | A click opens the sheet with a country and three probability bars |
| 7 | `test_07_example_panel_scores_real_records` | 30 rows, 3×3 confusion matrix, 75.68% held-out figure |
| 8 | `test_08_world_2d_page_draws` | `/world` draws cells and events, stats fill in |
| 9 | `test_09_world_2d_layers_toggle` | The events layer leaves and returns |
| 10 | `test_10_world_2d_magnitude_filter` | Raising the floor to M7 cuts the plotted events |
| 11 | `test_11_docs_page_lists_endpoints` | `/docs` documents the pages and the API |
| 12 | `test_12_api_health` | `ok`, `model_loaded`, `hazard_ready` |
| 13 | `test_13_api_predict` | Grade 1–3, probabilities sum to 1 |
| 14 | `test_14_api_predict_rejects_bad_input` | Out-of-range field returns 422 |
| 15 | `test_15_examples_api_matches_the_panel` | 30 rows, 3×3 confusion, diagonal equals `correct` |
| 16 | `test_16_world_status` | Countries, cities, catalogue and grid counts |
| 17 | `test_17_world_countries_geojson` | 177 countries, India/Nepal/Japan/Chile present |
| 18 | `test_18_world_events_catalogue` | Magnitude floor honoured, coordinates in range |
| 19 | `test_19_world_hazard_grid` | 6,000 cells with tercile classes |
| 20 | `test_20_redirects_and_scrollbars` | `/3d`, `/lab` → `/globe`, `/2d` → `/world`, no sideways scroll |

## Notes

- Check 7 needs the model in `models/damage_model.joblib` and `data/examples.json`.
- The 3D checks (5 and 6) need WebGL. The fixture launches
  Chromium with `--use-gl=angle --use-angle=swiftshader --enable-webgl
  --enable-unsafe-swiftshader`, and a `webgl` fixture skips those checks when the
  browser still has no working GL — so the suite passes on a headless container
  without a GPU, and the globe shows a "3D needs WebGL" note (with a link to the
  2D map) instead of hanging on its loader.
- On failure, capture a screenshot for context:
  `pytest tests/test_app.py --screenshot=only-on-failure` is not supported by
  pytest-playwright unless it is installed; the fixture here is plain Playwright.
