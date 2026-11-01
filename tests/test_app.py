"""Browser and API checks for the site (Playwright): the India 3D map

Start the site first, then run:
    python -m playwright install chromium
    pytest tests/test_app.py -v

The tests expect the server at BASE (default http://127.0.0.1:8000). Set
BASE_URL to point somewhere else. They start their own uvicorn unless
--no-server is given via the EXTERNAL_SERVER=1 environment variable.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("TEST_PORT", "8011"))
BASE = os.environ.get("BASE_URL", f"http://127.0.0.1:{PORT}")
EXTERNAL = os.environ.get("EXTERNAL_SERVER") == "1"

sync_playwright = pytest.importorskip(
    "playwright.sync_api", reason="install playwright: pip install playwright"
).sync_playwright


# --------------------------------------------------------------------- server
def _wait_for_port(host: str, port: int, timeout: float = 40.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                return
        time.sleep(0.25)
    raise RuntimeError(f"server did not start on {host}:{port}")


@pytest.fixture(scope="session")
def server():
    proc = None
    if not EXTERNAL:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1",
             "--port", str(PORT)],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _wait_for_port("127.0.0.1", PORT)
    yield BASE
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="session")
def webgl(page, server):
    """Skip GPU-only checks where the headless browser has no WebGL (some CI containers)."""
    page.goto(server + "/2d", wait_until="domcontentloaded")
    ok = page.evaluate(
        "() => { const c = document.createElement('canvas');"
        " return !!(c.getContext('webgl2') || c.getContext('webgl')); }"
    )
    if not ok:
        pytest.skip("this browser has no WebGL, so the 3D views cannot be rendered here")
    return True


@pytest.fixture(scope="session")
def page(server):
    with sync_playwright() as p:
        browser = p.chromium.launch(args=[
            "--use-gl=angle", "--use-angle=swiftshader", "--enable-webgl",
            "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist",
        ])
        ctx = browser.new_context(viewport={"width": 1400, "height": 900})
        pg = ctx.new_page()
        yield pg
        browser.close()


def api(path: str):
    with urllib.request.urlopen(server_url() + path, timeout=30) as r:
        return json.load(r)


def server_url() -> str:
    return BASE


# ---------------------------------------------------------------------- pages
def test_01_home_is_india_in_3d(page, server):
    resp = page.goto(server + "/", wait_until="domcontentloaded")
    assert resp.status == 200
    assert "India in 3D" in page.title()
    assert page.locator("#info h1").inner_text().strip() == "India in 3D"


def test_02_page_nav(page, server):
    page.goto(server + "/", wait_until="domcontentloaded")
    nav = page.locator("nav.pagenav").first
    hrefs = [nav.locator("a").nth(i).get_attribute("href") for i in range(2)]
    assert hrefs == ["/", "/2d"]


def test_03_old_routes_redirect(page, server):
    page.goto(server + "/globe", wait_until="domcontentloaded")
    assert page.url.rstrip("/") == server.rstrip("/"), "/globe should land on the country map"
    page.goto(server + "/world", wait_until="domcontentloaded")
    assert page.url.rstrip("/").endswith("/2d"), "/world should land on the world risk map"
    page.goto(server + "/3d", wait_until="domcontentloaded")
    assert page.url.rstrip("/") == server.rstrip("/")
    # the per-state 3D pages are retired; the country view contains them
    for old in ("/place/kutch", "/place/koyna"):
        page.goto(server + old, wait_until="domcontentloaded")
        assert page.url.rstrip("/") == server.rstrip("/"), f"{old} should land on the country map"


def test_04_national_counts(page, server, webgl):
    page.goto(server + "/", wait_until="domcontentloaded")
    page.wait_for_function(
        "() => { const el = document.getElementById('st-events');"
        " return el && el.textContent.replace(/[^0-9]/g,'') !== ''; }",
        timeout=60000,
    )
    assert page.locator("#st-states").inner_text().strip() == "36"
    assert page.locator("#st-events").inner_text().strip() == "1,463"
    assert page.locator("#st-cells").inner_text().strip() == "2,651"
    total = page.locator("#st-buildings").inner_text().replace(",", "").strip()
    assert total == "13302299", f"expected 13,302,299 buildings, saw {total!r}"


def test_05_loader_clears_and_canvas_draws(page, server, webgl):
    page.goto(server + "/", wait_until="domcontentloaded")
    page.wait_for_selector("#loader.done", timeout=60000)
    assert page.locator("#scene canvas").count() >= 1
    assert page.evaluate("() => !!window.__india3d")


def test_06_example_panel_scores_real_records(page, server):
    """The Example panel scores 30 real held-out records on the Assam page."""
    page.goto(server + "/", wait_until="domcontentloaded")
    page.click("#example-btn")
    page.wait_for_function(
        "() => { const rows = document.querySelectorAll('#e-rows tbody tr');"
        " const st = document.getElementById('e-status');"
        " return rows.length > 0 && st && /correct/i.test(st.textContent); }",
        timeout=90000,
    )
    assert page.get_attribute("#example-btn", "aria-expanded") == "true"
    assert page.locator("#e-rows tbody tr").count() == 30
    assert page.locator("#e-cm tbody tr").count() == 3
    assert page.locator("#e-acc").inner_text().strip().endswith("%")
    assert page.locator("#e-full").inner_text().strip() == "75.68%"
    page.click("#eclose")
    assert page.get_attribute("#example-btn", "aria-expanded") == "false"


def test_06b_state_sheet_from_the_dock(page, server, webgl):
    """Picking a state opens its sheet, fills its record and scores a building profile."""
    page.goto(server + "/", wait_until="domcontentloaded")
    page.wait_for_selector("#loader.done", timeout=90000)
    page.click('.city[data-state="gujarat"]')
    page.wait_for_function(
        "() => document.getElementById('btitle').textContent.trim() === 'Gujarat'",
        timeout=30000,
    )
    facts = page.locator("#bfacts").inner_text()
    assert "151" in facts and "18" in facts and "M7.7" in facts
    page.wait_for_function(
        "() => /grade|damage|[123] ·/i.test(document.getElementById('bbadge-v').textContent)",
        timeout=30000,
    )
    assert page.locator("#bbars .pbar").count() == 3
    assert page.locator("#statePick option").count() == 37


def test_06c_layer_toggles(page, server, webgl):
    page.goto(server + "/", wait_until="domcontentloaded")
    page.wait_for_selector("#loader.done", timeout=90000)
    page.click("#t-hazard")
    assert page.get_attribute("#t-hazard", "aria-pressed") == "false"
    page.click("#t-events")
    assert page.get_attribute("#t-events", "aria-pressed") == "false"
    assert page.get_attribute("#t-states", "aria-pressed") == "true"
    page.click("#t-hazard")
    page.click("#t-events")
    assert page.get_attribute("#t-hazard", "aria-pressed") == "true"


def test_07_world_2d_page_draws(page, server):
    resp = page.goto(server + "/2d", wait_until="domcontentloaded")
    assert resp.status == 200
    page.wait_for_selector(".leaflet-container", timeout=20000)
    page.wait_for_function(
        "() => { const el = document.getElementById('stEvents');"
        " return el && /[0-9]/.test(el.textContent); }",
        timeout=40000,
    )
    assert page.locator("#map path").count() > 1000
    assert page.locator("#stCells").inner_text().strip() == "6,000"
    assert page.locator("#eventsList div").count() >= 5


def test_08_world_2d_layers_toggle(page, server):
    # the layers arrive asynchronously, so compare the event dots with a settled page
    page.goto(server + "/2d", wait_until="domcontentloaded")
    page.wait_for_function(
        "() => document.querySelectorAll('#map path').length > 30000", timeout=60000
    )
    page.wait_for_timeout(1500)
    before = page.locator("#map path").count()
    page.click("#btnEvents")
    page.wait_for_timeout(1000)
    off = page.locator("#map path").count()
    page.click("#btnEvents")
    page.wait_for_timeout(1500)
    on = page.locator("#map path").count()
    assert off < before, f"events layer should leave {before} paths, saw {off}"
    assert on >= before - 2000, f"events should come back, saw {on} of {before}"


def test_09_world_2d_magnitude_filter(page, server):
    page.goto(server + "/2d", wait_until="domcontentloaded")
    page.wait_for_function(
        "() => /[0-9]/.test(document.getElementById('stEvents').textContent)",
        timeout=40000,
    )
    page.evaluate("() => { const s = document.getElementById('mag');"
                  " s.value = '7'; s.dispatchEvent(new Event('input', { bubbles: true })); }")
    page.wait_for_timeout(1500)
    shown = int(page.locator("#stEvents").inner_text().replace(",", ""))
    assert shown < 30000, "raising the magnitude floor should cut the plotted events"
    assert shown >= 1000


def test_10_docs_page_lists_endpoints(page, server):
    resp = page.goto(server + "/docs", wait_until="domcontentloaded")
    assert resp.status == 200
    body = page.inner_text("main")
    for path in ["/2d/assam", "/api/predict", "/api/examples", "/api/india/status",
                 "/api/india/hazard", "/api/india/events", "/api/india/state/{id}",
                 "/api/world/events", "/api/world/hazard", "/api/world/countries"]:
        assert path in body, f"{path} missing from the docs page"


def test_11_api_health(page, server):
    h = api("/api/health")
    assert h["ok"] is True
    assert h["model_loaded"] is True
    assert h["hazard_ready"] is True


def test_12_api_predict(page, server):
    opts = api("/api/options")
    payload = {f["name"]: f["default"] for f in opts["fields"]}
    req = urllib.request.Request(
        server + "/api/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.load(r)
    assert out["damage_grade"] in (1, 2, 3)
    assert abs(sum(out["probabilities"].values()) - 1) < 0.01


def test_13_api_predict_rejects_bad_input(page, server):
    opts = api("/api/options")
    payload = {f["name"]: f["default"] for f in opts["fields"]}
    payload["age"] = 99999
    req = urllib.request.Request(
        server + "/api/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req, timeout=30)
    assert err.value.code == 422


def test_14_examples_api_matches_the_panel(page, server):
    d = api("/api/examples?n=30")
    assert len(d["rows"]) == 30
    assert len(d["confusion"]) == 3
    assert 0.5 <= d["test_accuracy"] <= 0.9
    assert d["correct"] == sum(d["confusion"][i][i] for i in range(3))


def test_15_world_status(page, server):
    st = api("/api/world/status")
    assert st["countries"] == 177
    assert st["cities"] >= 4000
    assert st["catalogue"]["events"] > 5000
    assert st["hazard"]["cells"] > 1000


def test_16_world_countries_geojson(page, server):
    with urllib.request.urlopen(server + "/api/world/countries.geojson", timeout=30) as r:
        gj = json.load(r)
    assert gj["type"] == "FeatureCollection"
    names = {f["properties"]["name"] for f in gj["features"]}
    assert {"India", "Nepal", "Japan", "Chile"} <= names


def test_17_world_events_catalogue(page, server):
    d = api("/api/world/events?min_mag=7&limit=5000")
    assert d["count"] == len(d["events"])
    assert d["events"][0]["mag"] >= 7
    assert d["coverage"] in ("global", "regional")
    assert all(-180 <= e["lon"] <= 180 and -90 <= e["lat"] <= 90 for e in d["events"][:200])


def test_18_world_hazard_grid(page, server):
    d = api("/api/world/hazard")
    meta, cells = d["meta"], d["cells"]
    assert len(cells) == meta["cells_total"] > 1000
    assert meta["cells"]["red"] > 0 and meta["cells"]["green"] > 0
    assert all(c[2] in ("r", "y", "g") for c in cells[:500])


def test_19_places_api(page, server):
    d = api("/api/places")
    ids = [p["id"] for p in d["places"]]
    assert ids == ["assam", "kutch", "koyna"]
    buildings = {p["id"]: p["has_buildings"] for p in d["places"]}
    assert buildings["assam"] is True
    assert buildings["kutch"] is False and buildings["koyna"] is False

    kutch = api("/api/places/kutch")
    assert kutch["meta"]["largest"]["mag"] == 7.7
    assert kutch["meta"]["largest"]["time"] == "2001-01-26"
    assert len(kutch["cells"]) == kutch["meta"]["cells_total"] > 100
    assert all(c[2] in ("r", "y", "g") for c in kutch["cells"])
    assert 68 <= kutch["bbox"][0] and kutch["bbox"][2] <= 73

    koyna = api("/api/places/koyna")
    assert koyna["meta"]["events"] > 10
    assert all(73.0 <= c[0] <= 74.8 for c in koyna["cells"])


def test_20_india_api(page, server):
    st = api("/api/india/status")
    assert st["states"] == 36
    assert st["events"] == 1463
    assert st["events_catalogue"] == 18501
    assert st["cells"] == 2651
    assert st["footprint_states"] == ["assam"]
    assert st["footprint_buildings"] == 13302299
    assert sum(row["events"] for row in st["per_state"]) == st["events"]
    top = st["per_state"][0]
    assert top["id"] == "jammu-kashmir" and top["events"] == 500

    haz = api("/api/india/hazard")
    cells = haz["cells"]
    assert len(cells) == haz["meta"]["cells_total"] == 2651
    assert all(c[2] in ("r", "y", "g") for c in cells)
    assert {c[2] for c in cells} == {"r", "y", "g"}
    assert 67 <= min(c[0] for c in cells) and max(c[0] for c in cells) <= 98
    assert 6 <= min(c[1] for c in cells) and max(c[1] for c in cells) <= 38
    assert "not a forecast" in haz["meta"]["disclaimer"].lower()

    gujarat = api("/api/india/state/gujarat")
    assert gujarat["name"] == "Gujarat"
    assert gujarat["meta"]["largest"]["mag"] == 7.7
    assert gujarat["meta"]["largest"]["time"] == "2001-01-26"
    assert gujarat["meta"]["events"] == 151
    assert len(gujarat["rings"]) >= 1

    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(server + "/api/india/state/atlantis", timeout=30)
    assert err.value.code == 404

    with urllib.request.urlopen(server + "/static/india/states.json", timeout=30) as r:
        gj = json.load(r)
    assert len(gj["states"]) == 36
    assert sum(len(ring) for s in gj["states"] for ring in s["rings"]) == 29503


def test_20b_india_events_list(page, server):
    d = api("/api/india/events?min_mag=5")
    assert d["count"] == len(d["events"])
    assert d["count"] == 158
    assert all(e["mag"] >= 5 for e in d["events"])
    assert all(67 <= e["lon"] <= 98 and 6 <= e["lat"] <= 38 for e in d["events"])
    everything = api("/api/india/events?min_mag=4")
    assert everything["count"] == 1463
    assert everything["events"][0]["mag"] == 7.7


def test_21_docs_and_scrollbars(page, server):
    for path in ("/docs", "/2d", "/", "/2d/assam"):
        page.goto(server + path, wait_until="domcontentloaded")
        overflow = page.evaluate("() => document.documentElement.scrollWidth - window.innerWidth")
        assert overflow <= 1, f"{path} scrolls sideways by {overflow}px"
