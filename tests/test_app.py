"""Browser checks for the damage-risk site (Playwright, 15 checks).

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
def page(server):
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=angle", "--enable-webgl"])
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
def test_01_home_returns_200(page, server):
    resp = page.goto(server + "/", wait_until="domcontentloaded")
    assert resp.status == 200
    assert "Assam in 3D" in page.title()


def test_02_home_has_page_nav(page, server):
    page.goto(server + "/", wait_until="domcontentloaded")
    nav = page.locator("nav.pagenav a")
    assert nav.count() == 3
    hrefs = [nav.nth(i).get_attribute("href") for i in range(3)]
    assert hrefs == ["/", "/2d", "/docs"]


def test_03_3d_reports_building_total(page, server):
    page.goto(server + "/", wait_until="domcontentloaded")
    page.wait_for_function(
        "() => { const el = document.getElementById('st-total');"
        " return el && el.textContent.replace(/[^0-9]/g,'') !== ''; }",
        timeout=45000,
    )
    total = page.locator("#st-total").inner_text().replace(",", "").strip()
    assert total.startswith("13"), f"expected 13,3xx,xxx buildings, saw {total!r}"


def test_04_3d_loader_clears(page, server):
    page.goto(server + "/", wait_until="domcontentloaded")
    page.wait_for_selector("#loader.done", timeout=45000)
    assert page.locator("#scene canvas").count() >= 1


def test_05_example_panel_runs_model(page, server):
    page.goto(server + "/", wait_until="domcontentloaded")
    page.wait_for_selector("#loader.done", timeout=45000)
    page.click("#example-btn")
    page.wait_for_function(
        "() => { const rows = document.querySelectorAll('#e-rows tbody tr');"
        " const st = document.getElementById('e-status');"
        " return rows.length > 0 && st && /correct/i.test(st.textContent); }",
        timeout=60000,
    )
    assert page.get_attribute("#example-btn", "aria-expanded") == "true"
    assert page.locator("#e-rows tbody tr").count() == 30
    assert page.locator("#e-cm tbody tr").count() == 3
    assert "%" in page.locator("#e-acc").inner_text()


def test_06_2d_page_and_map(page, server):
    resp = page.goto(server + "/2d", wait_until="domcontentloaded")
    assert resp.status == 200
    page.wait_for_selector(".leaflet-container", timeout=20000)
    assert page.locator("#map .leaflet-pane").count() >= 1


def test_07_hazard_cells_drawn(page, server):
    page.goto(server + "/2d", wait_until="domcontentloaded")
    page.wait_for_function(
        "() => { const el = document.getElementById('cellCount');"
        " return el && /\\d+ cells/.test(el.textContent); }",
        timeout=30000,
    )
    cells = int(page.locator("#cellCount").inner_text().split()[0])
    assert cells == 912
    assert page.locator("#map path.leaflet-interactive").count() > 100
    # the load pill must clear once the grid is drawn
    page.wait_for_function(
        "() => { const el = document.getElementById('status');"
        " return !el || el.hidden || el.textContent.trim() === ''; }",
        timeout=15000,
    )


def test_08_events_layer_toggles(page, server):
    page.goto(server + "/2d", wait_until="domcontentloaded")
    page.wait_for_selector("#map path.leaflet-interactive", timeout=30000)
    before = page.locator("#map path.leaflet-interactive").count()
    page.click("#btnEvents")           # toggle off
    page.wait_for_timeout(600)
    off = page.locator("#map path.leaflet-interactive").count()
    page.click("#btnEvents")           # toggle on
    page.wait_for_timeout(600)
    on = page.locator("#map path.leaflet-interactive").count()
    assert off < before, "events layer should be removed when toggled off"
    assert on >= before, "events layer should come back when toggled on"


def test_09_damage_table_has_31_areas(page, server):
    page.goto(server + "/2d", wait_until="domcontentloaded")
    page.wait_for_function(
        "() => document.querySelectorAll('#zoneBody tr').length > 0", timeout=30000
    )
    assert page.locator("#zoneBody tr").count() == 31


def test_10_damage_table_sorts(page, server):
    page.goto(server + "/2d", wait_until="domcontentloaded")
    page.wait_for_selector("#zoneBody tr", timeout=30000)
    page.click("#zoneTable th[data-sort='g3']")
    page.wait_for_timeout(300)
    first = page.locator("#zoneBody tr td:nth-child(4)").first.inner_text()
    vals = [page.locator("#zoneBody tr td:nth-child(4)").nth(i).inner_text()
            for i in range(page.locator("#zoneBody tr").count())]
    assert vals[0] == max(vals), f"sorting by G3 actual should put the max first, saw {first}"


def test_11_docs_page_lists_endpoints(page, server):
    resp = page.goto(server + "/docs", wait_until="domcontentloaded")
    assert resp.status == 200
    body = page.inner_text("main")
    for path in ["/api/predict", "/api/examples", "/api/hazard", "/api/damage-zones", "/2d"]:
        assert path in body, f"{path} missing from the docs page"


def test_12_api_health(page, server):
    h = api("/api/health")
    assert h["ok"] is True
    assert h["model_loaded"] is True
    assert h["hazard_ready"] is True


def test_13_api_predict(page, server):
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


def test_14_api_predict_rejects_bad_input(page, server):
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


def test_15_redirects_and_scrollbars(page, server):
    for path in ("/3d", "/lab"):
        resp = page.goto(server + path, wait_until="domcontentloaded")
        assert page.url.rstrip("/") == server.rstrip("/"), f"{path} should redirect to /"
    page.goto(server + "/docs", wait_until="domcontentloaded")
    overflow = page.evaluate(
        "() => document.documentElement.scrollWidth - window.innerWidth"
    )
    assert overflow <= 1, f"docs page scrolls sideways by {overflow}px"
