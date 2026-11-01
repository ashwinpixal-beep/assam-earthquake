"""World-scale data for the /globe (3D) and /world (2D) pages.

Three real datasets, no fabrication:

  countries  Natural Earth 1:110m admin-0 polygons, built by scripts/build_world_data.py
  cities     GeoNames cities with population >= 100,000, same script
  events     USGS FDSN catalogue, fetched live and cached on disk. When the host has no
             outbound access, the bundled global catalogue is served instead: M5.0+
             worldwide 2016-2025 and M6.0+ since 1900, from the USGS via the Live Atlas
             release (CC BY 4.0).

The hazard grid is computed from whichever catalogue is available: Gutenberg-Richter
energy per cell, Gaussian smoothed, split into terciles. It ranks where past events
clustered; it is not a forecast and not a design-level hazard map.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.parse
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
WORLD_DIR = os.path.join(ROOT, "public", "static", "world")
QUAKE_DIR = os.path.join(ROOT, "data", "quakes")
CACHE_DIR = os.path.join(ROOT, "data", "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "usgs_global_m55.json")
BUNDLED = os.path.join(QUAKE_DIR, "world_catalogue.json")

USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_PARAMS = {
    "format": "geojson",
    "starttime": "2000-01-01",
    "minmagnitude": "5.5",
    "orderby": "time-asc",
}
CACHE_TTL = 7 * 24 * 3600          # refresh the live catalogue weekly
FETCH_TIMEOUT = 25

CELL = 2.0                          # degrees, world grid
SIGMA = 2.0                         # degrees, Gaussian smoothing
MAX_CELLS = 6000                    # strongest cells kept (the map is a ranking, not a raster)

_lock = threading.Lock()
_state: dict = {
    "events": None,          # list of event dicts
    "source": None,
    "coverage": None,
    "fetched_at": None,
    "warming": False,
    "error": None,
}
_countries = None
_cities = None
_grid_cache: dict = {}


# ------------------------------------------------------------------ static data
def countries() -> dict:
    global _countries
    if _countries is None:
        with open(os.path.join(WORLD_DIR, "countries.json")) as f:
            _countries = json.load(f)
    return _countries


def cities() -> dict:
    global _cities
    if _cities is None:
        with open(os.path.join(WORLD_DIR, "cities.json")) as f:
            _cities = json.load(f)
    return _cities


def countries_geojson() -> dict:
    doc = countries()
    feats = []
    for c in doc["countries"]:
        polys = []
        for ring in c["rings"]:
            polys.append([ring])
        # rings here are already split per polygon part by the builder, so wrap each
        feats.append({
            "type": "Feature",
            "properties": {"name": c["name"], "id": c["id"]},
            "geometry": {"type": "MultiPolygon", "coordinates": polys},
        })
    return {
        "type": "FeatureCollection",
        "meta": {
            "source": doc["source"],
            "licence": doc["licence"],
            "countries": doc["stats"]["countries"],
        },
        "features": feats,
    }


# ------------------------------------------------------------------ catalogue
def _load_disk_cache() -> dict | None:
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE) as f:
            doc = json.load(f)
        if time.time() - doc.get("fetched_at", 0) > CACHE_TTL:
            return None
        return doc
    except (OSError, ValueError):
        return None


def _fetch_usgs() -> dict:
    url = USGS + "?" + urllib.parse.urlencode(USGS_PARAMS)
    with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT) as r:
        raw = r.read()
    doc = json.loads(raw)
    events = []
    for f in doc.get("features", []):
        p = f.get("properties") or {}
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates") or [None, None, None]
        if coords[0] is None or coords[1] is None:
            continue
        mag = p.get("mag")
        if mag is None:
            continue
        events.append({
            "lon": round(float(coords[0]), 3),
            "lat": round(float(coords[1]), 3),
            "mag": round(float(mag), 1),
            "depth": round(float(coords[2]), 1) if coords[2] is not None else None,
            "time": time.strftime("%Y-%m-%d", time.gmtime((p.get("time") or 0) / 1000)),
            "place": (p.get("place") or "")[:120],
        })
    events.sort(key=lambda e: e["time"])
    out = {
        "source": "USGS FDSN event service, M5.5+, 2000-01-01 onwards",
        "coverage": "global",
        "url": url,
        "fetched_at": time.time(),
        "events": events,
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    return out


_bundled_doc = None


def _fallback() -> dict:
    """Bundled global catalogue, decoded from the compact arrays on first use."""
    global _bundled_doc
    if _bundled_doc is None:
        import datetime

        with open(BUNDLED) as f:
            doc = json.load(f)
        epoch = datetime.date(1900, 1, 1)
        events = []
        for lon, lat, mag, depth, day, place in doc["events"]:
            events.append({
                "lon": lon / 1000,
                "lat": lat / 1000,
                "mag": mag / 10,
                "depth": None if depth < 0 else float(depth),
                "time": (epoch + datetime.timedelta(days=day)).isoformat(),
                "place": place,
            })
        _bundled_doc = {
            "source": doc["source"],
            "coverage": "global",
            "licence": doc["licence"],
            "sets": doc["sets"],
            "stats": doc["stats"],
            "events": events,
            "fetched_at": None,
            "note": (
                "Live USGS feed unreachable from this host, so the bundled global catalogue "
                "is served: M5.0+ worldwide 2016-2025 and M6.0+ since 1900, frozen at the "
                "release date. The live feed is preferred whenever the host has outbound access."
            ),
        }
    return dict(_bundled_doc)


def _warm() -> None:
    """Fetch the global catalogue in the background; never block a request."""
    try:
        doc = _fetch_usgs()
        with _lock:
            _state.update(events=doc["events"], source=doc["source"], coverage="global",
                          fetched_at=doc["fetched_at"], warming=False, error=None)
    except Exception as exc:                      # noqa: BLE001 - any failure falls back
        with _lock:
            _state.update(warming=False, error=f"{type(exc).__name__}: {exc}")


def catalogue() -> dict:
    """Current catalogue: disk cache, then live fetch (background), then fallback."""
    with _lock:
        if _state["events"] is not None:
            return {
                "source": _state["source"],
                "coverage": _state["coverage"],
                "fetched_at": _state["fetched_at"],
                "warming": _state["warming"],
                "error": _state["error"],
                "events": _state["events"],
            }
        cached = _load_disk_cache()
        if cached:
            _state.update(events=cached["events"], source=cached["source"], coverage="global",
                          fetched_at=cached["fetched_at"], warming=False, error=None)
            return {
                "source": cached["source"], "coverage": "global",
                "fetched_at": cached["fetched_at"], "warming": False,
                "error": None, "events": cached["events"],
            }
        if not _state["warming"]:
            _state["warming"] = True
            threading.Thread(target=_warm, daemon=True, name="usgs-warm").start()

    doc = _fallback()
    doc["warming"] = _state.get("warming", False)
    doc["error"] = _state.get("error")
    return doc


def events_payload(min_mag: float = 5.5, limit: int = 6000) -> dict:
    cat = catalogue()
    rows = [e for e in cat["events"] if e["mag"] >= min_mag]
    rows.sort(key=lambda e: -e["mag"])
    rows = rows[:limit]
    lats = [e["lat"] for e in rows] or [0]
    lons = [e["lon"] for e in rows] or [0]
    return {
        "source": cat["source"],
        "coverage": cat["coverage"],
        "warming": cat.get("warming", False),
        "error": cat.get("error"),
        "note": cat.get("note"),
        "fetched_at": cat.get("fetched_at"),
        "min_magnitude": min_mag,
        "count": len(rows),
        "bbox": [min(lons), min(lats), max(lons), max(lats)],
        "events": rows,
    }


# ------------------------------------------------------------------ hazard grid
def hazard_grid(force: bool = False) -> dict:
    cat = catalogue()
    key = (cat["coverage"], len(cat["events"]), round(float(cat["events"][0]["mag"]) if cat["events"] else 0, 1))
    if not force and key in _grid_cache:
        return _grid_cache[key]

    ev = cat["events"]
    lons = np.array([e["lon"] for e in ev], dtype=float)
    lats = np.array([e["lat"] for e in ev], dtype=float)
    mags = np.array([e["mag"] for e in ev], dtype=float)
    energy = 10.0 ** (1.5 * mags)

    nx, ny = int(round(360 / CELL)), int(round(180 / CELL))
    ix = np.clip(((lons + 180.0) / CELL).astype(int), 0, nx - 1)
    iy = np.clip(((lats + 90.0) / CELL).astype(int), 0, ny - 1)

    E = np.zeros((ny, nx))
    N = np.zeros((ny, nx))
    M = np.zeros((ny, nx))
    np.add.at(E, (iy, ix), energy)
    np.add.at(N, (iy, ix), 1.0)
    np.maximum.at(M, (iy, ix), mags)

    r = int(math.ceil(3 * SIGMA / CELL))
    off = np.arange(-r, r + 1) * CELL
    k = np.exp(-0.5 * (off / SIGMA) ** 2)
    k /= k.sum()
    S = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, E)
    S = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, S)
    score = np.log10(S + 1e-12)

    active = S > 1e-6
    if not active.any():
        empty = {"cells": [], "meta": {"coverage": cat["coverage"], "cells": 0}}
        _grid_cache[key] = empty
        return empty

    ys, xs = np.nonzero(active)
    order = np.argsort(-score[ys, xs])
    if len(order) > MAX_CELLS:
        order = order[:MAX_CELLS]
    kept = score[ys[order], xs[order]]
    t1, t2 = np.quantile(kept, [1 / 3, 2 / 3])

    cells = []
    counts = {"red": 0, "yellow": 0, "green": 0}
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        s = float(score[y, x])
        cls = "red" if s > t2 else ("yellow" if s > t1 else "green")
        counts[cls] += 1
        cells.append([
            round(-180.0 + x * CELL, 2),
            round(-90.0 + y * CELL, 2),
            cls[0],
            round(s, 2),
            int(N[y, x]),
            round(float(M[y, x]), 1) if N[y, x] else 0.0,
        ])

    meta = {
        "title": "Relative earthquake hazard, world",
        "source": cat["source"],
        "coverage": cat["coverage"],
        "warming": cat.get("warming", False),
        "note": cat.get("note"),
        "events": len(ev),
        "cell_deg": CELL,
        "smoothing_sigma_deg": SIGMA,
        "thresholds_log10_energy": {"yellow_from": round(float(t1), 3), "red_from": round(float(t2), 3)},
        "cells": counts,
        "cells_total": len(cells),
        "max_cells": MAX_CELLS,
        "method": (
            f"Gutenberg-Richter energy 10^(1.5M) per {CELL:g} degree cell, Gaussian-smoothed "
            f"(sigma {SIGMA:g} degrees). The {MAX_CELLS:,} highest-scoring cells are kept and "
            "split into terciles of the score: red = top third, yellow = middle, green = bottom. "
            "Cells with no activity nearby are left blank rather than drawn green."
        ),
        "disclaimer": (
            "Ranks where magnitude 5.5+ earthquakes have clustered since 2000. Catalogue "
            "completeness varies by region and by decade. It is not a forecast, not a safety "
            "rating, and not a seismic design map."
        ),
    }
    doc = {"cells": cells, "meta": meta}
    _grid_cache[key] = doc
    return doc


def status() -> dict:
    cat = catalogue()
    grid = hazard_grid()
    tiles = os.path.join(WORLD_DIR, "tiles", "index.json")
    has_tiles = os.path.exists(tiles)
    detail = None
    if has_tiles:
        with open(tiles) as f:
            detail = json.load(f)
    return {
        "countries": countries()["stats"]["countries"],
        "cities": cities()["stats"]["cities"],
        "catalogue": {
            "source": cat["source"],
            "coverage": cat["coverage"],
            "events": len(cat["events"]),
            "warming": cat.get("warming", False),
            "error": cat.get("error"),
            "fetched_at": cat.get("fetched_at"),
        },
        "hazard": {
            "cells": grid["meta"].get("cells_total", 0),
            "cell_deg": CELL,
            "coverage": grid["meta"].get("coverage"),
        },
        "detail_buildings": (detail or {}).get("total_buildings"),
        "detail_tiles": (detail or {}).get("tiles"),
        "detail_ready": has_tiles,
    }
