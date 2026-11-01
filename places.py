"""Earthquake places: Assam (with building footprints) plus other Indian regions.

The 3D page started as Assam, where every building footprint is packaged. The same
page can show any other region where a big earthquake happened, but no building
footprints are bundled for those regions, so they are drawn from the earthquake
record instead: a real hazard grid computed from the USGS catalogue, the epicentres
themselves, and the same damage-grade model for a building profile you describe.

Nothing here is invented. Every number comes from data/quakes/india_quakes_m4_2000_2026.csv
(USGS FDSN, M4.0+ India region, 2000-01-01 to 2026-10-26) and is recomputed on
request. The hazard grid ranks where past events clustered; it is not a forecast
and not a design-level seismic hazard map.
"""
from __future__ import annotations

import csv
import math
import os
import threading

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
QUAKE_CSV = os.path.join(ROOT, "data", "quakes", "india_quakes_m4_2000_2026.csv")

# The catalogue in the repo has '?' where the USGS feed had accented characters.
# These are the names that actually occur inside the place boxes below.
NAME_FIX = {
    "Bhach?u": "Bhachau",
    "R?par": "Rapar",
    "Anj?r": "Anjar",
    "P?tan": "Patan",
    "M?khjan": "Makhjan",
    "L?nja": "Lanja",
    "K?r?d": "Karad",
    "S?t?ra": "Satara",
    "W?ngjing": "Wangjing",
    "Wāngjing": "Wangjing",
    "Imph?l": "Imphal",
    "Chur?ch?ndpur": "Churachandpur",
    "Hoj?i": "Hojai",
    "?mb?sa": "Ambasa",
    "Tezpur": "Tezpur",
}

PLACES = {
    "assam": {
        "id": "assam",
        "name": "Assam",
        "kind": "buildings",
        "route": "/",
        "bbox": (89.55, 24.00, 96.15, 28.15),
        "focus": (26.145, 91.736),
        "headline": "13.3 million building footprints, clickable",
        "blurb": (
            "Every Overture Maps footprint in the state, streamed as 0.1 degree tiles. "
            "Click a building to get a damage grade from the model."
        ),
    },
    "kutch": {
        "id": "kutch",
        "name": "Kutch, Gujarat",
        "kind": "record",
        "route": "/place/kutch",
        "bbox": (68.40, 22.00, 72.20, 24.90),
        "focus": (23.42, 70.23),
        "headline": "Bhuj earthquake, 26 January 2001, M7.7",
        "blurb": (
            "The strongest earthquake in India since 1950, 17 km north-west of Bhachau. "
            "No building footprints are bundled for Gujarat yet, so this view draws the "
            "hazard grid and the epicentres from the USGS catalogue."
        ),
        "cell_deg": 0.15,
        "max_cells": 320,
        "min_mag": 4.0,
    },
    "koyna": {
        "id": "koyna",
        "name": "Koyna, Maharashtra",
        "kind": "record",
        "route": "/place/koyna",
        "bbox": (73.10, 16.50, 74.70, 18.10),
        "focus": (17.40, 73.78),
        "headline": "Koyna earthquake, 11 December 1967, M6.6",
        "blurb": (
            "The reservoir-triggered earthquake that started the Koyna sequence, which is "
            "still the most active seismic zone in peninsular India. No building footprints "
            "are bundled for Maharashtra yet, so this view draws the hazard grid and the "
            "epicentres from the USGS catalogue."
        ),
        "cell_deg": 0.10,
        "max_cells": 260,
        "min_mag": 4.0,
    },
}

DEFAULT_PLACE = "assam"

_lock = threading.Lock()
_catalogue: list[dict] | None = None
_cache: dict[str, dict] = {}


def fix_name(text: str) -> str:
    if not text:
        return ""
    out = text
    for bad, good in NAME_FIX.items():
        out = out.replace(bad, good)
    return out.replace("?", "").strip()


def catalogue() -> list[dict]:
    """Every row of the bundled USGS India catalogue, as plain dicts."""
    global _catalogue
    with _lock:
        if _catalogue is not None:
            return _catalogue
        rows: list[dict] = []
        with open(QUAKE_CSV, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    lon = float(r["longitude"])
                    lat = float(r["latitude"])
                    mag = float(r["mag"])
                except (TypeError, ValueError, KeyError):
                    continue
                depth = r.get("depth") or ""
                try:
                    depth = float(depth)
                except ValueError:
                    depth = None
                rows.append({
                    "lon": lon,
                    "lat": lat,
                    "mag": mag,
                    "depth": None if depth is None else round(depth, 1),
                    "time": (r.get("time") or "")[:10],
                    "place": fix_name(r.get("place") or ""),
                })
        rows.sort(key=lambda e: e["time"])
        _catalogue = rows
        return rows


def in_box(ev: dict, bbox: tuple[float, float, float, float]) -> bool:
    west, south, east, north = bbox
    return west <= ev["lon"] <= east and south <= ev["lat"] <= north


def events_for(pid: str) -> list[dict]:
    spec = PLACES[pid]
    floor = spec.get("min_mag", 4.0)
    return [e for e in catalogue() if e["mag"] >= floor and in_box(e, spec["bbox"])]


def hazard_for(pid: str) -> dict:
    """Gutenberg-Richter energy per cell inside the place box, smoothed, in terciles."""
    with _lock:
        if pid in _cache:
            return _cache[pid]

    spec = PLACES[pid]
    west, south, east, north = spec["bbox"]
    cell = float(spec.get("cell_deg", 0.2))
    sigma = cell * 2.0
    max_cells = int(spec.get("max_cells", 300))
    evs = events_for(pid)

    nx = max(1, int(round((east - west) / cell)))
    ny = max(1, int(round((north - south) / cell)))
    grid = np.zeros((nx, ny), dtype=float)
    counts = np.zeros((nx, ny), dtype=int)
    peak = np.zeros((nx, ny), dtype=float)

    for e in evs:
        ix = min(nx - 1, max(0, int((e["lon"] - west) / cell)))
        iy = min(ny - 1, max(0, int((e["lat"] - south) / cell)))
        grid[ix, iy] += 10.0 ** (1.5 * e["mag"])
        counts[ix, iy] += 1
        peak[ix, iy] = max(peak[ix, iy], e["mag"])

    # Gaussian smoothing, wrapping is wrong at a box edge but conservative: the kernel
    # simply runs off the array, so edge cells are not inflated.
    r = max(1, int(math.ceil(2 * sigma / cell)))
    off = np.arange(-r, r + 1) * cell
    k = np.exp(-0.5 * (off / sigma) ** 2)
    smooth = np.zeros_like(grid)
    for i, wi in enumerate(k):
        for j, wj in enumerate(k):
            ar = np.roll(np.roll(grid, i - r, axis=0), j - r, axis=1)
            smooth += ar * wi * wj
    smooth = smooth / k.sum() ** 2

    vals = smooth[smooth > 0]
    if vals.size == 0:
        payload = {"cells": [], "meta": _meta(pid, cell, sigma, 0, 0, 0, evs)}
        with _lock:
            _cache[pid] = payload
        return payload

    # keep the strongest cells, then terciles of that selection
    flat = np.argsort(smooth, axis=None)[::-1]
    keep = flat[:max_cells]
    scores = np.sort(smooth.ravel()[keep])[::-1]
    n = len(scores)
    red_cut = float(scores[max(0, int(n * 1 / 3) - 1)])
    yellow_cut = float(scores[max(0, int(n * 2 / 3) - 1)])

    cells = []
    for idx in keep:
        ix, iy = np.unravel_index(idx, smooth.shape)
        s = float(smooth[ix, iy])
        if s <= 0:
            continue
        cls = "r" if s >= red_cut else ("y" if s >= yellow_cut else "g")
        cells.append([
            round(float(west + ix * cell), 2),
            round(float(south + iy * cell), 2),
            cls,
            round(float(math.log10(s)), 2),
            int(counts[ix, iy]),
            round(float(peak[ix, iy]), 1) if peak[ix, iy] else None,
        ])

    meta = _meta(pid, cell, sigma, len(cells), red_cut, yellow_cut, evs)
    payload = {"cells": cells, "meta": meta}
    with _lock:
        _cache[pid] = payload
    return payload


def _meta(pid: str, cell: float, sigma: float, n_cells: int,
          red_cut: float, yellow_cut: float, evs: list[dict]) -> dict:
    spec = PLACES[pid]
    strongest = max(evs, key=lambda e: e["mag"]) if evs else None
    return {
        "id": pid,
        "name": spec["name"],
        "title": f"Relative earthquake hazard, {spec['name']}",
        "source": "USGS FDSN catalogue, M4.0+ India region, 2000-01-01 to 2026-10-26",
        "cell_deg": cell,
        "smoothing_sigma_deg": sigma,
        "cells_total": n_cells,
        "events": len(evs),
        "events_m5": sum(1 for e in evs if e["mag"] >= 5),
        "largest": strongest,
        "thresholds_log10_energy": {"red_from": round(math.log10(red_cut), 2) if red_cut else None,
                                   "yellow_from": round(math.log10(yellow_cut), 2) if yellow_cut else None},
        "method": (
            f"Gutenberg-Richter energy 10^(1.5M) per {cell:g} degree cell, Gaussian-smoothed "
            f"(sigma {sigma:g} degrees), inside the box {spec['bbox']}. The strongest cells are "
            "kept and split into terciles: red = top third, yellow = middle, green = bottom."
        ),
        "disclaimer": (
            "Ranks where magnitude 4+ earthquakes have clustered since 2000. Catalogue "
            "completeness varies; this is not a forecast, not a safety rating, and not a "
            "seismic design map."
        ),
    }


def place(pid: str) -> dict:
    if pid not in PLACES:
        raise KeyError(pid)
    spec = PLACES[pid]
    out = {
        "id": spec["id"],
        "name": spec["name"],
        "kind": spec["kind"],
        "route": spec["route"],
        "bbox": list(spec["bbox"]),
        "focus": list(spec["focus"]),
        "headline": spec["headline"],
        "blurb": spec["blurb"],
        "has_buildings": spec["kind"] == "buildings",
    }
    if spec["kind"] != "buildings":
        haz = hazard_for(pid)
        meta = haz["meta"]
        out.update({
            "cells": haz["cells"],
            "meta": meta,
            "events": events_for(pid),
        })
    return out


def list_places() -> dict:
    items = []
    for pid, spec in PLACES.items():
        item = {
            "id": pid,
            "name": spec["name"],
            "kind": spec["kind"],
            "route": spec["route"],
            "headline": spec["headline"],
            "has_buildings": spec["kind"] == "buildings",
        }
        if spec["kind"] != "buildings":
            meta = hazard_for(pid)["meta"]
            item.update({
                "events": meta["events"],
                "events_m5": meta["events_m5"],
                "cells": meta["cells_total"],
                "largest": meta["largest"],
            })
        items.append(item)
    return {"places": items, "default": DEFAULT_PLACE}
