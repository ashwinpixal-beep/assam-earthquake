"""Whole-India data for the 3D map at /.

Three real sources, no fabrication:

  states   India state and union-territory outlines, built by
           scripts/build_india_states.py from the MIT-licensed Survey of India set.
  events   the bundled USGS India catalogue, M4.0+ from 2000-01-01 to 2026-10-26
           (data/quakes/india_quakes_m4_2000_2026.csv, 18,501 rows). Each event is
           assigned to a state by point-in-polygon.
  buildings only Assam has a packaged footprint tile set (837 tiles, 13.3 million
           buildings). Everything else is drawn from the earthquake record: hazard
           grid plus epicentres. The API says which states have footprints so the
           page never implies more than it has.

The hazard grid is Gutenberg-Richter energy per 0.25 degree cell, Gaussian smoothed,
split into terciles. It ranks where past events clustered; it is not a forecast and
not a seismic design map.
"""
from __future__ import annotations

import json
import math
import os
import threading

import numpy as np

import places

ROOT = os.path.dirname(os.path.abspath(__file__))
STATES_FILE = os.path.join(ROOT, "public", "static", "india", "states.json")

CELL = 0.25                  # degrees, national grid
SIGMA = 0.35                 # degrees, Gaussian smoothing
MAX_CELLS = 5000             # strongest cells kept

# Only these states have a packaged building-footprint tile set.
FOOTPRINT_STATES = {"assam"}

MASK_STEP = 0.05             # degrees, membership raster (about 5 km)
MASK_WEST, MASK_SOUTH = 66.0, 4.0
MASK_EAST, MASK_NORTH = 99.0, 39.0

_lock = threading.RLock()
_states: list[dict] | None = None
_state_cache: dict[str, dict] = {}
_hazard_cache: dict | None = None
_mask = None                 # int16 raster: index into states(), -1 outside India


def _ring_mask(ring, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Vectorised even-odd ray casting for one ring over an array of points."""
    inside = np.zeros(lons.shape, dtype=bool)
    n = len(ring)
    xs = np.array([p[0] for p in ring])
    ys = np.array([p[1] for p in ring])
    for i in range(n):
        xi, yi = xs[i], ys[i]
        xj, yj = xs[i - 1], ys[i - 1]
        if yi == yj:
            continue
        straddles = (yi > lats) != (yj > lats)
        if not straddles.any():
            continue
        t = (xj - xi) / (yj - yi)
        crosses = lons < t * (lats - yi) + xi
        inside ^= straddles & crosses
    return inside


def state_mask():
    """One raster for the whole country: every cell carries the state it falls in.

    A per-event point-in-polygon loop over 18,501 events and 36 detailed outlines takes
    minutes in Python; rasterising once takes about a second and answers both the event
    assignment and the hazard-cell land test as an array lookup.
    """
    global _mask
    with _lock:
        if _mask is not None:
            return _mask
        nx = int(round((MASK_EAST - MASK_WEST) / MASK_STEP))
        ny = int(round((MASK_NORTH - MASK_SOUTH) / MASK_STEP))
        mask = np.full((nx, ny), -1, dtype=np.int16)
        for idx, s in enumerate(states()):
            w, south, e, north = s["bbox"]
            ix0 = max(0, int((w - MASK_WEST) / MASK_STEP) - 1)
            ix1 = min(nx, int((e - MASK_WEST) / MASK_STEP) + 2)
            iy0 = max(0, int((south - MASK_SOUTH) / MASK_STEP) - 1)
            iy1 = min(ny, int((north - MASK_SOUTH) / MASK_STEP) + 2)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            lons = (MASK_WEST + (np.arange(ix0, ix1) + 0.5) * MASK_STEP)[:, None]
            lats = (MASK_SOUTH + (np.arange(iy0, iy1) + 0.5) * MASK_STEP)[None, :]
            lons, lats = np.broadcast_arrays(lons, lats)
            hit = np.zeros(lons.shape, dtype=bool)
            for ring in s["rings"]:
                hit |= _ring_mask(ring, lons, lats)
            block = mask[ix0:ix1, iy0:iy1]
            block[hit & (block < 0)] = idx
        _mask = mask
        return mask


def state_at(lon: float, lat: float) -> int:
    """Index into states(), or -1 for a point outside every outline."""
    mask = state_mask()
    ix = int((lon - MASK_WEST) / MASK_STEP)
    iy = int((lat - MASK_SOUTH) / MASK_STEP)
    if not (0 <= ix < mask.shape[0] and 0 <= iy < mask.shape[1]):
        return -1
    return int(mask[ix, iy])


# ------------------------------------------------------------------ geometry
def _ring_contains(ring, lon, lat) -> bool:
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def _in_state(state, lon, lat) -> bool:
    w, s, e, n = state["bbox"]
    if not (w <= lon <= e and s <= lat <= n):
        return False
    return any(_ring_contains(r, lon, lat) for r in state["rings"])


def states() -> list[dict]:
    global _states
    with _lock:
        if _states is not None:
            return _states
        data = json.load(open(STATES_FILE))
        out = []
        for s in data["states"]:
            out.append({
                "id": s["id"],
                "name": s["name"],
                "bbox": s["bbox"],
                "centre": s["centre"],
                "rings": s["rings"],
                "has_footprints": s["id"] in FOOTPRINT_STATES,
            })
        _states = out
        return out


def state_by_id(sid: str) -> dict:
    for s in states():
        if s["id"] == sid:
            return s
    raise KeyError(sid)


# ------------------------------------------------------------------ events
_events_by_state: dict[str, list[dict]] | None = None


def events_by_state() -> dict[str, list[dict]]:
    """Assign every catalogued event to the state that contains it."""
    global _events_by_state
    with _lock:
        if _events_by_state is not None:
            return _events_by_state
        st = states()
        buckets: dict[str, list[dict]] = {s["id"]: [] for s in st}
        stray = 0
        for ev in places.catalogue():
            idx = state_at(ev["lon"], ev["lat"])
            if idx < 0:
                stray += 1
                continue
            buckets[st[idx]["id"]].append(ev)
        _events_by_state = buckets
        return buckets


# ------------------------------------------------------------------ hazard grid
def _smooth(grid: np.ndarray, cell: float, sigma: float) -> np.ndarray:
    r = max(1, int(math.ceil(2 * sigma / cell)))
    off = np.arange(-r, r + 1) * cell
    k = np.exp(-0.5 * (off / sigma) ** 2)
    out = np.zeros_like(grid)
    for i, wi in enumerate(k):
        for j, wj in enumerate(k):
            out += np.roll(np.roll(grid, i - r, axis=0), j - r, axis=1) * wi * wj
    return out / k.sum() ** 2


def hazard_national() -> dict:
    """One 0.25 degree grid over the whole country, terciles of the strongest cells."""
    global _hazard_cache
    with _lock:
        if _hazard_cache is not None:
            return _hazard_cache

    west, south, east, north = 67.5, 6.0, 98.5, 37.5
    nx = int(round((east - west) / CELL))
    ny = int(round((north - south) / CELL))
    grid = np.zeros((nx, ny))
    counts = np.zeros((nx, ny), int)
    peak = np.zeros((nx, ny))

    evs = places.catalogue()
    for e in evs:
        ix = int((e["lon"] - west) / CELL)
        iy = int((e["lat"] - south) / CELL)
        if not (0 <= ix < nx and 0 <= iy < ny):
            continue
        grid[ix, iy] += 10.0 ** (1.5 * e["mag"])
        counts[ix, iy] += 1
        peak[ix, iy] = max(peak[ix, iy], e["mag"])

    smooth = _smooth(grid, CELL, SIGMA)

    # keep the strongest cells that actually sit over land
    flat = np.argsort(smooth, axis=None)[::-1]
    chosen: list[tuple[int, int]] = []
    for idx in flat:
        if len(chosen) >= MAX_CELLS:
            break
        ix, iy = np.unravel_index(idx, smooth.shape)
        if smooth[ix, iy] <= 0:
            break
        lon = west + (ix + 0.5) * CELL
        lat = south + (iy + 0.5) * CELL
        if state_at(lon, lat) < 0:          # keep the strongest cells that sit over land
            continue
        chosen.append((int(ix), int(iy)))

    scores = np.array([smooth[ix, iy] for ix, iy in chosen])
    order = np.argsort(scores)[::-1]
    n = len(order)
    red_cut = float(scores[order[max(0, n // 3 - 1)]]) if n else 0.0
    yellow_cut = float(scores[order[max(0, 2 * n // 3 - 1)]]) if n else 0.0

    cells = []
    for ix, iy in chosen:
        s = float(smooth[ix, iy])
        cls = "r" if s >= red_cut else ("y" if s >= yellow_cut else "g")
        cells.append([
            round(west + ix * CELL, 2), round(south + iy * CELL, 2), cls,
            round(math.log10(s), 2), int(counts[ix, iy]),
            round(float(peak[ix, iy]), 1) if peak[ix, iy] else None,
        ])

    meta = {
        "title": "Relative earthquake hazard, India",
        "source": "USGS FDSN catalogue, M4.0+ India region, 2000-01-01 to 2026-10-26",
        "events": len(evs),
        "events_m5": sum(1 for e in evs if e["mag"] >= 5),
        "events_note": (
            "The energy field uses every event in the bundled India-region catalogue, including "
            "neighbouring countries, so the border belts (Nepal, Myanmar, the Hindu Kush) are not "
            "artificially cut off. Per-state counts use only events inside that state's outline."
        ),
        "cell_deg": CELL,
        "smoothing_sigma_deg": SIGMA,
        "cells_total": len(cells),
        "thresholds_log10_energy": {"red_from": round(math.log10(red_cut), 2) if red_cut else None,
                                   "yellow_from": round(math.log10(yellow_cut), 2) if yellow_cut else None},
        "method": (
            f"Gutenberg-Richter energy 10^(1.5M) per {CELL:g} degree cell, Gaussian-smoothed "
            f"(sigma {SIGMA:g} degrees), for the whole country. Every land cell with any catalogued "
            f"energy is kept (at most {MAX_CELLS:,}) and split into terciles: red = top third, "
            "yellow = middle, green = bottom."
        ),
        "disclaimer": (
            "Ranks where magnitude 4+ earthquakes have clustered since 2000. Catalogue completeness "
            "varies by region and decade. Not a forecast, not a safety rating, not a seismic design map."
        ),
    }
    payload = {"cells": cells, "meta": meta}
    with _lock:
        _hazard_cache = payload
    return payload


def state_detail(sid: str) -> dict:
    """One state: its outline, its events, and a hazard grid computed inside its box."""
    with _lock:
        if sid in _state_cache:
            return _state_cache[sid]

    s = state_by_id(sid)
    evs = events_by_state().get(sid, [])
    payload = {
        "id": s["id"],
        "name": s["name"],
        "bbox": s["bbox"],
        "centre": s["centre"],
        "rings": s["rings"],
        "has_footprints": s["has_footprints"],
        "events": evs,
        "meta": {
            "events": len(evs),
            "events_m5": sum(1 for e in evs if e["mag"] >= 5),
            "largest": max(evs, key=lambda e: e["mag"]) if evs else None,
            "bounds": s["bbox"],
        },
    }
    with _lock:
        _state_cache[sid] = payload
    return payload


def in_india_events() -> list[dict]:
    """Only the catalogued events that fall inside an Indian state outline."""
    out = []
    for evs in events_by_state().values():
        out.extend(evs)
    out.sort(key=lambda e: -e["mag"])
    return out


def status() -> dict:
    haz = hazard_national()
    buckets = events_by_state()
    inside = in_india_events()
    per_state = [
        {
            "id": s["id"],
            "name": s["name"],
            "has_footprints": s["has_footprints"],
            "events": len(buckets.get(s["id"], [])),
            "events_m5": sum(1 for e in buckets.get(s["id"], []) if e["mag"] >= 5),
            "largest": max((e["mag"] for e in buckets.get(s["id"], [])), default=None),
        }
        for s in states()
    ]
    per_state.sort(key=lambda d: -d["events"])
    return {
        "states": len(per_state),
        "events_catalogue": haz["meta"]["events"],
        "events": len(inside),
        "events_m5": sum(1 for e in inside if e["mag"] >= 5),
        "largest": max((e["mag"] for e in inside), default=None),
        "cells": haz["meta"]["cells_total"],
        "cell_deg": CELL,
        "footprint_states": sorted(FOOTPRINT_STATES),
        "footprint_buildings": 13302299,
        "footprint_note": (
            "Only Assam has a packaged building-footprint tile set (837 tiles, 13,302,299 "
            "buildings). Every other state is drawn from the earthquake catalogue, not from "
            "invented geometry."
        ),
        "method": haz["meta"]["method"],
        "disclaimer": haz["meta"]["disclaimer"],
        "per_state": per_state,
    }
