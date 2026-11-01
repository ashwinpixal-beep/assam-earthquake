#!/usr/bin/env python3
"""Build public/static/india/states.json from an Indian state boundary GeoJSON.

Source: mraxays/india-states.geojson (MIT, 2024), 36 states and union territories,
originally from the Survey of India / Datameet boundary set.

The raw file is 7.8 MB. This script
  * rounds coordinates to 3 decimals (about 110 m, plenty for a national view),
  * drops rings smaller than MIN_AREA square degrees (specks and rocks),
  * snaps every remaining ring to a GRID-degree lattice (about 2 km),
and writes a compact file the browser can load in one request.

    python3 scripts/build_india_states.py path/to/states_india.geojson
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "public", "static", "india")
OUT = os.path.join(OUT_DIR, "states.json")

GRID = 0.02               # degrees (about 2 km), coordinates snap to this grid
MIN_AREA = 0.004          # square degrees, drop rings smaller than this

NAME_FIX = {
    "Arunanchal Pradesh": "Arunachal Pradesh",
    "Dadara & Nagar Havelli": "Dadra & Nagar Haveli",
    "Daman & Diu": "Daman & Diu",
    "Andaman & Nicobar Island": "Andaman & Nicobar",
    "NCT of Delhi": "Delhi",
}

# codes used by the earthquake catalogue and by the page headings
IDS = {
    "Assam": "assam", "Gujarat": "gujarat", "Maharashtra": "maharashtra",
    "Uttarakhand": "uttarakhand", "Himachal Pradesh": "himachal",
    "Jammu & Kashmir": "jammu-kashmir", "Bihar": "bihar", "Sikkim": "sikkim",
    "West Bengal": "west-bengal", "Meghalaya": "meghalaya", "Manipur": "manipur",
    "Mizoram": "mizoram", "Nagaland": "nagaland", "Tripura": "tripura",
    "Arunachal Pradesh": "arunachal", "Andaman & Nicobar": "andaman",
}


def ring_area(ring):
    a = 0.0
    for i in range(len(ring)):
        x1, y1 = ring[i - 1]
        x2, y2 = ring[i]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2


def snap_ring(ring):
    """Snap to a GRID-degree lattice, drop repeats and collinear runs.

    At 2 MB the raw Survey of India outlines are too heavy for a browser page. Snapping
    keeps every coastal bend that is wider than the grid (~2 km) and costs almost nothing
    to run, which a Douglas-Peucker pass on 359,000 points is not.
    """
    pts = []
    for x, y in ring:
        p = (round(x / GRID) * GRID, round(y / GRID) * GRID)
        if not pts or p != pts[-1]:
            pts.append(p)
    if len(pts) < 3:
        return [[round(x, 3), round(y, 3)] for x, y in pts]
    if pts[0] != pts[-1]:
        pts.append(pts[0])

    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = out[-1], pts[i], pts[i + 1]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cross) > 1e-12:
            out.append(b)
    out.append(pts[-1])
    if len(out) < 4:                       # a speck: keep the raw corners rather than nothing
        out = pts
    return [[round(x, 3), round(y, 3)] for x, y in out]


def main(src):
    gj = json.load(open(src))
    states = []
    for f in gj["features"]:
        props = f["properties"]
        name = NAME_FIX.get(props.get("st_nm") or "", props.get("st_nm") or "")
        geom = f["geometry"]
        if not name or not geom:
            continue
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        rings = []
        for poly in polys:
            outer = poly[0]
            if len(outer) < 4 or ring_area(outer) < MIN_AREA:
                continue
            rings.append(snap_ring(outer))
        if not rings:
            continue
        lons = [p[0] for r in rings for p in r]
        lats = [p[1] for r in rings for p in r]
        biggest = max(rings, key=ring_area)
        states.append({
            "id": IDS.get(name, name.lower().replace(" ", "-").replace("&", "and")),
            "name": name,
            "bbox": [round(min(lons), 3), round(min(lats), 3), round(max(lons), 3), round(max(lats), 3)],
            "centre": [round(sum(p[0] for p in biggest) / len(biggest), 3),
                       round(sum(p[1] for p in biggest) / len(biggest), 3)],
            "rings": rings,
        })

    states.sort(key=lambda s: s["name"])
    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {
        "source": "India state boundaries, mraxays/india-states.geojson (MIT)",
        "grid_deg": GRID,
        "states": states,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    pts = sum(len(r) for s in states for r in s["rings"])
    print(f"{len(states)} states, {pts:,} points, {os.path.getsize(OUT) / 1024:.0f} KB -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "states_india.geojson")
