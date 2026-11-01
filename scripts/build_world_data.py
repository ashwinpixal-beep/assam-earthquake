#!/usr/bin/env python3
"""Build the static world datasets used by the /globe and /world pages.

Inputs (installed with npm, not committed):
    npm i world-atlas all-the-cities

Outputs (committed, so the app needs no build step):
    public/static/world/countries.json    Natural Earth 1:110m countries, as lon/lat rings
    public/static/world/cities.json       every city with population >= 100,000
    data/quakes/world_catalogue.json      global M5+ 2016-2025 plus M6+ since 1900

The bundled catalogue is only used when the live USGS feed cannot be reached; see
world.py. Source: USGS via the Live Atlas "earthquake-catalogs" release (CC BY 4.0).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import time

EPOCH = datetime.date(1900, 1, 1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "public", "static", "world")
QUAKE_DIR = os.path.join(ROOT, "data", "quakes")
CSV = os.path.join(QUAKE_DIR, "india_quakes_m4_2000_2026.csv")


def decode_topojson(topo: dict, obj: str) -> list[dict]:
    """Decode a quantised TopoJSON object into [{name, id, rings:[[[lon,lat],...]]}]."""
    tr = topo.get("transform")
    sx, sy = tr["scale"] if tr else (1, 1)
    tx, ty = tr["translate"] if tr else (0, 0)

    arcs = []
    for arc in topo["arcs"]:
        x = y = 0
        pts = []
        for dx, dy in arc:
            x += dx
            y += dy
            pts.append([round(x * sx + tx, 3), round(y * sy + ty, 3)])
        arcs.append(pts)

    def ring(idxs):
        pts = []
        for i in idxs:
            a = arcs[i] if i >= 0 else arcs[~i][::-1]
            pts.extend(a if not pts else a[1:])
        # close the ring
        if pts and pts[0] != pts[-1]:
            pts.append(pts[0])
        return pts

    out = []
    for geom in topo["objects"][obj]["geometries"]:
        name = (geom.get("properties") or {}).get("name", "")
        polygons = []
        if geom["type"] == "Polygon":
            polygons = [geom["arcs"]]
        elif geom["type"] == "MultiPolygon":
            polygons = geom["arcs"]
        rings = []
        for poly in polygons:
            for i, r in enumerate(poly):
                pts = ring(r)
                if len(pts) < 5:
                    continue
                # drop specks: keep outer rings and holes above a rough area threshold
                lons = [p[0] for p in pts]
                lats = [p[1] for p in pts]
                if (max(lons) - min(lons)) * (max(lats) - min(lats)) < 0.01 and i == 0:
                    continue
                rings.append(pts)
        if rings:
            out.append({"id": geom.get("id"), "name": name, "rings": rings})
    out.sort(key=lambda c: c["name"])
    return out


def build_countries(npm: str) -> dict:
    src = os.path.join(npm, "node_modules", "world-atlas", "countries-110m.json")
    topo = json.load(open(src))
    countries = decode_topojson(topo, "countries")
    pts = sum(len(r) for c in countries for r in c["rings"])
    doc = {
        "generated": time.strftime("%Y-%m-%d"),
        "source": "Natural Earth 1:110m admin-0 countries, via the world-atlas npm package (2.0.2)",
        "licence": "Natural Earth is public domain",
        "countries": countries,
        "stats": {"countries": len(countries), "points": pts},
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "countries.json")
    json.dump(doc, open(path, "w"), separators=(",", ":"))
    print(f"countries.json  {len(countries)} countries, {pts} points, {os.path.getsize(path) // 1024} KB")
    return doc


def build_cities(npm: str, min_pop: int = 100_000) -> dict:
    import subprocess

    script = (
        "const cities=require('all-the-cities');"
        f"const out=cities.filter(c=>c.population>={min_pop}).map(c=>({{"
        "name:c.name,country:c.country,lat:+c.loc.coordinates[1].toFixed(3),"
        "lon:+c.loc.coordinates[0].toFixed(3),pop:c.population}));"
        "out.sort((a,b)=>b.pop-a.pop);console.log(JSON.stringify(out));"
    )
    raw = subprocess.run(["node", "-e", script], cwd=npm, capture_output=True, text=True, check=True).stdout
    rows = json.loads(raw)
    doc = {
        "generated": time.strftime("%Y-%m-%d"),
        "source": "GeoNames cities (all-the-cities npm package), population >= %d" % min_pop,
        "licence": "GeoNames data, CC BY 4.0",
        "cities": rows,
        "stats": {"cities": len(rows)},
    }
    path = os.path.join(OUT_DIR, "cities.json")
    json.dump(doc, open(path, "w"), separators=(",", ":"))
    print(f"cities.json     {len(rows)} cities, {os.path.getsize(path) // 1024} KB")
    return doc


LIVE_ATLAS = "https://codeload.github.com/liveatlasearth/earthquake-catalogs/tar.gz/refs/heads/main"
CATALOGUE = os.path.join(QUAKE_DIR, "world_catalogue.json")


def _download_catalogue(dest: str) -> str:
    """Fetch the Live Atlas USGS release (CC BY 4.0) and unpack the two world files."""
    import tarfile
    import tempfile
    import urllib.request

    if os.path.isdir(dest) and os.listdir(dest):
        return dest
    os.makedirs(dest, exist_ok=True)
    tmp = os.path.join(tempfile.gettempdir(), "eqcat.tar.gz")
    print("downloading", LIVE_ATLAS)
    urllib.request.urlretrieve(LIVE_ATLAS, tmp)
    with tarfile.open(tmp) as tf:
        for m in tf.getmembers():
            if m.name.endswith((".csv", "LICENSE", "README.md")) and "/data/" in m.name or m.name.endswith(("LICENSE",)) or m.name.endswith("README.md"):
                if m.isfile():
                    f = tf.extractfile(m)
                    if not f:
                        continue
                    name = os.path.basename(m.name)
                    target = os.path.join(dest, ("LICENSE.txt" if name == "LICENSE" else m.name.split("/")[-1]))
                    if name.startswith("world_") or name in ("LICENSE", "README.md"):
                        with open(target, "wb") as out:
                            out.write(f.read())
    return dest


def build_catalogue(src: str | None = None) -> dict:
    """Merge the two world catalogues into one compact JSON bundle."""
    import csv

    src = src or _download_catalogue("/tmp/eqcat/earthquake-catalogs-main")
    files = [
        os.path.join(src, "data", "world_m5_plus_2016_2025.csv"),
        os.path.join(src, "data", "world_m6_plus_since_1900.csv"),
    ]
    seen, rows = set(), []
    for path in files:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rid = r.get("usgs_id") or ""
                if rid and rid in seen:
                    continue
                if rid:
                    seen.add(rid)
                try:
                    lon = round(float(r["longitude"]), 3)
                    lat = round(float(r["latitude"]), 3)
                    mag = round(float(r["magnitude"]), 1)
                except (TypeError, ValueError, KeyError):
                    continue
                if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                    continue
                try:
                    depth = round(float(r["depth_km"]), 0) if r.get("depth_km") else None
                except ValueError:
                    depth = None
                date = (r.get("time") or "")[:10]
                try:
                    day = (datetime.date.fromisoformat(date) - EPOCH).days
                except ValueError:
                    continue
                # compact encoding: thousandths of a degree, tenths of a magnitude,
                # integer depth (-1 when unknown), days since 1900-01-01,
                # place name only for M6.5+ so the bundle stays small
                rows.append([
                    int(round(lon * 1000)), int(round(lat * 1000)), int(round(mag * 10)),
                    int(round(depth)) if depth is not None else -1,
                    day,
                    (r.get("place") or "").strip()[:64] if mag >= 6.5 else "",
                ])
    rows.sort(key=lambda e: e[4])
    doc = {
        "generated": time.strftime("%Y-%m-%d"),
        "source": "USGS earthquake catalogue, packaged by Live Atlas (earthquake-catalogs)",
        "coverage": "global",
        "licence": "CC BY 4.0 - credit Live Atlas (liveatlas.earth) and the USGS",
        "encoding": {
            "lon": "thousandths of a degree (int)",
            "lat": "thousandths of a degree (int)",
            "mag": "tenths of a magnitude (int)",
            "depth_km": "integer, -1 when unknown",
            "day": "days since 1900-01-01",
            "place": "text for M6.5+ only, otherwise empty",
        },
        "columns": ["lon", "lat", "mag", "depth_km", "day", "place"],
        "sets": [
            {"name": "M5.0+ worldwide, 2016-2025", "file": os.path.basename(files[0])},
            {"name": "M6.0+ worldwide, 1900-2026", "file": os.path.basename(files[1])},
        ],
        "bbox": [min(r[0] for r in rows) / 1000, min(r[1] for r in rows) / 1000,
                 max(r[0] for r in rows) / 1000, max(r[1] for r in rows) / 1000],
        "events": rows,
        "stats": {
            "events": len(rows),
            "m6_plus": sum(1 for r in rows if r[2] >= 60),
            "m7_plus": sum(1 for r in rows if r[2] >= 70),
            "m8_plus": sum(1 for r in rows if r[2] >= 80),
            "first_date": (EPOCH + datetime.timedelta(days=rows[0][4])).isoformat(),
            "last_date": (EPOCH + datetime.timedelta(days=rows[-1][4])).isoformat(),
        },
    }
    os.makedirs(QUAKE_DIR, exist_ok=True)
    json.dump(doc, open(CATALOGUE, "w"), separators=(",", ":"))
    print(f"world_catalogue {len(rows)} events "
          f"({doc['stats']['m7_plus']} M7+, {doc['stats']['m8_plus']} M8+), "
          f"{os.path.getsize(CATALOGUE) // 1024} KB")
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npm", default="/tmp/wdata", help="folder with world-atlas and all-the-cities installed")
    ap.add_argument("--min-pop", type=int, default=100_000)
    ap.add_argument("--catalogue-src", default=None,
                    help="folder holding the unpacked Live Atlas release (data/world_*.csv)")
    args = ap.parse_args()
    build_countries(args.npm)
    build_cities(args.npm, args.min_pop)
    build_catalogue(args.catalogue_src)
