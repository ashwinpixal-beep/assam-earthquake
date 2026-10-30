"""Relative earthquake hazard zones (red/yellow/green) from the USGS M>=4 catalogue, 2000-2026.

Method: gridded seismic energy (Gutenberg-Richter energy 10**(1.5*M)), smoothed with a Gaussian kernel,
then split into terciles of the score across grid cells inside Assam (red = top third, yellow = middle,
green = bottom). Cells outside Assam get the same thresholds. This ranks historical activity; it is not a forecast
and not a design-level seismic hazard map.
"""
import json, math
import numpy as np
import pandas as pd
from shapely.geometry import shape, Point, box, mapping
from shapely.strtree import STRtree

import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CAT = os.path.join(ROOT, "data", "quakes", "india_quakes_m4_2000_2026.csv")
BOUNDARY = os.path.join(ROOT, "public", "static", "assam3d", "assam_boundary.geojson")
OUT_DIR = os.path.join(ROOT, "public", "static", "hazard")
os.makedirs(OUT_DIR, exist_ok=True)

LON0, LON1, LAT0, LAT1 = 88.0, 97.5, 22.5, 28.5
CELL = 0.25
SIGMA = 0.35  # degrees, Gaussian smoothing radius

cat = pd.read_csv(CAT)
cat = cat[cat.longitude.between(LON0, LON1) & cat.latitude.between(LAT0, LAT1)].copy()
cat["energy"] = 10 ** (1.5 * cat.mag)

lons = np.arange(LON0, LON1, CELL)
lats = np.arange(LAT0, LAT1, CELL)
nx, ny = len(lons), len(lats)

# cell index for each event
ix = np.clip(((cat.longitude - LON0) / CELL).astype(int).values, 0, nx - 1)
iy = np.clip(((cat.latitude - LAT0) / CELL).astype(int).values, 0, ny - 1)
E = np.zeros((ny, nx)); N = np.zeros((ny, nx)); M = np.zeros((ny, nx))
for x, y, e, m in zip(ix, iy, cat.energy.values, cat.mag.values):
    E[y, x] += e; N[y, x] += 1; M[y, x] = max(M[y, x], m)

# Gaussian smoothing of energy (separable convolution, done with numpy)
r = int(math.ceil(3 * SIGMA / CELL))
k = np.exp(-0.5 * (np.arange(-r, r + 1) * CELL / SIGMA) ** 2); k /= k.sum()
def smooth(a):
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, a)
S = smooth(E)
score = np.log10(S + 1e-9)

# Assam mask for the reference terciles
bnd = shape(json.load(open(BOUNDARY))["features"][0]["geometry"]) if json.load(open(BOUNDARY))["type"] == "FeatureCollection" else shape(json.load(open(BOUNDARY))["geometry"])
cen = [(lons[j] + CELL / 2, lats[i] + CELL / 2) for i in range(ny) for j in range(nx)]
inside = np.array([bnd.contains(Point(x, y)) for x, y in cen]).reshape(ny, nx)
ref = score[inside]
t1, t2 = np.quantile(ref, [1 / 3, 2 / 3])

feats = []
counts = {"red": 0, "yellow": 0, "green": 0}
for i in range(ny):
    for j in range(nx):
        s = score[i, j]
        cls = "red" if s > t2 else ("yellow" if s > t1 else "green")
        counts[cls] += 1
        x0, y0 = lons[j], lats[i]
        poly = box(x0, y0, x0 + CELL, y0 + CELL)
        feats.append({
            "type": "Feature",
            "properties": {
                "cls": cls,
                "score": round(float(s), 3),
                "events_m4plus": int(N[i, j]),
                "max_mag": round(float(M[i, j]), 1) if N[i, j] else None,
                "in_assam": bool(inside[i, j]),
            },
            "geometry": mapping(poly),
        })

grid = {"type": "FeatureCollection", "features": feats}
json.dump(grid, open(os.path.join(OUT_DIR, "hazard_grid.geojson"), "w"), separators=(",", ":"))

big = cat[cat.mag >= 5.0]
pts = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"mag": round(float(r_.mag), 1), "time": r_.time[:10], "depth": round(float(r_.depth), 1), "place": str(r_.place)},
     "geometry": {"type": "Point", "coordinates": [round(float(r_.longitude), 4), round(float(r_.latitude), 4)]}}
    for r_ in big.itertuples()]}
json.dump(pts, open(os.path.join(OUT_DIR, "events_m5plus.geojson"), "w"), separators=(",", ":"))

meta = {
    "title": "Relative earthquake hazard, Assam and neighbouring NE India",
    "source": "USGS earthquake catalogue (M4.0+), 2000-01-01 to 2026-10-26, from terranetra_quake/data/india_quakes_m4_2000_2026.csv",
    "events_in_box": int(len(cat)),
    "events_m5_plus": int(len(big)),
    "box": {"lon": [LON0, LON1], "lat": [LAT0, LAT1]},
    "cell_deg": CELL,
    "smoothing_sigma_deg": SIGMA,
    "thresholds_log10_energy": {"yellow_from": round(float(t1), 3), "red_from": round(float(t2), 3)},
    "cells": counts,
    "method": "Gutenberg-Richter energy 10^(1.5M) per 0.25 deg cell, Gaussian-smoothed; terciles of the score across cells inside Assam.",
    "disclaimer": "Ranks where past earthquakes of magnitude 4 and above have clustered. It is not a forecast, not a safety rating, and not a seismic design map. The catalogue is short and uneven, and magnitude-4 completeness varies by region.",
}
json.dump(meta, open(os.path.join(OUT_DIR, "hazard_meta.json"), "w"), indent=1)
print(json.dumps({k: meta[k] for k in ["events_in_box", "events_m5_plus", "thresholds_log10_energy", "cells"]}, indent=1))
