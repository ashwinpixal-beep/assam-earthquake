"""Builds public/static/hazard/damage_zones.json: 2015 Gorkha damage by anonymised area.

Input : data/areas.json (31 anonymised geo_level_1 areas, held-out test split)
Output: public/static/hazard/damage_zones.json

Method: each area is ranked by its mean predicted probability of grade 3
(near-complete destruction). The 31 areas are split into terciles of that score:
red = top third, yellow = middle, green = bottom. The same thresholds are stored
so the 2D page and the API agree. This is a damage ranking from the 2015 Gorkha
survey, not a forecast for Assam and not a safety rating.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AREAS = os.path.join(ROOT, "data", "areas.json")
OUT = os.path.join(ROOT, "public", "static", "hazard", "damage_zones.json")

GRADE3_NAME = "Near-complete destruction"

areas = json.load(open(AREAS))
rows = areas["areas"]

scores = sorted(a["risk_score"] for a in rows)
n = len(scores)
t1 = scores[n // 3]           # yellow from
t2 = scores[(2 * n) // 3]     # red from

out_rows = []
counts = {"red": 0, "yellow": 0, "green": 0}
for a in sorted(rows, key=lambda r: -r["risk_score"]):
    s = a["risk_score"]
    zone = "red" if s > t2 else ("yellow" if s > t1 else "green")
    counts[zone] += 1
    out_rows.append({
        "area": f"Area {a['geo_level_1_id']}",
        "area_id": a["geo_level_1_id"],
        "zone": zone,
        "buildings_surveyed": a["buildings_test"],
        "g3_share_actual": a["actual_share"]["3"],
        "g3_share_predicted": a["predicted_share"]["3"],
        "mean_probability": a["mean_probability"],
        "accuracy": a["accuracy"],
        "risk_score": s,
    })

doc = {
    "title": "Gorkha 2015 earthquake damage by anonymised area",
    "source": "DrivenData Richter's Predictor, 2015 Nepal (Gorkha) earthquake survey",
    "basis": areas.get("basis", "held-out test split"),
    "grade_names": {
        "1": "Low damage",
        "2": "Medium damage",
        "3": GRADE3_NAME,
    },
    "method": ("Areas ranked by mean predicted probability of grade 3 (near-complete destruction); "
               "terciles of that score give red / yellow / green."),
    "thresholds_risk_score": {"yellow_from": round(t1, 4), "red_from": round(t2, 4)},
    "areas_total": len(out_rows),
    "buildings_total": sum(r["buildings_surveyed"] for r in out_rows),
    "counts": counts,
    "overall": areas.get("overall", {}),
    "areas": out_rows,
    "disclaimer": ("Reported damage from the 2015 Gorkha earthquake, Nepal. Areas are anonymised survey "
                   "IDs (geo level 1) and are not Assam districts. This is not a forecast and not a "
                   "structural or safety assessment."),
}

with open(OUT, "w") as f:
    json.dump(doc, f, indent=1)

print(json.dumps({"areas": len(out_rows), "counts": counts,
                  "thresholds": doc["thresholds_risk_score"]}, indent=1))
