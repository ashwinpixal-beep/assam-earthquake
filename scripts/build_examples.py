# Builds data/examples.json: 300 real held-out survey records (seed 2026) with true grades.
# Inputs: train_values.csv and train_labels.csv from the DrivenData competition page
# (not redistributed here). The saved model and data/options.json must exist.
#
#   python scripts/build_examples.py --values train_values.csv --labels train_labels.csv
import argparse, json, os, sys, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import model as dm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

_ap = argparse.ArgumentParser(description="Rebuild data/examples.json")
_ap.add_argument("--values", default="train_values.csv")
_ap.add_argument("--labels", default="train_labels.csv")
_ap.add_argument("--out", default=os.path.join(ROOT, "data", "examples.json"))
_args = _ap.parse_args()

RAW_VALUES = _args.values
RAW_LABELS = _args.labels
OUT = _args.out

raw, y = dm.load_train(RAW_VALUES, RAW_LABELS)
idx = np.arange(len(raw))
tr_i, te_i = train_test_split(idx, test_size=0.2, stratify=y, random_state=dm.SEED)
raw_te = raw.iloc[te_i].reset_index(drop=True)
y_te = y[te_i]
ids_te = pd.read_csv(RAW_VALUES).iloc[te_i][dm.ID_COL].astype(int).values

bundle = dm.DamageModel.load(os.path.join(ROOT, "models", "damage_model.joblib"))
proba = bundle.predict_proba(raw_te)
pred = proba.argmax(axis=1)
acc = accuracy_score(y_te, pred)
print("held-out test rows:", len(raw_te), "accuracy from bundle:", round(acc, 4))

opts = json.load(open(os.path.join(ROOT, "data", "options.json")))
names = [f["name"] for f in opts["fields"]]
assert set(names) <= set(raw.columns), (set(names) - set(raw.columns))

rng = np.random.default_rng(2026)
pick = rng.choice(len(raw_te), size=300, replace=False)

def native(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v) if not float(v).is_integer() else int(v)
    return str(v)

rows = []
for i in pick:
    payload = {n: native(raw_te.iloc[i][n]) for n in names}
    rows.append({
        "record": int(ids_te[i]),
        "payload": payload,
        "actual": int(y_te[i]) + 1,
    })

out = {
    "description": "Real records from the held-out 20% test split of the 2015 Gorkha survey (seed 42, stratified). Each record has its true damage grade. The model sees only the survey inputs.",
    "test_rows": int(len(raw_te)),
    "test_accuracy": round(float(acc), 4),
    "majority_accuracy": round(float((y_te == np.bincount(y_te).argmax()).mean()), 4),
    "rows": rows,
}
with open(OUT, "w") as f:
    json.dump(out, f, separators=(",", ":"))
print("wrote", OUT, os.path.getsize(OUT), "bytes; sample actual counts:", pd.Series([r["actual"] for r in rows]).value_counts().sort_index().to_dict())
