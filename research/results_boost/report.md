# Building Damage Grade Prediction: Full Comparison

Dataset: DrivenData *Richter's Predictor* (2015 Gorkha earthquake, Nepal), 260,601 buildings, grades 1 / 2 / 3.
Every model uses the same stratified 80/20 split (seed 42) and is scored once on the same 52,121 test buildings.
Early stopping for boosting uses a 10% validation slice carved from the training split only.

## Results

| Model | Accuracy | Macro F1 | Weighted F1 | Balanced acc. | F1 grade 1 | F1 grade 2 | F1 grade 3 |
|---|---|---|---|---|---|---|---|
| Baseline Categorical NB (your mid-review) | 69.85% | 0.662 | 0.703 | 0.699 | 0.560 | 0.730 | 0.697 |
| Improved NB (feature selection, 3 bins, α=0.1) | 70.38% | 0.667 | 0.707 | 0.700 | 0.567 | 0.738 | 0.696 |
| XGBoost (engineered features) | 75.02% | 0.701 | 0.746 | 0.678 | 0.605 | 0.796 | 0.700 |
| LightGBM (engineered features) | 75.32% | 0.708 | 0.750 | 0.691 | 0.622 | 0.798 | 0.705 |
| **LightGBM + XGBoost average** | 75.67% | 0.710 | 0.753 | 0.689 | 0.620 | 0.801 | 0.709 |

## Target check: 85% accuracy
Not reached, and not reachable honestly on this dataset. The best models here reach 75.7% accuracy.
The public DrivenData leaderboard for this competition tops out at roughly 0.75–0.76 micro-F1 (micro-F1 equals accuracy for single-label classification).
Reported results from other teams on the same data are in the same range, for example a public micro-F1 of 0.7515 (CatBoost + LightGBM ensemble) and a leader score of 0.7558.

## What drove the gains (NB 70.4% -> boosting 75.7%)
1. **Model class.** Gradient-boosted trees model feature interactions (e.g. foundation type x roof type x floors) that Naive Bayes treats as independent. This is the largest single gain (about +5 pp).
2. **Native handling of categoricals.** LightGBM uses categorical splits directly on the 8 text columns and the three geographic ID columns, with no one-hot blow-up.
3. **Geographic frequency features.** For each geo ID, the share of training buildings in the same area. This adds regional density information without leaking labels.
4. **Engineered ratios.** Height and area per floor, volume proxy, families per area, age x floors, and counts of superstructure and secondary-use flags.
5. **Early stopping on a validation slice of the training data.** This stops the trees growing too deep. The test set was never used for tuning.
6. **Ensembling.** Averaging LightGBM and XGBoost probabilities adds about +0.3 pp accuracy over LightGBM alone.

## What did not help
- Class balancing (oversampling) for Naive Bayes reduced macro F1.
- One-hot encoding the 11,000-value geo ID ran out of memory and gave no benefit, so the final models use native categoricals or plain codes instead.

## Limits
- Accuracy 75.7% is a single split. We did not repeat it across seeds, so the differences of a few tenths of a percent between boosting models are within noise.
- Grade 1 (the rarest class, 9.6%) remains the hardest, with F1 about 0.62.
- Earthquake damage depends on physics and construction quality that the survey only partly records, so a ceiling well below 85% is expected.

Files: `damage_boost.py` (boosting pipeline), `results_boost/boost_results.json` (all metrics), `damage_nb_compare.py` and `results/` (Naive Bayes comparison).
