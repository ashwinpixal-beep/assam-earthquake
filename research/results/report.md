# Building Damage Grade Prediction: Naive Bayes Baseline vs Improved Model

Dataset: DrivenData *Richter's Predictor: Modeling Earthquake Damage* (2015 Gorkha earthquake, Nepal).
260,601 buildings, 38 features, damage grade 1 / 2 / 3 (9.6% / 56.9% / 33.5%).

## Protocol
- Stratified 80/20 split, seed 42: 208,480 train / 52,121 test.
- Every choice (feature subset, bins, smoothing, balancing) was selected by 3–5 fold stratified CV on the **training split only**.
- The test split was scored once per model.

## Results (test set, 52,121 buildings)

| Model | Accuracy | Macro F1 | Weighted F1 | Balanced acc. | F1 grade 1 | F1 grade 2 | F1 grade 3 |
|---|---|---|---|---|---|---|---|
| Baseline Categorical NB (mid-review setup) | 69.85% | 0.662 | 0.703 | 0.699 | 0.560 | 0.730 | 0.697 |
| + Feature selection (15 features, MI) | 70.21% | 0.662 | 0.706 | 0.692 | 0.555 | 0.738 | 0.694 |
| **Improved NB** (15 features, 3 bins, α=0.1, no balancing) | **70.38%** | **0.667** | **0.708** | **0.700** | 0.567 | 0.738 | 0.696 |
| Base paper P4: SMOTE-XGBoost (Chen & Zhang 2022), same Nepal data | — | 0.729 | — | — | — | — | — |

Baseline reproduction: the mid-review reported 69.93% accuracy and macro F1 0.66. Our rerun gives 69.85% and 0.662, within normal split-to-split variation.

## Improvements (what helped and what did not)
1. **Feature selection by mutual information** kept 15 of 38 features. Accuracy rose slightly (+0.4 pp) with a much smaller model. Macro F1 did not change. The top features were `land_surface_condition`, `height_percentage`, `area_percentage`, `roof_type`, and `other_floor_type`.
2. **Coarser discretisation and stronger smoothing** (3 bins, α = 0.1) gave the best CV score. This added +0.005 macro F1 on test.
3. **Class balancing did not help.** Random oversampling and uniform class priors both reduced CV macro F1 (e.g. 0.658 → 0.636 with oversampling). With Naive Bayes, duplicating minority rows distorts the class-conditional estimates more than it helps. So the grader's preference for grade 1 is better handled by the decision threshold than by resampling.

## Interpretation
- Improvements over the baseline are small: +0.005 macro F1 and +0.5 pp accuracy. We did not run repeated splits, so we cannot claim they are statistically significant.
- The gap to P4 (0.667 vs 0.729 macro F1) is mostly model class. P4 uses gradient-boosted trees, which can model feature interactions that Naive Bayes assumes away. This comparison is a reference point, not a controlled test, because P4 used its own split and preprocessing.
- Naive Bayes stays interpretable: each feature contributes a per-grade probability that can be shown to the user.

## Possible next steps
- Run the same protocol with XGBoost to get a like-for-like comparison with P4.
- Repeat the split (e.g. 5 seeds) to put error bars on the Naive Bayes results.

Files: `comparison.json` (all metrics, confusion matrices, CV grid, mutual information), `damage_nb_compare.py` (pipeline).
